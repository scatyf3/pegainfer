#!/usr/bin/env python3
"""Side-by-side per-token breakdown of two EAGLE-3 nsys traces (pegainfer vs vLLM).

Each trace must be an ``nsys export --type sqlite`` of a run captured with
``--capture-range=cudaProfilerApi`` (so only the measured decode loop is in it)
and ``--cuda-graph-trace=node`` (so kernels inside CUDA graphs are visible).
The matching summary JSON gives the number of generated tokens in that range,
which every row is normalized by.

Rows: wall / GPU-busy / GPU-idle per token, kernel time per category, kernel and
graph launches, D2H copies and host-blocking sync calls per token, idle gaps by
size bucket (sub-3 µs = launch-bound, >20 µs = host work or sync), and the
largest gaps with the kernels on either side. Optionally folds in per-round
acceptance parsed from the pegainfer test log (``accepted_draft=N`` lines).

Usage:
  eagle3_gap_report.py --trace pegainfer=ours.sqlite:pegainfer_on.json \
                       --trace vllm=vllm.sqlite:vllm_on_profiled.json \
                       [--ours-log ours_ab.log --gamma 3] > report.md
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = [
    # order matters: first match wins
    ("attn.decode", r"decode|single_decode|BatchDecode|paged_attention_v|flash_fwd_splitkv|unified_attention"),
    ("attn.prefill", r"prefill|fmha|flash_fwd|flash_attn|BatchPrefill|attention|attn"),
    ("gemm", r"gemm|Gemm|GEMM|cutlass|cublas|nvjet|xmma|sm\d+_.*_tn|ampere_|hopper_|wgmma|matmul"),
    ("norm", r"rms|norm|Norm"),
    ("sample/argmax", r"argmax|topk|top_k|sampl|softmax|Softmax|reduce.*max|arg_max|rejection"),
    ("rope", r"rope|rotary|Rotary"),
    ("kv/copy", r"copy|Copy|reshape_and_cache|kv|scatter|gather|index|memcpy|memset"),
    ("elementwise", r"elementwise|vectorized|add|mul|silu|swiglu|act_and_mul|cast|fill|triton"),
]
CAT_RE = [(name, re.compile(rx)) for name, rx in CATEGORIES]
SYNC_RE = re.compile(r"Synchronize|^cudaMemcpy(_v\d+)?$|^cuMemcpyDtoH(_v\d+)?$|^cudaMemcpy2D|cudaEventQuery|cuStreamQuery")
LAUNCH_RE = re.compile(r"LaunchKernel|cuLaunch")
GRAPH_RE = re.compile(r"GraphLaunch")


def categorize(name: str) -> str:
    for cat, rx in CAT_RE:
        if rx.search(name):
            return cat
    return "other"


def strings(conn: sqlite3.Connection) -> dict[int, str]:
    return {i: s for i, s in conn.execute("select id, value from StringIds")}


def has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("select 1 from sqlite_master where type='table' and name=?", (name,)).fetchone() is not None


def analyze(path: Path, tokens: int) -> dict:
    conn = sqlite3.connect(path)
    s = strings(conn)
    cols = [r[1] for r in conn.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
    name_col = "demangledName" if "demangledName" in cols else "shortName"
    kernels = [
        (start, end, s.get(nid, "?"))
        for start, end, nid in conn.execute(f"select start, end, {name_col} from CUPTI_ACTIVITY_KIND_KERNEL order by start")
    ]
    if not kernels:
        raise SystemExit(f"{path}: no kernels — was the capture range empty?")
    t0, t1 = kernels[0][0], max(k[1] for k in kernels)
    wall = t1 - t0

    by_cat: dict[str, int] = defaultdict(int)
    by_name: dict[str, int] = defaultdict(int)
    busy = 0
    cur_end = t0
    gaps: list[tuple[int, str, str]] = []
    last_name = ""  # kernel that currently ends latest
    gap_by_next_cat: dict[str, int] = defaultdict(int)
    for start, end, name in kernels:
        cat = categorize(name)
        by_cat[cat] += end - start
        by_name[name] += end - start
        if start > cur_end:
            g = start - cur_end
            gaps.append((g, last_name, name))
            gap_by_next_cat[cat] += g
            busy += end - start
            cur_end, last_name = end, name
        elif end > cur_end:
            busy += end - cur_end
            cur_end, last_name = end, name
    idle = wall - busy

    buckets = Counter()
    bucket_ns = defaultdict(int)
    for g, _, _ in gaps:
        b = "<3us" if g < 3_000 else "3-20us" if g < 20_000 else "20-100us" if g < 100_000 else ">100us"
        buckets[b] += 1
        bucket_ns[b] += g

    d2h_n = d2h_ns = 0
    if has_table(conn, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for n_, ns in conn.execute(
            "select count(*), coalesce(sum(end-start),0) from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=2 and start>=? and end<=?",
            (t0, t1),
        ):
            d2h_n, d2h_ns = n_, ns

    api = Counter()
    api_ns = defaultdict(int)
    if has_table(conn, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        for nid, dur in conn.execute(
            "select nameId, end-start from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=?", (t0, t1)
        ):
            n = s.get(nid, "")
            for key, rx in (("sync", SYNC_RE), ("launch", LAUNCH_RE), ("graph", GRAPH_RE)):
                if rx.search(n):
                    api[key] += 1
                    api_ns[key] += dur

    per = 1e-3 / tokens  # ns -> us per token
    return {
        "tokens": tokens,
        "wall_us": wall * per,
        "busy_us": busy * per,
        "idle_us": idle * per,
        "cat_us": {k: v * per for k, v in by_cat.items()},
        "gap_next_cat_us": {k: v * per for k, v in gap_by_next_cat.items()},
        "kernels_per_tok": len(kernels) / tokens,
        "kernel_launch_api_per_tok": api["launch"] / tokens,
        "graph_launch_per_tok": api["graph"] / tokens,
        "sync_calls_per_tok": api["sync"] / tokens,
        "sync_us": api_ns["sync"] * per,
        "d2h_per_tok": d2h_n / tokens,
        "d2h_us": d2h_ns * per,
        "gap_buckets": {b: (buckets[b] / tokens, bucket_ns[b] * per) for b in ("<3us", "3-20us", "20-100us", ">100us")},
        "top_kernels": sorted(((v * per, k) for k, v in by_name.items()), reverse=True)[:12],
        "top_gaps": sorted(gaps, reverse=True)[:8],
    }


def parse_ours_log(path: Path, gamma: int) -> dict:
    acc = [int(m.group(1)) for m in re.finditer(r"accepted_draft=(\d+)", path.read_text(errors="replace"))]
    if not acc:
        return {}
    n = len(acc)
    return {
        "num_rounds": n,
        "tau": 1.0 + sum(acc) / n,
        "accept_rate": sum(acc) / (n * gamma),
        "accept_per_pos": [sum(a > i for a in acc) / n for i in range(gamma)],
    }


def short(name: str, n: int = 70) -> str:
    name = re.sub(r"\(.*", "", name)
    return name if len(name) <= n else name[: n - 1] + "…"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="append", required=True, help="label=trace.sqlite:summary.json")
    ap.add_argument("--ab", action="append", default=[], help="label=summary.json of the un-profiled A/B run")
    ap.add_argument("--ours-log", type=Path)
    ap.add_argument("--gamma", type=int, default=3)
    args = ap.parse_args()

    labels, res, summ = [], [], []
    for t in args.trace:
        label, rest = t.split("=", 1)
        sqlite, summary = rest.rsplit(":", 1)
        sj = json.loads(Path(summary).read_text())
        labels.append(label)
        summ.append(sj)
        res.append(analyze(Path(sqlite), sj["generated_tokens"]))

    def row(title, vals, fmt="{:.1f}"):
        print(f"| {title} | " + " | ".join(fmt.format(v) for v in vals) + " |")

    print(f"## EAGLE-3 per-token breakdown (γ={args.gamma}, bs=1, GSM8K, µs per generated token)\n")
    print("Trace rows are under nsys node tracing: compare proportions, not absolute µs.\n")
    print("| metric | " + " | ".join(labels) + " |")
    print("|---|" + "---|" * len(labels))
    # nsys --cuda-graph-trace=node inflates absolute time 30-60%: the real TPOT is
    # the un-profiled A/B; the trace is only for composition and proportions.
    ab = {}
    for a in args.ab:
        label, path = a.split("=", 1)
        if Path(path).exists():
            ab[label] = json.loads(Path(path).read_text())
    if ab:
        row("**TPOT ms, A/B without nsys**", [ab.get(l, {}).get("tpot_ms", float("nan")) for l in labels], "{:.2f}")
    row("TPOT ms under nsys (inflated)", [s["tpot_ms"] for s in summ], "{:.2f}")
    row("trace wall", [r["wall_us"] for r in res])
    row("GPU busy", [r["busy_us"] for r in res])
    row("GPU idle", [r["idle_us"] for r in res])
    cats = sorted({c for r in res for c in r["cat_us"]}, key=lambda c: -max(r["cat_us"].get(c, 0) for r in res))
    for c in cats:
        row(f"kernel: {c}", [r["cat_us"].get(c, 0.0) for r in res])
    for c in cats:
        vals = [r["gap_next_cat_us"].get(c, 0.0) for r in res]
        if max(vals) > 0.5:
            row(f"idle before {c}", vals)
    row("kernels / tok", [r["kernels_per_tok"] for r in res])
    row("cudaLaunchKernel / tok", [r["kernel_launch_api_per_tok"] for r in res])
    row("cudaGraphLaunch / tok", [r["graph_launch_per_tok"] for r in res])
    row("D2H copies / tok", [r["d2h_per_tok"] for r in res], "{:.2f}")
    row("host sync calls / tok", [r["sync_calls_per_tok"] for r in res], "{:.2f}")
    row("host sync time", [r["sync_us"] for r in res])
    for b in ("<3us", "3-20us", "20-100us", ">100us"):
        row(f"gaps {b}: count/tok", [r["gap_buckets"][b][0] for r in res], "{:.2f}")
        row(f"gaps {b}: µs/tok", [r["gap_buckets"][b][1] for r in res])

    accs = []
    if args.ours_log:
        accs.append(("pegainfer", parse_ours_log(args.ours_log, args.gamma)))
    for label, s in zip(labels, summ):
        src = ab.get(label, s)  # prefer the larger un-profiled A/B sample
        if src.get("engine") == "vllm" and "acceptance" in src:
            accs.append((label, src["acceptance"]))
    if accs:
        print("\n## Acceptance\n")
        print("| engine | rounds | τ (tokens/verify) | accept rate | per-pos accept |")
        print("|---|---|---|---|---|")
        for label, a in accs:
            if "tau" not in a:
                print(f"| {label} | n/a | | | |")
                continue
            pp = ", ".join(f"{x:.2f}" for x in (a.get("accept_per_pos") or []))
            print(f"| {label} | {a['num_rounds']} | {a['tau']:.2f} | {a['accept_rate']:.3f} | {pp} |")

    for label, r in zip(labels, res):
        print(f"\n## {label}: top kernels (µs/tok)\n")
        for us, name in r["top_kernels"]:
            print(f"- {us:7.1f}  `{short(name)}` ({categorize(name)})")
        print(f"\n{label}: largest idle gaps\n")
        for g, prev, nxt in r["top_gaps"]:
            print(f"- {g / 1e3:8.1f} µs  after `{short(prev, 40)}` → before `{short(nxt, 40)}`")


if __name__ == "__main__":
    main()
