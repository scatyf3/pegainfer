#!/usr/bin/env python3
"""Aggregate N repeated toolcall_itl.py runs into mean +/- std tables.

toolcall_itl.py measures one (dataset, concurrency) cell once. To turn a single
observation into a quotable number we run it N times -- each run warms both
conditions independently and scrapes /metrics around its own window, so the runs
are statistically independent -- then average here.

Input: a directory (or glob) of files named  <dataset>_c<c>.run<k>.json .
The (dataset, concurrency) key comes from each file's own contents, not its
name, so any naming scheme works as long as one file == one run of one cell.

Deltas are computed PER RUN and then averaged (paired), so the reported delta
std reflects run-to-run stability of the effect, not the spread of the two
absolute levels. That is the number to check before quoting a percentage.

Usage:
    python scripts/toolcall_itl_agg.py docs/benchmarks/data/toolcall-itl-5x/
"""
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

TTFT = "vllm:time_to_first_token_seconds"
TPOT = "vllm:request_time_per_output_token_seconds"
QUEUE = "vllm:request_queue_time_seconds"
PREFILL = "vllm:request_prefill_time_seconds"
INFER = "vllm:request_inference_time_seconds"


def sv(cond, hist, key="mean_ms"):
    stats = (cond.get("server") or {}).get(hist)
    return stats.get(key) if stats else None


def tok_per_req(cond):
    return cond["tok_s"] / cond["req_s"] if cond.get("req_s") else None


def load_runs(paths):
    """(dataset, c) -> list of {'off': summary, 'required': summary} per run."""
    cells = defaultdict(list)
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            report = json.load(fh)
        dataset = os.path.basename(report["dataset"]).replace(".jsonl", "")
        by_cond = {c["cond"]: c for c in report["conditions"]}
        cells[(dataset, report["concurrency"])].append(by_cond)
    return cells


def stat(xs):
    """mean, sample-std of a list, ignoring Nones."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    mean = statistics.mean(xs)
    std = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return mean, std


def paired_delta_pct(runs, extract):
    """Per-run off->required percent change, then mean/std across runs."""
    deltas = []
    for run in runs:
        a = extract(run["off"])
        b = extract(run.get("required", {}))
        if a and b:
            deltas.append((b - a) / a * 100.0)
    return stat(deltas)


def fmt(mean, std, prec=1, unit=""):
    if mean is None:
        return "-"
    return f"{mean:.{prec}f}±{std:.{prec}f}{unit}"


def fmt_pct(mean, std):
    if mean is None:
        return "-"
    return f"{mean:+.0f}±{std:.0f}%"


def cell_row(runs, extract, prec, unit="", min_base=0.0):
    off_m, off_s = stat([extract(r["off"]) for r in runs])
    req_m, req_s = stat([extract(r.get("required", {})) for r in runs])
    # Percent change off a ~0 baseline (e.g. queue time at high concurrency) is
    # numerically meaningless -- show absolute levels only.
    if off_m is not None and off_m < min_base:
        d = "  n/a"
    else:
        d = fmt_pct(*paired_delta_pct(runs, extract))
    return (f"{fmt(off_m, off_s, prec, unit):>14} ->{fmt(req_m, req_s, prec, unit):>14}"
            f"{d:>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="directory of, or glob for, per-run JSON files")
    args = ap.parse_args()

    paths = (sorted(glob.glob(os.path.join(args.path, "*.json")))
             if os.path.isdir(args.path) else sorted(glob.glob(args.path)))
    if not paths:
        raise SystemExit(f"no JSON found at {args.path}")
    cells = load_runs(paths)
    keys = sorted(cells, key=lambda k: (k[0], k[1]))

    n_runs = {k: len(v) for k, v in cells.items()}
    print(f"Aggregated over {min(n_runs.values())}-{max(n_runs.values())} runs "
          f"per cell (mean +/- sample std; delta is paired per-run).\n")

    metrics = [
        ("TTFT mean (ms)", lambda c: sv(c, TTFT), 1, ""),
        ("TPOT mean (ms)", lambda c: sv(c, TPOT), 2, ""),
        ("req/s", lambda c: c.get("req_s"), 1, ""),
        ("tok/req", tok_per_req, 1, ""),
    ]
    for title, extract, prec, unit in metrics:
        print(f"## {title}")
        hdr = f"{'dataset':<14}{'c':>3}{'runs':>5}|{'off -> required':>28}{'delta':>10}"
        print(hdr)
        print("-" * len(hdr))
        for key in keys:
            runs = cells[key]
            print(f"{key[0]:<14}{key[1]:>3}{len(runs):>5}|{cell_row(runs, extract, prec, unit)}")
        print()

    # Mechanism split, means only (the interesting one is inference under load).
    print("## Mechanism (mean ms, off -> required)")
    hdr = f"{'dataset':<14}{'c':>3}|{'queue':>22}|{'prefill':>22}|{'inference':>24}"
    print(hdr)
    print("-" * len(hdr))
    for key in keys:
        runs = cells[key]
        q = cell_row(runs, lambda c: sv(c, QUEUE), 2, min_base=0.5)
        p = cell_row(runs, lambda c: sv(c, PREFILL), 2)
        i = cell_row(runs, lambda c: sv(c, INFER), 1)
        print(f"{key[0]:<14}{key[1]:>3}|{q}|{p}|{i}")


if __name__ == "__main__":
    main()
