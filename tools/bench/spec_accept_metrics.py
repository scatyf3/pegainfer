#!/usr/bin/env python3
"""Derive spec-decode acceptance stats from the `/metrics` counters.

The `vllm:spec_decode_*` counters are cumulative over the *server's* lifetime,
so a per-case number is a difference of two scrapes. That is the whole reason
this tool exists: the previous acceptance study (`docs/models/qwen3/
dspark-integration.md`) recovered per-case rounds by grouping server-log lines
into benchmark time windows, which is the flakiest step in that method. Two
scrapes bracketing a case give exact boundaries instead.

`num_accepted_tokens_per_pos` is a *complementary CDF*, not a histogram:
`per_pos[i]` counts rounds that accepted **at least** `i+1` drafts. Differencing
adjacent positions recovers the accepted-length histogram, which is what the
prior study reported. `derive()` does that and then asserts the two identities
that must hold if the counters are sane:

    sum(hist)            == num_drafts
    sum(k * hist[k])     == num_accepted_tokens

Those are free correctness checks on the engine's counters — a violation is a
real bug, not a rounding artifact. (They are also exactly what breaks if the
drafter's K exceeds `MAX_SPEC_TOKENS`, since positions past the clamp land in
the scalar total but not the per-position vector.)

Subcommands:
    snapshot   scrape /metrics into a JSON blob
    diff       two snapshots -> one case's derived stats
    report     many case files -> the markdown tables
    selftest   run derive() against the published DSpark/DFlash numbers
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

# Counters are *registered* without the `_total` suffix (see the vLLM metrics
# crate, `rust/src/metrics/src/scheduler.rs`); `prometheus-client` appends it at
# exposition. Accept either spelling so a client-library change doesn't silently
# yield all-zero results.
SCALARS = {
    "drafts": "vllm:spec_decode_num_drafts",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens",
}
PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos"

SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>[0-9.eE+-]+)\s*$"
)
LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def _parse_labels(raw: str | None) -> dict[str, str]:
    return dict(LABEL_RE.findall(raw)) if raw else {}


def parse_metrics(text: str) -> dict:
    """Pull the spec-decode family out of a Prometheus exposition payload.

    Values are summed across `engine` labels. Speculative decoding is gated to
    the single-GPU path today, so there is normally exactly one engine; summing
    keeps the tool honest if that ever changes, and the engine set is recorded
    so a reader can tell an aggregate from a single-engine number.
    """
    scalars = {key: 0.0 for key in SCALARS}
    per_pos: dict[int, float] = {}
    engines: set[str] = set()
    seen_any = False

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        base = name[: -len("_total")] if name.endswith("_total") else name
        value = float(match.group("value"))
        labels = _parse_labels(match.group("labels"))

        for key, metric in SCALARS.items():
            if base == metric:
                scalars[key] += value
                seen_any = True
                if "engine" in labels:
                    engines.add(labels["engine"])
        if base == PER_POS:
            seen_any = True
            if "engine" in labels:
                engines.add(labels["engine"])
            position = labels.get("position")
            if position is None:
                continue
            per_pos[int(position)] = per_pos.get(int(position), 0.0) + value

    return {
        "found": seen_any,
        "engines": sorted(engines),
        "drafts": int(scalars["drafts"]),
        "draft_tokens": int(scalars["draft_tokens"]),
        "accepted_tokens": int(scalars["accepted_tokens"]),
        "per_pos": {str(k): int(v) for k, v in sorted(per_pos.items())},
    }


def scrape(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return parse_metrics(response.read().decode("utf-8"))


def derive(before: dict, after: dict, *, strict: bool = True) -> dict:
    """Difference two snapshots and recover the accepted-length histogram."""
    drafts = after["drafts"] - before["drafts"]
    draft_tokens = after["draft_tokens"] - before["draft_tokens"]
    accepted_tokens = after["accepted_tokens"] - before["accepted_tokens"]

    positions = sorted(int(p) for p in after["per_pos"])
    per_pos = [
        after["per_pos"].get(str(p), 0) - before["per_pos"].get(str(p), 0)
        for p in positions
    ]
    k = len(per_pos)

    problems: list[str] = []
    if drafts <= 0:
        # The usual cause is a non-greedy benchmark: `should_speculative_decode`
        # is all-or-nothing, so any non-greedy request in the batch drops the
        # whole step to plain decode and nothing ever drafts. Counters then read
        # a perfectly honest zero, which is easy to mistake for a wiring bug.
        problems.append(
            f"num_drafts delta is {drafts} — no verify step ran. "
            "Benchmark with --temperature 0; a non-greedy batch disables "
            "speculation entirely."
        )
    if any(b > a for a, b in zip(per_pos, per_pos[1:])):
        problems.append(f"per-position vector is not non-increasing: {per_pos}")

    hist: list[int] = []
    if drafts > 0 and k:
        hist = [drafts - per_pos[0]]
        hist += [per_pos[i - 1] - per_pos[i] for i in range(1, k)]
        hist.append(per_pos[k - 1])
        if any(count < 0 for count in hist):
            problems.append(f"negative histogram bin: {hist}")
        if sum(hist) != drafts:
            problems.append(f"sum(hist)={sum(hist)} != num_drafts={drafts}")
        weighted = sum(i * count for i, count in enumerate(hist))
        if weighted != accepted_tokens:
            problems.append(
                f"sum(k*hist[k])={weighted} != num_accepted_tokens="
                f"{accepted_tokens} — per-position and scalar totals disagree "
                "(expected if the drafter's K exceeds MAX_SPEC_TOKENS)"
            )

    if problems and strict:
        raise SystemExit("FATAL: " + "\n       ".join(problems))

    return {
        "rounds": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
        "k": k,
        "per_pos": per_pos,
        "hist": hist,
        # accepted / drafted — the same ratio dflash_lane.rs logs as
        # `cumulative_accept_rate`, so the two must agree exactly.
        "accept_rate": accepted_tokens / draft_tokens if draft_tokens else 0.0,
        # Excludes the bonus token, matching the prior study's `accepted_draft`;
        # committed tokens per round is this + 1.
        "mean_accepted_draft": accepted_tokens / drafts if drafts else 0.0,
        "zero_accept": hist[0] / drafts if hist and drafts else 0.0,
        "full_accept": hist[-1] / drafts if hist and drafts else 0.0,
        "engines": after.get("engines", []),
        "problems": problems,
    }


def _fmt_cell(stats: dict) -> str:
    return f"{stats['mean_accepted_draft']:.2f}"


def report(cells: list[dict]) -> str:
    """Render the two tables the prior acceptance study used."""
    out: list[str] = []
    by_config: dict[str, list[dict]] = {}
    for cell in cells:
        by_config.setdefault(cell.get("config", "?"), []).append(cell)

    out.append("Accepted-length distribution from the `/metrics` spec-decode")
    out.append("counters (per-case deltas of two scrapes). `accepted_draft`")
    out.append("excludes the bonus token, so `committed = accepted_draft + 1`.")
    out.append("")
    out.append(
        "| config | rounds | mean accepted draft | accept rate | "
        "zero-accept | full-K | hist 0..K |"
    )
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for config, group in by_config.items():
        rounds = sum(c["stats"]["rounds"] for c in group)
        accepted = sum(c["stats"]["accepted_tokens"] for c in group)
        drafted = sum(c["stats"]["draft_tokens"] for c in group)
        width = max(len(c["stats"]["hist"]) for c in group)
        hist = [0] * width
        for cell in group:
            for i, count in enumerate(cell["stats"]["hist"]):
                hist[i] += count
        mean = accepted / rounds if rounds else 0.0
        out.append(
            f"| {config} | {rounds:,} | {mean:.2f} | "
            f"{accepted / drafted if drafted else 0:.3f} | "
            f"{hist[0] / rounds if rounds else 0:.1%} | "
            f"{hist[-1] / rounds if rounds else 0:.1%} | `{hist}` |"
        )

    out.append("")
    out.append("Per-case mean accepted draft tokens:")
    out.append("")
    datasets = sorted({c.get("dataset", "?") for c in cells})
    concurrencies = sorted(
        {c.get("concurrency", 0) for c in cells}, key=lambda x: int(x)
    )
    configs = list(by_config)
    header = "| dataset | " + " | ".join(
        f"c{c} " + "/".join(configs) for c in concurrencies
    ) + " |"
    out.append(header)
    out.append("| --- |" + " --- |" * len(concurrencies))
    for dataset in datasets:
        row = [f"| {dataset} "]
        for concurrency in concurrencies:
            parts = []
            for config in configs:
                match = [
                    c
                    for c in cells
                    if c.get("dataset") == dataset
                    and c.get("concurrency") == concurrency
                    and c.get("config") == config
                ]
                parts.append(_fmt_cell(match[0]["stats"]) if match else "—")
            row.append("| " + " / ".join(parts) + " ")
        out.append("".join(row) + "|")
    return "\n".join(out)


def selftest() -> int:
    """Round-trip the published DSpark/DFlash numbers through derive().

    Rebuilds each histogram into the complementary-CDF shape the engine
    publishes, then checks derive() recovers the reported aggregates. This pins
    the CDF-differencing without needing a GPU, and it fails loudly if the
    per-position semantics are ever misread as a plain histogram.
    """
    published = {
        "DSpark": ([5636, 3942, 2394, 1549, 1042, 742, 639, 3350], 2.52, 0.292, 0.174),
        "DFlash": ([6838, 4340, 2685, 1648, 1071, 962, 731, 2939], 2.30, 0.322, 0.139),
    }
    failures = 0
    for name, (hist, mean, zero, full) in published.items():
        rounds = sum(hist)
        accepted = sum(i * c for i, c in enumerate(hist))
        drafted = rounds * (len(hist) - 1)  # every round drafts the full block
        # per_pos[i] = rounds accepting at least i+1 drafts.
        per_pos = [sum(hist[i + 1 :]) for i in range(len(hist) - 1)]
        zero_snap = {
            "drafts": 0,
            "draft_tokens": 0,
            "accepted_tokens": 0,
            "per_pos": {},
            "engines": [],
        }
        after = {
            "drafts": rounds,
            "draft_tokens": drafted,
            "accepted_tokens": accepted,
            "per_pos": {str(i): v for i, v in enumerate(per_pos)},
            "engines": ["0"],
        }
        stats = derive(zero_snap, after)
        checks = [
            ("hist", stats["hist"] == hist, stats["hist"], hist),
            ("rounds", stats["rounds"] == rounds, stats["rounds"], rounds),
            (
                "mean",
                abs(stats["mean_accepted_draft"] - mean) < 0.005,
                round(stats["mean_accepted_draft"], 4),
                mean,
            ),
            ("zero", abs(stats["zero_accept"] - zero) < 0.0005, round(stats["zero_accept"], 4), zero),
            ("full", abs(stats["full_accept"] - full) < 0.0005, round(stats["full_accept"], 4), full),
        ]
        for label, ok, got, want in checks:
            if not ok:
                print(f"FAIL {name}.{label}: got {got}, want {want}")
                failures += 1
        if not any(not ok for _, ok, _, _ in checks):
            print(
                f"ok   {name}: {rounds:,} rounds, mean {stats['mean_accepted_draft']:.4f}, "
                f"zero {stats['zero_accept']:.1%}, full {stats['full_accept']:.1%}"
            )
    print("selftest: " + ("PASS" if not failures else f"{failures} FAILURE(S)"))
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_snap = sub.add_parser("snapshot", help="scrape /metrics into JSON")
    p_snap.add_argument("--url", default="http://localhost:8000/metrics")
    p_snap.add_argument("--out", required=True, type=Path)
    p_snap.add_argument(
        "--allow-missing",
        action="store_true",
        help="do not fail when no spec_decode counters are exposed yet",
    )

    p_diff = sub.add_parser("diff", help="two snapshots -> one case's stats")
    p_diff.add_argument("--before", required=True, type=Path)
    p_diff.add_argument("--after", required=True, type=Path)
    p_diff.add_argument("--out", required=True, type=Path)
    p_diff.add_argument("--config", default="?")
    p_diff.add_argument("--dataset", default="?")
    p_diff.add_argument("--concurrency", default=0, type=int)
    p_diff.add_argument("--lenient", action="store_true", help="warn instead of failing")

    p_report = sub.add_parser("report", help="case files -> markdown tables")
    p_report.add_argument("cells", nargs="+", type=Path)

    sub.add_parser("selftest", help="validate derive() against published numbers")

    args = parser.parse_args()

    if args.cmd == "selftest":
        return selftest()

    if args.cmd == "snapshot":
        snap = scrape(args.url)
        if not snap["found"] and not args.allow_missing:
            raise SystemExit(
                f"FATAL: no vllm:spec_decode_* samples at {args.url}. Either no "
                "draft model is loaded, or the counters are not wired. Note the "
                "exposition name carries a `_total` suffix the registered name "
                "does not."
            )
        args.out.write_text(json.dumps(snap, indent=2))
        print(f"snapshot -> {args.out} (drafts={snap['drafts']})")
        return 0

    if args.cmd == "diff":
        before = json.loads(args.before.read_text())
        after = json.loads(args.after.read_text())
        stats = derive(before, after, strict=not args.lenient)
        cell = {
            "config": args.config,
            "dataset": args.dataset,
            "concurrency": args.concurrency,
            "stats": stats,
        }
        args.out.write_text(json.dumps(cell, indent=2))
        for problem in stats["problems"]:
            print(f"WARN: {problem}", file=sys.stderr)
        print(
            f"{args.config} {args.dataset} c{args.concurrency}: "
            f"rounds={stats['rounds']:,} mean_accepted={stats['mean_accepted_draft']:.3f} "
            f"accept_rate={stats['accept_rate']:.3f} hist={stats['hist']}"
        )
        return 0

    if args.cmd == "report":
        cells = [json.loads(path.read_text()) for path in args.cells]
        print(report(cells))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
