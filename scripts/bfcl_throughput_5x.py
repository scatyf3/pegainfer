#!/usr/bin/env python3
"""Robust throughput A/B: per (category, constraint), discard 1 warmup pass then
run 5 measured passes, reporting req/s and tok/s mean +/- std. Warmup absorbs
vLLM's one-time lazy init; the 5-run std quantifies short-run variance the single
pass hid. Greedy (temp 0) so accuracy is fixed and only timing varies.
"""
import json
import statistics
import subprocess
import sys

PY = ".venv/bin/python"
URL = "http://127.0.0.1:8000"
MODEL = "Qwen/Qwen3-0.6B"
EXTRA = '{"chat_template_kwargs": {"enable_thinking": false}}'
WARMUP = 1
MEASURED = 5

CATS = {
    "simple_python": "bfcl_data/scenarios/simple_python.jsonl",
    "multiple": "bfcl_data/scenarios/multiple.jsonl",
}


def one_pass(dataset, cond):
    out = subprocess.run(
        [PY, "scripts/toolcall_trace.py", "--base-url", URL, "--model", MODEL,
         "--dataset", dataset, "--repeat", "1", "--concurrency", "16",
         "--temperature", "0", "--max-tokens", "256", "--extra-body", EXTRA,
         "--constrained", cond, "--quiet"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        sys.stderr.write(out.stderr[-2000:])
        raise SystemExit(f"harness failed ({cond}): rc={out.returncode}")
    return json.loads(out.stdout)


def ms(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main():
    hdr = f"{'category':<16}{'cond':<10}{'acc':>7}{'req/s (mean±sd)':>20}{'tok/s (mean±sd)':>20}{'wall_s':>9}"
    print(hdr)
    print("-" * len(hdr))
    for cat, ds in CATS.items():
        for cond in ("off", "required"):
            for _ in range(WARMUP):
                one_pass(ds, cond)
            rps, tps, walls, acc = [], [], [], None
            for _ in range(MEASURED):
                s = one_pass(ds, cond)
                rps.append(s["throughput"]["requests_per_s"])
                tps.append(s["throughput"]["completion_tokens_per_s"])
                walls.append(s["throughput"]["wall_s"])
                acc = s["accuracy"]  # deterministic under greedy
            rm, rsd = ms(rps)
            tm, tsd = ms(tps)
            print(f"{cat:<16}{cond:<10}{acc*100:>6.1f}%"
                  f"{f'{rm:.1f}±{rsd:.1f}':>20}{f'{tm:.0f}±{tsd:.0f}':>20}"
                  f"{statistics.mean(walls):>9.1f}")
    print(f"\n(warmup passes discarded: {WARMUP}/group; measured: {MEASURED}/group; "
          f"greedy temp=0; concurrency=16)")


if __name__ == "__main__":
    main()
