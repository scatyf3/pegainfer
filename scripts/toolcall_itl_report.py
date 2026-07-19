#!/usr/bin/env python3
"""Render the constrained-decoding latency tables from toolcall_itl.py output.

Reads the per-(dataset, concurrency) JSON files produced by toolcall_itl.py and
emits the four tables that make up the report:

  A. headline off -> required deltas, server-side (authoritative)
  B. mechanism breakdown: queue / prefill / inference
  C. TTFT distribution (the only latency histogram whose buckets resolve here)
  D. output-length inflation

Deliberately omits TPOT/ITL percentiles: those histograms bucket from 0.01s up,
so on a small model every sample lands in the first bucket and any quantile is
interpolation fiction. Means come from _sum/_count and stay exact.

Usage:
    python scripts/toolcall_itl_report.py docs/benchmarks/data/toolcall-itl/
"""
import argparse
import glob
import json
import os
import sys

TTFT = "vllm:time_to_first_token_seconds"
TPOT = "vllm:request_time_per_output_token_seconds"
ITL = "vllm:inter_token_latency_seconds"
QUEUE = "vllm:request_queue_time_seconds"
PREFILL = "vllm:request_prefill_time_seconds"
INFER = "vllm:request_inference_time_seconds"


def load(paths):
    data = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            report = json.load(fh)
        dataset = os.path.basename(report["dataset"]).replace(".jsonl", "")
        data[(dataset, report["concurrency"])] = {
            c["cond"]: c for c in report["conditions"]
        }
    return data


def sv(cond, hist, key="mean_ms"):
    stats = (cond.get("server") or {}).get(hist)
    return stats.get(key) if stats else None


def delta_pct(a, b):
    return (b - a) / a * 100 if (a and b) else None


def pc(x):
    return f"{x:+.0f}%" if isinstance(x, (int, float)) else "-"


def table_a(data, keys):
    print("A. Headline -- server-side authoritative (off -> required)")
    hdr = (f"{'dataset':<14}{'c':>3}|{'TTFT mean':>19}{'D':>7}"
           f"|{'TPOT mean':>19}{'D':>7}|{'req/s':>15}{'D':>7}")
    print(hdr)
    print("-" * len(hdr))
    for key in keys:
        off, req = data[key]["off"], data[key]["required"]
        t1, t2 = sv(off, TTFT), sv(req, TTFT)
        p1, p2 = sv(off, TPOT), sv(req, TPOT)
        r1, r2 = off["req_s"], req["req_s"]
        print(f"{key[0]:<14}{key[1]:>3}|{t1:>8.1f} ->{t2:>8.1f}ms{pc(delta_pct(t1, t2)):>7}"
              f"|{p1:>8.2f} ->{p2:>8.2f}ms{pc(delta_pct(p1, p2)):>7}"
              f"|{r1:>6.1f} ->{r2:>6.1f}{pc(delta_pct(r1, r2)):>7}")


def table_b(data, keys):
    print("\nB. Mechanism -- where the cost lands (mean ms)")
    hdr = f"{'dataset':<14}{'c':>3}|{'queue':>16}|{'prefill':>16}|{'inference':>18}"
    print(hdr)
    print("-" * len(hdr))
    for key in keys:
        off, req = data[key]["off"], data[key]["required"]

        def pair(metric, width=6, prec=2):
            return f"{sv(off, metric):>{width}.{prec}f} ->{sv(req, metric):>{width}.{prec}f}"

        print(f"{key[0]:<14}{key[1]:>3}|{pair(QUEUE)}|{pair(PREFILL)}|{pair(INFER, 7, 1)}")


def table_c(data, keys):
    print("\nC. TTFT distribution (buckets start at 1ms -> usable)")
    hdr = f"{'dataset':<14}{'c':>3}|{'p50 off->req':>18}|{'p95 off->req':>18}"
    print(hdr)
    print("-" * len(hdr))
    for key in keys:
        off, req = data[key]["off"], data[key]["required"]
        print(f"{key[0]:<14}{key[1]:>3}"
              f"|{sv(off, TTFT, 'p50_ms'):>7.1f} ->{sv(req, TTFT, 'p50_ms'):>7.1f}ms"
              f"|{sv(off, TTFT, 'p95_ms'):>7.1f} ->{sv(req, TTFT, 'p95_ms'):>7.1f}ms")


def table_d(data, keys):
    print("\nD. Output-length inflation -- the third cost channel")
    hdr = f"{'dataset':<14}{'c':>3}|{'tok/req off->req':>24}{'D':>8}"
    print(hdr)
    print("-" * len(hdr))
    for key in keys:
        off, req = data[key]["off"], data[key]["required"]
        a = off["tok_s"] / off["req_s"]
        b = req["tok_s"] / req["req_s"]
        print(f"{key[0]:<14}{key[1]:>3}|{a:>10.1f} ->{b:>10.1f}{pc(delta_pct(a, b)):>8}")


def table_client_vs_server(data, keys):
    print("\nE. Client-side estimates vs server truth (ms/token) -- why client ITL lies")
    hdr = (f"{'dataset':<14}{'c':>3} {'cond':<9}{'SERVER':>8}{'cli_usg':>9}"
           f"{'cli_chk':>9}{'srv_ITL':>9}{'tok/chk':>8}")
    print(hdr)
    print("-" * len(hdr))
    for key in keys:
        for name in ("off", "required"):
            cond = data[key][name]
            cli = cond["client"]
            print(f"{key[0]:<14}{key[1]:>3} {name:<9}{sv(cond, TPOT):>8.2f}"
                  f"{cli['tpot_ms_usage']:>9.2f}{cli['tpot_ms_chunk']:>9.2f}"
                  f"{sv(cond, ITL):>9.2f}{cli['tok_per_chunk']:>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="directory of, or glob for, toolcall_itl.py JSON output")
    args = ap.parse_args()

    paths = (sorted(glob.glob(os.path.join(args.path, "*.json")))
             if os.path.isdir(args.path) else sorted(glob.glob(args.path)))
    if not paths:
        raise SystemExit(f"no JSON found at {args.path}")
    data = load(paths)

    # Group each dataset's concurrencies together, low concurrency first.
    keys = sorted(data, key=lambda k: (k[0], k[1]))
    table_a(data, keys)
    table_b(data, keys)
    table_c(data, keys)
    table_d(data, keys)
    table_client_vs_server(data, keys)

    total = sum(c["ok"] for k in keys for c in data[k].values())
    failed = sum(c["failed"] for k in keys for c in data[k].values())
    print(f"\n({total} requests, {failed} failed; greedy temp=0, max_tokens=256, "
          f"per-condition warmup x2)", file=sys.stdout)


if __name__ == "__main__":
    main()
