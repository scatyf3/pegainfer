#!/usr/bin/env python3
"""Streaming latency A/B for tool-call constrained decoding: TTFT / ITL / TPOT.

Companion to toolcall_trace.py (which is throughput+accuracy, non-streaming).
Measures the same quantity two ways and reports both, because client-side
per-token latency is not trustworthy on tool-call workloads:

  CLIENT side (this process' wall clock):
    * streams with stream_options.include_usage=True and takes the token count
      from `usage.completion_tokens` -- the authoritative denominator.
    * ALSO reports the naive chunk-count denominator. A tool-call parser emits
      parsed-argument fragments, not raw tokens, so `tok_per_chunk` != 1 and
      chunk-denominated TPOT is inflated by exactly that factor.

  SERVER side (scraped from vLLM /metrics, differenced around each condition):
    * vllm:request_time_per_output_token_seconds -- engine-internal
      decode_time/(tokens-1). Exact, no client/GIL/SSE contamination.
    * queue / prefill / inference split, which the client cannot see at all.
    * NOTE: the TPOT and ITL histograms bucket from 0.01s up, so on a small
      fast model (TPOT ~2ms) every sample lands in the first bucket and the
      percentiles are meaningless -- `_sum/_count` means stay exact, so we
      report means and flag saturated percentiles instead of pretending.

Greedy (temp 0) so only timing varies.
"""
import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# Server-side histograms worth differencing per condition.
SERVER_HISTS = [
    "vllm:time_to_first_token_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_inference_time_seconds",
]


def resolve_tool_choice(expect_tool, constrained):
    """Mirror of toolcall_trace.resolve_tool_choice: engine-agnostic on/off."""
    if constrained == "off":
        return "auto"
    if constrained == "required":
        return "required"
    if constrained == "named":
        return {"type": "function", "function": {"name": expect_tool}} if expect_tool else "required"
    return "auto"


def load_scenarios(path, limit):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


# ---------------------------------------------------------------------------
# Prometheus /metrics scraping. We aggregate across label sets (single engine
# here) and keep _sum/_count plus cumulative buckets for each histogram.
# ---------------------------------------------------------------------------
def scrape_metrics(base_url, timeout=10.0):
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=timeout) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception:
        return None
    snap = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_part, _, value_part = line.rpartition(" ")
        if not name_part:
            continue
        try:
            value = float(value_part)
        except ValueError:
            continue
        base, _, labels = name_part.partition("{")
        labels = labels.rstrip("}")
        for hist in SERVER_HISTS:
            if base == f"{hist}_sum":
                snap.setdefault(hist, {"sum": 0.0, "count": 0.0, "buckets": {}})["sum"] += value
            elif base == f"{hist}_count":
                snap.setdefault(hist, {"sum": 0.0, "count": 0.0, "buckets": {}})["count"] += value
            elif base == f"{hist}_bucket":
                le = None
                for part in labels.split(","):
                    k, _, v = part.partition("=")
                    if k.strip() == "le":
                        le = v.strip().strip('"')
                if le is None:
                    continue
                edge = float("inf") if le in ("+Inf", "Inf") else float(le)
                d = snap.setdefault(hist, {"sum": 0.0, "count": 0.0, "buckets": {}})
                d["buckets"][edge] = d["buckets"].get(edge, 0.0) + value
    return snap


def hist_quantile(cum_buckets, q):
    """Prometheus-style histogram_quantile over cumulative (le -> count)."""
    edges = sorted(cum_buckets)
    if not edges:
        return None
    total = cum_buckets[edges[-1]]
    if total <= 0:
        return None
    target = q * total
    prev_edge, prev_count = 0.0, 0.0
    for edge in edges:
        count = cum_buckets[edge]
        if count >= target:
            if edge == float("inf"):
                return prev_edge
            if count == prev_count:
                return edge
            return prev_edge + (edge - prev_edge) * (target - prev_count) / (count - prev_count)
        prev_edge, prev_count = edge, count
    return None


def diff_metrics(before, after):
    """Per-condition server-side stats from two snapshots."""
    if not before or not after:
        return None
    out = {}
    for hist in SERVER_HISTS:
        b = before.get(hist)
        a = after.get(hist)
        if not a:
            continue
        b = b or {"sum": 0.0, "count": 0.0, "buckets": {}}
        dsum = a["sum"] - b["sum"]
        dcount = a["count"] - b["count"]
        if dcount <= 0:
            continue
        cum = {}
        for edge, v in a["buckets"].items():
            cum[edge] = v - b["buckets"].get(edge, 0.0)
        edges = sorted(cum)
        # Fraction of samples in the lowest bucket -> percentiles unusable.
        lowest = cum[edges[0]] if edges else 0.0
        saturated = bool(edges and dcount > 0 and lowest / dcount > 0.9)
        out[hist] = {
            "mean_ms": round(dsum / dcount * 1000.0, 3),
            "n": int(dcount),
            "p50_ms": _ms(hist_quantile(cum, 0.50)),
            "p95_ms": _ms(hist_quantile(cum, 0.95)),
            "pct_unusable": saturated,
            "lowest_bucket_s": edges[0] if edges else None,
        }
    return out


def _ms(x):
    return round(x * 1000.0, 2) if x is not None else None


@dataclass
class Result:
    ok: bool
    ttft_ms: float | None = None
    itl_ms: list[float] = field(default_factory=list)   # inter-chunk gaps
    span_ms: float | None = None                         # first->last output chunk
    n_chunks: int = 0                                    # output-bearing chunks
    completion_tokens: int | None = None
    error: str | None = None


def delta_has_output(delta):
    if delta.get("content"):
        return True
    for tc in delta.get("tool_calls") or []:
        fn = tc.get("function") or {}
        if fn.get("name") or fn.get("arguments"):
            return True
    return False


def stream_once(base_url, model, scenario, constrained, temperature, max_tokens, timeout, extra_body):
    body = {
        "model": model,
        "messages": scenario["messages"],
        "tools": scenario["tools"],
        "tool_choice": resolve_tool_choice(scenario.get("expect_tool"), constrained),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra_body:
        body.update(extra_body)
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = None
    last = None
    itl = []
    n = 0
    completion_tokens = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                payload = json.loads(data)
                # usage arrives on its own trailing chunk (choices == [])
                usage = payload.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    completion_tokens = usage["completion_tokens"]
                choices = payload.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if not delta_has_output(delta):
                    continue
                now = time.perf_counter()
                if first is None:
                    first = now
                else:
                    itl.append((now - last) * 1000.0)
                last = now
                n += 1
    except Exception as exc:  # noqa: BLE001 - benchmark records the error
        return Result(ok=False, error=f"{type(exc).__name__}: {exc}")
    if first is None:
        return Result(ok=False, error="no output chunks")
    return Result(
        ok=True,
        ttft_ms=(first - start) * 1000.0,
        itl_ms=itl,
        span_ms=(last - first) * 1000.0,
        n_chunks=n,
        completion_tokens=completion_tokens,
    )


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def summarize(cond, results, wall_s, server):
    ok = [r for r in results if r.ok]
    ttfts = sorted(r.ttft_ms for r in ok)
    itls = sorted(v for r in ok for v in r.itl_ms)
    # TPOT two ways: authoritative token count vs naive chunk count.
    tpot_usage, tpot_chunk, tokphunk = [], [], []
    tot_tokens = 0
    for r in ok:
        # Throughput counts every generated token, including 1-token replies.
        if r.completion_tokens:
            tot_tokens += r.completion_tokens
        if r.span_ms is None:
            continue
        if r.completion_tokens and r.completion_tokens > 1:
            tpot_usage.append(r.span_ms / (r.completion_tokens - 1))
        if r.n_chunks > 1:
            tpot_chunk.append(r.span_ms / (r.n_chunks - 1))
        if r.completion_tokens and r.n_chunks:
            tokphunk.append(r.completion_tokens / r.n_chunks)
    return {
        "cond": cond,
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "wall_s": round(wall_s, 2),
        "req_s": round(len(ok) / wall_s, 2) if wall_s else 0.0,
        "tok_s": round(tot_tokens / wall_s, 1) if wall_s else 0.0,
        "client": {
            "ttft_ms": {"p50": _r(pct(ttfts, .5)), "p95": _r(pct(ttfts, .95))},
            "itl_ms": {"p50": _r(pct(itls, .5)), "p95": _r(pct(itls, .95))},
            "tpot_ms_usage": _r(statistics.mean(tpot_usage)) if tpot_usage else None,
            "tpot_ms_chunk": _r(statistics.mean(tpot_chunk)) if tpot_chunk else None,
            "tok_per_chunk": round(statistics.mean(tokphunk), 3) if tokphunk else None,
        },
        "server": server,
    }


def _r(x):
    return round(x, 2) if x is not None else None


def run(base_url, model, scenarios, constrained, concurrency, temperature, max_tokens,
        timeout, extra_body, scrape=True):
    def task(sc):
        return stream_once(base_url, model, sc, constrained, temperature, max_tokens, timeout, extra_body)
    before = scrape_metrics(base_url) if scrape else None
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(task, scenarios))
    wall = time.perf_counter() - start
    after = None
    if scrape:
        time.sleep(1.0)  # let the engine flush finished-request stats
        after = scrape_metrics(base_url)
    return summarize(constrained, results, wall, diff_metrics(before, after))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="cap scenarios (0 = all)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--extra-body", default="")
    ap.add_argument("--warmup", type=int, default=2, help="warmup passes (per condition)")
    ap.add_argument("--no-metrics", action="store_true", help="skip /metrics scraping")
    args = ap.parse_args()

    extra_body = json.loads(args.extra_body) if args.extra_body else None
    scenarios = load_scenarios(args.dataset, args.limit)
    warm = scenarios[: max(2, args.concurrency * 2)]

    report = {"dataset": args.dataset, "n": len(scenarios), "concurrency": args.concurrency,
              "conditions": []}
    for cond in ("off", "required"):
        # Warm each condition on its own path (guided decoding has a distinct
        # first-request cost), and keep warmup traffic out of the scraped window.
        for _ in range(args.warmup):
            run(args.base_url, args.model, warm, cond, args.concurrency,
                args.temperature, args.max_tokens, args.timeout, extra_body, scrape=False)
        report["conditions"].append(
            run(args.base_url, args.model, scenarios, cond, args.concurrency,
                args.temperature, args.max_tokens, args.timeout, extra_body,
                scrape=not args.no_metrics)
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
