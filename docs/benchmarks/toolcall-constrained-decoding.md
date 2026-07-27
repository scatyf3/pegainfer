# Tool-call constrained decoding: latency and throughput cost

**Created**: 2026-07-19 · **Revised**: 2026-07-27 (5× repeat overturned the concurrency framing)
**TL;DR**: vLLM 0.21 / Qwen3-0.6B / RTX 5070 Ti, BFCL `simple_python`+`multiple`, `tool_choice=required` vs `auto`, **5 independent runs per cell**. The throughput cost of constrained decoding is **`-11~21%` req/s, steady-state and stable** (std ±3%), and comes from two **cache-independent** channels: output-length inflation (`+7~17%` tokens, deterministic) and per-token mask work (TPOT `+3~10%`). A third channel — grammar compilation — is a **one-time cold-cache cost (~8–12 ms queue per distinct schema) that vLLM already amortizes across requests**: it shows up only on the *first* request for each schema, is independent of concurrency, and vanishes once cached. **The earlier single-run table mis-attributed this to low concurrency** — it happened to measure a cache-cold cell at `c=1` and a cache-warm cell at `c=16`; repeating each cell 5× shows the spike follows cache state, not concurrency. Client-side ITL/TPOT stays untrustworthy here (~1.8 tokens per SSE delta); all numbers are from vLLM `/metrics`.

## Why this exists

@xiaguan's question was whether constrained decoding harms throughput enough that tool-call accuracy should instead be bought with frontend caching. The accuracy side was measured separately (`required` removes all `no_tool_call` and `schema_violation` failures, +4.5pp overall). This doc measures the cost side.

The short answer for @xiaguan: **the caching he proposed already exists inside the engine** — vLLM caches compiled grammars by schema, so the compilation cost is paid once per distinct tool schema and never again. But that channel was never the throughput problem. The steady-state `-11~21%` req/s hit is per-token mask work plus longer outputs, and **neither is cacheable**. So "cache it instead of constraining" does not recover the throughput at production concurrency.

## Method

Server:

```bash
export PATH="$PWD/.venv/bin:$PATH"       # else flashinfer JIT cannot exec ninja
export VLLM_USE_FLASHINFER_SAMPLER=0     # avoids the JIT entirely; greedy is unaffected
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B --gpu-memory-utilization 0.6 --max-model-len 4096 \
  --enable-auto-tool-choice --tool-call-parser hermes --port 8000
```

`scripts/toolcall_itl.py` runs both conditions once, warms each independently, and scrapes
`/metrics` before and after each measurement window so server-side histograms are differenced
per condition. To turn a single observation into a quotable number, run it N times per cell and
aggregate with `scripts/toolcall_itl_agg.py` (mean ± sample std; deltas are computed per run
then averaged, so the reported delta std is run-to-run stability of the *effect*):

```bash
for k in 1 2 3 4 5; do
  .venv/bin/python scripts/toolcall_itl.py \
    --dataset bfcl_data/scenarios/simple_python.jsonl \
    --concurrency 16 --warmup 2 --max-tokens 256 --temperature 0 \
    --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
    > docs/benchmarks/data/toolcall-itl-5x/simple_python_c16.run${k}.json
done
.venv/bin/python scripts/toolcall_itl_agg.py docs/benchmarks/data/toolcall-itl-5x/
```

Greedy (`temp=0`), full datasets (400 `simple_python` + 200 `multiple`), 5 runs × 2 conditions ×
(2 warmup + 1 measured) per cell, 0 failures throughout. `off` = `tool_choice:"auto"`,
`required` = `tool_choice:"required"`; everything else identical. **Run order matters for the
compilation channel** — see the cold-cache section; here `c=16` cells ran before `c=1`, which is
why `c=1` shows no cold compile at all (its grammars were already cached by the `c=16` runs).

## Results (5 runs per cell, mean ± std)

Throughput and its two cache-independent drivers — **stable across all 5 runs**, `off → required`:

```
                       req/s               tok/req              TPOT mean (ms)
simple_python  c=1   9.3±0.2 -> 7.8±0.2 -17%   37.9 -> 44.5 +17%   2.48±0.06 -> 2.58±0.07  +4±4%
simple_python  c=16 80.7±0.9 -> 63.1±2.1 -22%  37.8 -> 44.4 +17%   4.39±0.08 -> 4.89±0.12 +11±4%
multiple       c=1   8.9±0.4 -> 7.8±0.4 -12%   38.1 -> 41.2  +8%   2.52±0.13 -> 2.68±0.14  +6±4%
multiple       c=16 58.4±0.7 -> 51.9±1.6 -11%  38.3 -> 41.1  +7%   5.91±0.12 -> 6.30±0.20  +7±5%
```

- **req/s `-11~22%`** — the headline cost. `std ±3%`, present at both concurrencies, present in
  every run including the cache-warm ones. This is the number that matters.
- **tok/req `+7~17%`** — `required` generates more tokens per request. `std ≈ 0` (deterministic
  under greedy). For `simple_python` this alone explains most of the throughput loss.
- **TPOT `+3~11%`** — per-token mask application. Real (std well below the mean) and it grows with
  concurrency: `+3~6%` at `c=1`, `+7~11%` at `c=16`. This is what the single-run `+9~14%` was
  gesturing at; the repeat confirms it and tightens it.

TTFT is dominated by the cold-cache artifact and is shown separately below.

## The grammar-compilation channel is a cold-cache cost, not a concurrency cost

This is the correction to the original doc. Per-run `required`-side queue time (ms), which is where
compilation shows up (the request sits in `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`,
`vllm/v1/request.py`, until its grammar future resolves):

```
cell                run1   run2   run3   run4   run5
simple_python c=16  8.27   0.016  0.019  0.008  0.016
multiple      c=16 12.25   0.007  0.018  0.030  0.010
simple_python c=1   0.043  0.052  0.024  0.024  0.013   <- never cold: grammars cached by c=16 runs
multiple      c=1   0.054  0.050  0.041  0.022  0.040
```

The `~8–12 ms` spike is present **only on run 1 of the first cell to touch each dataset**, and only
because that run compiles ~400 distinct schemas cold inside its window (warmup covers just 2
scenarios). Every run after it reads vLLM's compiled-grammar cache and pays `~0`. The `c=1` cells,
which ran *after* the `c=16` cells, never see a cold compile at all — same schemas, already cached.

So the original table's "TTFT `+60~72%` at `c=1`, flat at `c=16`" was **cache ordering, not
concurrency**: that run measured a cold `c=1` cell first and a warm `c=16` cell second. Compilation
cost does not depend on concurrency; it depends on whether the schema is in cache.

What remains once caches are warm is a small, stable TTFT residual — mask setup, not compilation.
Cache-warm steady state (runs 2–5), `off → required`:

```
cell                TTFT off->req    D
simple_python c=1   13.6 -> 14.6ms  +8%
simple_python c=16  27.3 -> 29.9ms  +9%
multiple      c=1   17.3 -> 19.1ms +10%
multiple      c=16  36.4 -> 41.1ms +13%
```

## The three channels, restated

| Channel | Magnitude | Depends on | Fixable by grammar cache |
| --- | --- | --- | --- |
| Grammar compilation | one-time `~8–12 ms` queue per distinct schema | cache-cold first hit (not concurrency) | **Already cached by vLLM** |
| Output-length inflation | `+7~17%` tokens, every run/concurrency | model + schema | No |
| Per-token mask | TPOT `+3~11%`, grows with concurrency | concurrency | No |

The cost model still closes. For `simple_python c=16` cache-warm, `e2e ≈ TTFT + tokens × TPOT`:

```
off      : 27.3 + 37.8 × 4.42 = 194 ms
required : 29.9 + 44.4 × 4.88 = 247 ms
ratio 194/247 = 0.79 → -21%   (measured req/s -21%)
```

Both channels that drive that ratio — the `+17%` token count and the `+10%` TPOT — are
cache-independent, which is why req/s loses the same `-21~22%` on every run.

## Why the headline numbers are server-side

Client-observed per-token latency does not survive contact with a tool-call parser. The hermes
parser emits parsed-argument fragments, not raw tokens, so one SSE delta carries **~1.8 tokens**
and both naive client estimates are wrong in opposite directions:

```
dataset         c cond       SERVER  cli_usg  cli_chk  srv_ITL tok/chk
simple_python   1 off          2.81     1.88     3.46     2.82    1.85
simple_python  16 required     5.24     3.05     5.32     5.27    1.77
```

- `cli_chk` (divide by chunk count) is inflated by exactly `tok/chunk`.
- `cli_usg` (divide by the true token count) is *deflated* ~33%, because the parser buffers before
  the first delta so the measured span starts late while the denominator counts all tokens.
  **`TPOT_usage` is therefore not a safe proxy either.**
- `srv_ITL` matches server TPOT to two decimals, confirming one token per engine step here.

(These per-token illustration rows come from the original single-run capture in
`docs/benchmarks/data/toolcall-itl/`; regenerate with `scripts/toolcall_itl_report.py`.)

## Gotchas

- **Grammar compilation is cached across requests — measure it deliberately.** A single A/B run
  that compiles cold in one condition and warm in the other will mis-attribute the spike (here,
  to concurrency). Either warm every distinct schema first, or run ≥2 rounds and read the cold
  first hit and the warm steady state as separate numbers. This is the single biggest trap in this
  benchmark.
- **`vllm:inter_token_latency_seconds` is step-level, not token-level.** One sample per
  `EngineCoreOutput` regardless of how many tokens that step emitted (`vllm/v1/metrics/stats.py`,
  "batch-level" comment). Equal to per-token ITL only for plain autoregressive decode; **it will
  understate speculative-decoding gains**. Use `vllm:request_time_per_output_token_seconds`
  (`decode_time/(tokens-1)`) instead.
- **TPOT/ITL histogram percentiles are unusable on small models.** Those histograms bucket from
  `0.01s`; at ~2–6 ms every sample lands in the first bucket and any quantile is interpolation
  fiction. `_sum/_count` means stay exact. `toolcall_itl.py` flags this as `pct_unusable`. TTFT
  buckets start at `0.001s` and do resolve. A 4B/8B model (TPOT 20–50 ms) would not hit this.
- **Percent change off a ~0 baseline is noise.** Queue time at `c=16` warm is `~0.01 ms`; the
  aggregator prints `n/a` for its delta rather than a fabricated `+30000%`.
- flashinfer's sampler JIT needs `ninja` on `PATH`; `VLLM_USE_FLASHINFER_SAMPLER=0` sidesteps it.

## Conclusion and next step

Grammar caching removes the compilation penalty — and vLLM already does it, so there is nothing
left there to win with a frontend cache. The throughput cost that remains, `-11~21%` req/s, is
per-token mask work plus `+7~17%` longer outputs; both are cache-independent and both reproduce on
every run. "Use caching instead of constrained decoding" is therefore not a throughput-neutral
substitution. The levers that remain are selective enablement (constrain only high-value requests)
and cheaper mask application; reducing the output-length inflation would need a decoding-side change.

**Confidence:** each cell is 5 independent runs; the throughput (`-11~22%`), token-inflation
(`+7~17%`) and TPOT (`+3~11%`) effects all have std well below their means. The one number still
worth more data is the cold first-hit compile cost — it rests on the single run-1 observation per
dataset (`8.3` / `12.2 ms`); a schema-count sweep would characterize how it scales.
Next step: concurrency sweep `1/4/8/16/32/64` with **caches pre-warmed**, to isolate how the
per-token mask channel (TPOT) scales with batch size now that the compilation confound is understood.
