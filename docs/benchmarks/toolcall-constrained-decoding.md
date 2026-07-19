# Tool-call constrained decoding: latency and throughput cost

**Created**: 2026-07-19
**TL;DR**: vLLM 0.21 / Qwen3-0.6B / RTX 5070 Ti, BFCL `simple_python`+`multiple`, `tool_choice=required` vs `auto`. Constrained decoding costs throughput through **three independent channels**, and which one dominates depends on concurrency: at `c=1` it is grammar compilation (TTFT `+60~72%`, entirely queue time, TPOT flat); at `c=16` compilation is fully hidden (TTFT flat) but per-token mask cost appears (TPOT `+9~14%`). A third channel — `required` generates `+7~18%` more tokens per request — is present at every concurrency. Only the first channel is fixable by grammar caching, so "cache it away" does not hold at production concurrency. **Client-side ITL/TPOT is untrustworthy here** (one SSE delta carries ~1.8 tokens); all headline numbers come from vLLM `/metrics`.

## Why this exists

The question from @xiaguan was whether constrained decoding harms throughput enough that tool-call accuracy should instead be bought with frontend caching. The accuracy side was measured separately (`required` removes all `no_tool_call` and `schema_violation` failures, +4.5pp overall). This doc measures the cost side, and specifically corrects an earlier client-side-only measurement that got two of three conclusions wrong.

## Method

Server:

```bash
export PATH="$PWD/.venv/bin:$PATH"       # else flashinfer JIT cannot exec ninja
export VLLM_USE_FLASHINFER_SAMPLER=0     # avoids the JIT entirely; greedy is unaffected
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B --gpu-memory-utilization 0.6 --max-model-len 4096 \
  --enable-auto-tool-choice --tool-call-parser hermes --port 8000
```

Client — `scripts/toolcall_itl.py` runs both conditions, warms each independently, and
scrapes `/metrics` before and after each measurement window so server-side histograms can
be differenced per condition:

```bash
.venv/bin/python scripts/toolcall_itl.py \
  --dataset bfcl_data/scenarios/simple_python.jsonl \
  --concurrency 16 --warmup 2 --max-tokens 256 --temperature 0 \
  --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
```

Regenerate the tables from the stored JSON with:

```bash
.venv/bin/python scripts/toolcall_itl_report.py docs/benchmarks/data/toolcall-itl/
```

Greedy (`temp=0`), full datasets (400 + 200 scenarios), 2400 requests total, 0 failures.
`off` = `tool_choice:"auto"`, `required` = `tool_choice:"required"`; everything else identical.

## Results

Authoritative numbers, from vLLM's own accounting (`_sum/_count`, exact):

```
dataset         c|          TTFT mean      D|          TPOT mean      D|          req/s      D
simple_python   1|    15.4 ->    24.7ms   +60%|    2.81 ->    2.80ms    -0%|   8.2 ->   6.7   -18%
simple_python  16|    34.0 ->    34.3ms    +1%|    4.59 ->    5.24ms   +14%|  75.0 ->  58.6   -22%
multiple        1|    19.5 ->    33.6ms   +72%|    2.92 ->    2.90ms    -1%|   7.7 ->   6.6   -15%
multiple       16|    43.6 ->    41.4ms    -5%|    5.85 ->    6.36ms    +9%|  56.6 ->  51.3    -9%
```

Where the cost lands (mean ms):

```
dataset         c|           queue|         prefill|         inference
simple_python   1|  0.02 ->   8.86|  9.69 ->   8.98|  113.8 ->  131.1
simple_python  16|  0.01 ->   0.01| 16.32 ->  16.63|  185.2 ->  245.6
multiple        1|  0.01 ->  13.91| 13.47 ->  11.79|  121.3 ->  127.8
multiple       16|  0.26 ->   0.02| 24.26 ->  21.47|  240.8 ->  276.3
```

Output-length inflation, the channel that is easy to miss:

```
dataset         c|  tok/req off->req       D
simple_python   1|  37.9 ->      44.5    +17%
simple_python  16|  37.8 ->      44.4    +18%
multiple        1|  38.1 ->      41.3     +8%
multiple       16|  38.4 ->      41.1     +7%
```

## The three cost channels

| Channel | c=1 | c=16 | Fixable by grammar cache |
| --- | --- | --- | --- |
| Grammar compilation | `+9~14ms`, all of it queue time | ~0 (hidden) | **Yes** |
| Per-token mask | ~0 | TPOT `+9~14%` | No |
| Output-length inflation | `+8~17%` tokens | `+7~18%` tokens | No |

At `c=1` the TTFT increase *is* the compile, to within 0.5ms: `simple` TTFT `+9.3ms` vs queue
`+8.84ms`; `multiple` TTFT `+14.1ms` vs queue `+13.90ms`. Prefill does not move, so this is
not a prefill effect — the request sits in `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`
(`vllm/v1/request.py:320`) until its grammar future resolves.

At `c=16` that wait disappears from the critical path. Compilation is async on a
`ThreadPoolExecutor` sized `(cpu_count+1)//2` (8 workers on a 16-core host,
`vllm/v1/structured_output/__init__.py`), so compiles run in parallel *and* overlap with the
15 other requests occupying the GPU. The cost does not vanish — it reappears as per-token
mask work, visible as inference time `185 -> 246ms` and TPOT `+14%`.

The cost model closes with no unexplained residual. Using `e2e ~= TTFT + tokens x TPOT`
for `simple_python c=1`:

```
off      : 15.4 + 37.9 x 2.81 = 122ms   (measured e2e 119.5)
required : 24.7 + 44.5 x 2.80 = 149ms   (measured e2e 146.5)
ratio 122/149 = 0.82 -> -18%            (measured req/s -18%)
```

`c=16` behaves the same way (e2e `203.1 -> 263.4`, ratio 0.77 vs measured `-22%`).

## Why the headline numbers are server-side

Client-observed per-token latency does not survive contact with a tool-call parser. The
hermes parser emits parsed-argument fragments, not raw tokens, so one SSE delta carries
**~1.8 tokens** and both naive client estimates are wrong in opposite directions:

```
dataset         c cond       SERVER  cli_usg  cli_chk  srv_ITL tok/chk
simple_python   1 off          2.81     1.88     3.46     2.82    1.85
simple_python  16 required     5.24     3.05     5.32     5.27    1.77
```

- `cli_chk` (divide by chunk count) is inflated by exactly `tok/chunk`.
- `cli_usg` (divide by the true token count) is *deflated* ~33%, because the parser buffers
  before the first delta so the measured span starts late while the denominator counts all
  tokens. **`TPOT_usage` is therefore not a safe proxy either.**
- `srv_ITL` matches server TPOT to two decimals, confirming one token per engine step here.

Client TTFT overstates server TTFT by 28–92ms, and the gap *grows* under `required` at
`c=16` (46 -> 81ms) — which is how a client-only measurement produced a phantom "TTFT +44%"
at `c=16` where the engine shows `+1%`.

## Gotchas

- **`vllm:inter_token_latency_seconds` is step-level, not token-level.** One sample per
  `EngineCoreOutput` regardless of how many tokens that step emitted
  (`vllm/v1/metrics/stats.py`, "batch-level" comment). Equal to per-token ITL only for plain
  autoregressive decode; **it will understate speculative-decoding gains**, since one step
  there emits k tokens. Use `vllm:request_time_per_output_token_seconds`
  (`decode_time/(tokens-1)`) instead.
- **TPOT/ITL histogram percentiles are unusable on small models.** Those histograms bucket
  from `0.01s`; at ~2-6ms every sample lands in the first bucket and any quantile is
  interpolation fiction. `_sum/_count` means stay exact. `toolcall_itl.py` flags this as
  `pct_unusable`. TTFT buckets start at `0.001s` and do resolve. A 4B/8B model
  (TPOT 20-50ms) would not hit this.
- **Warm each condition separately.** An earlier run warmed only `off` and measured
  `required` paying cold compile costs, reporting `-38%` throughput instead of `-22%`.
- flashinfer's sampler JIT needs `ninja` on `PATH`; `VLLM_USE_FLASHINFER_SAMPLER=0` sidesteps it.

## Conclusion and next step

Grammar caching removes the low-concurrency penalty essentially completely, but nothing at
production concurrency: there the cost is per-token mask work plus longer outputs, neither
of which caching touches. "Use caching instead of constrained decoding" is therefore not a
throughput-neutral substitution at `c=16`. Selective enablement (constrain only
high-value requests) or optimizing mask application are the levers that remain.

**Single run per cell — no 5x averaging, no order counterbalancing.** The TTFT (`+60~72%`),
throughput (`-9~22%`) and token-inflation (`+7~18%`) effects are large enough to survive
that, but **TPOT `+9~14%` rests on one observation and should be repeated 3-5x before being
quoted.** Next step: concurrency sweep `1/4/8/16/32/64` — the absorption model predicts the
required-vs-off TTFT gap shrinks as slack grows, bottoms out near the 8-worker compile pool,
then rises again as the pool saturates.
