# Tool-call constrained-decoding A/B (throughput × accuracy)

**TL;DR:** `scripts/toolcall_trace.py` drives `/v1/chat/completions` with `tools`
against a vLLM/SGLang server and buckets each response (ok / wrong_tool /
malformed_json / schema_violation / value_mismatch / leaked_tool_call /
no_tool_call), reporting accuracy **and** throughput. Run it twice — once
`--constrained off`, once `--constrained required` — and diff. `scripts/bfcl_to_scenarios.py`
converts BFCL into the scenario format. First warm, order-controlled smoke result
(Qwen3-0.6B, vLLM 0.21): constrained decoding buys accuracy 83→99% for ~35% lower
req/s (~24% lower tok/s) — the throughput cost is real, as su predicted. (A first
un-warmed run showed the reverse; it was a cold-start artifact — see the warmup
lesson.)

Last touched: 2026-07

## Why this exists

Issue #655 proposed grammar-constrained decoding (llguidance / XGrammar) to make
small models reliably emit tool-call JSON. Maintainer pushback: constrained
decoding is widely reported to cost throughput, so lead with measurement, not a
build. The right experiment (their steer): take a tool-call dataset, have an
engine that *already* has constrained decoding (vLLM / SGLang — openinfer does
not yet) generate tool calls with the constraint on vs. off, and measure both
throughput and accuracy. That delta is the input to "should openinfer build it."

## What the harness measures

openinfer's own frontend can't run this A/B (no guided-decoding knob — see
`openinfer-dynamo-backend/src/convert.rs`, `SamplingParams` has none), so the
harness is a client that owns the round-trip against vLLM/SGLang. Per sample:

| bucket | meaning | what fixes it |
|---|---|---|
| `ok` | right tool, valid-JSON args, schema-valid, ground-truth-matching | — |
| `wrong_tool` | valid call, wrong tool | prompt / model capability |
| `malformed_json` | `arguments` not valid JSON | **constrained decoding** |
| `schema_violation` | args parse but break the schema (missing required / bad type / bad enum) | **constrained decoding** |
| `value_mismatch` | right function, schema-valid, but a value disagrees with BFCL ground truth | model capability (constraint can't know the answer) |
| `leaked_tool_call` | tool-call JSON in `content`, not parsed into `tool_calls` | **frontend parser** config |
| `no_tool_call` | answered in prose | prompt / `tool_choice` |
| `error` | transport / HTTP failure | infra |

accuracy = ok / total. The split is the point: `malformed`+`schema_violation` is
the slice a grammar engine removes; `value_mismatch`/`wrong_tool`/`no_tool_call`
it cannot.

## Constrained toggle

Engine-agnostic — both vLLM and SGLang gate guided decoding on `tool_choice`:

- `--constrained off` → `tool_choice="auto"` — free generation, parser extracts.
- `--constrained required` → `tool_choice="required"` — engine forces *some* schema-valid call.
- `--constrained named` → `tool_choice={function: expect_tool}` — forces the expected function.

Tools, parser, prompt are otherwise identical, so the off↔required delta is
attributable to the constraint. Note `required` conflates two effects: it
engages the grammar **and** shortens output (no prose) — see the caveat below.

## Dataset: BFCL

Recommended: BFCL (Berkeley Function-Calling Leaderboard) non-executable
categories — `simple` (single fn, isolates arg-filling), `multiple` (fn
selection), `parallel` (multi-call). They carry ground truth (function + allowed
values per param) and per-function JSON schemas (what a grammar backend needs).
Convert:

```bash
git clone https://github.com/ShishirPatil/gorilla
python3 scripts/bfcl_to_scenarios.py \
  --prompt  gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_simple.json \
  --answers gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v3_simple.json \
  --out data/bfcl_simple.jsonl --limit 200
```

The converter normalizes BFCL's schema dialect (`dict`→`object`, `float`→`number`,
`tuple`→`array`, drops `any`) and embeds ground truth so accuracy is an AST-style
value match, not just schema validity. The bundled
`test_data/toolcall_scenarios.jsonl` is a 12-case smoke set (no ground truth) for
wiring checks. NB: BFCL's *official* harness gives the canonical AST accuracy but
does **not** measure the throughput A/B — that's this harness's job.

## Run it

```bash
# vLLM (0.21). Two gotchas on this box, both non-fatal once handled:
#  - FlashInfer JIT-builds its sampler with `ninja`; put .venv/bin on PATH or set
#    VLLM_USE_FLASHINFER_SAMPLER=0 (we run greedy anyway).
#  - Qwen3 <think> reasoning wastes tokens on tool calls — disable per request.
PATH="$PWD/.venv/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve Qwen/Qwen3-0.6B --gpu-memory-utilization 0.6 --max-model-len 8192 \
    --enable-auto-tool-choice --tool-call-parser hermes --port 8000

EXTRA='{"chat_template_kwargs": {"enable_thinking": false}}'
for c in off required; do
  python3 scripts/toolcall_trace.py --base-url http://127.0.0.1:8000 \
    --model Qwen/Qwen3-0.6B --dataset test_data/toolcall_scenarios.jsonl \
    --repeat 10 --concurrency 16 --temperature 0.7 --max-tokens 256 \
    --extra-body "$EXTRA" --constrained $c --trace-out ab_$c.jsonl --quiet
done
```

`--extra-body` merges arbitrary JSON into the request (thinking toggle, guided
backend, etc.). `--concurrency N` drives throughput; `--fail-under R` gates CI.

## First measurement (Qwen3-0.6B, vLLM 0.21, RTX 5070 Ti)

12-case smoke set, 30 samples/scenario (360 req), concurrency 16, temp 0.7,
max_tokens 256. **Engine warmed first, and each condition run in both orders**
— see the warmup lesson below for why that is mandatory. Averaged over the two
orders (per-round spread in parens):

| | accuracy | req/s | tok/s | p50 latency |
|---|---|---|---|---|
| `off` (auto) | 83.2% | ~90 (85–95) | ~3030 (2863–3203) | ~159 ms |
| `required` | **98.8%** | ~58 (53–62) | ~2315 (2130–2499) | ~201 ms |
| delta | **+15.6 pp** | **−36%** | **−24%** | **+26%** |

Two findings:

1. **Constraint fixed exactly the buckets predicted.** The off-condition
   `schema_violation`s were all `enum_strict` emitting `priority="urgent"` (enum
   is low/medium/high/critical); guided decoding makes the invalid token
   unreachable → 0 violations. `no_tool_call` also drops (`required` forces a
   call). This accuracy gain is warmup-independent.
2. **Constrained decoding costs throughput here — as expected.** ~35% fewer
   req/s, ~24% fewer tok/s, ~26% higher p50. The tok/s drop is the per-token
   grammar-mask overhead; the req/s drop compounds it with `required` emitting
   slightly longer outputs (mean 42 vs 33 tokens/req). So on Qwen3-0.6B the
   trade is **+15.6 pp accuracy for ~⅓ of throughput** — su's concern is real,
   now quantified. Whether it's worth it is a product call (a coding agent that
   retries malformed tool calls may prefer the accuracy).

### Warmup lesson (why the first run of this A/B lied)

The *first* attempt reported the opposite — constrained *faster* (+84% req/s).
That was a measurement bug, not a finding: the run was 3–7 s wall, and vLLM's
one-time lazy init (first-batch compile / CUDA-graph capture) stalled the first
wave of exactly `concurrency` (16) requests by ~5.7 s. Whichever condition ran
**first** ate that cold-start and looked slow. Fix: discard a warmup pass, run
≥300 requests, and run both orders. The per-request `tok/s` median (batch-count
independent) is a more robust signal than aggregate req/s on short runs.

## Caveats

- `required` changes **two** things at once (grammar on + output shortened). To
  isolate pure per-token mask overhead, compare at matched output length or
  measure decode-phase per-token latency. The p50 +68 ms is the closest proxy
  here to the fixed grammar/compile cost.
- 12-case smoke set, no ground truth → accuracy = right-tool + schema-valid, not
  BFCL AST. Small N; one scenario (`enum_strict`) drives the whole off-accuracy gap.
- Single model / engine / GPU. With **many distinct** schemas, XGrammar's
  per-schema compile cost grows (here schemas repeat and amortize) — the
  throughput story can shift. Re-measure on the real target before generalizing.
- Minimal schema validator (type / required / properties / enum / array items);
  lenient on `anyOf` / `pattern` / numeric bounds.

## Next step

Run on real BFCL `simple` + `multiple` (200+ cases each) at both constraint
settings (warm, both orders) to get a standard AST accuracy number and confirm
the +accuracy / −throughput trade at schema variety. Record the table in
`benchmarks/`. The open question for #655 is not *whether* constrained decoding
costs throughput (it does, ~⅓ here) but whether the accuracy it buys is
reachable more cheaply — su's frontend tool-call cache / parser-repair idea — or
whether the ~⅓ hit is acceptable for the coding-agent use case. Measure the
frontend-repair alternative on the same traces before committing to a grammar
engine.
