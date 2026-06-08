# Mixed-Load ITL — long prompts arriving into steady-state decode (Qwen3-4B)

**Created**: 2026-06-07
**TL;DR**: When a long prompt is admitted mid-decode, Qwen3-4B's unified step
(no per-step token budget) freezes **every** active decode for the whole
prefill — the stalled inter-token gap ≈ the prefill wall-time (4096-tok prompt →
~490ms, 10000-tok → ~2730ms), while gaps with no prefill in flight stay at
baseline TPOT (~14ms). Whether that reaches headline **ITL p99** is a *frequency*
question: stalls hit p99 once they exceed ~1% of decode gaps, and that fraction
grows with both arrival rate *and* prompt length (a long stall starves decode,
inflating its own share). Measured on RTX 5070 Ti: a moderate 4k prompt at
**0.5 req/s keeps ITL p99 at baseline order (15.6 → ~15–19 ms, at the knee)** — the
490ms stall lives only at max; but **1 req/s (4k) → p99 487ms** and **0.3 req/s with
a 10k prompt → p99 2818ms**. **Decision: conditional no-go on chunked prefill** — justified only
behind a hard ITL-p99 SLA in a sustained / long-prompt regime; not for the
low-QPS moderate-prompt profile in #244.

Reproduce with `bench_serving mixed` ([below](#reproduce)); the canonical
low-QPS profile is committed as `mixed-load-itl.qwen3-4b.json`.

## Why this measurement exists

[Issue #244](https://github.com/xiaguan/pegainfer/issues/244). Qwen3-4B's
scheduler admits a pending prefill into a **unified step** when decodes are
active (`pending && active → Unified{pending}`,
[plan.rs:55-68](../../pegainfer-qwen3-4b/src/scheduler/plan.rs#L55-L68)) with no
per-step token budget: the long prefill and all active decode rows run in one
forward pass, so that step's wall-time balloons to ~the prefill time and every
decoding request stalls for that one inter-token gap. The scheduler doc carried
an anecdotal "+38% ITL p99 tail vs vLLM (291 vs 211ms)"
([scheduler.md](../subsystems/scheduler/scheduler.md)); the maintainer stance is
that **chunked prefill is not automatically the fix — measure first**. This is
that measurement. Chunked prefill itself is out of scope for #244.

## Method

In-process, deterministic, single-GPU. `bench_serving mixed` (added in this PR)
is an **open-loop** driver:

- **Background (decode-heavy steady state):** `bg_concurrency` long-lived greedy
  decode streams (`ignore_eos`), kept alive for the whole run, timestamping
  every emitted token.
- **Injector (long prompts at low QPS):** one thread submits a long prompt every
  `1/qps` seconds (greedy, `output_len=1` → prefill-dominated), draining each.
  Each `[submit, last-token]` is an *in-flight-prefill window*. Each injection
  uses a **distinct** synthetic prompt (`synthetic_prompt_tokens_salted`) so it
  is a real prefill — reusing one prompt would let Qwen3's default-on prefix
  cache serve every injection after the first as a ~37ms cache hit, measuring
  caching instead of a prefill stall.
- **Metric:** inter-token gaps of the background streams. A gap is a **stall** if
  it overlaps any prefill window, else **steady**. Reported as `mixed_itl.{all,
  steady, stall}` p50/p95/p99/max. A **baseline** phase runs the same background
  with no injector over the same wall-clock for the control.
- **Determinism:** greedy + `ignore_eos` → identical token streams run-to-run;
  ITL deltas are scheduling, not sampling. A head-start gate ensures all
  background streams are in steady-state decode before injection begins. The
  driver surfaces caveats as `warnings` (stream hit `bg_output_len` early →
  concurrency drift; prefill outran the `1/qps` slot → QPS not sustained).

### Sizing on this GPU (RTX 5070 Ti, 16 GB)

Admission reserves the **full** `prompt + max_tokens` KV per request upfront
([scheduler.rs](../../pegainfer-qwen3-4b/src/scheduler.rs)). KV pool here =
**2332 blocks ≈ 18,656 tokens** (8 tokens/block; ~6 GB free after the 7.6 GB
weights). So `bg_concurrency × (bg_prompt + bg_out) + (inj_prompt + 1)` must fit
≈ 18.6k tokens *and* `bg_out` must outlast `num_injections / qps` seconds at
~14ms TPOT. The plan's `conc=8 / bg_out=8192 / inj=10000` cannot coexist on
16 GB; the configs below trade concurrency vs prompt length to stay in budget
with margin.

## Results

RTX 5070 Ti, Qwen3-4B, commit `f5aa646` + this PR's harness, greedy,
`ignore_eos`, warmup=5, seed=42. Baseline = decode-only control at the same
concurrency, same wall-clock. ITL in ms.

| # | qps | inj prompt | bg conc | prefill (med) | baseline p50 / p99 / max | mixed p50 / **p99** / max | big-stall frac |
|---|-----|-----------|---------|---------------|--------------------------|---------------------------|----------------|
| A | 0.5 | 4096  | 6 | 500 ms  | 14.2 / 15.5 / 16.1 | 14.1 / **18.9** / 596 | 0.9% |
| B | 1.0 | 4096  | 6 | 496 ms  | 14.0 / 15.2 / 18.6 | 13.7 / **486.6** / 491 | 2.7% |
| C | 0.3 | 10000 | 4 | 2736 ms | 14.1 / 29.1 / 36.6 | 17.0 / **2818.4** / 2819 | 2.3% |
| D | 0.5 | 4096 (**warm**) | 6 | 36 ms (first 498) | 14.3 / 15.5 / 16.8 | 14.2 / **16.7** / 485 | — |

> **A sits right at the knee.** Its big-stall fraction (~0.9%) ≈ the 1% p99
> cutoff, so A's mixed p99 wobbles run-to-run as p99 catches 0–few stalls —
> observed **15.7–19.2 ms** across runs (the committed JSON is one such run). That's
> still **baseline order** (≤1.3× baseline, vs 32–185× for B/C), so the decision is
> unchanged; the wobble just pinpoints where the tail starts to bite.

(big-stall frac = `bg_conc × num_injections` truly-stalled gaps over total background
gaps — the fraction that determines whether the stall reaches p99.)

Config A is the committed canonical profile (`mixed-load-itl.qwen3-4b.json`) and
matches #244's "long prompts at **low QPS** over a decode-heavy steady state."
A–C use **distinct** (cold) injection prompts; **D** is A with `--inj-warm-frac 1.0`
(all warm). The `--inj-warm-frac` knob interpolates between them — e.g. `0.5`
evenly interleaves cold and warm injections (4 cold ~490ms + 4 warm ~30ms over 8),
modelling a workload where only some long prompts reuse a cached prefix. (An earlier attempt at C with qps=0.5 oversaturated — a 10k prefill ~2.7s
exceeds the 2s arrival slot — which the harness flagged via an overrun
`warning`; C above uses qps=0.3 so prefills don't overlap.)

### Cold vs warm prefill (config D)

Whether each injected long prompt actually pays a full prefill depends on the
**prefix cache** (default-on since #216). A–C force a cache *miss* (distinct
prompt each injection → real prefill = the worst realistic case: every
long-context request has different content). Config D reuses one prompt, so only
the **first** injection is a real prefill (498ms) and the rest are cache hits
(~30ms median) — modelling a workload where long prompts share a prefix (same
system prompt, shared RAG context). Result: the stall essentially **vanishes** —
the only residual is the one-time cold fill at `max` (485ms), and even that stays
below p99 at low QPS (mixed p99 16.7 ≈ baseline 15.5). So if your real long
prompts reuse prefixes, the tail is even milder than A; cold A is the
conservative bound. (This also means a naive repeated-prompt benchmark would
*hide* the stall entirely — the harness salts injection prompts for exactly this
reason; see Gotchas.)

### The two independent knobs

1. **Severity (per event) = the prefill wall-time.** The gap containing the
   unified step ≈ the entire prefill: `stall`-conditional p99 = 490ms @ 4k,
   2819ms @ 10k. Every active decode freezes together; `steady` gaps and **p50
   stay at baseline (~14ms) in every config** — decode is untouched when no
   prefill is in flight. This is purely a tail phenomenon.

2. **Frequency (does it reach p99) = the stall-gap fraction.** With `streams`
   decodes all freezing per injection, the fraction of background gaps that are
   stalls is

   ```
   frac ≈ qps / ( qps + (1 − qps·prefill_s) / TPOT )
   ```

   — it rises with **both** arrival rate and prefill length, because a long
   stall consumes decode time that would otherwise produce steady gaps,
   inflating its own share. Headline ITL p99 is unaffected while `frac < ~1%`
   and climbs to the per-event stall once `frac > ~1%`:
   - **A**: frac ~0.9% ≈ the 1% cutoff → **p99 stays baseline-order** (~15–19ms,
     wobbles run-to-run at the knee); the 490ms stall lives at max/p99.9.
   - **B**: raise qps to 1.0 → frac 2.7% → **p99 = 487ms** (32× baseline).
   - **C**: even at low qps=0.3, a 10k prompt → frac 2.3% → **p99 = 2818ms**
     (96× baseline). The long stall breaches p99 *despite* low QPS.

## Decision: chunked prefill — conditional no-go

**Do not implement chunked prefill unconditionally.** For #244's profile —
moderate prompts at genuinely low QPS into a decode-heavy batch (config A) — the
ITL p99 tail stays **baseline-order (15.6 → ~15–19 ms, at the knee)**; the
"varied-length workloads break the waves naturally" stance holds *at p99*, and the
residual cost is a single ~490ms stall at p99.9/max, tolerable for most decode SLAs.

**Implement chunked prefill only if** a deployment has a **hard ITL-p99 SLA**
*and* a regime where the stall-gap fraction exceeds ~1% — i.e. either sustained
arrival rate (≳1 req/s at these shapes, config B) **or** routinely long prompts
(≥10k, config C) where each stall is long enough to breach p99 even at low QPS.
In those regimes the measured tail is **487–2818ms at p99** (32–96× baseline),
which no per-token SLA survives, and chunked prefill (bounding per-step prefill
tokens so decode rows are serviced between chunks) is the correct fix.

This keeps the maintainer's "measure first / conditional" position but replaces
the anecdote with a severity-vs-frequency model and the p50/p99/max numbers
required by #244's acceptance boundary.

## Reproduce

```bash
# Build (RTX 5070 Ti / Arch WSL2 env; SM 120, absolute Triton python path):
CUDA_HOME=/opt/cuda NVCC_PREPEND_FLAGS="-ccbin g++-13" \
  LIBRARY_PATH=/usr/lib/wsl/lib:/opt/cuda/lib64 \
  PEGAINFER_CUDA_SM=120 PEGAINFER_TRITON_PYTHON=/abs/path/.venv/bin/python \
  cargo build -r -p pegainfer-server --bin bench_serving

# Canonical low-QPS profile (config A), writes the committed JSON:
CUDA_HOME=/opt/cuda LIBRARY_PATH=/usr/lib/wsl/lib:/opt/cuda/lib64 \
  ./target/release/bench_serving --model-path models/Qwen3-4B \
  --format json --out docs/benchmarks/mixed-load-itl.qwen3-4b.json \
  mixed --bg-prompt-len 512 --bg-concurrency 6 --bg-output-len 1536 \
        --inj-prompt-len 4096 --inj-output-len 1 --qps 0.5 --num-injections 8 --warmup 5

# Frequency knob (B):     --qps 1.0 --num-injections 12
# Severity knob (C):      --inj-prompt-len 10000 --bg-concurrency 4 --bg-output-len 2048 --qps 0.3 --num-injections 5
# Prefix-cache knob (D):  --inj-warm-frac 1.0  (fraction of injections that hit the cache; 0.5 = half)
```

`mixed` is a **standalone diagnostic**, not part of the shape-guarded snapshot
regression set ([bench-regression.md](../conventions/bench-regression.md)) — it
runs open-loop and is not committed into `bench_snapshots/`.

### Gotchas

- **KV budget binds the config.** Reservation is `prompt + max_tokens` per
  active request; a rejected request aborts the run with a clear message. Size
  per [Sizing](#sizing-on-this-gpu-rtx-5070-ti-16-gb).
- **`bg_output_len` must outlast the run** or background concurrency drops
  mid-run — the harness emits a `warning` if a stream finishes early.
- **`qps · prefill_s` must stay < 1** or prefills overlap and decode starves
  continuously (overrun `warning`). For a 10k prompt (~2.7s) keep qps ≤ ~0.3.
- **Distinct injection prompts are mandatory** — the prefix cache otherwise
  hides the stall after the first injection (verified: median prefill collapses
  to ~37ms). The harness salts each injection prompt.
- **Benign teardown log.** At process exit the still-active background requests
  can log one `WARN ... Sync failed: DriverError(CUDA_ERROR_DEINITIALIZED)`
  *after* the report — a shutdown race, not a measurement error. `RUST_LOG=error`
  suppresses it.
- **First long prefill is cold** (config C: ~2.4s vs ~2.7s steady) — warmup, and
  read the median prefill.

## Next

If a deployment surfaces a hard ITL-p99 SLA under sustained or long-prompt
load, this profile is the gate to re-open the chunked-prefill decision: re-run
at that deployment's QPS/prompt mix and compare mixed vs baseline p99.
