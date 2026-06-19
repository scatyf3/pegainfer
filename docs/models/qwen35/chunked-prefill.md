# Qwen3.5 scheduler-level chunked prefill

> **TL;DR:** Issue #375. Qwen3.5 already chunked prefill at the *KV level*
> (`PREFILL_CHUNK_LEN = 20000`, an OOM guard) but ran every chunk inside one
> scheduler step, so a long prompt stalled the whole decode batch for one unified
> pass — the same ITL tail #368 measured on Qwen3. The scheduler now owns a
> per-step prefill **token budget**: each step prefills at most
> `OPENINFER_QWEN35_PREFILL_BUDGET` (default 1024, matching Qwen3) prompt tokens off
> the FIFO front and services the decode batch between chunks. Validated: 24
> scheduler unit tests + `e2e_scheduler` green at the 1024 default and at budget 4
> (heavy slicing, concurrent interleave). **Next:** the #375 part-2 mixed-load ITL benchmark (chunked vs
> standard) — run with `OPENINFER_QWEN35_PREFILL_BUDGET` huge for the standard A/B.
>
> **Last touched:** 2026-06.

## Why it was needed

Qwen3.5's executor already prefills a prompt as serial chunks that advance the
paged KV and the linear-attention recurrent/conv state *in place*
([`prefill.rs`](../../../openinfer-qwen35-4b/src/prefill.rs) — `prefill_forward`
loops `prefill_chunk_forward`). But the chunk loop lived *inside one
`batch_prefill`/`unified_step` call*, so the scheduler saw prefill as atomic: a
`Unified` step packed the entire prompt + every active decode row into one
forward pass, ballooning that step's wall-time to ≈ the whole prefill and
stalling every decoding request for one inter-token gap. This is the Qwen3
behaviour #368 profiled; Qwen3.5 had the same shape because its scheduler — a
separate implementation from Qwen3's — had a KV/slot admission budget but **no
per-step prefill-token budget**.

## The key enabler (no new kernel)

`prefill_forward(window, &mut kv, &mut rec)` reads its base position from
`kv.seq_len()` and advances `kv` and `rec` in place
([`prefill.rs:146`](../../../openinfer-qwen35-4b/src/prefill.rs#L146),
[`prefill.rs:178`](../../../openinfer-qwen35-4b/src/prefill.rs#L178)). So calling
it repeatedly with *successive prompt slices* evolves KV/recurrent state
bit-identically to one whole-prompt call — only the final window's logits are
kept; intermediate windows' last-token LM-head output is discarded. This is the
same continuation `prefill_forward` already exercises across its internal
`PREFILL_CHUNK_LEN` chunks, so scheduler-level chunking needed **only scheduler
bookkeeping**, no executor/kernel change.

## Design ([`scheduler.rs`](../../../openinfer-qwen35-4b/src/scheduler.rs), [`scheduler/plan.rs`](../../../openinfer-qwen35-4b/src/scheduler/plan.rs))

- **`prefilling` queue.** A new FIFO of `PrefillingRequest35` — each owns its
  growing `KvState` + `RecurrentState` + a `cursor` (prompt tokens prefilled so
  far). State persists here across steps until the prompt is exhausted.
- **Per-step token budget.** `plan_prefill_chunks(remaining, budget)` (pure,
  unit-tested) packs front requests up to `budget` total tokens; a request that
  doesn't fit takes a partial chunk and stays at the front (packing stops), so
  one long prompt is sliced across steps while short prompts behind it still get
  serviced once it finishes. `budget ≥ Σ remaining` ⇒ whole-prompt prefill in one
  step (the pre-#375 behaviour, used for benchmark A/B).
- **`run_step`.** Each step: advance the scheduled prefill chunks
  (`prefill_forward` per request) → `decode_step` the active batch → promote any
  prompt that finished this step. Prefill and decode are independent forward
  passes on disjoint state, so "prefill chunk then `batch_decode_graph`" equals a
  standalone decode — the invariant the retained `unified_step` equivalence test
  (now `#[cfg(test)]`) guards.
- **Decode-then-promote ordering.** `decode_step` may retire requests and compact
  slots; promoting finished prefills *after* it sees the freed slots and assigns
  dense graph-slot indices.

## Admission accounts for in-flight prefills

A partially-prefilled request is committal in a way Qwen3's isn't: from its first
chunk it holds live recurrent state and KV pages, and it has a graph slot
reserved for promotion. So before admitting *new* prompts the loop shrinks the
budgets passed to `admit_pending_requests`:

- **Slots:** `max_batch − prefilling.len()` (each in-flight prefill reserves the
  slot it will promote into).
- **Pages:** `available_pages() − prefilling_future_pages(...)`. `available_pages`
  already excludes pages held by in-flight prefill KV (the `cursor` tokens);
  `prefilling_future_pages` (pure, unit-tested) subtracts the *future* growth
  those prefills will still claim — the same future-growth reservation
  `active_future_pages` makes for decoding requests.

This keeps `admit_pending_requests` and its tests untouched (the scalars are
reduced at the call site) and guarantees `active + prefilling ≤ max_batch`, so a
promotion's `slot_for_new_request` never fails.

## The tuning knob

`OPENINFER_QWEN35_PREFILL_BUDGET` overrides `DEFAULT_MAX_PREFILL_TOKENS` (1024,
the same name and value as Qwen3's default). Smaller → lower decode ITL tail,
higher prompt TTFT; absurdly large → standard one-pass prefill. It is the Qwen3.5
analogue of Qwen3's `max_prefill_tokens`. 1024 mirrors Qwen3's empirically-tuned
default, but Qwen3.5's hybrid architecture may shift the sweet spot — the part-2
benchmark calibrates it.

## Next step — #375 part 2 (benchmark)

Deliver a mixed-load ITL comparison (chunked vs standard) aligned with #368's
profile, sweeping `OPENINFER_QWEN35_PREFILL_BUDGET` (set it huge for the standard
baseline) over the QPS × prompt-length × prefix-reuse space, and record the
go/no-go and a calibrated default. See [`../../benchmarks/mixed-load-itl.md`](../../benchmarks/mixed-load-itl.md)
for the Qwen3 method and `scripts/sweep_mixed_itl.sh`.
