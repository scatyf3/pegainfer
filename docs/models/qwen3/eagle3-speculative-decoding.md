# EAGLE-3 Speculative Decoding (Qwen3-4B)

**TL;DR:** An opt-in EAGLE-3 chain drafter (`--eagle3-draft-model-path`, e.g. `AngelSlim/Qwen3-4B_eagle3`) now runs through the same speculative transaction as DFlash: the scheduler, `DraftPlan → DraftResult → VerifyPlan → VerifyResult`, and the executor's verify/accept/KV commit are shared, and only the proposer lane is EAGLE-specific. v1 speculates greedy requests whose whole prompt fits one prefill chunk. Every other request, and any batch containing one, decodes normally. The code is a **draft that has not been compiled or run yet**: it was written off-GPU, and none of the gates or A/B numbers below exist. Next step: build it on a CUDA box and run the drafter golden gate, then the lossless gate, then the A/B.

Last touched: 2026-09

## Why this shape

PR #707 landed the drafter forward on its own. Review (2026-09-07) rejected it because 883 lines sat behind `allow(dead_code)` with no caller, no GPU gate exercising the forward, and no path through the shared transaction. This slice answers that review directly. It wires the smallest usable opt-in path end to end, so the lossless gate, the golden gate, and the A/B all run against code that production can reach.

## What is shared, what is EAGLE-3

| Layer | Owner | EAGLE-3 change |
| --- | --- | --- |
| Scheduler plan (`should_speculative_decode`, span truncation to the remaining budget) | shared | none: `speculative_enabled()` now also covers EAGLE-3 |
| `executor/spec.rs` (reserve K+1, verify step, `apply_speculative`, counters) | shared | readiness check renamed to `draft_ready_requests` |
| Verify forward (`batch_prefill_into` + `VerifyGraphBuffers`, per-request params) | shared call | `execute_eagle3_verify` runs the plain pass with no hedge ladder |
| Prefill capture → ready set → drop on retire/plain decode | shared | one `draft_ready_requests` set and one `DropDraft` worker command for both drafters |
| Memory reservation | per drafter | `Eagle3MemoryReservation`: fixed bytes via the margin, single-layer draft K/V per pool token |
| Proposer lane | per drafter | `executor/eagle3_lane.rs` + `eagle3/forward.rs` |

Two proposers now exist, and the part they genuinely share is the executor transaction plus the capture/readiness bookkeeping. The lanes still differ: DFlash appends pending context and drafts a block, while EAGLE-3 rebuilds draft K/V and rolls a chain. A proposer trait therefore stays deferred; the shared state is expressed as the renamed ready set and drop command rather than a trait.

## Mechanics that are easy to get wrong

- **Feature↔token shift.** Draft K/V slot `j` pairs target feature `f_j` with the embedding of token `j+1`. Prompt prefill teacher-forces the `P-1` pairs and keeps `f_{P-1}` as the boundary feature. A chain round starts from `fc(boundary)` plus the current token.
- **K/V-only teacher forcing.** A single-layer drafter's K/V depends only on the attention input, so production prefill and the post-verify rebuild stop after the K/V append. Only the golden gate asks `prefill_batched` for logits. `eagle3_kv_only_prefill_matches_full_forward_kv` pins that the two paths leave identical K/V.
- **Chain rewind.** `chain_round` writes transient K/V for its drafts and resets `cached_len`. `rebuild_after_verify` then teacher-forces the accepted prefix (`matched + 1` tokens) from the verify capture.
- **Cache edge.** An all-accepted round commits `k + 1` tokens. `chain_capacity` shrinks `k` near `max_cache_len`, down to a zero-draft span.
- **Position budget.** A request is speculated only if `prompt + max_output + EAGLE3_CHAIN_LENGTH <= draft max positions`. Otherwise it decodes plainly instead of failing at its last positions. There is deliberately no admission cap like DFlash's.
- **Host sync.** Each draft token does a device argmax and reads back one `i32`, because the `d2t` remap is host-side. The earlier prototype copied full draft logits to the host.

## v1 limits (by design, revisit with data)

- Greedy only. The gate covers greedy; sampled-verify would be lossless for a one-hot draft, but it is untested here.
- Single-chunk prompts only. Chunked prompts decode without speculation.
- Drafting is per request through a single-token scratch (bs>1 works serially), with no draft-side CUDA Graph and no `tune_gemm_algos` for drafter shapes.
- `EAGLE3_CHAIN_LENGTH = 3` is inherited from the prototype and has no same-head evidence yet.
- Single GPU only. Rejected at startup with LoRA, KV offload, decode overlap, `--batch-invariant`, and DFlash.

## Validation plan (unverified — none of these have been run on this head)

1. `cargo test --release -p pegainfer-qwen3 --lib eagle3` covers the CPU tests: chain capacity, capture eligibility, reservation.
2. `PEGAINFER_EAGLE3_TEST_MODEL_PATH=models/Qwen3-4B_eagle3 cargo test --release -p pegainfer-qwen3 --lib eagle3 -- --ignored --nocapture` runs the drafter golden gate against SafeAILab/EAGLE (`test_data/qwen3-4b-eagle3-golden.safetensors`), batched-vs-sequential prefill, K/V-only prefill equivalence, and chain mechanics.
3. `PEGAINFER_EAGLE3_TEST_MODEL_PATH=models/Qwen3-4B_eagle3 cargo test --release -p pegainfer-qwen3 --test eagle3_speculative_gate -- --nocapture` runs the lossless gate: single request, concurrent staggered budgets, abort, and a greedy request batched with a sampled one.
4. `PEGAINFER_EAGLE3_TEST_MODEL_PATH=models/Qwen3-4B_eagle3 cargo test --release -p pegainfer-qwen3 --test eagle3_speculative_perf -- --nocapture` is the same-head A/B at c1/c4. Record accepted-length per position from the server's spec-decode counters, and host-sync counts from an nsys trace of one chain round.

Promote any command to verified only after running it, and record the results here.
