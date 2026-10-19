# EAGLE-3 Speculative Decoding (Qwen3-4B)

**TL;DR:** An opt-in EAGLE-3 chain drafter (`--eagle3-draft-model-path`, e.g. `AngelSlim/Qwen3-4B_eagle3`) now runs through the same speculative transaction as DFlash: the scheduler, `DraftPlan → DraftResult → VerifyPlan → VerifyResult`, and the executor's verify/accept/KV commit are shared, and only the proposer lane is EAGLE-specific. v1 speculates greedy requests whose whole prompt fits one prefill chunk. Every other request, and any batch containing one, decodes normally. The code is a **draft that has not been compiled or run yet**: it was written off-GPU, and none of the gates or A/B numbers below exist. Next step: build it on a CUDA box and run the drafter golden gate, then the lossless gate, then the A/B.

Last touched: 2026-10

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
- γ is fixed at `EAGLE3_CHAIN_LENGTH = 3` (public, the single source of truth). The prototype's chain sweep put the optimum there (2→1.30×, 3→1.32×, 4→1.26× on 5070 Ti); this head has no same-head evidence yet. The vLLM comparison reads γ back from the harness, so the two engines cannot drift apart.
- Single GPU only. Rejected at startup with LoRA, KV offload, decode overlap, `--batch-invariant`, and DFlash.

## Validation plan (unverified — none of these have been run on this head)

1. `cargo test --release -p pegainfer-qwen3 --lib eagle3` covers the CPU tests: chain capacity, capture eligibility, reservation.
2. `PEGAINFER_EAGLE3_TEST_MODEL_PATH=models/Qwen3-4B_eagle3 cargo test --release -p pegainfer-qwen3 --lib eagle3 -- --ignored --nocapture` runs the drafter golden gate against SafeAILab/EAGLE (`test_data/qwen3-4b-eagle3-golden.safetensors`), batched-vs-sequential prefill, K/V-only prefill equivalence, and chain mechanics.
3. `PEGAINFER_EAGLE3_TEST_MODEL_PATH=models/Qwen3-4B_eagle3 cargo test --release -p pegainfer-qwen3 --test eagle3_speculative_gate -- --nocapture` runs the lossless gate: single request, concurrent staggered budgets, abort, and a greedy request batched with a sampled one.
4. `PEGAINFER_EAGLE3_TEST_MODEL_PATH=models/Qwen3-4B_eagle3 cargo test --release -p pegainfer-qwen3 --test eagle3_speculative_perf -- --nocapture` is the same-head A/B at c1/c4. Record accepted-length per position from the server's spec-decode counters, and host-sync counts from an nsys trace of one chain round.

5. `MODEL=models/Qwen3-4B DRAFT=models/Qwen3-4B_eagle3 VLLM_PY=<python with vllm> tools/bench/eagle3_vllm_profile.sh` runs the γ-aligned comparison against vLLM's EAGLE-3. Both engines get the same drafter, the same GSM8K 8-shot prompt **token ids** (`eagle3_gsm8k_ab` dumps them and vLLM consumes the dump), `num_speculative_tokens = EAGLE3_CHAIN_LENGTH`, greedy with `ignore_eos`, 256 tokens, and bs=1. It reports decode TPOT separately from TTFT, and τ / per-position accept from each engine's own counters. It then captures nsys traces of the decode loop only (`cudaProfilerApi` range, `--cuda-graph-trace=node`) and writes `report.md` with a per-token breakdown: GPU busy/idle, kernel time by family, launch/graph counts, D2H and host-sync counts, gap buckets, and the largest gaps.

Why this exists: the prototype compared *speedup ratios* (ours 1.32× vs vLLM 0.21 ~1.23×). The two engines have different baselines and the vLLM γ was not recorded, so the ratios don't say which engine decodes faster. The comparison that counts is absolute spec-on TPOT at equal γ and τ. If τ matches, the remaining gap is per-round overhead, and that is what the nsys breakdown attributes. The suspects are this head's eager per-token draft (no draft graph, one argmax D2H sync per draft token, host-side `d2t`) against vLLM's graphed drafter, and the post-verify K/V rebuild.

Promote any command to verified only after running it, and record the results here.
