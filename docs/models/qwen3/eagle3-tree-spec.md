# EAGLE-3 tree speculative decoding

**TL;DR:** Moving EAGLE-3 from a top-1 linear chain (γ=3, ~1.30×) to a static draft **tree** with tree-attention verify, to raise the accepted-tokens-per-step ceiling toward ~1.8–2×. Phase 1 (custom-mask attention foundation) is **done and bit-exact**; Phase 2 (static tree topology + draft tree decode) is in progress.

Last touched: 2026-07

## Why tree (the chain ceiling)

The shipped chain is structurally capped. Measured on a Qwen3-4B GSM8K A/B (RTX 5070 Ti): per-draft acceptance decays geometrically **d₁≈62% → d₂≈45% → d₃≈30% → …**, so mean accepted length saturates at **τ≈2.0 by γ≈3**, while each extra chain draft costs a full forward. γ sweep: 2→1.30×, **3→1.32×**, 4→1.26×, 7→1.13×. The standard EAGLE-3 γ=7 is a *tree* budget (7 nodes fanned out, verified in one target pass); on a linear chain it is pure overhead.

vLLM 0.21's EAGLE-3 also runs a chain and also only gets ~1.23× — direct evidence the ceiling is the **chain structure**, not per-step overhead. Only a draft tree (multiple candidates per depth, verified together under a tree-attention mask) lifts τ toward 3–4.

## Components (6, not 3)

The naïve "draft kernel / verify kernel / KV buffer" split hides three things:

| # | Work | Status |
|---|------|--------|
| A | **custom-mask attention** (single-seq for draft + paged for verify) — FlashInfer `DefaultAttention<custom_mask=true>` + `MaskMode::kCustom`, plumbed through FFI/ops | single-seq **done** (Phase 1); paged pending (Phase 3) |
| B | static tree topology (parent/depth/packed ancestor mask, host-built constant) | Phase 2 |
| C | draft tree decode: per-depth batched forward + tree mask + device-side top-k selection | Phase 2 |
| D | target tree verify: paged custom-mask, **depth-indexed** positions (not contiguous), aux capture per node | Phase 3 |
| E | tree-aware accept: longest matching **path** (replaces the linear prefix walk in `speculative.rs::num_accepted`) | Phase 3 |
| F | KV: draft buffer grows to tree_size (easy); target **accepted-path gather** into contiguous slots (new, the real work in "KV part") | Phase 3 |

Key constraints found in the codebase:
- Attention was hardcoded causal (`paged_attention.cu`); Phase 1 added the custom-mask variant.
- Verify positions are contiguous (`verify_graph.rs`); a tree needs position = base + node depth (siblings share a position).
- `apply_speculative` (openinfer-kv-cache) assumes the accepted prefix is the physically-first N slots — a tree path is scattered, so F needs a gather.

## Phases

- **Phase 1 — custom-mask foundation (done).** `single_prefill_nhd_custom_mask_into`; validated bit-exact vs the causal kernel by feeding an equivalent causal mask (`max|Δ|=0`). Commit on `feat/eagle3-tree`.
- **Phase 2 — static tree + draft decode.** Topology module (`eagle3/tree.rs`), device-side top-k selection (reuse `select_batch`, subsumes the chain's per-step `to_host`), per-depth draft rollout writing tree KV slots. Start small: depth 3, per-depth branching (tunable), ~15–30 nodes.
- **Phase 3 — verify + accept + KV.** Paged custom-mask, depth-indexed positions, longest-path accept, accepted-path KV gather, drafter reseed along the path.
- **Phase 4 — CUDA graph + tune.** Static mask → graphable; A/B tree shape on GSM8K like the γ=3 chain tuning; extend the losslessness golden gate to `tree output == greedy output`.

## Next action

Phase 2: land `eagle3/tree.rs` (topology + packed ancestor mask + unit tests), then wire per-depth draft rollout on top of `single_prefill_nhd_custom_mask_into`.

Branch: `feat/eagle3-tree` (on top of the rebased chain branch `feat/eagle3-loader`).
