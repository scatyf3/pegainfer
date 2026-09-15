//! Worker-side EAGLE-3 draft lane: the drafter plus per-request draft state.
//!
//! Parallels [`super::dflash_lane`] for the EAGLE-3 chain drafter. It lives on
//! the worker thread next to the target because the rollout reuses the target's
//! `embed_tokens` and consumes its captured low/mid/high hidden states; those
//! features stay private to this lane, so the draft/verify boundary remains a
//! pure token span.
//!
//! Flow: the prefill capture seeds an [`Eagle3RequestState`] for each eligible
//! request, `execute_eagle3_draft` rolls the top-1 chain, and
//! `execute_eagle3_verify` runs the shared verify forward and rebuilds the
//! draft K/V for the accepted prefix. Accept/commit is the drafter-agnostic
//! transaction in [`super::spec`] — losslessness comes from there, not here.

use std::collections::HashMap;

use anyhow::Result;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_kv_cache::KvView;

use super::LocalQwen3Lane;
use super::PrefillStepItem;
use super::RequestId;
use super::dflash_prefill::dflash_prefill_can_capture;
use crate::batch_decode_buffers::BATCH_BUCKETS;
use crate::eagle3::EAGLE3_CHAIN_LENGTH;
use crate::eagle3::Eagle3DraftModel;
use crate::eagle3::Eagle3RequestState;
use crate::eagle3::Eagle3Scratch;
use crate::speculative::DraftRequestResult;
use crate::speculative::DraftResult;
use crate::speculative::DraftStepItem;
use crate::speculative::VerifyRequestResult;
use crate::speculative::VerifyResult;
use crate::speculative::VerifyStepItem;
use crate::speculative::build_verify_results;
use crate::verify_graph::VerifyGraphBuffers;

pub(super) struct Eagle3LaneState {
    model: Eagle3DraftModel,
    /// Per-request single-layer draft K/V plus the boundary target feature.
    /// Populated by the prefill capture once a prompt finishes prefilling.
    requests: HashMap<RequestId, Eagle3RequestState>,
    /// Single-token draft scratch, reused across steps. v1 drafts one request at
    /// a time; batching the chain is a follow-up.
    scratch: Eagle3Scratch,
    /// Target layers whose hidden states feed the drafter (low/mid/high).
    aux_layer_ids: [usize; 3],
    verified_draft_tokens: usize,
    accepted_draft_tokens: usize,
}

impl Eagle3LaneState {
    pub(super) fn new(
        ctx: &DeviceContext,
        model: Eagle3DraftModel,
        aux_layer_ids: [usize; 3],
    ) -> Result<Self> {
        let scratch = model.new_scratch(ctx)?;
        Ok(Self {
            model,
            requests: HashMap::new(),
            scratch,
            aux_layer_ids,
            verified_draft_tokens: 0,
            accepted_draft_tokens: 0,
        })
    }
}

/// Whether a prefill request can seed the EAGLE-3 drafter (v1 scope).
///
/// On top of DFlash's clean-capture rules (no LoRA, no prefix hit, no
/// logprobs) a request must:
/// - use a greedy sampler — the losslessness gate covers greedy only;
/// - prefill its whole prompt in one chunk — the feature↔token shift must not
///   cross a chunk boundary;
/// - have at least two prompt tokens — the shift consumes one;
/// - fit `prompt + max_output + EAGLE3_CHAIN_LENGTH` inside the drafter's
///   position table — every transient chain slot needs a RoPE row, and a
///   length-capped generation must not fail at its final positions.
///
/// Anything else decodes without speculation.
fn eagle3_prefill_can_capture(req: &PrefillStepItem, max_positions: usize) -> bool {
    let prompt_len = req.prompt_tokens.len();
    dflash_prefill_can_capture(req, false)
        && req.params.is_greedy()
        && req.chunk_start == 0
        && req.chunk_tokens == prompt_len
        && prompt_len >= 2
        && prompt_len
            .saturating_add(req.max_output_tokens)
            .saturating_add(EAGLE3_CHAIN_LENGTH)
            <= max_positions
}

impl LocalQwen3Lane {
    /// Target layers whose hidden states the drafter consumes (None when
    /// EAGLE-3 is not loaded).
    pub(super) fn eagle3_capture_layer_ids(&self) -> Option<Vec<usize>> {
        self.eagle3
            .as_ref()
            .map(|eagle3| eagle3.aux_layer_ids.to_vec())
    }

    pub(super) fn should_capture_eagle3_prefill_context(
        &self,
        requests: &[PrefillStepItem],
    ) -> bool {
        let Some(eagle3) = self.eagle3.as_ref() else {
            return false;
        };
        let max_positions = eagle3.model.config.max_position_embeddings;
        requests
            .iter()
            .any(|req| eagle3_prefill_can_capture(req, max_positions))
    }

    /// Seed each eligible request's draft state from the hidden states captured
    /// during prefill: teacher-force the prompt into the draft K/V and keep the
    /// boundary feature. Returns the requests that captured context this step.
    pub(super) fn record_prefill_eagle3_context(
        &mut self,
        requests: &[PrefillStepItem],
        capture_requested: bool,
        captured_hidden: Option<&HiddenStates>,
    ) -> Result<Vec<RequestId>> {
        let Some(captured_hidden) = captured_hidden else {
            anyhow::ensure!(
                !capture_requested,
                "EAGLE-3 prefill context capture was requested but no hidden states were returned"
            );
            return Ok(Vec::new());
        };
        anyhow::ensure!(
            capture_requested,
            "EAGLE-3 prefill hidden states were returned without a capture request"
        );
        let expected_tokens: usize = requests.iter().map(|req| req.chunk_tokens).sum();
        anyhow::ensure!(
            captured_hidden.seq_len == expected_tokens,
            "EAGLE-3 prefill captured {} hidden rows for {} scheduled tokens",
            captured_hidden.seq_len,
            expected_tokens
        );
        // Split borrow: the draft prefill reads the target's embeddings while the
        // lane's draft state is mutated.
        let LocalQwen3Lane { model, eagle3, .. } = self;
        let Some(eagle3) = eagle3.as_mut() else {
            anyhow::bail!("EAGLE-3 prefill context record requested without EAGLE-3");
        };
        let max_positions = eagle3.model.config.max_position_embeddings;
        let mut captured_requests = Vec::new();
        let mut token_offset = 0usize;
        for req in requests {
            // v1 captures whole prompts only, so any earlier state is stale.
            eagle3.requests.remove(&req.request_id);
            if eagle3_prefill_can_capture(req, max_positions) {
                // Committed prompt + decode, plus one chain of transient draft slots.
                let max_cache_len =
                    req.prompt_tokens.len() + req.max_output_tokens + EAGLE3_CHAIN_LENGTH;
                let mut state = eagle3
                    .model
                    .new_request_state(model.device_ctx(), max_cache_len)?;
                eagle3.model.prefill_prompt(
                    model,
                    &mut state,
                    captured_hidden,
                    token_offset,
                    &req.prompt_tokens,
                )?;
                eagle3.requests.insert(req.request_id, state);
                captured_requests.push(req.request_id);
            }
            token_offset += req.chunk_tokens;
        }
        Ok(captured_requests)
    }

    /// One draft round per request: roll the top-1 chain forward from the
    /// carried boundary feature. Returns the verify span
    /// `[current_token, draft_1, …]`; near the draft cache edge the chain
    /// shrinks, down to no drafts at all.
    pub(super) fn execute_eagle3_draft(
        &mut self,
        requests: &[DraftStepItem],
    ) -> Result<DraftResult> {
        anyhow::ensure!(
            !requests.is_empty(),
            "EAGLE-3 draft requested without active requests"
        );
        let LocalQwen3Lane { model, eagle3, .. } = self;
        let Some(eagle3) = eagle3.as_mut() else {
            anyhow::bail!("EAGLE-3 draft requested but EAGLE-3 is not loaded");
        };
        let Eagle3LaneState {
            model: draft,
            requests: states,
            scratch,
            ..
        } = eagle3;
        let mut outputs = Vec::with_capacity(requests.len());
        for req in requests {
            let state = states
                .get_mut(&req.request_id)
                .ok_or_else(|| anyhow::anyhow!("missing EAGLE-3 state for {:?}", req.request_id))?;
            let drafts = draft.chain_round(
                model,
                state,
                scratch,
                req.current_token,
                EAGLE3_CHAIN_LENGTH,
            )?;
            let mut token_ids = Vec::with_capacity(drafts.len() + 1);
            token_ids.push(req.current_token);
            token_ids.extend(drafts);
            outputs.push(DraftRequestResult {
                request_id: req.request_id,
                token_ids,
            });
        }
        Ok(DraftResult { requests: outputs })
    }

    /// EAGLE-3 verify forward: one target prefill over every request's span into
    /// the fixed [`VerifyGraphBuffers`], all-position committed tokens under each
    /// request's params, capture at the EAGLE aux layers, then the draft K/V
    /// rebuild for the accepted prefix. The same plain pass as
    /// `execute_dflash_verify`, without the hedge ladder.
    pub(super) fn execute_eagle3_verify(
        &mut self,
        requests: &[VerifyStepItem],
        kv_views: &[KvView],
        sample_seed: u64,
    ) -> Result<VerifyResult> {
        let capture_layer_ids = self
            .eagle3_capture_layer_ids()
            .ok_or_else(|| anyhow::anyhow!("EAGLE-3 verify requested but no drafter is loaded"))?;

        if self.verify_bufs.is_none() {
            // Base page lists are disjoint pool pages (sum <= total_blocks): a
            // loaded drafter force-disables the prefix cache, and EAGLE-3 never
            // hedges, so no page list is duplicated.
            let max_batch = *BATCH_BUCKETS.last().unwrap();
            self.verify_bufs = Some(VerifyGraphBuffers::new(
                &self.model,
                max_batch,
                EAGLE3_CHAIN_LENGTH + 1,
                capture_layer_ids.len(),
                self.total_blocks,
            )?);
        }

        // Take the buffers out of `self` so the forward, the sampling, and the
        // draft rebuild don't alias a `self` borrow.
        let mut bufs = self.verify_bufs.take().expect("verify buffers just set");
        let result = (|| -> Result<VerifyResult> {
            let spans: Vec<&[u32]> = requests.iter().map(VerifyStepItem::as_slice).collect();
            self.model.batch_prefill_into(
                &spans,
                kv_views,
                self.kv_buffer.buffer(),
                &self.layout,
                &capture_layer_ids,
                &mut bufs,
            )?;
            let params: Vec<&SamplingParams> = requests
                .iter()
                .flat_map(|req| std::iter::repeat_n(&req.params, req.as_slice().len()))
                .collect();
            let target_tokens = self.select_step_tokens(bufs.all_logits(), &params, sample_seed)?;
            let request_results = build_verify_results(requests, &target_tokens)?;
            self.record_verify_eagle3_context(requests, &request_results, bufs.captured_hidden())?;
            Ok(VerifyResult {
                requests: request_results,
            })
        })();
        self.verify_bufs = Some(bufs);
        result
    }

    /// Rebuild each request's draft K/V for the accepted prefix from the verify
    /// capture and store the next boundary feature. The chain's own K/V writes
    /// were already rewound in `chain_round`.
    fn record_verify_eagle3_context(
        &mut self,
        requests: &[VerifyStepItem],
        results: &[VerifyRequestResult],
        captured_hidden: &HiddenStates,
    ) -> Result<()> {
        anyhow::ensure!(
            requests.len() == results.len(),
            "EAGLE-3 verify result count {} does not match request count {}",
            results.len(),
            requests.len()
        );
        let expected_tokens: usize = requests.iter().map(|req| req.token_ids.len()).sum();
        anyhow::ensure!(
            captured_hidden.seq_len == expected_tokens,
            "EAGLE-3 verify captured {} hidden rows for {} span tokens",
            captured_hidden.seq_len,
            expected_tokens
        );
        let LocalQwen3Lane { model, eagle3, .. } = self;
        let Some(eagle3) = eagle3.as_mut() else {
            anyhow::bail!("EAGLE-3 verify context record requested without EAGLE-3");
        };
        let mut token_offset = 0usize;
        for (req, result) in requests.iter().zip(results) {
            anyhow::ensure!(
                req.request_id == result.request_id,
                "EAGLE-3 verify result {:?} does not match request {:?}",
                result.request_id,
                req.request_id
            );
            let state = eagle3.requests.get_mut(&req.request_id).ok_or_else(|| {
                anyhow::anyhow!("missing EAGLE-3 state after verify for {:?}", req.request_id)
            })?;
            eagle3.model.rebuild_after_verify(
                model,
                state,
                captured_hidden,
                token_offset,
                &req.token_ids,
                result.matched_draft_tokens,
            )?;
            eagle3.verified_draft_tokens += req.token_ids.len().saturating_sub(1);
            eagle3.accepted_draft_tokens += result.matched_draft_tokens;
            let rate = if eagle3.verified_draft_tokens == 0 {
                0.0
            } else {
                eagle3.accepted_draft_tokens as f64 / eagle3.verified_draft_tokens as f64
            };
            log::debug!(
                "Qwen3 EAGLE-3 request={} accepted_draft={} committed_tokens={} cumulative_accept_rate={:.3}",
                req.request_id.raw(),
                result.matched_draft_tokens,
                result.accepted_tokens.len(),
                rate,
            );
            token_offset += req.token_ids.len();
        }
        Ok(())
    }

    pub(super) fn drop_eagle3_request(&mut self, request_id: RequestId) {
        if let Some(eagle3) = self.eagle3.as_mut() {
            eagle3.requests.remove(&request_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_POSITIONS: usize = 64;

    fn prefill_item(prompt_len: usize, max_output_tokens: usize) -> PrefillStepItem {
        PrefillStepItem {
            request_id: RequestId::new(1),
            prompt_tokens: vec![7; prompt_len],
            max_output_tokens,
            params: SamplingParams::default(),
            logprobs: None,
            prompt_logprobs: None,
            lora_adapter: None,
            cached_tokens: 0,
            chunk_budget: prompt_len,
            chunk_start: 0,
            chunk_tokens: prompt_len,
        }
    }

    #[test]
    fn single_chunk_greedy_prompt_is_eligible() {
        assert!(eagle3_prefill_can_capture(
            &prefill_item(8, 16),
            MAX_POSITIONS
        ));
    }

    #[test]
    fn position_budget_counts_the_transient_chain() {
        let fits = MAX_POSITIONS - 8 - EAGLE3_CHAIN_LENGTH;
        assert!(eagle3_prefill_can_capture(
            &prefill_item(8, fits),
            MAX_POSITIONS
        ));
        assert!(!eagle3_prefill_can_capture(
            &prefill_item(8, fits + 1),
            MAX_POSITIONS
        ));
    }

    #[test]
    fn unsupported_requests_decode_without_speculation() {
        let mut first_of_many_chunks = prefill_item(8, 16);
        first_of_many_chunks.chunk_tokens = 4;
        let mut later_chunk = prefill_item(8, 16);
        later_chunk.chunk_start = 4;
        later_chunk.chunk_tokens = 4;
        let mut sampled = prefill_item(8, 16);
        sampled.params = SamplingParams {
            temperature: 0.7,
            top_k: 50,
            ..SamplingParams::default()
        };
        let mut prefix_hit = prefill_item(8, 16);
        prefix_hit.cached_tokens = 4;
        let mut logprobs = prefill_item(8, 16);
        logprobs.logprobs = Some(0);
        for (name, req) in [
            ("first of many chunks", first_of_many_chunks),
            ("later chunk", later_chunk),
            ("single prompt token", prefill_item(1, 16)),
            ("sampled", sampled),
            ("prefix hit", prefix_hit),
            ("logprobs", logprobs),
        ] {
            assert!(
                !eagle3_prefill_can_capture(&req, MAX_POSITIONS),
                "{name} must not seed the drafter"
            );
        }
    }
}
