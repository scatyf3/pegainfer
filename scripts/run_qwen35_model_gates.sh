#!/usr/bin/env bash
#
# Qwen3.5 maintained model gates.
#
# All eleven gates are `#[ignore]`d and spread across four cargo targets — the
# crate's own lib target plus three integration binaries. A repository-wide
# `-- --ignored` sweep neither reaches them uniformly nor proves any of them
# loaded a checkpoint. This runner names each gate, runs it on its own, and
# asserts libtest actually executed it.
#
# Requires: two CUDA devices, NCCL, and Qwen3.5 weights.
#
#   PEGAINFER_TEST_MODEL_PATH=models/Qwen3.5-4B \
#     scripts/run_qwen35_model_gates.sh
#
# Optional: PEGAINFER_TEST_TP_DEVICES=0,1 to pick the TP2 ordinals.
#
set -uo pipefail

CRATE="pegainfer-qwen35"
CARGO_ARGS=(--release --locked -p "$CRATE" --features qwen35)

# "<cargo target selector>|<exact libtest name>"
# Keep this manifest explicit: no globbing, no source grep, no naming
# convention. Adding a gate means adding a line here.
#
# The lib-target names carry their module path; the integration-target ones are
# bare because each lives at the root of its own test binary.
GATES=(
  "--lib|tp_executor::tests::tp2_drop_expectations_detect_rank_lifecycle_divergence"
  "--lib|tp_executor::tests::tp2_partial_dispatch_gate_prevents_rank_local_mutation"
  "--lib|tp_executor::tests::tp2_worker_receiver_disconnect_poisons_without_snapshot_claim"
  "--lib|tp_executor::tests::tp2_unified_step_advances_prefill_and_decode_together"
  "--lib|tp_executor::tests::tp2_drop_all_restores_complete_request_capacity"
  "--lib|tp_executor::tests::tp2_readmission_matches_clean_first_token_artifact"
  "--lib|scheduler::tests::tp2_scheduler_runs_forced_mixed_steps"
  "--test e2e_scheduler|test_e2e_qwen35_scheduler_tp2"
  "--test hf_golden_gate|pega_logprobs_match_hf_golden_within_qwen35_tolerance_tp2"
  "--test hf_golden_gate|pega_logprobs_match_hf_long_golden_within_qwen35_tolerance_tp2"
  "--test serving_tp2|qwen35_tp2_serves_openai_completions_over_http"
)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
dirty=""
if ! git diff --quiet HEAD 2>/dev/null; then
  dirty=" (dirty worktree)"
fi

selected=${#GATES[@]}
completed=0
failures=()

echo "=== Qwen3.5 model gates ==="
echo "commit:   ${commit}${dirty}"
echo "crate:    ${CRATE} (--features qwen35)"
echo "selected: ${selected}"
echo

log_dir="$(mktemp -d)"
trap 'rm -rf "$log_dir"' EXIT

for entry in "${GATES[@]}"; do
  target="${entry%%|*}"
  name="${entry##*|}"
  read -r -a target_args <<<"$target"

  echo "--- ${name} (${target}) ---"
  log="${log_dir}/$(tr '/:' '__' <<<"$name").log"

  # --ignored runs *only* ignored tests; --exact pins us to this one gate.
  if ! cargo test "${CARGO_ARGS[@]}" "${target_args[@]}" \
       -- --exact "$name" --ignored --nocapture 2>&1 | tee "$log"; then
    failures+=("$name: cargo test exited non-zero")
    echo
    continue
  fi

  # A green exit code is not enough. libtest reports `ok. 0 passed` when the
  # filter matched nothing (renamed gate, dropped #[ignore], moved module,
  # typo in the manifest), which reads as success. Require exactly one
  # executed test.
  summary="$(grep -E '^test result:' "$log" | tail -1)"
  passed="$(sed -nE 's/^test result: ok\. ([0-9]+) passed.*/\1/p' <<<"$summary")"
  if [[ "$passed" != "1" ]]; then
    failures+=("$name: expected 1 passed, libtest said '${summary:-<no result line>}'")
    echo
    continue
  fi

  completed=$((completed + 1))
  echo
done

echo "=== Qwen3.5 model gates: summary ==="
echo "commit:    ${commit}${dirty}"
echo "selected:  ${selected}"
echo "completed: ${completed}"

if ((${#failures[@]} > 0)); then
  echo
  echo "FAILED gates:"
  printf '  - %s\n' "${failures[@]}"
  exit 1
fi

if ((completed != selected)); then
  echo
  echo "FAILED: completed (${completed}) != selected (${selected})"
  exit 1
fi

echo "OK: all ${selected} Qwen3.5 model gates executed"
