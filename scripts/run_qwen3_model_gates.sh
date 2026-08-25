#!/usr/bin/env bash
#
# Qwen3 maintained model gates.
#
# Every gate below is `#[ignore]`d, so an ordinary `cargo test` never touches it
# and a repository-wide `-- --ignored` sweep gives no evidence about which
# checkpoint actually loaded. This runner names each gate, runs it on its own,
# and asserts libtest actually executed it — a gate that silently selects
# nothing is a failure here, not a quiet pass.
#
# Requires: one CUDA device, Qwen3-4B weights, DFlash draft weights, a LoRA
# adapter fixture.
#
#   PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B \
#   PEGAINFER_DFLASH_TEST_MODEL_PATH=models/Qwen3-4B-DFlash-b16 \
#     scripts/run_qwen3_model_gates.sh
#
set -uo pipefail

CRATE="pegainfer-qwen3"
CARGO_ARGS=(--release --locked -p "$CRATE")

# "<cargo target selector>|<exact libtest name>"
# Keep this manifest explicit: no globbing, no source grep, no naming
# convention. Adding a gate means adding a line here.
GATES=(
  "--test dflash_sampled_equivalence|dflash_sampled_equivalence_gate"
  "--test dflash_sampled_equivalence|dflash_sampled_equivalence_null_check"
  "--test dflash_sampled_equivalence|dflash_sampled_seeded_determinism"
  "--test lora_smoke|qwen3_lora_loads_adapter_and_generates"
  "--test lora_smoke|qwen3_lora_loads_rank64_adapter_and_generates"
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

echo "=== Qwen3 model gates ==="
echo "commit:   ${commit}${dirty}"
echo "crate:    ${CRATE}"
echo "selected: ${selected}"
echo

log_dir="$(mktemp -d)"
trap 'rm -rf "$log_dir"' EXIT

for entry in "${GATES[@]}"; do
  target="${entry%%|*}"
  name="${entry##*|}"
  read -r -a target_args <<<"$target"

  echo "--- ${name} (${target}) ---"
  log="${log_dir}/${name}.log"

  # --ignored runs *only* ignored tests; --exact pins us to this one gate.
  if ! cargo test "${CARGO_ARGS[@]}" "${target_args[@]}" \
       -- --exact "$name" --ignored --nocapture 2>&1 | tee "$log"; then
    failures+=("$name: cargo test exited non-zero")
    echo
    continue
  fi

  # A green exit code is not enough. libtest reports `ok. 0 passed` when the
  # filter matched nothing (renamed gate, dropped #[ignore], typo in the
  # manifest), which reads as success. Require exactly one executed test.
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

echo "=== Qwen3 model gates: summary ==="
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

echo "OK: all ${selected} Qwen3 model gates executed"
