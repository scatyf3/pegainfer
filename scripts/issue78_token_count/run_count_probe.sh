#!/bin/bash
# Issue #78 repro (per-request): start pegainfer, compare streamed vs reported tokens.
# Usage: bash scripts/issue78_token_count/run_count_probe.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

start_server /tmp/pega_issue78_count_server.log || exit 1
PORT="$PORT" MODEL_NAME="$MODEL_PATH" python3 "$REPO_ROOT/scripts/issue78_token_count/count_probe.py"
rc=$?
stop_server
exit $rc
