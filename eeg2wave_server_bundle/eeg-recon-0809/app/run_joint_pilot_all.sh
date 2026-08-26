#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/joint_pilot_common.sh"
start_joint_log complete
require_joint_runtime

set +e
joint_run "$SCRIPT_DIR/run_joint_stage0.sh"
STATUS=$?
set -e
if [[ "$STATUS" -eq 3 ]]; then
  echo "Pipeline paused at the registered human evidence gate (expected and safe)." >&2
  echo "After completing the review CSV, run: $SCRIPT_DIR/run_joint_after_review.sh" >&2
  exit 3
fi
if [[ "$STATUS" -ne 0 ]]; then
  echo "Stage 0 failed with status $STATUS; later stages were not started." >&2
  exit "$STATUS"
fi
joint_run "$SCRIPT_DIR/run_joint_after_review.sh"
