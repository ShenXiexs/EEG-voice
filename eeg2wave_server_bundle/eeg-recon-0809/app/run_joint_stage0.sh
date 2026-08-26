#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/joint_pilot_common.sh"
start_joint_log stage0
require_joint_runtime
require_local_hubert

cd "$PROJECT_ROOT"
joint_run "$PYTHON_BIN" scripts/prepare_training_data.py --config "$DATA_CONFIG" audit --strict --fetch-aux
joint_run "$PYTHON_BIN" scripts/prepare_training_data.py --config "$DATA_CONFIG" make-splits
joint_run "$PYTHON_BIN" scripts/prepare_stage2_split.py --data-config "$DATA_CONFIG" --pilot-config "$PILOT_CONFIG"

M0_ARGS=(--data-config "$DATA_CONFIG" --pilot-config "$PILOT_CONFIG" --hubert-local-path "$HUBERT_LOCAL_PATH")
if [[ "${REBUILD_M0:-0}" == "1" ]]; then
  M0_ARGS+=(--rebuild)
fi
joint_run "$PYTHON_BIN" scripts/prepare_m0_artifacts.py "${M0_ARGS[@]}"
joint_run "$PYTHON_BIN" scripts/prepare_training_data.py --config "$DATA_CONFIG" validate --strict

if ! joint_run require_formal_stage0; then
  echo "Stage 0 machine gates passed, but formal M0 remains blocked." >&2
  echo "Review and listen to all rows in:" >&2
  echo "$ARTIFACT_ROOT/qc/ds004940_pair_review_20.csv" >&2
  echo "Set human_listen_transcript_status=pass only after real human verification, rerun this script, then use app/run_joint_after_review.sh." >&2
  exit 3
fi
echo "Stage 0 and the human evidence gate are complete."
