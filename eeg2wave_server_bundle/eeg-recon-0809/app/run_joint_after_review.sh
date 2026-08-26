#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/joint_pilot_common.sh"
start_joint_log after_review
require_joint_runtime

cd "$PROJECT_ROOT"
joint_run "$PYTHON_BIN" scripts/prepare_training_data.py --config "$DATA_CONFIG" validate --strict
joint_run require_formal_stage0
joint_run "$SCRIPT_DIR/run_joint_audio_oracle.sh"
joint_run "$SCRIPT_DIR/run_joint_m0.sh" all
joint_run "$SCRIPT_DIR/run_joint_stage2.sh"
