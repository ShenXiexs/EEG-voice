#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/joint_pilot_common.sh"
start_joint_log audio_oracle
require_joint_runtime

cd "$PROJECT_ROOT"
joint_run "$PYTHON_BIN" scripts/prepare_training_data.py --config "$DATA_CONFIG" validate --strict
ARGS=(--config "$PILOT_CONFIG")
if [[ -n "${MAX_STEPS:-}" ]]; then ARGS+=(--max-steps "$MAX_STEPS"); fi
joint_run "$PYTHON_BIN" app/train_audio_renderer.py "${ARGS[@]}"
echo "Audio-only MFCC-to-acoustic renderer gate passed. Waveform generation remains disabled without a validated vocoder."
