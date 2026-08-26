#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/joint_pilot_common.sh"
start_joint_log smoke
require_joint_runtime

cd "$PROJECT_ROOT"
joint_run "$PYTHON_BIN" scripts/prepare_training_data.py --config "$DATA_CONFIG" validate --strict
joint_run "$PYTHON_BIN" app/train_joint.py --config "$PILOT_CONFIG" --mode joint --stage overfit --dry-run
joint_run "$PYTHON_BIN" app/train_audio_renderer.py --config "$PILOT_CONFIG" --dry-run

SEED="${SEED:-925}"
SMOKE_STEPS="${SMOKE_STEPS:-5}"
joint_run "$PYTHON_BIN" app/train_joint.py --config "$PILOT_CONFIG" --mode joint --stage overfit \
  --seed "$SEED" --smoke-model --max-steps "$SMOKE_STEPS"
CHECKPOINT="$RUN_ROOT/smoke/overfit/joint/seed-$SEED/checkpoint.pt"
for dataset in ds004940 ds006104; do
  joint_run "$PYTHON_BIN" app/evaluate_joint.py --checkpoint "$CHECKPOINT" --dataset "$dataset" --role train
done
echo "Smoke completed. These outputs are engineering diagnostics and never satisfy formal M0 gates."
