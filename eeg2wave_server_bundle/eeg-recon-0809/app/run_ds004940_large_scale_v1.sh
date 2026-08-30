#!/usr/bin/env bash
# Isolated, resumable DS004940 N400Active large-scale double-OOD experiment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PILOT_CONFIG="${PILOT_CONFIG:-$PROJECT_ROOT/configs/ds004940_large_scale_v1.yaml}"
source "$SCRIPT_DIR/lib/joint_pilot_common.sh"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-ds004940_large_scale_v1}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$RUN_ROOT/$EXPERIMENT_NAME}"
MAX_EPOCHS="${DS004940_LARGE_MAX_EPOCHS:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"

start_joint_log "$EXPERIMENT_NAME"
require_joint_runtime
require_local_hubert
cd "$PROJECT_ROOT"

IFS=' ' read -r -a SEEDS <<< "${DS004940_SEEDS:-$(pilot_seeds)}"
echo "WARNING: exploratory only. Human DS004940 pair review is still pending."
echo "experiment_root=$EXPERIMENT_ROOT"
echo "config=$PILOT_CONFIG"
echo "epochs_per_seed=$MAX_EPOCHS seeds=${SEEDS[*]} checkpoint_every=$CHECKPOINT_EVERY"
echo "This runner is resumable. Re-run this exact command after an interrupt; do not use --restart."

STAGE2_ARGS=(--data-config "$DATA_CONFIG" --pilot-config "$PILOT_CONFIG" --explore --materialize \
  --hubert-local-path "$HUBERT_LOCAL_PATH")
if [[ "${REBUILD_DS004940_LARGE:-0}" == "1" ]]; then STAGE2_ARGS+=(--rebuild); fi
joint_run "$PYTHON_BIN" scripts/prepare_stage2_split.py "${STAGE2_ARGS[@]}"
joint_run "$PYTHON_BIN" scripts/prepare_stage2_split.py --data-config "$DATA_CONFIG" \
  --pilot-config "$PILOT_CONFIG" --explore --check-readiness

for seed in "${SEEDS[@]}"; do
  joint_run "$PYTHON_BIN" app/train_joint.py --config "$PILOT_CONFIG" --mode ds004940 \
    --stage generalization --seed "$seed" --explore --max-epochs "$MAX_EPOCHS" \
    --checkpoint-every "$CHECKPOINT_EVERY" --output-root "$EXPERIMENT_ROOT"
  CHECKPOINT="$EXPERIMENT_ROOT/generalization/ds004940/seed-$seed/checkpoint.pt"
  for role in validation test; do
    joint_run "$PYTHON_BIN" app/evaluate_joint.py --checkpoint "$CHECKPOINT" \
      --dataset ds004940 --role "$role" \
      --renderer-checkpoint "$RUN_ROOT/audio_renderer_explore/checkpoint.pt"
  done
done

joint_run "$PYTHON_BIN" app/plot_ds004940_large_scale.py \
  --large-root "$EXPERIMENT_ROOT" \
  --small-root "$RUN_ROOT/ds004940_explore_10h_v1" \
  --seeds "$(IFS=,; echo "${SEEDS[*]}")" --formats png,pdf --dpi 300

echo "DS004940 large-scale exploratory training complete. Outputs are not registered evidence."
echo "results=$EXPERIMENT_ROOT/generalization/ds004940"
echo "figures=$EXPERIMENT_ROOT/generalization/figures"
echo "Next: ./app/run_ds004940_large_scale_audio_comparisons.sh"
