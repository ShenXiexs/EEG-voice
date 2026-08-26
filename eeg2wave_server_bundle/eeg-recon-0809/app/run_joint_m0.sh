#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/joint_pilot_common.sh"
start_joint_log m0
require_joint_runtime

MODE="${1:-all}"
case "$MODE" in
  ds004940|ds006104|joint) MODES=("$MODE") ;;
  all) MODES=(ds004940 ds006104 joint) ;;
  *) echo "usage: $0 [ds004940|ds006104|joint|all]" >&2; exit 2 ;;
esac

cd "$PROJECT_ROOT"
joint_run "$PYTHON_BIN" scripts/prepare_training_data.py --config "$DATA_CONFIG" validate --strict
IFS=' ' read -r -a SEEDS <<< "${SEEDS:-$(pilot_seeds)}"

for mode in "${MODES[@]}"; do
  if [[ "$mode" != "ds006104" ]]; then
    joint_run require_formal_stage0
  fi
  for seed in "${SEEDS[@]}"; do
    TRAIN_ARGS=(--config "$PILOT_CONFIG" --mode "$mode" --stage overfit --seed "$seed")
    if [[ -n "${MAX_STEPS:-}" ]]; then TRAIN_ARGS+=(--max-steps "$MAX_STEPS"); fi
    joint_run "$PYTHON_BIN" app/train_joint.py "${TRAIN_ARGS[@]}"
    CHECKPOINT="$RUN_ROOT/pilot/overfit/$mode/seed-$seed/checkpoint.pt"
    DATASETS=("$mode")
    if [[ "$mode" == "joint" ]]; then DATASETS=(ds004940 ds006104); fi
    for dataset in "${DATASETS[@]}"; do
      EVAL_ARGS=(--checkpoint "$CHECKPOINT" --dataset "$dataset" --role train)
      RENDERER="$RUN_ROOT/audio_renderer/checkpoint.pt"
      if [[ "$dataset" == "ds004940" && -f "$RENDERER" ]]; then
        EVAL_ARGS+=(--renderer-checkpoint "$RENDERER")
      fi
      joint_run "$PYTHON_BIN" app/evaluate_joint.py "${EVAL_ARGS[@]}"
      joint_run require_evaluation_gate "$RUN_ROOT/pilot/overfit/$mode/seed-$seed/evaluation_${dataset}_train.json"
    done
  done
done
echo "Requested registered M0 runs passed."
