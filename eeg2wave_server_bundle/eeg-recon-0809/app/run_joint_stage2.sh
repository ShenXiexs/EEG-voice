#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/joint_pilot_common.sh"
start_joint_log stage2
require_joint_runtime
require_local_hubert
joint_run require_formal_stage0

cd "$PROJECT_ROOT"
joint_run "$PYTHON_BIN" scripts/prepare_stage2_split.py --data-config "$DATA_CONFIG" --pilot-config "$PILOT_CONFIG" \
  --materialize --hubert-local-path "$HUBERT_LOCAL_PATH"
joint_run "$PYTHON_BIN" scripts/prepare_stage2_split.py --data-config "$DATA_CONFIG" --pilot-config "$PILOT_CONFIG" --check-readiness

IFS=' ' read -r -a SEEDS <<< "${SEEDS:-$(pilot_seeds)}"
for mode in ds004940 ds006104 joint; do
  for seed in "${SEEDS[@]}"; do
    TRAIN_ARGS=(--config "$PILOT_CONFIG" --mode "$mode" --stage generalization --seed "$seed")
    if [[ -n "${MAX_STEPS:-}" ]]; then TRAIN_ARGS+=(--max-steps "$MAX_STEPS"); fi
    joint_run "$PYTHON_BIN" app/train_joint.py "${TRAIN_ARGS[@]}"
    CHECKPOINT="$RUN_ROOT/pilot/generalization/$mode/seed-$seed/checkpoint.pt"
    DATASETS=("$mode")
    if [[ "$mode" == "joint" ]]; then DATASETS=(ds004940 ds006104); fi
    for dataset in "${DATASETS[@]}"; do
      for role in validation test; do
        joint_run "$PYTHON_BIN" app/evaluate_joint.py --checkpoint "$CHECKPOINT" --dataset "$dataset" --role "$role"
      done
    done
  done
done

for seed in "${SEEDS[@]}"; do
  for dataset in ds004940 ds006104; do
    for role in validation test; do
      SINGLE="$RUN_ROOT/pilot/generalization/$dataset/seed-$seed/evaluation_${dataset}_${role}.json"
      JOINT="$RUN_ROOT/pilot/generalization/joint/seed-$seed/evaluation_${dataset}_${role}.json"
      OUTPUT="$RUN_ROOT/pilot/generalization/comparisons/seed-$seed/${dataset}_${role}_single_vs_joint.json"
      joint_run "$PYTHON_BIN" app/evaluate_joint.py --compare-single "$SINGLE" --compare-joint "$JOINT" --output "$OUTPUT"
    done
  done
done
echo "Stage 2 finished. With one held subject per dataset/role, bootstrap intervals are descriptive; do not claim population-level positive transfer from this pilot alone."
