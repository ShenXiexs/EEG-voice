#!/usr/bin/env bash
# Isolated eight-hour exploratory Stage-2 rerun with automatic comparison figures.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/joint_pilot_common.sh"

PILOT_CONFIG="${PILOT_CONFIG:-$PROJECT_ROOT/configs/joint_explore_8h_v1.yaml}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-explore_8h_v1}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$RUN_ROOT/$EXPERIMENT_NAME}"
MAX_STEPS="${EXPLORE_8H_MAX_STEPS:-2400}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
BUDGET_HOURS="${BUDGET_HOURS:-8}"
EXPECTED_SECONDS_PER_STEP="${EXPECTED_SECONDS_PER_STEP:-0.92}"

start_joint_log "$EXPERIMENT_NAME"
require_joint_runtime
require_local_hubert
cd "$PROJECT_ROOT"

echo "WARNING: this is a new isolated exploratory run; scientific gates remain bypassed."
echo "experiment_root=$EXPERIMENT_ROOT"
echo "total_budget_hours=$BUDGET_HOURS"
echo "stage2_max_steps_per_run=$MAX_STEPS"
echo "checkpoint_every=$CHECKPOINT_EVERY"

IFS=' ' read -r -a SEEDS <<< "${EXPLORE_SEEDS:-$(pilot_seeds)}"
"$PYTHON_BIN" - "$MAX_STEPS" "$EXPECTED_SECONDS_PER_STEP" "$BUDGET_HOURS" "${#SEEDS[@]}" <<'PY'
import sys
steps, seconds, budget, seeds = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
runs = 3 * seeds
training_hours = runs * steps * seconds / 3600.0
reserve = budget - training_hours
print({"planned_runs": runs, "planned_optimizer_steps": runs * steps,
       "estimated_training_hours": round(training_hours, 2),
       "estimated_nontraining_reserve_hours": round(reserve, 2)})
if reserve < 1.5:
    raise SystemExit("Configured training estimate leaves <1.5 h for preparation/evaluation/figures; lower EXPLORE_8H_MAX_STEPS")
PY

# This config writes a new 4/3/3-subject split and named artifacts. Existing
# explore_stage2 data and completed checkpoints are not overwritten.
STAGE2_ARGS=(--data-config "$DATA_CONFIG" --pilot-config "$PILOT_CONFIG" --explore --materialize \
  --hubert-local-path "$HUBERT_LOCAL_PATH")
if [[ "${REBUILD_EXPLORE_8H:-0}" == "1" ]]; then STAGE2_ARGS+=(--rebuild); fi
joint_run "$PYTHON_BIN" scripts/prepare_stage2_split.py "${STAGE2_ARGS[@]}"
joint_run "$PYTHON_BIN" scripts/prepare_stage2_split.py --data-config "$DATA_CONFIG" \
  --pilot-config "$PILOT_CONFIG" --explore --check-readiness

for mode in ds004940 ds006104 joint; do
  for seed in "${SEEDS[@]}"; do
    joint_run "$PYTHON_BIN" app/train_joint.py --config "$PILOT_CONFIG" --mode "$mode" \
      --stage generalization --seed "$seed" --explore --max-steps "$MAX_STEPS" \
      --checkpoint-every "$CHECKPOINT_EVERY" --output-root "$EXPERIMENT_ROOT"
    CHECKPOINT="$EXPERIMENT_ROOT/generalization/$mode/seed-$seed/checkpoint.pt"
    DATASETS=("$mode")
    if [[ "$mode" == "joint" ]]; then DATASETS=(ds004940 ds006104); fi
    for dataset in "${DATASETS[@]}"; do
      for role in validation test; do
        joint_run "$PYTHON_BIN" app/evaluate_joint.py --checkpoint "$CHECKPOINT" \
          --dataset "$dataset" --role "$role"
      done
    done
  done
done

for seed in "${SEEDS[@]}"; do
  for dataset in ds004940 ds006104; do
    for role in validation test; do
      SINGLE="$EXPERIMENT_ROOT/generalization/$dataset/seed-$seed/evaluation_${dataset}_${role}.json"
      JOINT="$EXPERIMENT_ROOT/generalization/joint/seed-$seed/evaluation_${dataset}_${role}.json"
      OUTPUT="$EXPERIMENT_ROOT/generalization/comparisons/seed-$seed/${dataset}_${role}_single_vs_joint.json"
      joint_run "$PYTHON_BIN" app/evaluate_joint.py --compare-single "$SINGLE" \
        --compare-joint "$JOINT" --output "$OUTPUT"
    done
  done
done

SEED_CSV="$(IFS=,; echo "${SEEDS[*]}")"
joint_run "$PYTHON_BIN" app/plot_joint_comparison.py --input-root "$EXPERIMENT_ROOT" \
  --seeds "$SEED_CSV" --formats png,pdf --dpi 300

echo "Explore 8h run complete. Outputs are exploratory, not registered evidence."
echo "results=$EXPERIMENT_ROOT/generalization"
echo "figures=$EXPERIMENT_ROOT/generalization/figures"
echo "summary=$EXPERIMENT_ROOT/generalization/figures/comparison_summary.json"
