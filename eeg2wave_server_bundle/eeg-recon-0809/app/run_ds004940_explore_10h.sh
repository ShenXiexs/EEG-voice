#!/usr/bin/env bash
# Isolated, resumable DS004940-only rich Stage-2 exploratory training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -z "${PILOT_CONFIG+x}" ]]; then
  PILOT_CONFIG="$PROJECT_ROOT/configs/ds004940_explore_10h_v1.yaml"
fi
source "$SCRIPT_DIR/lib/joint_pilot_common.sh"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-ds004940_explore_10h_v1}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$RUN_ROOT/$EXPERIMENT_NAME}"
MAX_STEPS="${DS004940_10H_MAX_STEPS:-9000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
EXPECTED_SECONDS_PER_STEP="${EXPECTED_SECONDS_PER_STEP:-0.92}"
BUDGET_HOURS="${BUDGET_HOURS:-10}"

start_joint_log "$EXPERIMENT_NAME"
require_joint_runtime
require_local_hubert
cd "$PROJECT_ROOT"

IFS=' ' read -r -a SEEDS <<< "${DS004940_SEEDS:-$(pilot_seeds)}"
"$PYTHON_BIN" - "$MAX_STEPS" "$EXPECTED_SECONDS_PER_STEP" "$BUDGET_HOURS" "${#SEEDS[@]}" <<'PY'
import sys
steps, seconds, budget, seeds = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
training_hours = steps * seconds * seeds / 3600.0
print({"mode": "ds004940_only", "seeds": seeds, "steps_per_seed": steps,
       "estimated_training_hours": round(training_hours, 2),
       "estimated_nontraining_reserve_hours": round(budget - training_hours, 2)})
if budget - training_hours < 1.5:
    raise SystemExit("Configured training estimate leaves <1.5 h for artifacts/evaluation; lower DS004940_10H_MAX_STEPS")
PY

echo "WARNING: exploratory only. This creates an isolated DS004940 artifact set and does not modify joint results."
echo "experiment_root=$EXPERIMENT_ROOT"
echo "config=$PILOT_CONFIG"

STAGE2_ARGS=(--data-config "$DATA_CONFIG" --pilot-config "$PILOT_CONFIG" --explore --materialize \
  --hubert-local-path "$HUBERT_LOCAL_PATH")
if [[ "${REBUILD_DS004940_10H:-0}" == "1" ]]; then STAGE2_ARGS+=(--rebuild); fi
joint_run "$PYTHON_BIN" scripts/prepare_stage2_split.py "${STAGE2_ARGS[@]}"
joint_run "$PYTHON_BIN" scripts/prepare_stage2_split.py --data-config "$DATA_CONFIG" \
  --pilot-config "$PILOT_CONFIG" --explore --check-readiness

for seed in "${SEEDS[@]}"; do
  joint_run "$PYTHON_BIN" app/train_joint.py --config "$PILOT_CONFIG" --mode ds004940 \
    --stage generalization --seed "$seed" --explore --max-steps "$MAX_STEPS" \
    --checkpoint-every "$CHECKPOINT_EVERY" --output-root "$EXPERIMENT_ROOT"
  CHECKPOINT="$EXPERIMENT_ROOT/generalization/ds004940/seed-$seed/checkpoint.pt"
  for role in validation test; do
    joint_run "$PYTHON_BIN" app/evaluate_joint.py --checkpoint "$CHECKPOINT" \
      --dataset ds004940 --role "$role" \
      --renderer-checkpoint "$RUN_ROOT/audio_renderer_explore/checkpoint.pt"
  done
done

echo "DS004940 10h exploratory run complete. Outputs are not registered evidence."
echo "results=$EXPERIMENT_ROOT/generalization/ds004940"
