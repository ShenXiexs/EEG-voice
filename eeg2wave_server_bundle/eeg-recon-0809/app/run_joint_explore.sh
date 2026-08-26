#!/usr/bin/env bash
# Complete exploratory execution.  It never writes registered `pilot` outputs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/joint_pilot_common.sh"
start_joint_log explore
require_joint_runtime
require_local_hubert

cd "$PROJECT_ROOT"
echo "WARNING: explore mode bypasses human/M0 scientific gates. Outputs are not registered results."

# EXPLORE_FROM=m0 resumes after a completed audit/split instead of repeating it.
# EXPLORE_FROM=overfit resumes directly at model training after explore M0
# artifacts have been materialized.
case "${EXPLORE_FROM:-start}" in
  start)
    # Stage 0 remains mechanically strict: only scientific approval is bypassed.
    joint_run "$PYTHON_BIN" scripts/prepare_training_data.py --config "$DATA_CONFIG" audit --strict --fetch-aux
    joint_run "$PYTHON_BIN" scripts/prepare_training_data.py --config "$DATA_CONFIG" make-splits
    joint_run "$PYTHON_BIN" scripts/prepare_stage2_split.py --data-config "$DATA_CONFIG" --pilot-config "$PILOT_CONFIG"
    ;;
  m0)
    [[ -f "$ARTIFACT_ROOT/manifests/manifest_all.csv" ]] || { echo "Cannot resume at M0: missing audited manifest" >&2; exit 2; }
    [[ -f "$ARTIFACT_ROOT/splits/joint_ood_fold-0.csv" ]] || { echo "Cannot resume at M0: missing frozen split" >&2; exit 2; }
    [[ -f "$ARTIFACT_ROOT/splits/stage2_joint_ood_fold-0.csv" ]] || { echo "Cannot resume at M0: missing Stage-2 split" >&2; exit 2; }
    echo "Resuming at isolated explore M0 artifact construction."
    ;;
  overfit)
    [[ -f "$ARTIFACT_ROOT/manifests/manifest_explore_m0.csv" ]] || { echo "Cannot resume at overfit: missing explore M0 manifest" >&2; exit 2; }
    [[ -f "$ARTIFACT_ROOT/normalizers/explore_m0_joint_ood_fold-0.json" ]] || { echo "Cannot resume at overfit: missing explore M0 normalizer" >&2; exit 2; }
    [[ -f "$ARTIFACT_ROOT/speech_targets/speech_targets_explore_m0.h5" ]] || { echo "Cannot resume at overfit: missing explore M0 speech targets" >&2; exit 2; }
    echo "Resuming at exploratory M0 model training."
    ;;
  *)
    echo "EXPLORE_FROM must be start, m0, or overfit" >&2
    exit 2
    ;;
esac

if [[ "${EXPLORE_FROM:-start}" != "overfit" ]]; then
  M0_ARGS=(--data-config "$DATA_CONFIG" --pilot-config "$PILOT_CONFIG" --hubert-local-path "$HUBERT_LOCAL_PATH" --artifact-set explore_m0)
  if [[ "${REBUILD_M0:-0}" == "1" ]]; then M0_ARGS+=(--rebuild); fi
  joint_run "$PYTHON_BIN" scripts/prepare_m0_artifacts.py "${M0_ARGS[@]}"

  # This oracle run is retained separately and returns success even if its quality gate fails.
  ORACLE_ARGS=(--config "$PILOT_CONFIG" --explore)
  if [[ -n "${EXPLORE_MAX_STEPS:-}" ]]; then ORACLE_ARGS+=(--max-steps "$EXPLORE_MAX_STEPS"); fi
  joint_run "$PYTHON_BIN" app/train_audio_renderer.py "${ORACLE_ARGS[@]}"
fi

IFS=' ' read -r -a SEEDS <<< "${EXPLORE_SEEDS:-$(pilot_seeds)}"
for mode in ds004940 ds006104 joint; do
  for seed in "${SEEDS[@]}"; do
    TRAIN_ARGS=(--config "$PILOT_CONFIG" --mode "$mode" --stage overfit --seed "$seed" --explore)
    if [[ -n "${EXPLORE_MAX_STEPS:-}" ]]; then TRAIN_ARGS+=(--max-steps "$EXPLORE_MAX_STEPS"); fi
    joint_run "$PYTHON_BIN" app/train_joint.py "${TRAIN_ARGS[@]}"
    CHECKPOINT="$RUN_ROOT/explore/overfit/$mode/seed-$seed/checkpoint.pt"
    DATASETS=("$mode")
    if [[ "$mode" == "joint" ]]; then DATASETS=(ds004940 ds006104); fi
    for dataset in "${DATASETS[@]}"; do
      joint_run "$PYTHON_BIN" app/evaluate_joint.py --checkpoint "$CHECKPOINT" --dataset "$dataset" --role train
    done
  done
done

# Explore Stage 2 has an independent manifest, normalizer, target cache, and shards.
STAGE2_ARGS=(--data-config "$DATA_CONFIG" --pilot-config "$PILOT_CONFIG" --explore --materialize --hubert-local-path "$HUBERT_LOCAL_PATH")
if [[ "${REBUILD_EXPLORE:-0}" == "1" ]]; then STAGE2_ARGS+=(--rebuild); fi
joint_run "$PYTHON_BIN" scripts/prepare_stage2_split.py "${STAGE2_ARGS[@]}"
joint_run "$PYTHON_BIN" scripts/prepare_stage2_split.py --data-config "$DATA_CONFIG" --pilot-config "$PILOT_CONFIG" \
  --explore --check-readiness

for mode in ds004940 ds006104 joint; do
  for seed in "${SEEDS[@]}"; do
    TRAIN_ARGS=(--config "$PILOT_CONFIG" --mode "$mode" --stage generalization --seed "$seed" --explore)
    if [[ -n "${EXPLORE_MAX_STEPS:-}" ]]; then TRAIN_ARGS+=(--max-steps "$EXPLORE_MAX_STEPS"); fi
    joint_run "$PYTHON_BIN" app/train_joint.py "${TRAIN_ARGS[@]}"
    CHECKPOINT="$RUN_ROOT/explore/generalization/$mode/seed-$seed/checkpoint.pt"
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
      SINGLE="$RUN_ROOT/explore/generalization/$dataset/seed-$seed/evaluation_${dataset}_${role}.json"
      JOINT="$RUN_ROOT/explore/generalization/joint/seed-$seed/evaluation_${dataset}_${role}.json"
      OUTPUT="$RUN_ROOT/explore/generalization/comparisons/seed-$seed/${dataset}_${role}_single_vs_joint.json"
      joint_run "$PYTHON_BIN" app/evaluate_joint.py --compare-single "$SINGLE" --compare-joint "$JOINT" --output "$OUTPUT"
    done
  done
done

echo "Explore run complete. Do not treat outputs under outputs/joint_pilot_v1/explore as registered evidence."
