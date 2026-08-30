#!/usr/bin/env bash
# Export qualitative source/model/control audio pairs without re-training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/joint_pilot_common.sh"

# Default to the already-completed first eight-hour exploratory run.  Pass an
# explicit path as $1 for the corrected/future experiment root.
EXPERIMENT_ROOT="${1:-${EXPERIMENT_ROOT:-$RUN_ROOT/explore_8h_v1}}"
RENDERER_CHECKPOINT="${RENDERER_CHECKPOINT:-$RUN_ROOT/audio_renderer_explore/checkpoint.pt}"
AUDIO_PAIR_SEEDS="${AUDIO_PAIR_SEEDS:-31}"
AUDIO_PAIR_DATASETS="${AUDIO_PAIR_DATASETS:-ds004940,ds006104}"
AUDIO_PAIR_ROLES="${AUDIO_PAIR_ROLES:-validation,test}"
AUDIO_PAIR_MAX_PAIRS="${AUDIO_PAIR_MAX_PAIRS:-3}"
GRIFFIN_LIM_ITERATIONS="${GRIFFIN_LIM_ITERATIONS:-32}"
AUDIO_PAIR_MANIFEST_NAME="${AUDIO_PAIR_MANIFEST_NAME:-export_manifest}"
SINGLE_ONLY_ARGS=()
if [[ "${AUDIO_PAIR_SINGLE_ONLY:-0}" == "1" ]]; then SINGLE_ONLY_ARGS+=(--single-only); fi
TRAIN_REPRESENTATIVE_ARGS=()
if [[ "${AUDIO_PAIR_TRAIN_REPRESENTATIVE:-0}" == "1" ]]; then TRAIN_REPRESENTATIVE_ARGS+=(--one-train-representative-per-content); fi

start_joint_log audio_pair_export
require_joint_runtime
cd "$PROJECT_ROOT"

[[ -d "$EXPERIMENT_ROOT/generalization" ]] || { echo "Missing experiment root: $EXPERIMENT_ROOT" >&2; exit 2; }
[[ -f "$RENDERER_CHECKPOINT" ]] || { echo "Missing audio renderer checkpoint: $RENDERER_CHECKPOINT" >&2; exit 2; }

echo "Exporting qualitative audio pairs only; this does not train or alter checkpoints."
echo "experiment_root=$EXPERIMENT_ROOT"
echo "renderer_checkpoint=$RENDERER_CHECKPOINT"
echo "generated audio is deterministic Griffin-Lim diagnostic output, not neural-vocoder output."

joint_run "$PYTHON_BIN" app/export_audio_pair_comparisons.py \
  --experiment-root "$EXPERIMENT_ROOT" \
  --renderer-checkpoint "$RENDERER_CHECKPOINT" \
  --seeds "$AUDIO_PAIR_SEEDS" \
  --datasets "$AUDIO_PAIR_DATASETS" \
  --roles "$AUDIO_PAIR_ROLES" \
  --max-pairs "$AUDIO_PAIR_MAX_PAIRS" \
  --griffin-lim-iterations "$GRIFFIN_LIM_ITERATIONS" \
  --manifest-name "$AUDIO_PAIR_MANIFEST_NAME" \
  "${TRAIN_REPRESENTATIVE_ARGS[@]}" \
  "${SINGLE_ONLY_ARGS[@]}"

echo "audio_pairs=$EXPERIMENT_ROOT/generalization/audio_pair_comparisons"
