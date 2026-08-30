#!/usr/bin/env bash
# Export all held-out and compact training-representative qualitative bundles.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT_ROOT="${1:-outputs/joint_pilot_v1/ds004940_large_scale_v1}"
ITERATIONS="${GRIFFIN_LIM_ITERATIONS:-32}"

echo "Phase 1/2: all validation/test bundles for every completed seed."
AUDIO_PAIR_SINGLE_ONLY=1 \
  AUDIO_PAIR_DATASETS=ds004940 \
  AUDIO_PAIR_ROLES=validation,test \
  AUDIO_PAIR_SEEDS="${DS004940_LARGE_EVALUATION_SEEDS:-31,47,73}" \
  AUDIO_PAIR_MAX_PAIRS=0 \
  AUDIO_PAIR_MANIFEST_NAME=heldout_export_manifest \
  GRIFFIN_LIM_ITERATIONS="$ITERATIONS" \
  "$SCRIPT_DIR/run_joint_audio_pair_comparisons.sh" "$EXPERIMENT_ROOT"

echo "Phase 2/2: one deterministic seed-31 training representative per content."
AUDIO_PAIR_SINGLE_ONLY=1 \
  AUDIO_PAIR_DATASETS=ds004940 \
  AUDIO_PAIR_ROLES=train \
  AUDIO_PAIR_SEEDS="${DS004940_LARGE_TRAIN_REPRESENTATIVE_SEED:-31}" \
  AUDIO_PAIR_MAX_PAIRS=0 \
  AUDIO_PAIR_TRAIN_REPRESENTATIVE=1 \
  AUDIO_PAIR_MANIFEST_NAME=train_representative_export_manifest \
  GRIFFIN_LIM_ITERATIONS="$ITERATIONS" \
  "$SCRIPT_DIR/run_joint_audio_pair_comparisons.sh" "$EXPERIMENT_ROOT"

echo "large_scale_audio_pairs=$EXPERIMENT_ROOT/generalization/audio_pair_comparisons"
