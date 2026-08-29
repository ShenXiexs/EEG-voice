#!/usr/bin/env bash
# Export DS004940-only qualitative audio/energy pairs after its 10-hour run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT_ROOT="${1:-outputs/joint_pilot_v1/ds004940_explore_10h_v1}"

export AUDIO_PAIR_SINGLE_ONLY=1
export AUDIO_PAIR_DATASETS=ds004940
export AUDIO_PAIR_ROLES="${AUDIO_PAIR_ROLES:-validation,test}"
export AUDIO_PAIR_SEEDS="${AUDIO_PAIR_SEEDS:-31,47,73}"
export AUDIO_PAIR_MAX_PAIRS="${AUDIO_PAIR_MAX_PAIRS:-3}"
export GRIFFIN_LIM_ITERATIONS="${GRIFFIN_LIM_ITERATIONS:-32}"

exec "$SCRIPT_DIR/run_joint_audio_pair_comparisons.sh" "$EXPERIMENT_ROOT"
