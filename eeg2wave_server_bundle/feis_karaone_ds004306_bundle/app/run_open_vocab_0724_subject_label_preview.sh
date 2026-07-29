#!/usr/bin/env bash
# Generate a compact, descriptive v0724 reconstruction preview.
#
# The 500 primary EEG trials are stratified within each dataset/split by
# (subject_group_id, label). Every selected primary trial still emits all ten
# counterfactual reconstruction modes, but expensive numerical metrics are
# deliberately skipped: this script is for qualitative inspection only.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${BUNDLE_DIR}/app/run_open_vocab_0724_v1.sh"
CONFIG_PATH="${BUNDLE_DIR}/app/configs/open_vocab_0724_factorized_v1_exploratory.yaml"

export DEVICE="${DEVICE:-mps}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
SEED="${SEED:-15}"
PLOT_DPI="${PLOT_DPI:-140}"
ROOT="${BUNDLE_DIR}/artifacts/open_vocab_0724_factorized_v1_exploratory/synthesis_preview_subject_label_500"

# Current eligible-pool stratum counts are KaraOne train=132, FEIS train=288,
# KaraOne validation=11, and FEIS validation=16. These limits total 500 and
# leave at least one primary trial per available subject-label stratum, with
# the remaining 53 slots distributed round-robin by the Python sampler.
KARAONE_TRAIN_LIMIT="${KARAONE_TRAIN_LIMIT:-145}"
FEIS_TRAIN_LIMIT="${FEIS_TRAIN_LIMIT:-315}"
KARAONE_VALIDATION_LIMIT="${KARAONE_VALIDATION_LIMIT:-18}"
FEIS_VALIDATION_LIMIT="${FEIS_VALIDATION_LIMIT:-22}"
TOTAL=$((KARAONE_TRAIN_LIMIT + FEIS_TRAIN_LIMIT + KARAONE_VALIDATION_LIMIT + FEIS_VALIDATION_LIMIT))

if (( TOTAL != 500 )); then
  echo "ERROR: the four preview limits must total 500; observed ${TOTAL}." >&2
  exit 2
fi

run_partition() {
  local dataset="$1"
  local split="$2"
  local limit="$3"
  local root="${ROOT}/${dataset}/${split}"

  printf '\n===== v0724 qualitative preview: %s/%s (%s primary trials) =====\n' \
    "${dataset}" "${split}" "${limit}"
  CONFIG="${CONFIG_PATH}" DEVICE="${DEVICE}" PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
    bash "${RUNNER}" synthesize "${dataset}" "${split}" \
      --seed "${SEED}" \
      --stratified-limit "${limit}" \
      --visual-preview \
      --output "${root}" \
      --resume-existing

  # Three rows are enough to inspect the intended condition and two decisive
  # EEG-validity controls without rendering every counterfactual column.
  CONFIG="${CONFIG_PATH}" DEVICE="${DEVICE}" PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
    bash "${RUNNER}" plot "${dataset}" "${split}" \
      --seed "${SEED}" \
      --synthesis-root "${root}" \
      --output "${root}/comparison_pairs" \
      --modes "correct_content_correct_realization,shuffled_eeg,zero_eeg" \
      --dpi "${PLOT_DPI}" \
      --resume-existing
}

printf '[0724 subject-label preview] device=%s seed=%s total_primary_trials=%s\n' \
  "${DEVICE}" "${SEED}" "${TOTAL}"
printf '[0724 subject-label preview] qualitative only: no test split, no aggregate numerical claims\n'
printf '[0724 subject-label preview] each selected primary trial still has 10 generated modes\n'

run_partition karaone train "${KARAONE_TRAIN_LIMIT}"
run_partition feis train "${FEIS_TRAIN_LIMIT}"
run_partition karaone validation "${KARAONE_VALIDATION_LIMIT}"
run_partition feis validation "${FEIS_VALIDATION_LIMIT}"

printf '\n[0724 subject-label preview] complete: %s\n' "${ROOT}"
