#!/usr/bin/env bash
# Export every reconstruction-eligible KaraOne and FEIS EEG trial for v0724.
# This is visualization-only: it never trains, evaluates, or writes a formal test claim.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${BUNDLE_DIR}/app/run_open_vocab_0724_v1.sh"
CONFIG_PATH="${BUNDLE_DIR}/app/configs/open_vocab_0724_factorized_v1_exploratory.yaml"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"

export DEVICE="${DEVICE:-mps}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
SEED="${SEED:-15}"
PLOT_DPI="${PLOT_DPI:-140}"
INCLUDE_EXPLORATORY_TEST="${INCLUDE_EXPLORATORY_TEST:-1}"
ROOT="${BUNDLE_DIR}/artifacts/open_vocab_0724_factorized_v1_exploratory/synthesis"

echo "[0724 full pairs] device=${DEVICE} seed=${SEED} dpi=${PLOT_DPI}"
echo "[0724 full pairs] scope=KaraOne+FEIS only; ds004306 is excluded (no paired audio target)"
echo "[0724 full pairs] train=KaraOne 1615 + FEIS 2832; validation=165 + 160"

run_partition() {
  local dataset="$1"
  local split="$2"
  local output_split="$3"
  shift 3
  local root="${ROOT}/${dataset}/${output_split}"
  echo
  echo "===== v0724 synthesize: ${dataset}/${output_split} ====="
  CONFIG="${CONFIG_PATH}" DEVICE="${DEVICE}" PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
    bash "${RUNNER}" synthesize "${dataset}" "${split}" --seed "${SEED}" \
    --output "${root}" --resume-existing "$@"
  echo "===== v0724 plot: ${dataset}/${output_split} ====="
  CONFIG="${CONFIG_PATH}" DEVICE="${DEVICE}" PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
    bash "${RUNNER}" plot "${dataset}" "${split}" --seed "${SEED}" \
    --synthesis-root "${root}" --output "${root}/comparison_pairs" --dpi "${PLOT_DPI}" \
    --resume-existing
}

for dataset in karaone feis; do
  run_partition "${dataset}" train train
  run_partition "${dataset}" validation validation
done

if [[ "${INCLUDE_EXPLORATORY_TEST}" == "1" ]]; then
  echo "[0724 full pairs] exploratory test is enabled; outputs will be labeled exploratory_test."
  for dataset in karaone feis; do
    run_partition "${dataset}" test exploratory_test --exploratory-test
  done
else
  echo "[0724 full pairs] exploratory test disabled (INCLUDE_EXPLORATORY_TEST=${INCLUDE_EXPLORATORY_TEST})."
fi

if [[ "${INCLUDE_EXPLORATORY_TEST}" == "1" ]]; then
  "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/audit_open_vocab_0724_full_pairs.py" --synthesis-root "${ROOT}"
fi

echo "[0724 full pairs] complete: ${ROOT}"
