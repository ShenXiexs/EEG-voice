#!/usr/bin/env bash
# Re-render every existing v0724 exploratory reference/reconstruction pair.
# No training and no synthesis are run by this script.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${BUNDLE_DIR}/app/run_open_vocab_0724_v1.sh"
CONFIG_PATH="${BUNDLE_DIR}/app/configs/open_vocab_0724_factorized_v1_exploratory.yaml"

export DEVICE="${DEVICE:-mps}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
SEED="${SEED:-15}"
PLOT_DPI="${PLOT_DPI:-140}"

printf '[0724 all pairs] device=%s seed=%s dpi=%s\n' "${DEVICE}" "${SEED}" "${PLOT_DPI}"
printf '[0724 all pairs] scope=existing exploratory WAV/mel only; no training and no new synthesis\n'

for dataset in karaone feis
do
  for split in validation test
  do
    printf '\n===== v0724 redraw: %s / %s =====\n' "${dataset}" "${split}"
    CONFIG="${CONFIG_PATH}" DEVICE="${DEVICE}" PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
      bash "${RUNNER}" plot "${dataset}" "${split}" --seed "${SEED}" --dpi "${PLOT_DPI}"
  done
done

printf '\n[0724 all pairs] complete.\n'
printf '%s\n' "${BUNDLE_DIR}/artifacts/open_vocab_0724_factorized_v1_exploratory/synthesis/{karaone,feis}/{validation,test}/comparison_pairs"
