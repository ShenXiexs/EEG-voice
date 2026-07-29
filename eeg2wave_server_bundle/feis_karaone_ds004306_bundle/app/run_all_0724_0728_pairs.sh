#!/usr/bin/env bash
# Rebuild every available exploratory reference-vs-reconstruction pair figure
# without retraining either model. v0728 locked test remains protected.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V0724_RUNNER="${BUNDLE_DIR}/app/run_open_vocab_0724_v1.sh"
V0728_RUNNER="${BUNDLE_DIR}/app/run_open_vocab_0728_all_nonlocked_pairs.sh"
V0724_CONFIG="${BUNDLE_DIR}/app/configs/open_vocab_0724_factorized_v1_exploratory.yaml"

export DEVICE="${DEVICE:-mps}"
export RUN_ID="${RUN_ID:-all_0724_0728_pairs_$(date -u +%Y%m%dT%H%M%SZ)}"
V0724_SEED="${V0724_SEED:-15}"
V0724_DPI="${V0724_DPI:-140}"
V0728_SYNTH_LIMIT="${V0728_SYNTH_LIMIT:-0}"
V0728_PAIR_LIMIT="${V0728_PAIR_LIMIT:-0}"

printf '[all pairs] device=%s run_id=%s\n' "${DEVICE}" "${RUN_ID}"
printf '[all pairs] v0728: train + validation + diagnostic only; locked test is intentionally excluded\n'

printf '\n===== v0728: synthesize and render all non-locked KaraOne pairs =====\n'
SYNTH_LIMIT="${V0728_SYNTH_LIMIT}" PAIR_LIMIT="${V0728_PAIR_LIMIT}" \
  RUN_ID="${RUN_ID}_v0728" DEVICE="${DEVICE}" bash "${V0728_RUNNER}"

printf '\n===== v0724: redraw KaraOne validation pairs =====\n'
CONFIG="${V0724_CONFIG}" DEVICE="${DEVICE}" bash "${V0724_RUNNER}" \
  plot karaone validation --seed "${V0724_SEED}" --dpi "${V0724_DPI}"

printf '\n===== v0724: redraw KaraOne exploratory-test pairs =====\n'
CONFIG="${V0724_CONFIG}" DEVICE="${DEVICE}" bash "${V0724_RUNNER}" \
  plot karaone test --seed "${V0724_SEED}" --dpi "${V0724_DPI}"

printf '\n===== v0724: redraw FEIS validation pairs =====\n'
CONFIG="${V0724_CONFIG}" DEVICE="${DEVICE}" bash "${V0724_RUNNER}" \
  plot feis validation --seed "${V0724_SEED}" --dpi "${V0724_DPI}"

printf '\n===== v0724: redraw FEIS exploratory-test pairs =====\n'
CONFIG="${V0724_CONFIG}" DEVICE="${DEVICE}" bash "${V0724_RUNNER}" \
  plot feis test --seed "${V0724_SEED}" --dpi "${V0724_DPI}"

printf '\n[all pairs] complete.\n'
printf 'v0728: %s\n' "${BUNDLE_DIR}/artifacts/open_vocab_0728_duallatent_v1/synthesis/full11/{train,validation,diagnostic}/comparison_pairs"
printf 'v0724: %s\n' "${BUNDLE_DIR}/artifacts/open_vocab_0724_factorized_v1_exploratory/synthesis/{karaone,feis}/{validation,test}/comparison_pairs"
