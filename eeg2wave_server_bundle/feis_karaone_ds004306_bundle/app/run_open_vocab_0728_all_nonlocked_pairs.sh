#!/usr/bin/env bash
# Generate every non-locked KaraOne v0728 full11 reference/reconstruction pair.
# This deliberately excludes locked_test: it must remain fail-closed until the
# validation gate passes and a formal TEST_ACCESS_ID is supplied.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${BUNDLE_DIR}/app/run_open_vocab_0728_v1.sh"

export DEVICE="${DEVICE:-mps}"
export RUN_ID="${RUN_ID:-v0728_all_nonlocked_pairs_$(date -u +%Y%m%dT%H%M%SZ)}"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${MPS_HIGH_WATERMARK:-1.2}"
export PYTORCH_MPS_LOW_WATERMARK_RATIO="${MPS_LOW_WATERMARK:-1.0}"

SYNTH_LIMIT="${SYNTH_LIMIT:-0}"
PAIR_LIMIT="${PAIR_LIMIT:-0}"

printf '[0728 nonlocked pairs] run_id=%s device=%s\n' "${RUN_ID}" "${DEVICE}"
printf '[0728 nonlocked pairs] scope=train (1077) + validation (264) + diagnostic (297); locked test excluded\n'

for split in train validation diagnostic
do
  printf '\n===== Synthesize full11 %s =====\n' "${split}"
  bash "${RUNNER}" synthesize full11 "${split}" --limit "${SYNTH_LIMIT}" --resume-existing
  printf '\n===== Render energy-structure pairs: %s =====\n' "${split}"
  bash "${RUNNER}" plot full11 "${split}" --limit "${PAIR_LIMIT}" --resume-existing
done

printf '\n[0728 nonlocked pairs] complete. Outputs are under artifacts/open_vocab_0728_duallatent_v1/synthesis/full11/{train,validation,diagnostic}/comparison_pairs\n'
