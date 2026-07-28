#!/usr/bin/env bash
# Resume v0728 from a completed audio checkpoint and produce exploratory
# full11 validation counterfactual pair figures. This script deliberately
# does not run LOSO or the locked test.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${BUNDLE_DIR}/app/run_open_vocab_0728_v1.sh"
ROOT="${BUNDLE_DIR}/artifacts/open_vocab_0728_duallatent_v1"

export RUN_ID="${RUN_ID:-v0728_resume_pairs_$(date -u +%Y%m%dT%H%M%SZ)}"
export DEVICE="${DEVICE:-mps}"
# A lower MPS watermark asks PyTorch to reclaim memory earlier, making the
# post-training audio audit less likely to be terminated by macOS.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.8}"

AUDIO_RESUME="${AUDIO_RESUME:-${ROOT}/audio/checkpoints/latest.pt}"
if [[ ! -f "${AUDIO_RESUME}" ]]; then
  printf 'Missing resumable audio checkpoint: %s\n' "${AUDIO_RESUME}" >&2
  exit 1
fi

soft_stage() {
  local label="$1"
  shift
  if bash "${RUNNER}" "$@"; then
    return 0
  else
    local exit_code=$?
    printf '[0728 exploratory] %s ended with exit=%s; retaining its logs and continuing diagnostic generation.\n' "${label}" "${exit_code}" >&2
  fi
}

printf '[0728 exploratory] run_id=%s device=%s\n' "${RUN_ID}" "${DEVICE}"
printf '[0728 exploratory] audio resume=%s\n' "${AUDIO_RESUME}"
printf '[0728 exploratory] scope=audio resume -> semantic4 -> dual4 -> full11 validation -> pair figures; no LOSO; no locked test\n'

# The audio loop is already complete at epoch 20. Resume only so the model can
# write its gate/freeze outputs; a failed gate is intentionally non-blocking.
soft_stage 'audio-resume-or-gate' train-audio --resume "${AUDIO_RESUME}"
soft_stage 'audio-disentanglement-audit' audit-disentanglement

bash "${RUNNER}" train-semantic4
bash "${RUNNER}" synthesize semantic4 validation
soft_stage 'semantic4-validation-gate' gate semantic4 validation

bash "${RUNNER}" train-dual4
bash "${RUNNER}" synthesize dual4 validation
soft_stage 'dual4-validation-gate' gate dual4 validation

bash "${RUNNER}" train-full11
bash "${RUNNER}" synthesize full11 validation
soft_stage 'full11-validation-gate' gate full11 validation

bash "${RUNNER}" plot full11 validation --limit "${PAIR_LIMIT:-48}"

PAIR_DIR="${ROOT}/synthesis/full11/validation/figures"
printf '[0728 exploratory] complete. Pair figures: %s\n' "${PAIR_DIR}"
