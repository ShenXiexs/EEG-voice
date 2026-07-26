#!/usr/bin/env bash
# Finish the primary-seed v0724 exploratory run after EEG pretraining.
#
# This script deliberately uses --exploratory-test.  It produces diagnostic
# test WAVs and comparison figures without claiming or representing a formal
# locked-test result.
set -eo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${BUNDLE_DIR}/app/run_open_vocab_0724_v1.sh"
CONFIG="${BUNDLE_DIR}/app/configs/open_vocab_0724_factorized_v1_exploratory.yaml"
OUTPUT_ROOT="${BUNDLE_DIR}/artifacts/open_vocab_0724_factorized_v1_exploratory"
PRETRAIN_BEST="${OUTPUT_ROOT}/eeg_pretrain/checkpoints/best.pt"
EEG_BEST="${OUTPUT_ROOT}/eeg/checkpoints/best.pt"
EEG_LATEST="${OUTPUT_ROOT}/eeg/checkpoints/latest.pt"

DEVICE="${DEVICE:-mps}"
SEED="${SEED:-15}"
EEG_EPOCHS="${EEG_EPOCHS:-30}"
EEG_PATIENCE="${EEG_PATIENCE:-30}"
SOFT_DTW_FRAMES="${SOFT_DTW_FRAMES:-32}"
SOFT_DTW_EVERY="${SOFT_DTW_EVERY:-4}"
PLOT_DPI="${PLOT_DPI:-140}"
RUN_EXPLORATORY_TEST="${RUN_EXPLORATORY_TEST:-1}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  DEVICE=mps bash app/run_open_vocab_0724_exploratory_finish.sh

Optional environment controls:
  EEG_EPOCHS=30             paired EEG epoch ceiling
  EEG_PATIENCE=30           exploratory early-stopping patience
  SOFT_DTW_FRAMES=32        training-only soft-DTW temporal resolution
  SOFT_DTW_EVERY=4          compute training soft-DTW once per N batches
  PLOT_DPI=140              pair-figure resolution
  RUN_EXPLORATORY_TEST=1    set to 0 to stop after validation WAVs/figures
  DRY_RUN=1                 print commands without running them

The test outputs are exploratory diagnostics. This script never creates a
formal locked-test claim and must not be used to report formal test results.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "ERROR: this script accepts environment controls only; see --help." >&2
  exit 2
fi
if [[ "${SEED}" != "15" ]]; then
  echo "ERROR: exploratory reconstruction uses the primary seed 15." >&2
  exit 2
fi
if ! [[ "${EEG_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EEG_EPOCHS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${EEG_PATIENCE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EEG_PATIENCE must be a positive integer." >&2
  exit 2
fi
if ! [[ "${SOFT_DTW_FRAMES}" =~ ^[0-9]+$ ]] || (( SOFT_DTW_FRAMES < 2 )); then
  echo "ERROR: SOFT_DTW_FRAMES must be an integer of at least two." >&2
  exit 2
fi
if ! [[ "${SOFT_DTW_EVERY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SOFT_DTW_EVERY must be a positive integer." >&2
  exit 2
fi
if ! [[ "${PLOT_DPI}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PLOT_DPI must be a positive integer." >&2
  exit 2
fi
if [[ "${RUN_EXPLORATORY_TEST}" != "0" && "${RUN_EXPLORATORY_TEST}" != "1" ]]; then
  echo "ERROR: RUN_EXPLORATORY_TEST must be 0 or 1." >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "ERROR: DRY_RUN must be 0 or 1." >&2
  exit 2
fi
if [[ ! -f "${PRETRAIN_BEST}" ]]; then
  echo "ERROR: completed EEG-pretrain best checkpoint is missing: ${PRETRAIN_BEST}" >&2
  exit 2
fi

export CONFIG DEVICE

run_command() {
  printf '\n+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" == "0" ]]; then
    "$@"
  fi
}

run_step() {
  local label="$1"
  shift
  printf '\n===== %s =====\n' "${label}"
  run_command "$@"
}

train_paired_eeg() {
  local args=(
    bash "${RUNNER}" train-eeg
    --seed "${SEED}"
    --epochs "${EEG_EPOCHS}"
    --early-stopping-patience "${EEG_PATIENCE}"
    --soft-dtw-train-frames "${SOFT_DTW_FRAMES}"
    --soft-dtw-every-batches "${SOFT_DTW_EVERY}"
  )
  if [[ -f "${EEG_LATEST}" ]]; then
    args+=(--resume "${EEG_LATEST}")
    printf '[0724 exploratory] resuming paired EEG from latest: %s\n' "${EEG_LATEST}"
  elif [[ -f "${EEG_BEST}" ]]; then
    args+=(--resume "${EEG_BEST}")
    printf '[0724 exploratory] latest missing; resuming paired EEG from best: %s\n' "${EEG_BEST}"
  else
    printf '[0724 exploratory] starting paired EEG from pretrain best: %s\n' "${PRETRAIN_BEST}"
  fi
  run_command "${args[@]}"
}

printf '[0724 exploratory] config=%s\n' "${CONFIG}"
printf '[0724 exploratory] output=%s\n' "${OUTPUT_ROOT}"
printf '[0724 exploratory] device=%s seed=%s epochs=%s patience=%s\n' \
  "${DEVICE}" "${SEED}" "${EEG_EPOCHS}" "${EEG_PATIENCE}"
printf '[0724 exploratory] fast soft-DTW: frames=%s every=%s batches\n' \
  "${SOFT_DTW_FRAMES}" "${SOFT_DTW_EVERY}"
printf '[0724 exploratory] formal locked-test claim: disabled\n'

printf '\n===== Train or resume paired EEG =====\n'
train_paired_eeg

run_step "Evaluate validation latent metrics" \
  bash "${RUNNER}" validate --seed "${SEED}"

for dataset in karaone feis; do
  run_step "Synthesize ${dataset} validation WAV and mel tensors" \
    bash "${RUNNER}" synthesize "${dataset}" validation --seed "${SEED}"
  run_step "Render ${dataset} validation comparison pairs" \
    bash "${RUNNER}" plot "${dataset}" validation \
      --seed "${SEED}" --dpi "${PLOT_DPI}"
done

if [[ "${RUN_EXPLORATORY_TEST}" == "1" ]]; then
  for dataset in karaone feis; do
    run_step "Synthesize ${dataset} exploratory-test WAV and mel tensors" \
      bash "${RUNNER}" synthesize "${dataset}" test \
        --seed "${SEED}" --exploratory-test
    run_step "Render ${dataset} exploratory-test comparison pairs" \
      bash "${RUNNER}" plot "${dataset}" test \
        --seed "${SEED}" --dpi "${PLOT_DPI}"
  done
fi

printf '\nOpenVoice-EEG v0724 exploratory finish completed.\n'
printf 'KaraOne validation pairs: %s\n' \
  "${OUTPUT_ROOT}/synthesis/karaone/validation/comparison_pairs"
printf 'FEIS validation pairs: %s\n' \
  "${OUTPUT_ROOT}/synthesis/feis/validation/comparison_pairs"
if [[ "${RUN_EXPLORATORY_TEST}" == "1" ]]; then
  printf 'KaraOne exploratory-test pairs: %s\n' \
    "${OUTPUT_ROOT}/synthesis/karaone/test/comparison_pairs"
  printf 'FEIS exploratory-test pairs: %s\n' \
    "${OUTPUT_ROOT}/synthesis/feis/test/comparison_pairs"
fi
