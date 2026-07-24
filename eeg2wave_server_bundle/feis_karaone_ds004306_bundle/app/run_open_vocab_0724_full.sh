#!/usr/bin/env bash
# One-command formal OpenVoice-EEG v0724 run.
#
# Sequence (fail closed): read-only preflight -> cache v2 -> audio factorizer
# -> strict audio oracle -> registered three-seed EEG/LOSO development ->
# strict validation gate -> one locked-test session -> numerical comparison PNGs.
#
# The test session is intentionally last and irreversible.  If it succeeds but
# plotting is interrupted, rerun only `app/run_open_vocab_0724_v1.sh plot ...`;
# never rerun this full script against the same artifact root.
set -eo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${BUNDLE_DIR}/app/run_open_vocab_0724_v1.sh"
PREFLIGHT="${BUNDLE_DIR}/app/scripts/preflight_open_vocab_0724_final.py"
PLOTTER="${BUNDLE_DIR}/app/scripts/plot_open_vocab_0724_pairs.py"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
CONFIG="${CONFIG:-${BUNDLE_DIR}/app/configs/open_vocab_0724_factorized_v1.yaml}"
DEVICE="${DEVICE:-}"
ALLOW_NETWORK="${ALLOW_NETWORK:-0}"
PLOT_LIMIT="${PLOT_LIMIT:--1}"
PLOT_MODES="${PLOT_MODES:-}"
PLOT_DPI="${PLOT_DPI:-140}"

usage() {
  cat <<'EOF'
Usage:
  DEVICE=mps bash app/run_open_vocab_0724_full.sh

Environment controls:
  CONFIG=/absolute/path/to/open_vocab_0724_config.yaml
  DEVICE=mps|cuda|cpu       optional; unset selects CUDA/MPS/CPU automatically
  ALLOW_NETWORK=1           allow the cache builder to download a missing WavLM teacher
  PLOT_LIMIT=-1             -1 renders every final-test reconstruction pair
  PLOT_MODES=a,b,...        optional subset of counterfactual columns for presentation
  PLOT_DPI=140              PNG rasterization DPI
  TEST_ACCESS_ID=...        optional safe ID for the one final-test session

The registered seeds are fixed to 15, 31, and 47.  This command cannot resume
or replay a locked test; after a completed test, regenerate plots with:
  bash app/run_open_vocab_0724_v1.sh plot karaone test
  bash app/run_open_vocab_0724_v1.sh plot feis test
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "ERROR: configure this formal run through environment variables; see --help." >&2
  exit 2
fi
if [[ "${ALLOW_NETWORK}" != "0" && "${ALLOW_NETWORK}" != "1" ]]; then
  echo "ERROR: ALLOW_NETWORK must be 0 or 1." >&2
  exit 2
fi
if ! [[ "${PLOT_LIMIT}" =~ ^-?[0-9]+$ ]] || (( PLOT_LIMIT == 0 || PLOT_LIMIT < -1 )); then
  echo "ERROR: PLOT_LIMIT must be -1 or a positive integer." >&2
  exit 2
fi
if ! [[ "${PLOT_DPI}" =~ ^[0-9]+$ ]] || (( PLOT_DPI < 1 )); then
  echo "ERROR: PLOT_DPI must be a positive integer." >&2
  exit 2
fi
if [[ -n "${SEEDS:-}" && "${SEEDS}" != "15 31 47" ]]; then
  echo "ERROR: v0724 formal gate requires exactly SEEDS='15 31 47'." >&2
  exit 2
fi
if [[ "${PYTHON_BIN}" == */* ]]; then
  [[ -x "${PYTHON_BIN}" ]] || {
    echo "ERROR: Python executable is missing or not executable: ${PYTHON_BIN}" >&2
    exit 2
  }
elif ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable is not on PATH: ${PYTHON_BIN}" >&2
  exit 2
fi
for required in "${RUNNER}" "${PREFLIGHT}" "${PLOTTER}"; do
  [[ -f "${required}" ]] || {
    echo "ERROR: required v0724 entrypoint is missing: ${required}" >&2
    exit 2
  }
done

# Do not allow a caller's stale development manifest to unlock the formal test.
OUTPUT_ROOT="$("${PYTHON_BIN}" "${PREFLIGHT}" --config "${CONFIG}" --print-output-root)"
SYNTHESIS_MANIFEST="${OUTPUT_ROOT}/synthesis/karaone/validation/synthesis_manifest.json"
TEST_ACCESS_ID="${TEST_ACCESS_ID:-v0724_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
SEEDS="15 31 47"
if ! [[ "${TEST_ACCESS_ID}" =~ ^[A-Za-z0-9_.-]{8,128}$ ]]; then
  echo "ERROR: TEST_ACCESS_ID must contain 8-128 letters, digits, ., _, or -." >&2
  exit 2
fi
if [[ -n "${PLOT_MODES}" ]]; then
  IFS=',' read -r -a requested_plot_modes <<< "${PLOT_MODES}"
  for mode in "${requested_plot_modes[@]}"; do
    trimmed_mode="${mode#"${mode%%[![:space:]]*}"}"
    trimmed_mode="${trimmed_mode%"${trimmed_mode##*[![:space:]]}"}"
    case "${trimmed_mode}" in
      correct_content_correct_realization|correct_content_wrong_realization|wrong_content_correct_realization|wrong_content_wrong_realization|content_only|realization_only|shuffled_eeg|zero_eeg) ;;
      *)
        echo "ERROR: unsupported PLOT_MODES entry: ${mode}" >&2
        exit 2
        ;;
    esac
  done
fi
export PYTHON_BIN CONFIG DEVICE SEEDS SYNTHESIS_MANIFEST TEST_ACCESS_ID

TOTAL_STEPS=12
CURRENT_STEP=0
CURRENT_LABEL="starting"

draw_progress() {
  local completed="$1" label="$2" state="$3" width=32 filled empty bar=""
  filled=$((completed * width / TOTAL_STEPS))
  empty=$((width - filled))
  (( filled == 0 )) || bar="$(printf '%*s' "${filled}" '' | tr ' ' '#')"
  (( empty == 0 )) || bar+="$(printf '%*s' "${empty}" '' | tr ' ' '.')"
  printf '\r[openvoice-0724] [%s] %d/%d %-46s %s' \
    "${bar}" "${completed}" "${TOTAL_STEPS}" "${label}" "${state}"
  if (( completed >= TOTAL_STEPS )) && [[ "${state}" == "done" ]]; then
    printf '\n'
  fi
}

on_error() {
  local status="$?"
  printf '\n'
  draw_progress "${CURRENT_STEP}" "${CURRENT_LABEL}" "FAILED (exit ${status})"
  printf '\nPipeline stopped before completion.  If this was after the locked-test step, do not rerun this script; use the read-only plot commands from --help.\n' >&2
  exit "${status}"
}
trap on_error ERR

run_step() {
  local label="$1"
  shift
  CURRENT_LABEL="${label}"
  draw_progress "${CURRENT_STEP}" "${label}" "running"
  printf '\n\n===== %s =====\n' "${label}"
  "$@"
  CURRENT_STEP=$((CURRENT_STEP + 1))
  draw_progress "${CURRENT_STEP}" "${label}" "done"
  printf '\n'
}

build_cache() {
  local args=(cache)
  [[ "${ALLOW_NETWORK}" == "1" ]] && args+=(--allow-network)
  bash "${RUNNER}" "${args[@]}"
}

plot_final_pairs() {
  local dataset="$1"
  local args=(plot "${dataset}" test --limit "${PLOT_LIMIT}" --dpi "${PLOT_DPI}")
  [[ -n "${PLOT_MODES}" ]] && args+=(--modes "${PLOT_MODES}")
  bash "${RUNNER}" "${args[@]}"
}

printf '[openvoice-0724] config=%s\n' "${CONFIG}"
printf '[openvoice-0724] output=%s\n' "${OUTPUT_ROOT}"
printf '[openvoice-0724] device=%s; formal seeds=%s\n' \
  "${DEVICE:-auto}" "${SEEDS}"
printf '[openvoice-0724] final test remains unclaimed until step 10/12.\n'

run_step "Read-only source and locked-test preflight" \
  "${PYTHON_BIN}" "${PREFLIGHT}" --config "${CONFIG}" --stage initial
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_ROOT}/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${OUTPUT_ROOT}/xdg_cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"
run_step "Build or verify waveform-consistent teacher cache v2" build_cache
run_step "Train audio-only content/realization factorizer" \
  bash "${RUNNER}" train-audio
run_step "Strict audio oracle audit and freeze" \
  bash "${RUNNER}" audit-audio --strict
run_step "Train/evaluate all registered EEG seeds (15/31/47)" \
  bash "${RUNNER}" seeds
run_step "Run KaraOne subject-LOSO development suite" \
  bash "${RUNNER}" loso-all
run_step "Generate full primary KaraOne validation counterfactuals" \
  bash "${RUNNER}" synthesize karaone validation
run_step "Require strict validation gate before final test" \
  bash "${RUNNER}" gate --strict
run_step "Metadata-only preflight immediately before locked test" \
  "${PYTHON_BIN}" "${PREFLIGHT}" --config "${CONFIG}" --stage before-test
run_step "One locked test session: latent + KaraOne + FEIS reconstructions" \
  bash "${RUNNER}" test
run_step "Render KaraOne final-test reconstruction comparisons" \
  plot_final_pairs karaone
run_step "Render FEIS final-test reconstruction comparisons" \
  plot_final_pairs feis

draw_progress "${CURRENT_STEP}" "formal pipeline" "done"
printf '\nOpenVoice-EEG v0724 completed successfully.\n'
printf 'KaraOne final-test WAVs and comparison PNGs: %s\n' \
  "${OUTPUT_ROOT}/synthesis/karaone/test/comparison_pairs"
printf 'FEIS final-test WAVs and comparison PNGs: %s\n' \
  "${OUTPUT_ROOT}/synthesis/feis/test/comparison_pairs"
printf 'Each panel keeps 80 mel bins fixed, distinguishes predicted energy from decoded-WAV log-mel, and never derives metrics from PNG pixels.\n'
