#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
CONFIG="${CONFIG:-${BUNDLE_DIR}/app/configs/open_vocab_0728_duallatent_v1.yaml}"
root="${BUNDLE_DIR}/artifacts/open_vocab_0728_duallatent_v1"

# Every invocation has durable logs, including Python tracebacks.  Child
# invocations inherit RUN_ID/RUN_LOG_DIR, so `full.log` is the chronological
# record of the complete pipeline while each stage gets its own focused log.
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${root}/logs/runs/${RUN_ID}}"
export RUN_ID RUN_LOG_DIR
mkdir -p "${RUN_LOG_DIR}"

stage_raw="${1:-usage}"
stage_key="$(printf '%s' "${stage_raw}" | tr -cs '[:alnum:]_.-' '_')"
stage_log="${RUN_LOG_DIR}/${stage_key}.log"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
command_line="$(printf '%q ' "$0" "$@")"

# Run the actual stage in a child shell.  This captures every stdout/stderr
# line in a normal tee pipeline, while keeping `set -e` active inside the
# child (wrapping a shell function in a pipeline would weaken fail-closed
# behaviour in Bash).  The guard prevents recursive tee setup.
if [[ "${V0728_LOGGING_ACTIVE:-0}" != "1" ]]; then
  export V0728_LOGGING_ACTIVE=1
  if bash "$0" "$@" 2>&1 | tee -a "${stage_log}"; then
    exit 0
  else
    exit_code=${PIPESTATUS[0]}
    exit "${exit_code}"
  fi
fi

write_stage_status() {
  local exit_code=$?
  local finished_at
  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'stage=%s\n' "${stage_raw}"
    printf 'status=%s\n' "$([[ ${exit_code} -eq 0 ]] && printf passed || printf failed)"
    printf 'exit_code=%s\n' "${exit_code}"
    printf 'started_at=%s\n' "${started_at}"
    printf 'finished_at=%s\n' "${finished_at}"
    printf 'command=%s\n' "${command_line}"
    printf 'log=%s\n' "${stage_log}"
  } > "${RUN_LOG_DIR}/${stage_key}.status"
  if [[ ${exit_code} -ne 0 ]]; then
    local failure_file="${RUN_LOG_DIR}/${stage_key}.failure"
    {
      printf 'v0728 stage failed; inspect the referenced stage log for the complete traceback.\n'
      printf 'run_id=%s\nstage=%s\nexit_code=%s\nlog=%s\n' "${RUN_ID}" "${stage_raw}" "${exit_code}" "${stage_log}"
    } > "${failure_file}"
    cp "${failure_file}" "${root}/logs/latest_failure.txt"
    printf '[0728] FAILED stage=%s exit=%s; traceback: %s\n' "${stage_raw}" "${exit_code}" "${stage_log}" >&2
  else
    printf '[0728] completed stage=%s; log: %s\n' "${stage_raw}" "${stage_log}"
  fi
}
trap write_stage_status EXIT

printf '[0728] run_id=%s stage=%s started=%s\n' "${RUN_ID}" "${stage_raw}" "${started_at}"
printf '[0728] log=%s\n' "${stage_log}"

DEVICE_ARGS=()
[[ -n "${DEVICE:-}" ]] && DEVICE_ARGS=(--device "${DEVICE}")

train() { "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/train_open_vocab_0728.py" --config "${CONFIG}" --phase "$1" "${DEVICE_ARGS[@]}" "${@:2}"; }
synth() { "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/synthesize_open_vocab_0728.py" --config "${CONFIG}" --phase "$1" --split "$2" "${DEVICE_ARGS[@]}" "${@:3}"; }

case "${1:-}" in
  preflight) "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/preflight_open_vocab_0728.py" --config "${CONFIG}" ;;
  cache) "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/build_open_vocab_0728_cache.py" --config "${CONFIG}" "${DEVICE_ARGS[@]}" "${@:2}" ;;
  metric) "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/validate_open_vocab_0728_metric.py" --config "${CONFIG}" "${@:2}" ;;
  ceiling) "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/audit_open_vocab_0728_griffin_lim.py" --config "${CONFIG}" "${DEVICE_ARGS[@]}" "${@:2}" ;;
  audit-disentanglement) "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/audit_open_vocab_0728_disentanglement.py" --config "${CONFIG}" "${DEVICE_ARGS[@]}" ;;
  train-audio) train audio "${@:2}" ;;
  train-semantic4) train semantic4 "${@:2}" ;;
  train-dual4) train dual4 "${@:2}" ;;
  train-full11) train full11 "${@:2}" ;;
  synthesize)
    # Keep the parameter-expansion error text free of a second literal `}`;
    # otherwise bash treats that brace as part of the phase value.
    phase="${2:?missing phase (semantic4|dual4|full11)}"; split="${3:?missing split}"
    shift 3; synth "${phase}" "${split}" "$@" ;;
  plot)
    phase="${2:?missing phase}"; split="${3:?missing split}"
    "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/plot_open_vocab_0728_pairs.py" --manifest "${root}/synthesis/${phase}/${split}/synthesis_manifest.json" "${@:4}" ;;
  gate)
    phase="${2:?missing phase (semantic4|dual4|full11)}"; split="${3:-validation}"
    "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/gate_open_vocab_0728.py" --config "${CONFIG}" --phase "${phase}" --manifest "${root}/synthesis/${phase}/${split}/synthesis_manifest.json" ;;
  freeze-test)
    phase="${2:-full11}"
    "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/gate_open_vocab_0728.py" --config "${CONFIG}" --phase "${phase}" --manifest "${root}/synthesis/${phase}/validation/synthesis_manifest.json" --freeze-locked-test ;;
  all-development)
    bash "$0" preflight
    bash "$0" cache
    bash "$0" metric
    bash "$0" development-after-metric
    ;;
  # Resume point after the train-only cache and frozen STSS metric manifest
  # have completed successfully.  This intentionally does not rerun either
  # stage, so a metric-gate repair can continue without invalidating them.
  development-after-metric)
    bash "$0" ceiling
    bash "$0" train-audio
    bash "$0" audit-disentanglement
    bash "$0" train-semantic4
    bash "$0" synthesize semantic4 validation
    bash "$0" gate semantic4 validation
    bash "$0" train-dual4
    bash "$0" synthesize dual4 validation
    bash "$0" gate dual4 validation
    bash "$0" train-full11
    bash "$0" synthesize full11 validation
    bash "$0" gate full11 validation
    bash "$0" plot full11 validation
    ;;
  locked-test)
    access="${TEST_ACCESS_ID:?Set TEST_ACCESS_ID to a stable resumable formal-test id}"
    bash "$0" freeze-test full11
    bash "$0" synthesize full11 locked_test --access-id "${access}"
    bash "$0" gate full11 locked_test
    bash "$0" plot full11 locked_test
    ;;
  full)
    : "${TEST_ACCESS_ID:?Set TEST_ACCESS_ID before the preregistered one-time formal run}"
    bash "$0" all-development
    bash "$0" loso-all
    bash "$0" aggregate-loso
    bash "$0" locked-test
    ;;
  continue-full)
    : "${TEST_ACCESS_ID:?Set TEST_ACCESS_ID before the preregistered one-time formal run}"
    bash "$0" development-after-metric
    bash "$0" loso-all
    bash "$0" aggregate-loso
    bash "$0" locked-test
    ;;
  loso-shared)
    subject="${2:?usage: $0 loso-shared SUBJECT [SEED]}"; seed="${3:-15}"
    bash "$0" train-semantic4 --loso-subject "${subject}" --seed "${seed}"
    bash "$0" train-dual4 --loso-subject "${subject}" --seed "${seed}"
    bash "$0" train-full11 --loso-subject "${subject}" --seed "${seed}"
    ;;
  loso-strict)
    subject="${2:?usage: $0 loso-strict SUBJECT}"; seed="${3:-15}"
    bash "$0" train-audio --loso-subject "${subject}" --strict-audio-loso --seed "${seed}"
    bash "$0" train-semantic4 --loso-subject "${subject}" --strict-audio-loso --seed "${seed}"
    bash "$0" train-dual4 --loso-subject "${subject}" --strict-audio-loso --seed "${seed}"
    bash "$0" train-full11 --loso-subject "${subject}" --strict-audio-loso --seed "${seed}"
    ;;
  loso-all)
    while IFS= read -r subject; do
      for seed in ${SEEDS:-15 31 47}; do bash "$0" loso-shared "${subject}" "${seed}"; done
      bash "$0" loso-strict "${subject}" 15
    done <<'SUBJECTS'
karaone:MM05
karaone:MM08
karaone:MM09
karaone:MM10
karaone:MM11
karaone:MM12
karaone:MM14
karaone:MM15
karaone:MM16
karaone:MM18
karaone:MM19
karaone:MM20
SUBJECTS
    ;;
  aggregate-loso) "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/aggregate_open_vocab_0728_loso.py" --config "${CONFIG}" ;;
  *) echo "usage: $0 {preflight|cache|metric|ceiling|audit-disentanglement|train-audio|train-semantic4|train-dual4|train-full11|synthesize|plot|gate|freeze-test|all-development|development-after-metric|locked-test|loso-shared|loso-strict|loso-all|aggregate-loso|full|continue-full}" >&2; exit 2 ;;
esac
