#!/usr/bin/env bash
set -eo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
CONFIG="${CONFIG:-${BUNDLE_DIR}/app/configs/open_vocab_0724_factorized_v1.yaml}"
DEVICE_ARGS=()
[[ -n "${DEVICE:-}" ]] && DEVICE_ARGS=(--device "${DEVICE}")

run_train() {
  "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/train_open_vocab_0724.py" \
    --config "${CONFIG}" --phase "$1" "${DEVICE_ARGS[@]}" "${@:2}"
}

case "${1:-}" in
  cache)
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/build_open_vocab_teacher_cache_0724.py" \
      --config "${CONFIG}" "${DEVICE_ARGS[@]}" "${@:2}"
    ;;
  train-audio)
    run_train audio "${@:2}"
    ;;
  audit-audio)
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/audit_open_vocab_0724_audio_oracle.py" \
      --config "${CONFIG}" "${DEVICE_ARGS[@]}" "${@:2}"
    ;;
  pretrain-eeg)
    run_train eeg-pretrain "${@:2}"
    ;;
  train-eeg)
    run_train eeg "${@:2}"
    ;;
  validate)
    run_train evaluate --split validation "${@:2}"
    ;;
  synthesize)
    dataset="${2:?usage: $0 synthesize {karaone|feis} [validation|test] [options]}"
    shift 2
    split="validation"
    if [[ "${1:-}" == "validation" || "${1:-}" == "test" ]]; then
      split="$1"
      shift
    fi
    args=(--config "${CONFIG}" --dataset "${dataset}" --split "${split}" "${DEVICE_ARGS[@]}")
    [[ "${split}" == "test" ]] && args+=(--allow-final-test)
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/synthesize_open_vocab_0724.py" "${args[@]}" "$@"
    ;;
  gate)
    manifest="${SYNTHESIS_MANIFEST:-${BUNDLE_DIR}/artifacts/open_vocab_0724_factorized_v1/synthesis/karaone/validation/synthesis_manifest.json}"
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/gate_open_vocab_0724.py" \
      --config "${CONFIG}" --synthesis-manifest "${manifest}" "${@:2}"
    ;;
  seeds)
    for seed in ${SEEDS:-15 31 47}; do
      "${BASH_SOURCE[0]}" pretrain-eeg --seed "${seed}"
      "${BASH_SOURCE[0]}" train-eeg --seed "${seed}"
      "${BASH_SOURCE[0]}" validate --seed "${seed}"
    done
    ;;
  loso)
    subject="${2:?usage: $0 loso SUBJECT_GROUP_ID}"
    for seed in ${SEEDS:-15 31 47}; do
      "${BASH_SOURCE[0]}" pretrain-eeg --seed "${seed}" --loso-subject "${subject}"
      "${BASH_SOURCE[0]}" train-eeg --seed "${seed}" --loso-subject "${subject}"
      "${BASH_SOURCE[0]}" validate --seed "${seed}" --loso-subject "${subject}"
      if [[ "${seed}" == "15" ]]; then
        "${BASH_SOURCE[0]}" synthesize karaone validation --seed "${seed}" --loso-subject "${subject}"
      fi
    done
    ;;
  loso-all)
    while IFS= read -r subject; do
      [[ -z "${subject}" ]] && continue
      "${BASH_SOURCE[0]}" loso "${subject}"
    done < <(
      "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/gate_open_vocab_0724.py" \
        --config "${CONFIG}" --list-required-loso-subjects
    )
    ;;
  held-label)
    label="${2:?usage: $0 held-label LABEL}"
    for setting in g2 g3; do
      for seed in ${SEEDS:-15 31 47}; do
        "${BASH_SOURCE[0]}" pretrain-eeg --seed "${seed}" --generalization "${setting}" --holdout-label "${label}"
        "${BASH_SOURCE[0]}" train-eeg --seed "${seed}" --generalization "${setting}" --holdout-label "${label}"
        "${BASH_SOURCE[0]}" validate --seed "${seed}" --generalization "${setting}" --holdout-label "${label}"
      done
    done
    ;;
  ablation-config)
    name="${2:?usage: $0 ablation-config ABLATION OUTPUT_YAML [--contentvec-model MODEL]}"
    output="${3:?usage: $0 ablation-config ABLATION OUTPUT_YAML [--contentvec-model MODEL]}"
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/make_open_vocab_0724_ablation_config.py" \
      --base-config "${CONFIG}" --ablation "${name}" --output "${output}" "${@:4}"
    ;;
  test)
    access_id="${TEST_ACCESS_ID:-v0724_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
    run_train evaluate --split test --allow-final-test --test-access-id "${access_id}" "${@:2}"
    "${BASH_SOURCE[0]}" synthesize karaone test --test-access-id "${access_id}" "${@:2}"
    "${BASH_SOURCE[0]}" synthesize feis test --test-access-id "${access_id}" "${@:2}"
    ;;
  all)
    "${BASH_SOURCE[0]}" cache
    "${BASH_SOURCE[0]}" train-audio
    "${BASH_SOURCE[0]}" audit-audio --strict
    "${BASH_SOURCE[0]}" seeds
    "${BASH_SOURCE[0]}" loso-all
    "${BASH_SOURCE[0]}" synthesize karaone validation
    "${BASH_SOURCE[0]}" gate --strict
    ;;
  *)
    echo "usage: $0 {cache|train-audio|audit-audio|pretrain-eeg|train-eeg|validate|synthesize|gate|seeds|loso|loso-all|held-label|ablation-config|test|all} [options]" >&2
    exit 2
    ;;
esac
