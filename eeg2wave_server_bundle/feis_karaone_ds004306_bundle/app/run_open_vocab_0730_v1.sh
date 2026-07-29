#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${1:-$APP_DIR/configs/open_vocab_0730_explicit_cp_v1.yaml}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -x "$APP_DIR/.venv_0730/bin/python" ]]; then
  PYTHON_BIN="$APP_DIR/.venv_0730/bin/python"
elif [[ -x /opt/anaconda3/bin/python ]]; then
  PYTHON_BIN=/opt/anaconda3/bin/python
else
  PYTHON_BIN=python3
fi

OUTPUT_ROOT="$APP_DIR/../artifacts/open_vocab_0730_explicit_cp_v1"
RENDERER_CHECKPOINT="$OUTPUT_ROOT/renderer/checkpoints/best.pt"
EEG_CHECKPOINT="$OUTPUT_ROOT/eeg_cp/checkpoints/best.pt"
RUN_MANIFEST="$OUTPUT_ROOT/run_manifest.json"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

cd "$APP_DIR"
if [[ "${RUN_TRAIN:-1}" == "1" ]]; then
  "$PYTHON_BIN" scripts/prepare_open_vocab_0730.py --config "$CFG"
  if [[ "$FORCE_RETRAIN" == "1" || ! -s "$RENDERER_CHECKPOINT" || ! -s "$EEG_CHECKPOINT" || ! -s "$RUN_MANIFEST" ]]; then
    "$PYTHON_BIN" scripts/train_open_vocab_0730.py --config "$CFG" --phase all --wall-hours 9.5 --fresh
  else
    echo "[0730] reusing completed v0730 training (set FORCE_RETRAIN=1 to retrain)"
  fi
  "$PYTHON_BIN" scripts/evaluate_open_vocab_0730.py --config "$CFG" --device "$EVAL_DEVICE"
fi

if [[ "${DOWNLOAD_SPEECHT5_HIFIGAN:-1}" == "1" ]]; then
  "$PYTHON_BIN" scripts/download_speecht5_hifigan.py --config "$CFG"
fi

if [[ "${EXPORT_ALL_PAIRS:-1}" == "1" ]]; then
  "$PYTHON_BIN" scripts/export_open_vocab_0730_pairs.py --config "$CFG" --device "$EXPORT_DEVICE" --resume
  "$PYTHON_BIN" scripts/verify_open_vocab_0730_pairs.py --config "$CFG"
fi
