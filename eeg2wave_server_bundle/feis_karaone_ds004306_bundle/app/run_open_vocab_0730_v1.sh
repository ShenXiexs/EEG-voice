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

cd "$APP_DIR"
if [[ "${RUN_TRAIN:-1}" == "1" ]]; then
  "$PYTHON_BIN" scripts/prepare_open_vocab_0730.py --config "$CFG"
  "$PYTHON_BIN" scripts/train_open_vocab_0730.py --config "$CFG" --phase all --wall-hours 9.5 --fresh
  "$PYTHON_BIN" scripts/evaluate_open_vocab_0730.py --config "$CFG"
fi

if [[ "${DOWNLOAD_SPEECHT5_HIFIGAN:-1}" == "1" ]]; then
  "$PYTHON_BIN" scripts/download_speecht5_hifigan.py --config "$CFG"
fi

if [[ "${EXPORT_ALL_PAIRS:-1}" == "1" ]]; then
  "$PYTHON_BIN" scripts/export_open_vocab_0730_pairs.py --config "$CFG"
  "$PYTHON_BIN" scripts/verify_open_vocab_0730_pairs.py --config "$CFG"
fi
