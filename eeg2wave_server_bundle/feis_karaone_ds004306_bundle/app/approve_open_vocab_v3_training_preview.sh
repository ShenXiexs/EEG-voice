#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${CFG:-$APP_DIR/configs/open_vocab_v3_mfcc_training_first.yaml}"
PY="${PYTHON_BIN:-$APP_DIR/.venv_0730/bin/python}"
REVIEWER="${1:-samxie}"
NOTE="${2:-training WAV pairs manually inspected}"
cd "$APP_DIR"
"$PY" scripts/approve_open_vocab_v3_training_preview.py --config "$CFG" --approve --reviewer "$REVIEWER" --note "$NOTE"
