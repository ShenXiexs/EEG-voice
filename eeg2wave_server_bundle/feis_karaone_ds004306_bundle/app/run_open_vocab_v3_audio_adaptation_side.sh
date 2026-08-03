#!/usr/bin/env bash
# Non-blocking side experiment. Its checkpoints are never selected by the CP
# main runner without an explicit, metric-based manual configuration change.
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${1:-$APP_DIR/configs/open_vocab_v3_cp_temporal_large_v1.yaml}"
PY="${PYTHON_BIN:-$APP_DIR/.venv_0730/bin/python}"
DEVICE="${TRAIN_DEVICE:-auto}"
HOURS="${BUDGET_HOURS:-4}"
DEADLINE="$(( $(date +%s) + $($PY -c 'import sys; print(int(float(sys.argv[1])*3600))' "$HOURS") ))"
cd "$APP_DIR"
echo "[v3 CP side adaptation] non-blocking; frozen models remain the main pipeline default."
"$PY" scripts/finetune_open_vocab_v3_encodec_audio_models.py --config "$CFG" --scope fit --device "$DEVICE" --deadline-epoch "$DEADLINE" --explore
echo "[v3 CP side adaptation] complete; review relative metrics before any manual promotion."
