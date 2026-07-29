#!/usr/bin/env bash
# Assemble a ready-to-play v1/v2 control-audio showcase. No training.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="$APP_DIR/configs/open_vocab_0730_explicit_cp_fixed_v2.yaml"
OUT="${PRESENTATION_OUTPUT:-$APP_DIR/../reports/v1_v2_eeg_to_speech_technical_report/presentation_audio_trials}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$APP_DIR/.venv_0730/bin/python" ]]; then
  PY="$APP_DIR/.venv_0730/bin/python"
else
  PY=python3
fi

export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/v1_v2_presentation_pycache}"
cd "$APP_DIR"
echo "[presentation pairs] source=v1 existing WAVs + v2 existing checkpoint"
echo "[presentation pairs] output=$OUT"
"$PY" scripts/assemble_v1_v2_presentation_pairs.py \
  --config "$CFG" \
  --output "$OUT" \
  --device cpu \
  --resume
