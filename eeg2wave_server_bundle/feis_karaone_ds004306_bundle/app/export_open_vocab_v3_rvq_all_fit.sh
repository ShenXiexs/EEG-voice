#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ID="${RUN_ID:-20260804_232133}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-2}"
OUT_NAME="open_vocab_v3_mfcc_encodec_rvq_repair_v3_run_${RUN_ID}_explore/pairs/all_fit_m0b_1016_explore"
OUT="$APP_DIR/../artifacts/$OUT_NAME"
CFG="$APP_DIR/configs/open_vocab_v3_mfcc_encodec_rvq_repair_v3.yaml"
PY="$APP_DIR/.venv_0730/bin/python"
[[ -x "$PY" ]] || { echo "missing python: $PY" >&2; exit 2; }
[[ -f "$CFG" ]] || { echo "missing config: $CFG" >&2; exit 2; }
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OPEN_VOCAB_V3_EXPLORATION=1
export OPEN_VOCAB_V3_ARTIFACT_ROOT_NAME="open_vocab_v3_mfcc_encodec_rvq_repair_v3_run_${RUN_ID}_explore"
cd "$APP_DIR"
if [[ "${RESUME:-0}" == "1" ]]; then
  exec "$PY" scripts/export_open_vocab_v3_encodec_rvq_repair_all_fit.py \
    --config "$CFG" --device "$DEVICE" --batch-size "$BATCH_SIZE" \
    --output-root "$OUT" --explore --resume
else
  exec "$PY" scripts/export_open_vocab_v3_encodec_rvq_repair_all_fit.py \
    --config "$CFG" --device "$DEVICE" --batch-size "$BATCH_SIZE" \
    --output-root "$OUT" --explore
fi
