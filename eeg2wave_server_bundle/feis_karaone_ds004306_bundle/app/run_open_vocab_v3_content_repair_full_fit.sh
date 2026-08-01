#!/usr/bin/env bash
# Stage 2 of the fail-closed v3 content-chain repair protocol.
# This script may only produce full-fit *training* WAVs.  It never reads a
# held-out EEG/audio role; use the after-review script only after listening.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${1:-$APP_DIR/configs/open_vocab_v3_content_repair_v2.yaml}"
PY="${PYTHON_BIN:-$APP_DIR/.venv_0730/bin/python}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
BUDGET_HOURS="${BUDGET_HOURS:-20}"
OUTPUT_ROOT="$APP_DIR/../artifacts/open_vocab_v3_content_repair_v2"
LOG_ROOT="$OUTPUT_ROOT/logs"
STATE_PATH="$OUTPUT_ROOT/run_state/training_budget.json"
mkdir -p "$LOG_ROOT"
RUN_LOG="$LOG_ROOT/run_content_repair_full_fit_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1
cd "$APP_DIR"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

# A caller may pass the deadline created by stage 1.  When review happens in
# a separate session, deliberately starting a new budget is explicit through
# BUDGET_HOURS rather than silently inheriting a stale shell timestamp.
if [[ -n "${DEADLINE_EPOCH:-}" ]]; then
  : # Explicit override is recorded by the surrounding terminal log.
elif [[ -f "$STATE_PATH" ]]; then
  DEADLINE_EPOCH="$("$PY" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["deadline_epoch"]))' "$STATE_PATH")"
else
  BUDGET_SECONDS="$("$PY" -c 'import sys; print(int(float(sys.argv[1])*3600))' "$BUDGET_HOURS")"
  DEADLINE_EPOCH="$(( $(date +%s) + BUDGET_SECONDS ))"
fi

echo "[v3 repair full-fit] config=$CFG"
echo "[v3 repair full-fit] deadline_epoch=$DEADLINE_EPOCH"
if [[ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]]; then
  echo "[v3 repair full-fit] the shared 20-hour deadline has elapsed; refusing to begin another training phase."
  exit 2
fi
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase micro --device "$EVAL_DEVICE"
"$PY" scripts/train_open_vocab_v3_encodec_clip.py --config "$CFG" --phase fit --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase fit --device "$EVAL_DEVICE"
"$PY" scripts/export_open_vocab_v3_encodec_clip_pairs.py --config "$CFG" --stage fit --device "$EVAL_DEVICE" --resume
echo "[v3 repair full-fit] training preview is ready; held-out access remains blocked."
echo "[v3 repair full-fit] preview=$OUTPUT_ROOT/pairs/full_fit_preview"
