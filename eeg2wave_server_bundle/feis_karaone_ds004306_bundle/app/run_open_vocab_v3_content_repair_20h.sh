#!/usr/bin/env bash
# Fail-closed v3 content-chain repair.  Existing exploratory artifacts are
# never reused; held-out EEG remains inaccessible until a human approves the
# exported training WAVs.
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
RUN_LOG="$LOG_ROOT/run_content_repair_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1
cd "$APP_DIR"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
BUDGET_SECONDS="$("$PY" -c 'import sys; print(int(float(sys.argv[1])*3600))' "$BUDGET_HOURS")"
START_EPOCH="$(date +%s)"
DEADLINE_EPOCH="$(( START_EPOCH + BUDGET_SECONDS ))"
mkdir -p "$(dirname "$STATE_PATH")"
"$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"schema":"openvoice-v3-content-repair-budget-v1","budget_hours":float(sys.argv[2]),"started_epoch":int(sys.argv[3]),"deadline_epoch":int(sys.argv[4])},indent=2)+"\n")' "$STATE_PATH" "$BUDGET_HOURS" "$START_EPOCH" "$DEADLINE_EPOCH"
echo "[v3 repair] config=$CFG"
echo "[v3 repair] root=$OUTPUT_ROOT"
echo "[v3 repair] budget_hours=$BUDGET_HOURS"
echo "[v3 repair] deadline_epoch=$DEADLINE_EPOCH"
"$PY" scripts/audit_open_vocab_v3_audio.py --config "$CFG"
"$PY" scripts/denoise_open_vocab_v3.py --config "$CFG" --device "$EVAL_DEVICE"
"$PY" scripts/prepare_open_vocab_v3.py --config "$CFG" --device "$EVAL_DEVICE" --force
"$PY" scripts/download_open_vocab_v3_models.py --config "$CFG"
"$PY" scripts/finetune_open_vocab_v3_encodec_audio_models.py --config "$CFG" --scope fit --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH"
"$PY" scripts/prepare_open_vocab_v3.py --config "$CFG" --with-speaker --device "$EVAL_DEVICE" --force
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t0 --device "$EVAL_DEVICE"
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t0b --device "$EVAL_DEVICE"
"$PY" scripts/build_open_vocab_v3_encodec_cache.py --config "$CFG" --device "$EVAL_DEVICE" --force
"$PY" scripts/train_open_vocab_v3_encodec_clip.py --config "$CFG" --phase audio_content --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t1 --device "$EVAL_DEVICE"
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t1d --device "$EVAL_DEVICE"
"$PY" scripts/train_open_vocab_v3_encodec_clip.py --config "$CFG" --phase cvae --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t2 --device "$EVAL_DEVICE"
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t2v --device "$EVAL_DEVICE"
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t3 --device "$EVAL_DEVICE"
"$PY" scripts/train_open_vocab_v3_encodec_clip.py --config "$CFG" --phase micro --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase micro --device "$EVAL_DEVICE"
"$PY" scripts/export_open_vocab_v3_encodec_clip_pairs.py --config "$CFG" --stage micro --device "$EVAL_DEVICE" --resume
echo "[v3 repair] micro training WAVs are ready. Review them before the approval/held-out phase."
