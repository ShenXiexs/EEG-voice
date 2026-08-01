#!/usr/bin/env bash
# Exploratory-only full v3 content-repair runner.  It deliberately bypasses
# scientific gates and the listening gate, but uses a physically separate
# artifact namespace and labels every report/export as exploratory.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${1:-$APP_DIR/configs/open_vocab_v3_content_repair_v2.yaml}"
PY="${PYTHON_BIN:-$APP_DIR/.venv_0730/bin/python}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
BUDGET_HOURS="${BUDGET_HOURS:-20}"
OUTPUT_ROOT="$APP_DIR/../artifacts/open_vocab_v3_content_repair_v2_explore"
LOG_ROOT="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_ROOT"
RUN_LOG="$LOG_ROOT/run_content_repair_explore_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1
cd "$APP_DIR"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OPEN_VOCAB_V3_EXPLORATION=1

BUDGET_SECONDS="$("$PY" -c 'import sys; print(int(float(sys.argv[1])*3600))' "$BUDGET_HOURS")"
DEADLINE_EPOCH="$(( $(date +%s) + BUDGET_SECONDS ))"
echo "[v3 repair explore] WARNING: all gate failures and the listening gate are bypassed."
echo "[v3 repair explore] outputs are exploratory only; never use held-out results as primary evidence."
echo "[v3 repair explore] config=$CFG"
echo "[v3 repair explore] artifact_root=$OUTPUT_ROOT"
echo "[v3 repair explore] budget_hours=$BUDGET_HOURS deadline_epoch=$DEADLINE_EPOCH"

"$PY" scripts/audit_open_vocab_v3_audio.py --config "$CFG"
"$PY" scripts/denoise_open_vocab_v3.py --config "$CFG" --device "$EVAL_DEVICE"
"$PY" scripts/prepare_open_vocab_v3.py --config "$CFG" --device "$EVAL_DEVICE" --force
"$PY" scripts/download_open_vocab_v3_models.py --config "$CFG"
"$PY" scripts/finetune_open_vocab_v3_encodec_audio_models.py --config "$CFG" --scope fit --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --explore
"$PY" scripts/prepare_open_vocab_v3.py --config "$CFG" --with-speaker --device "$EVAL_DEVICE" --force
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t0 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t0b --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/build_open_vocab_v3_encodec_cache.py --config "$CFG" --device "$EVAL_DEVICE" --force
"$PY" scripts/train_open_vocab_v3_encodec_clip.py --config "$CFG" --phase audio_content --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t1 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t1d --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/train_open_vocab_v3_encodec_clip.py --config "$CFG" --phase cvae --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t2 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t2v --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t3 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/train_open_vocab_v3_encodec_clip.py --config "$CFG" --phase micro --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase micro --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/export_open_vocab_v3_encodec_clip_pairs.py --config "$CFG" --stage micro --device "$EXPORT_DEVICE" --resume --explore
"$PY" scripts/train_open_vocab_v3_encodec_clip.py --config "$CFG" --phase fit --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase fit --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/export_open_vocab_v3_encodec_clip_pairs.py --config "$CFG" --stage fit --device "$EXPORT_DEVICE" --resume --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase validation --device "$EVAL_DEVICE" --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase locked --device "$EVAL_DEVICE" --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase locked_unseen --device "$EVAL_DEVICE" --explore
"$PY" scripts/export_open_vocab_v3_encodec_clip_pairs.py --config "$CFG" --stage final --device "$EXPORT_DEVICE" --resume --explore
echo "[v3 repair explore] complete"
echo "[v3 repair explore] final_pairs=$OUTPUT_ROOT/pairs/encodec_clip_mfcc_training_fit_v1"
