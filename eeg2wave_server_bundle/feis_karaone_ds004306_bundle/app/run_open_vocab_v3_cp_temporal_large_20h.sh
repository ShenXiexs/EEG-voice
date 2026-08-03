#!/usr/bin/env bash
# Fail-closed v3 CP-temporal-large runner. Held-out data remain inaccessible
# until the exact training preview lineage has explicit human approval.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${1:-$APP_DIR/configs/open_vocab_v3_cp_temporal_large_v1.yaml}"
PY="${PYTHON_BIN:-$APP_DIR/.venv_0730/bin/python}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
BUDGET_HOURS="${BUDGET_HOURS:-30}"
OUTPUT_ROOT="$APP_DIR/../artifacts/open_vocab_v3_cp_temporal_large_v1"
LOG_ROOT="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_ROOT"
mkdir -p "$OUTPUT_ROOT/.matplotlib" "$OUTPUT_ROOT/.cache"
RUN_LOG="$LOG_ROOT/run_cp_temporal_large_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1
cd "$APP_DIR"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MPLCONFIGDIR="$OUTPUT_ROOT/.matplotlib" XDG_CACHE_HOME="$OUTPUT_ROOT/.cache"

BUDGET_SECONDS="$($PY -c 'import sys; print(int(float(sys.argv[1])*3600))' "$BUDGET_HOURS")"
DEADLINE_EPOCH="$(( $(date +%s) + BUDGET_SECONDS ))"
echo "[v3 CP] config=$CFG"
echo "[v3 CP] artifact_root=$OUTPUT_ROOT"
echo "[v3 CP] train_device=$TRAIN_DEVICE evaluation_device=$EVAL_DEVICE export_device=$EXPORT_DEVICE"
echo "[v3 CP] training_budget_hours=$BUDGET_HOURS deadline_epoch=$DEADLINE_EPOCH"

"$PY" scripts/audit_open_vocab_v3_audio.py --config "$CFG"
"$PY" scripts/denoise_open_vocab_v3.py --config "$CFG" --device "$EVAL_DEVICE"
"$PY" scripts/prepare_open_vocab_v3.py --config "$CFG" --device "$EVAL_DEVICE" --force
"$PY" scripts/download_open_vocab_v3_models.py --config "$CFG"
"$PY" scripts/prepare_open_vocab_v3.py --config "$CFG" --with-speaker --device "$EVAL_DEVICE" --force
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase t0 --device "$EVAL_DEVICE"
"$PY" scripts/train_open_vocab_v3_cp_temporal.py --config "$CFG" --phase oracle --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase oracle --device "$EVAL_DEVICE"
"$PY" scripts/build_open_vocab_v3_cp_temporal_encodec_cache.py --config "$CFG" --device "$EVAL_DEVICE" --force
"$PY" scripts/train_open_vocab_v3_cp_temporal.py --config "$CFG" --phase prosody --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase prosody --device "$EVAL_DEVICE"
"$PY" scripts/train_open_vocab_v3_cp_temporal.py --config "$CFG" --phase content --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase content --device "$EVAL_DEVICE"
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase intervention --device "$EVAL_DEVICE"
"$PY" scripts/train_open_vocab_v3_cp_temporal.py --config "$CFG" --phase cvae --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase cvae --device "$EVAL_DEVICE"
"$PY" scripts/train_open_vocab_v3_cp_temporal.py --config "$CFG" --phase micro --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase micro --device "$EVAL_DEVICE"
"$PY" scripts/export_open_vocab_v3_cp_temporal_pairs.py --config "$CFG" --stage micro --device "$EXPORT_DEVICE" --resume
"$PY" scripts/train_open_vocab_v3_cp_temporal.py --config "$CFG" --phase fit --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase fit --device "$EVAL_DEVICE"
"$PY" scripts/export_open_vocab_v3_cp_temporal_pairs.py --config "$CFG" --stage fit --device "$EXPORT_DEVICE" --resume
"$PY" scripts/train_open_vocab_v3_cp_temporal.py --config "$CFG" --phase eeg_prosody --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase eeg_prosody --device "$EVAL_DEVICE" --no-fail
"$PY" scripts/finalize_open_vocab_v3_cp_temporal_run.py --config "$CFG" --phase training_preview

REVIEW="$OUTPUT_ROOT/gates/E_training_wav_human_review.json"
if [[ ! -f "$REVIEW" ]]; then
  echo "[v3 CP] training WAVs are ready at $OUTPUT_ROOT/pairs/full_fit_preview"
  echo "[v3 CP] approve the exact checkpoint/preview before held-out evaluation."
  exit 0
fi
"$PY" scripts/approve_open_vocab_v3_training_preview.py --config "$CFG" --check

"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase validation --device "$EVAL_DEVICE"
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase locked --device "$EVAL_DEVICE"
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase locked_unseen --device "$EVAL_DEVICE"
"$PY" scripts/export_open_vocab_v3_cp_temporal_pairs.py --config "$CFG" --stage final --device "$EXPORT_DEVICE" --resume
"$PY" scripts/finalize_open_vocab_v3_cp_temporal_run.py --config "$CFG" --phase complete
echo "[v3 CP] complete"
echo "[v3 CP] final_pairs=$OUTPUT_ROOT/pairs/final"
