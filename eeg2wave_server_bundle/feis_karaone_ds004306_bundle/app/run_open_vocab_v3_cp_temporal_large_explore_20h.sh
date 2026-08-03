#!/usr/bin/env bash
# Exploratory CP-temporal-large runner: records every failed gate and continues.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${1:-$APP_DIR/configs/open_vocab_v3_cp_temporal_large_v1.yaml}"
PY="${PYTHON_BIN:-$APP_DIR/.venv_0730/bin/python}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
BUDGET_HOURS="${BUDGET_HOURS:-30}"
OUTPUT_ROOT="$APP_DIR/../artifacts/open_vocab_v3_cp_temporal_large_v1_explore"
LOG_ROOT="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_ROOT"
mkdir -p "$OUTPUT_ROOT/.matplotlib" "$OUTPUT_ROOT/.cache"
RUN_LOG="$LOG_ROOT/run_cp_temporal_large_explore_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1
cd "$APP_DIR"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OPEN_VOCAB_V3_EXPLORATION=1 MPLCONFIGDIR="$OUTPUT_ROOT/.matplotlib" XDG_CACHE_HOME="$OUTPUT_ROOT/.cache"

BUDGET_SECONDS="$($PY -c 'import sys; print(int(float(sys.argv[1])*3600))' "$BUDGET_HOURS")"
DEADLINE_EPOCH="$(( $(date +%s) + BUDGET_SECONDS ))"
echo "[v3 CP explore] WARNING: gates and human approval are bypassed; all held-out output is exploratory."
echo "[v3 CP explore] config=$CFG root=$OUTPUT_ROOT budget_hours=$BUDGET_HOURS"

"$PY" scripts/audit_open_vocab_v3_audio.py --config "$CFG"
"$PY" scripts/denoise_open_vocab_v3.py --config "$CFG" --device "$EVAL_DEVICE"
"$PY" scripts/prepare_open_vocab_v3.py --config "$CFG" --device "$EVAL_DEVICE" --force
"$PY" scripts/download_open_vocab_v3_models.py --config "$CFG"
"$PY" scripts/prepare_open_vocab_v3.py --config "$CFG" --with-speaker --device "$EVAL_DEVICE" --force
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase t0 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/train_open_vocab_v3_cp_temporal.py --config "$CFG" --phase oracle --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh --explore
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase oracle --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/build_open_vocab_v3_cp_temporal_encodec_cache.py --config "$CFG" --device "$EVAL_DEVICE" --force
"$PY" scripts/train_open_vocab_v3_cp_temporal.py --config "$CFG" --phase prosody --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh --explore
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase prosody --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/train_open_vocab_v3_cp_temporal.py --config "$CFG" --phase content --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh --explore
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase content --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase intervention --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/train_open_vocab_v3_cp_temporal.py --config "$CFG" --phase cvae --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh --explore
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase cvae --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/train_open_vocab_v3_cp_temporal.py --config "$CFG" --phase micro --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh --explore
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase micro --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/export_open_vocab_v3_cp_temporal_pairs.py --config "$CFG" --stage micro --device "$EXPORT_DEVICE" --resume --explore
"$PY" scripts/train_open_vocab_v3_cp_temporal.py --config "$CFG" --phase fit --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh --explore
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase fit --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/export_open_vocab_v3_cp_temporal_pairs.py --config "$CFG" --stage fit --device "$EXPORT_DEVICE" --resume --explore
"$PY" scripts/train_open_vocab_v3_cp_temporal.py --config "$CFG" --phase eeg_prosody --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh --explore
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase eeg_prosody --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase validation --device "$EVAL_DEVICE" --explore
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase locked --device "$EVAL_DEVICE" --explore
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase locked_unseen --device "$EVAL_DEVICE" --explore
"$PY" scripts/export_open_vocab_v3_cp_temporal_pairs.py --config "$CFG" --stage final --device "$EXPORT_DEVICE" --resume --explore
"$PY" scripts/finalize_open_vocab_v3_cp_temporal_run.py --config "$CFG" --phase complete --explore
echo "[v3 CP explore] complete"
echo "[v3 CP explore] final_pairs=$OUTPUT_ROOT/pairs/final"
