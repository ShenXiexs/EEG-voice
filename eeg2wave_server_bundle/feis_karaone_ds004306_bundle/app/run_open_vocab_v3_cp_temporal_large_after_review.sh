#!/usr/bin/env bash
# Continue the strict CP-temporal run after approving the exact training preview.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${1:-$APP_DIR/configs/open_vocab_v3_cp_temporal_large_v1.yaml}"
PY="${PYTHON_BIN:-$APP_DIR/.venv_0730/bin/python}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
OUTPUT_ROOT="$APP_DIR/../artifacts/open_vocab_v3_cp_temporal_large_v1"
LOG_ROOT="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_ROOT"
mkdir -p "$OUTPUT_ROOT/.matplotlib" "$OUTPUT_ROOT/.cache"
RUN_LOG="$LOG_ROOT/run_cp_temporal_after_review_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1
cd "$APP_DIR"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MPLCONFIGDIR="$OUTPUT_ROOT/.matplotlib" XDG_CACHE_HOME="$OUTPUT_ROOT/.cache"

"$PY" scripts/approve_open_vocab_v3_training_preview.py --config "$CFG" --check
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase validation --device "$EVAL_DEVICE"
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase locked --device "$EVAL_DEVICE"
"$PY" scripts/evaluate_open_vocab_v3_cp_temporal.py --config "$CFG" --phase locked_unseen --device "$EVAL_DEVICE"
"$PY" scripts/export_open_vocab_v3_cp_temporal_pairs.py --config "$CFG" --stage final --device "$EXPORT_DEVICE" --resume
"$PY" scripts/finalize_open_vocab_v3_cp_temporal_run.py --config "$CFG" --phase complete
echo "[v3 CP] held-out evaluation and final export complete"
echo "[v3 CP] final_pairs=$OUTPUT_ROOT/pairs/final"
