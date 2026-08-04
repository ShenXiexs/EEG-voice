#!/usr/bin/env bash
# Explore runner: records failed fit-only gates but never reads held-out data.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${1:-$APP_DIR/configs/open_vocab_v3_mfcc_encodec_bridge_v2.yaml}"
PY="${PYTHON_BIN:-$APP_DIR/.venv_0730/bin/python}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
BRIDGE_DEVICE="${BRIDGE_DEVICE:-cpu}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
BUDGET_HOURS="${BUDGET_HOURS:-30}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[v3 bridge explore] RUN_ID may contain only letters, digits, dot, underscore, and dash." >&2
  exit 2
fi
export OPEN_VOCAB_V3_EXPLORATION=1
export OPEN_VOCAB_V3_ARTIFACT_ROOT_NAME="open_vocab_v3_mfcc_encodec_bridge_v2_run_${RUN_ID}"
OUTPUT_ROOT="$APP_DIR/../artifacts/${OPEN_VOCAB_V3_ARTIFACT_ROOT_NAME}_explore"
LOG_ROOT="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT/.matplotlib" "$OUTPUT_ROOT/.cache"
RUN_LOG="$LOG_ROOT/run_explore_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1
cd "$APP_DIR"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MPLCONFIGDIR="$OUTPUT_ROOT/.matplotlib" XDG_CACHE_HOME="$OUTPUT_ROOT/.cache" PYTHONPYCACHEPREFIX="$OUTPUT_ROOT/.pycache"
BUDGET_SECONDS="$($PY -c 'import sys; print(int(float(sys.argv[1])*3600))' "$BUDGET_HOURS")"
DEADLINE_EPOCH="$(( $(date +%s) + BUDGET_SECONDS ))"
echo "[v3 bridge explore] WARNING: failed gates are recorded and bypassed; no held-out data are accessed."
echo "[v3 bridge explore] config=$CFG artifact_root=$OUTPUT_ROOT run_id=$RUN_ID total_training_budget_hours=$BUDGET_HOURS"
echo "[v3 bridge explore] bridge_device=$BRIDGE_DEVICE (frozen EnCodec decoder backward is CPU-default for MPS safety)"

"$PY" scripts/audit_open_vocab_v3_audio.py --config "$CFG"
"$PY" scripts/denoise_open_vocab_v3.py --config "$CFG" --device "$EVAL_DEVICE"
"$PY" scripts/prepare_open_vocab_v3.py --config "$CFG" --device "$EVAL_DEVICE" --force
"$PY" scripts/prepare_open_vocab_v3.py --config "$CFG" --with-speaker --device "$EVAL_DEVICE"
"$PY" scripts/evaluate_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase a0 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/build_open_vocab_v3_encodec_bridge_cache.py --config "$CFG" --device "$EVAL_DEVICE" --force
"$PY" scripts/evaluate_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase e0 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/train_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase bridge --device "$BRIDGE_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase e1 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase e2 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase b0 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/train_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase audio_c --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase c1 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase c2 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/train_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase m0 --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase m0 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/train_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase m1 --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase m1 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/export_open_vocab_v3_encodec_bridge_pairs.py --config "$CFG" --device "$EXPORT_DEVICE" --resume --explore
"$PY" scripts/finalize_open_vocab_v3_encodec_bridge_run.py --config "$CFG" --explore
echo "[v3 bridge explore] complete"
echo "[v3 bridge explore] fit-only preview=$OUTPUT_ROOT/pairs/training_and_micro_preview"
