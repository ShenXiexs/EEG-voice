#!/usr/bin/env bash
# Resume a fit-only exploratory bridge-v2 run after a non-training failure.
# It never reruns audit, prepare, EnCodec caching, bridge training, or Audio-C
# training.  Use the exact RUN_ID printed by the interrupted runner.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${1:-$APP_DIR/configs/open_vocab_v3_mfcc_encodec_bridge_v2.yaml}"
PY="${PYTHON_BIN:-$APP_DIR/.venv_0730/bin/python}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
BUDGET_HOURS="${BUDGET_HOURS:-30}"
RUN_ID="${RUN_ID:?Set RUN_ID to the interrupted run ID, e.g. 20260804_121213}"
RESUME_FROM="${RESUME_FROM:-c1}"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[v3 bridge resume] RUN_ID may contain only letters, digits, dot, underscore, and dash." >&2
  exit 2
fi
case "$RESUME_FROM" in c1|c2|m0|m1|export) ;; *)
  echo "[v3 bridge resume] RESUME_FROM must be c1, c2, m0, m1, or export." >&2
  exit 2
esac

export OPEN_VOCAB_V3_EXPLORATION=1
export OPEN_VOCAB_V3_ARTIFACT_ROOT_NAME="open_vocab_v3_mfcc_encodec_bridge_v2_run_${RUN_ID}"
OUTPUT_ROOT="$APP_DIR/../artifacts/${OPEN_VOCAB_V3_ARTIFACT_ROOT_NAME}_explore"
for required in \
  "$OUTPUT_ROOT/cache/prepared_encodec_bridge_v2.npz" \
  "$OUTPUT_ROOT/cache/frozen_encodec_bridge_targets_v2.npz" \
  "$OUTPUT_ROOT/encodec_latent_bridge_v2/checkpoints/best.pt" \
  "$OUTPUT_ROOT/audio_c_teacher_v2/checkpoints/best.pt"; do
  if [[ ! -f "$required" ]]; then
    echo "[v3 bridge resume] prerequisite is missing: $required" >&2
    echo "[v3 bridge resume] use the full explore runner for a new run instead." >&2
    exit 2
  fi
done

LOG_ROOT="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT/.matplotlib" "$OUTPUT_ROOT/.cache"
RUN_LOG="$LOG_ROOT/resume_${RESUME_FROM}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1
cd "$APP_DIR"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MPLCONFIGDIR="$OUTPUT_ROOT/.matplotlib" XDG_CACHE_HOME="$OUTPUT_ROOT/.cache" PYTHONPYCACHEPREFIX="$OUTPUT_ROOT/.pycache"
BUDGET_SECONDS="$($PY -c 'import sys; print(int(float(sys.argv[1])*3600))' "$BUDGET_HOURS")"
DEADLINE_EPOCH="$(( $(date +%s) + BUDGET_SECONDS ))"
echo "[v3 bridge resume] exploratory only; no held-out data are accessed."
echo "[v3 bridge resume] artifact_root=$OUTPUT_ROOT run_id=$RUN_ID from=$RESUME_FROM budget_hours=$BUDGET_HOURS"

if [[ "$RESUME_FROM" == "c1" ]]; then
  "$PY" scripts/evaluate_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase c1 --device "$EVAL_DEVICE" --no-fail --explore
fi
if [[ "$RESUME_FROM" == "c1" || "$RESUME_FROM" == "c2" ]]; then
  "$PY" scripts/evaluate_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase c2 --device "$EVAL_DEVICE" --no-fail --explore
fi
if [[ "$RESUME_FROM" == "c1" || "$RESUME_FROM" == "c2" || "$RESUME_FROM" == "m0" ]]; then
  "$PY" scripts/train_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase m0 --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
  "$PY" scripts/evaluate_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase m0 --device "$EVAL_DEVICE" --no-fail --explore
fi
if [[ "$RESUME_FROM" != "export" ]]; then
  "$PY" scripts/train_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase m1 --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
  "$PY" scripts/evaluate_open_vocab_v3_encodec_bridge.py --config "$CFG" --phase m1 --device "$EVAL_DEVICE" --no-fail --explore
fi
"$PY" scripts/export_open_vocab_v3_encodec_bridge_pairs.py --config "$CFG" --device "$EXPORT_DEVICE" --resume --explore
"$PY" scripts/finalize_open_vocab_v3_encodec_bridge_run.py --config "$CFG" --explore
echo "[v3 bridge resume] complete: $OUTPUT_ROOT/pairs/training_and_micro_preview"
