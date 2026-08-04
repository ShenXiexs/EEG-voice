#!/usr/bin/env bash
# Exploratory repair-v3 runner.  It continues fit-only diagnostics after gates
# fail, but never reads validation/locked-test roles.
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${1:-$APP_DIR/configs/open_vocab_v3_mfcc_encodec_rvq_repair_v3.yaml}"
PY="${PYTHON_BIN:-$APP_DIR/.venv_0730/bin/python}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"; EVAL_DEVICE="${EVAL_DEVICE:-cpu}"; BUDGET_HOURS="${BUDGET_HOURS:-30}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsafe RUN_ID" >&2; exit 2; }
export OPEN_VOCAB_V3_EXPLORATION=1 OPEN_VOCAB_V3_ARTIFACT_ROOT_NAME="open_vocab_v3_mfcc_encodec_rvq_repair_v3_run_${RUN_ID}"
OUT="$APP_DIR/../artifacts/${OPEN_VOCAB_V3_ARTIFACT_ROOT_NAME}_explore"
if [[ -e "$OUT" ]]; then echo "[v3 RVQ explore] refusing to overwrite existing run: $OUT" >&2; exit 2; fi
mkdir -p "$OUT/logs" "$OUT/.matplotlib" "$OUT/.cache"
exec > >(tee -a "$OUT/logs/run_explore_$(date +%Y%m%d_%H%M%S).log") 2>&1
cd "$APP_DIR"; export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MPLCONFIGDIR="$OUT/.matplotlib" XDG_CACHE_HOME="$OUT/.cache" PYTHONPYCACHEPREFIX="$OUT/.pycache"
DEADLINE_EPOCH="$($PY -c 'import sys;print(int(float(sys.argv[1])*3600))' "$BUDGET_HOURS")"; DEADLINE_EPOCH="$(( $(date +%s)+DEADLINE_EPOCH ))"
echo "[v3 RVQ explore] warning: fit-only exploratory diagnostics; held-out access is prohibited."; echo "[v3 RVQ explore] root=$OUT hours=$BUDGET_HOURS"
"$PY" scripts/audit_open_vocab_v3_audio.py --config "$CFG" --fit-only
"$PY" scripts/denoise_open_vocab_v3.py --config "$CFG" --device "$EVAL_DEVICE"
"$PY" scripts/prepare_open_vocab_v3.py --config "$CFG" --device "$EVAL_DEVICE" --fit-only --force
"$PY" scripts/prepare_open_vocab_v3.py --config "$CFG" --with-speaker --device "$EVAL_DEVICE" --fit-only
"$PY" scripts/evaluate_open_vocab_v3_encodec_rvq_repair.py --config "$CFG" --phase a0 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/build_open_vocab_v3_encodec_rvq_repair_cache.py --config "$CFG" --device "$EVAL_DEVICE" --force
"$PY" scripts/evaluate_open_vocab_v3_encodec_rvq_repair.py --config "$CFG" --phase r0 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/train_open_vocab_v3_encodec_rvq_repair.py --config "$CFG" --phase rvq_micro --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_encodec_rvq_repair.py --config "$CFG" --phase e1a --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/train_open_vocab_v3_encodec_rvq_repair.py --config "$CFG" --phase rvq --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_encodec_rvq_repair.py --config "$CFG" --phase e1b --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_rvq_repair.py --config "$CFG" --phase b0 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/train_open_vocab_v3_encodec_rvq_repair.py --config "$CFG" --phase audio_c --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_encodec_rvq_repair.py --config "$CFG" --phase c1 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_rvq_repair.py --config "$CFG" --phase c2 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/train_open_vocab_v3_encodec_rvq_repair.py --config "$CFG" --phase m0a --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_encodec_rvq_repair.py --config "$CFG" --phase m0a --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/train_open_vocab_v3_encodec_rvq_repair.py --config "$CFG" --phase m0b --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_encodec_rvq_repair.py --config "$CFG" --phase m0b --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/train_open_vocab_v3_encodec_rvq_repair.py --config "$CFG" --phase m1 --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --fresh
"$PY" scripts/evaluate_open_vocab_v3_encodec_rvq_repair.py --config "$CFG" --phase m1 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/export_open_vocab_v3_encodec_rvq_repair_preview.py --config "$CFG" --device "$EVAL_DEVICE" --explore
"$PY" scripts/finalize_open_vocab_v3_encodec_rvq_repair_run.py --config "$CFG" --explore
echo "[v3 RVQ explore] complete: $OUT/pairs/training_and_micro_preview"
