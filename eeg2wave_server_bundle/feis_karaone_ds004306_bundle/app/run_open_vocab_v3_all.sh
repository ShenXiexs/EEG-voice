#!/usr/bin/env bash
# Fail-closed v3 entry point.  It deliberately does not delete or alter any
# v0724/v0730 artifacts; all v3 results live below its own output root.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${1:-$APP_DIR/configs/open_vocab_v3_mfcc_training_first.yaml}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$APP_DIR/.venv_0730/bin/python" ]]; then
  PY="$APP_DIR/.venv_0730/bin/python"
elif [[ -x /opt/anaconda3/bin/python ]]; then
  PY=/opt/anaconda3/bin/python
else
  PY=python3
fi

OUTPUT_ROOT="$APP_DIR/../artifacts/open_vocab_v3_mfcc_training_first"
LOG_ROOT="$OUTPUT_ROOT/logs"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
BUDGET_HOURS="${BUDGET_HOURS:-9.5}"
FRESH="${FRESH:-1}"
mkdir -p "$LOG_ROOT"
RUN_LOG="$LOG_ROOT/run_all_v3_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "$APP_DIR"

echo "[v3] python=$PY"
echo "[v3] config=$CFG"
echo "[v3] log=$RUN_LOG"
echo "[v3] train_device=$TRAIN_DEVICE evaluation_device=$EVAL_DEVICE export_device=$EXPORT_DEVICE"
echo "[v3] total_training_budget_hours=$BUDGET_HOURS"

if ! "$PY" -c 'import speechbrain, librosa; import df' >/dev/null 2>&1; then
  echo "[v3] missing v3 audio/CVAE/denoise dependencies. Run: ./bootstrap_open_vocab_v3.sh"
  exit 1
fi
"$PY" -c 'import importlib.metadata, librosa, numpy, scipy, speechbrain, torch, transformers; print({"speechbrain": speechbrain.__version__, "deepfilternet": importlib.metadata.version("deepfilternet"), "librosa": librosa.__version__, "torch": torch.__version__, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available(), "transformers": transformers.__version__})'

FRESH_FLAG=()
PREPARE_FLAG=()
if [[ "$FRESH" == "1" ]]; then
  FRESH_FLAG=(--fresh)
  PREPARE_FLAG=(--force)
fi

# 0. Raw audit -> explicit selective denoising -> feature preparation. Source
# WAVs remain immutable; rejected enhancement never enters the prepared cache.
"$PY" scripts/audit_open_vocab_v3_audio.py --config "$CFG"
"$PY" scripts/denoise_open_vocab_v3.py --config "$CFG" --device "$EVAL_DEVICE"
"$PY" scripts/prepare_open_vocab_v3.py --config "$CFG" --with-speaker --device "$EVAL_DEVICE" "${PREPARE_FLAG[@]}"
"$PY" scripts/download_open_vocab_v3_models.py --config "$CFG"
"$PY" scripts/evaluate_open_vocab_v3.py --config "$CFG" --phase v0 --device "$EVAL_DEVICE"

# One absolute deadline begins after audit/model preparation/V0. Audio-oracle,
# 50-pair, and full-fit optimization therefore share the full 9.5h budget.
BUDGET_SECONDS="$($PY -c "print(int(float('$BUDGET_HOURS') * 3600))")"
DEADLINE_EPOCH="$(( $(date +%s) + BUDGET_SECONDS ))"

# 1. Audio-only content/timbre oracles.  No EEG model has been trained yet.
"$PY" scripts/train_open_vocab_v3.py --config "$CFG" --phase audio --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" "${FRESH_FLAG[@]}"
"$PY" scripts/evaluate_open_vocab_v3.py --config "$CFG" --phase v1 --device "$EVAL_DEVICE"
"$PY" scripts/evaluate_open_vocab_v3.py --config "$CFG" --phase v2 --device "$EVAL_DEVICE"

# 2. Direct 50-pair EEG->MFCC sanity check.  A failed report exits before fit.
"$PY" scripts/train_open_vocab_v3.py --config "$CFG" --phase micro --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" "${FRESH_FLAG[@]}"
"$PY" scripts/evaluate_open_vocab_v3.py --config "$CFG" --phase micro --device "$EVAL_DEVICE"
"$PY" scripts/export_open_vocab_v3_training_preview.py --config "$CFG" --stage micro --device "$EXPORT_DEVICE" --resume

# 3. Only after the overfit gate passes: full fit, then its training gate.
"$PY" scripts/train_open_vocab_v3.py --config "$CFG" --phase fit --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" "${FRESH_FLAG[@]}"
"$PY" scripts/evaluate_open_vocab_v3.py --config "$CFG" --phase fit --device "$EVAL_DEVICE"
"$PY" scripts/export_open_vocab_v3_training_preview.py --config "$CFG" --stage fit --device "$EXPORT_DEVICE" --resume

# 4. Hard human listening gate. Held-out roles are not loaded or evaluated
# until the exact full-fit preview has been approved.
if ! "$PY" scripts/approve_open_vocab_v3_training_preview.py --config "$CFG" --check; then
  echo "[v3] stopped before validation/test: listen to $OUTPUT_ROOT/pairs/full_fit_preview"
  echo "[v3] approve with: ./approve_open_vocab_v3_training_preview.sh samxie \"listening note\""
  echo "[v3] then continue with: ./run_open_vocab_v3_after_review.sh"
  exit 3
fi

# 5. These are reports, not extra optimization.  They are unreachable unless
# all preceding gates pass. CPU avoids the known PyTorch/MPS counterfactual
# Transformer buffer assertion observed in v0730 evaluation/export.
"$PY" scripts/evaluate_open_vocab_v3.py --config "$CFG" --phase validation --device "$EVAL_DEVICE"
"$PY" scripts/evaluate_open_vocab_v3.py --config "$CFG" --phase locked --device "$EVAL_DEVICE"
"$PY" scripts/evaluate_open_vocab_v3.py --config "$CFG" --phase locked_unseen --device "$EVAL_DEVICE"
"$PY" scripts/export_open_vocab_v3_pairs.py --config "$CFG" --device "$EXPORT_DEVICE" --resume

echo "[v3] complete"
echo "[v3] audit=$OUTPUT_ROOT/audit/audio_audit.json"
echo "[v3] gates=$OUTPUT_ROOT/gates"
echo "[v3] validation=$OUTPUT_ROOT/evaluation/subject_holdout_seen.json"
echo "[v3] locked=$OUTPUT_ROOT/evaluation/locked_seen_label.json"
echo "[v3] locked_unseen_exploratory=$OUTPUT_ROOT/evaluation/locked_unseen_pot_exploratory.json"
echo "[v3] pairs=$OUTPUT_ROOT/pairs/training_fit_eligible/manifest.csv"
