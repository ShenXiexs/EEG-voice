#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${1:-$APP_DIR/configs/open_vocab_0730_explicit_cp_v1.yaml}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$APP_DIR/.venv_0730/bin/python" ]]; then
  PY="$APP_DIR/.venv_0730/bin/python"
elif [[ -x /opt/anaconda3/bin/python ]]; then
  PY=/opt/anaconda3/bin/python
else
  PY=python3
fi

OUTPUT_ROOT="$APP_DIR/../artifacts/open_vocab_0730_explicit_cp_v1"
LOG_ROOT="$OUTPUT_ROOT/logs"
RENDERER_CHECKPOINT="$OUTPUT_ROOT/renderer/checkpoints/best.pt"
EEG_CHECKPOINT="$OUTPUT_ROOT/eeg_cp/checkpoints/best.pt"
RUN_MANIFEST="$OUTPUT_ROOT/run_manifest.json"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
mkdir -p "$LOG_ROOT"
RUN_LOG="$LOG_ROOT/run_all_0730_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "$APP_DIR"

echo "[0730] python=$PY"
echo "[0730] config=$CFG"
echo "[0730] log=$RUN_LOG"
echo "[0730] evaluation_device=$EVAL_DEVICE export_device=$EXPORT_DEVICE"

"$PY" -c 'import numpy, scipy, sklearn, torch, transformers; print({"torch": torch.__version__, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()})'

"$PY" scripts/prepare_open_vocab_0730.py --config "$CFG"
"$PY" scripts/download_speecht5_hifigan.py --config "$CFG"
if [[ "$FORCE_RETRAIN" == "1" || ! -s "$RENDERER_CHECKPOINT" || ! -s "$EEG_CHECKPOINT" || ! -s "$RUN_MANIFEST" ]]; then
  echo "[0730] training v0730 from scratch"
  "$PY" scripts/train_open_vocab_0730.py --config "$CFG" --phase all --wall-hours 9.5 --fresh
else
  echo "[0730] reusing completed v0730 training (set FORCE_RETRAIN=1 to retrain)"
  echo "[0730] renderer_checkpoint=$RENDERER_CHECKPOINT"
  echo "[0730] eeg_checkpoint=$EEG_CHECKPOINT"
fi

# PyTorch 2.8 MPS can abort in repeated counterfactual Transformer forwards with
# an MPSNDArray buffer-size assertion.  Evaluation and pair export are therefore
# CPU-safe by default; training can still use MPS.  Explicit env overrides remain
# available for a future PyTorch build: EVAL_DEVICE=mps EXPORT_DEVICE=mps.
"$PY" scripts/evaluate_open_vocab_0730.py --config "$CFG" --device "$EVAL_DEVICE"
"$PY" scripts/export_open_vocab_0730_pairs.py --config "$CFG" --device "$EXPORT_DEVICE" --resume
"$PY" scripts/verify_open_vocab_0730_pairs.py --config "$CFG"

echo "[0730] complete"
echo "[0730] report=$OUTPUT_ROOT/evaluation/evaluation.json"
echo "[0730] pairs=$OUTPUT_ROOT/pairs/all_1341/manifest.csv"
echo "[0730] pair_audit=$OUTPUT_ROOT/pairs/all_1341/pairs_audit.json"
