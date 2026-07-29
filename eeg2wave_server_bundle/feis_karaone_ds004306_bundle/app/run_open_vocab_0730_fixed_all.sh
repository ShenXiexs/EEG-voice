#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${1:-$APP_DIR/configs/open_vocab_0730_explicit_cp_fixed_v2.yaml}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$APP_DIR/.venv_0730/bin/python" ]]; then
  PY="$APP_DIR/.venv_0730/bin/python"
elif [[ -x /opt/anaconda3/bin/python ]]; then
  PY=/opt/anaconda3/bin/python
else
  PY=python3
fi

OUTPUT_ROOT="$APP_DIR/../artifacts/open_vocab_0730_explicit_cp_fixed_v2"
LOG_ROOT="$OUTPUT_ROOT/logs"
RENDERER_CHECKPOINT="$OUTPUT_ROOT/renderer/checkpoints/best.pt"
EEG_CHECKPOINT="$OUTPUT_ROOT/eeg_cp/checkpoints/best.pt"
RUN_MANIFEST="$OUTPUT_ROOT/run_manifest.json"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
mkdir -p "$LOG_ROOT"
RUN_LOG="$LOG_ROOT/run_all_0730_fixed_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/open_vocab_0730_fixed_pycache}"

cd "$APP_DIR"
echo "[0730-fixed] python=$PY"
echo "[0730-fixed] config=$CFG"
echo "[0730-fixed] log=$RUN_LOG"
echo "[0730-fixed] evaluation_device=$EVAL_DEVICE export_device=$EXPORT_DEVICE"

"$PY" -c 'import numpy, scipy, sklearn, torch, transformers, tqdm; print({"torch": torch.__version__, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()})'
"$PY" scripts/prepare_open_vocab_0730_fixed.py --config "$CFG"
"$PY" scripts/download_speecht5_hifigan.py --config "$CFG"

if [[ "$FORCE_RETRAIN" == "1" || ! -s "$RENDERER_CHECKPOINT" || ! -s "$EEG_CHECKPOINT" || ! -s "$RUN_MANIFEST" ]]; then
  echo "[0730-fixed] training fixed v2 from scratch"
  "$PY" scripts/train_open_vocab_0730_fixed.py --config "$CFG" --phase all --wall-hours 9.5 --fresh
else
  echo "[0730-fixed] reusing completed fixed-v2 training (set FORCE_RETRAIN=1 to retrain)"
fi

"$PY" scripts/evaluate_open_vocab_0730_fixed.py --config "$CFG" --device "$EVAL_DEVICE"
"$PY" scripts/export_open_vocab_0730_fixed_pairs.py --config "$CFG" --device "$EXPORT_DEVICE" --resume
"$PY" scripts/verify_open_vocab_0730_fixed_pairs.py --config "$CFG"

echo "[0730-fixed] complete"
echo "[0730-fixed] report=$OUTPUT_ROOT/evaluation/evaluation.json"
echo "[0730-fixed] generated_gate=$OUTPUT_ROOT/evaluation/generated_gate.json"
echo "[0730-fixed] pairs=$OUTPUT_ROOT/pairs/all_1341/manifest.csv"
echo "[0730-fixed] pair_audit=$OUTPUT_ROOT/pairs/all_1341/pairs_audit.json"
