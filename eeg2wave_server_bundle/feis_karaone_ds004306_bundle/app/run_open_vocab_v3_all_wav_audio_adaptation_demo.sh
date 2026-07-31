#!/usr/bin/env bash
# Optional transductive audio-only adaptation. These checkpoints have seen all
# eligible KaraOne WAVs and are deliberately isolated from held-out EEG reports.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${1:-$APP_DIR/configs/open_vocab_v3_mfcc_training_first.yaml}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$APP_DIR/.venv_0730/bin/python" ]]; then
  PY="$APP_DIR/.venv_0730/bin/python"
else
  PY=python3
fi

cd "$APP_DIR"
CACHE="$APP_DIR/../artifacts/open_vocab_v3_mfcc_training_first/cache/prepared_encodec_clip_mfcc_v1.npz"
if [[ ! -f "$CACHE" ]]; then
  "$PY" scripts/audit_open_vocab_v3_audio.py --config "$CFG"
  "$PY" scripts/denoise_open_vocab_v3.py --config "$CFG" --device cpu
  "$PY" scripts/prepare_open_vocab_v3.py --config "$CFG" --device cpu --force
fi
"$PY" scripts/download_open_vocab_v3_models.py --config "$CFG"
"$PY" scripts/finetune_open_vocab_v3_encodec_audio_models.py \
  --config "$CFG" --scope all --device "${TRAIN_DEVICE:-auto}"

OUTPUT="$APP_DIR/../artifacts/open_vocab_v3_mfcc_training_first/audio_adaptation/transductive_all_encodec_clip_v1"
echo "[v3 all-WAV audio demo] complete"
echo "[v3 all-WAV audio demo] artifacts=$OUTPUT"
echo "[v3 all-WAV audio demo] these weights are audio-demo-only and are not used by validation/test"
