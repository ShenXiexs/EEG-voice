#!/usr/bin/env bash
# Continue only the held-out/report/export portion after a bound human review.
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${1:-$APP_DIR/configs/open_vocab_v3_mfcc_training_first.yaml}"
PY="${PYTHON_BIN:-$APP_DIR/.venv_0730/bin/python}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
cd "$APP_DIR"

"$PY" scripts/approve_open_vocab_v3_training_preview.py --config "$CFG" --check
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase validation --device "$EVAL_DEVICE"
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase locked --device "$EVAL_DEVICE"
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase locked_unseen --device "$EVAL_DEVICE"
"$PY" scripts/export_open_vocab_v3_encodec_clip_pairs.py --config "$CFG" --stage final --device "$EXPORT_DEVICE" --resume
echo "[v3] held-out evaluation and final training-pair export complete"
