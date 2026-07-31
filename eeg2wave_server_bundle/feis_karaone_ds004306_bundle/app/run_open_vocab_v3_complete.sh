#!/usr/bin/env bash
# Single interactive entry point from dependency bootstrap through final pair
# WAV/PNG export.  Held-out evaluation remains behind an explicit listening
# confirmation in this same shell invocation.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${1:-$APP_DIR/configs/open_vocab_v3_mfcc_training_first.yaml}"
REVIEWER="${REVIEWER:-samxie}"
OUTPUT_ROOT="$APP_DIR/../artifacts/open_vocab_v3_mfcc_training_first"
PREVIEW_ROOT="$OUTPUT_ROOT/pairs/full_fit_preview"
FINAL_PAIR_ROOT="$OUTPUT_ROOT/pairs/encodec_clip_mfcc_training_fit_v1"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$APP_DIR/.venv_0730/bin/python" ]]; then
  PY="$APP_DIR/.venv_0730/bin/python"
else
  PY=python3
fi

cd "$APP_DIR"

if ! "$PY" -c 'import speechbrain, librosa' >/dev/null 2>&1; then
  echo "[v3 complete] installing missing v3 dependencies"
  ./bootstrap_open_vocab_v3.sh
fi

set +e
./run_open_vocab_v3_all.sh "$CFG"
FIRST_STAGE_STATUS=$?
set -e

if [[ "$FIRST_STAGE_STATUS" -eq 3 ]]; then
  echo
  echo "[v3 complete] training stages passed and previews are ready:"
  echo "[v3 complete] $PREVIEW_ROOT"
  echo "[v3 complete] listen to cleaned reference, analytic/CVAE oracles, EEG prior,"
  echo "[v3 complete] and zero/time/channel controls before continuing."
  echo
  if [[ ! -t 0 ]]; then
    echo "[v3 complete] interactive input is unavailable; refusing to touch held-out data."
    echo "[v3 complete] rerun this script in a terminal, or use the documented approve/continue commands."
    exit 3
  fi
  read -r -p "Type YES to approve these exact training WAVs and continue to validation/test: " DECISION
  if [[ "$DECISION" != "YES" ]]; then
    echo "[v3 complete] not approved; held-out evaluation was not run."
    exit 3
  fi
  ./approve_open_vocab_v3_training_preview.sh "$REVIEWER" "approved interactively inside run_open_vocab_v3_complete.sh"
  ./run_open_vocab_v3_after_review.sh "$CFG"
elif [[ "$FIRST_STAGE_STATUS" -ne 0 ]]; then
  echo "[v3 complete] pipeline stopped with status $FIRST_STAGE_STATUS; no automatic retry."
  exit "$FIRST_STAGE_STATUS"
fi

if [[ ! -f "$FINAL_PAIR_ROOT/manifest.csv" || ! -f "$FINAL_PAIR_ROOT/export_manifest.json" ]]; then
  echo "[v3 complete] final pair export is incomplete: $FINAL_PAIR_ROOT"
  exit 1
fi

echo
echo "[v3 complete] complete"
echo "[v3 complete] final_pair_folder=$FINAL_PAIR_ROOT"
echo "[v3 complete] wav_png_manifest=$FINAL_PAIR_ROOT/manifest.csv"
echo "[v3 complete] export_audit=$FINAL_PAIR_ROOT/export_manifest.json"
