#!/usr/bin/env bash
# Exploratory v3 runner.  It records every gate but deliberately continues
# after threshold failures.  It is NOT a replacement for run_open_vocab_v3_all.sh
# and may not be used for a held-out/generalization claim.
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

OUTPUT_ROOT="$APP_DIR/../artifacts/open_vocab_v3_mfcc_training_first_explore"
LOG_ROOT="$OUTPUT_ROOT/logs"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
BUDGET_HOURS="${BUDGET_HOURS:-20}"
FRESH="${FRESH:-1}"

mkdir -p "$LOG_ROOT"
RUN_LOG="$LOG_ROOT/run_explore_v3_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OPEN_VOCAB_V3_EXPLORATION=1
cd "$APP_DIR"

echo "[v3 explore] WARNING: gate failures and the human listening gate are bypassed."
echo "[v3 explore] all output is exploratory only; do not report held-out results as main evidence."
echo "[v3 explore] python=$PY"
echo "[v3 explore] config=$CFG"
echo "[v3 explore] artifact_root=$OUTPUT_ROOT"
echo "[v3 explore] log=$RUN_LOG"
echo "[v3 explore] train_device=$TRAIN_DEVICE evaluation_device=$EVAL_DEVICE export_device=$EXPORT_DEVICE"
echo "[v3 explore] total_training_budget_hours=$BUDGET_HOURS"

if ! "$PY" -c 'import speechbrain, librosa' >/dev/null 2>&1; then
  echo "[v3 explore] missing v3 dependencies. Run: ./bootstrap_open_vocab_v3.sh"
  exit 1
fi
"$PY" -c 'import librosa, numpy, scipy, speechbrain, torch, transformers; print({"speechbrain": speechbrain.__version__, "librosa": librosa.__version__, "torch": torch.__version__, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available(), "transformers": transformers.__version__})'

PREPARE_FLAG=()
if [[ "$FRESH" == "1" ]]; then
  PREPARE_FLAG=(--force)
fi
BUDGET_SECONDS="$("$PY" -c 'import sys; print(int(float(sys.argv[1]) * 3600))' "$BUDGET_HOURS")"
DEADLINE_EPOCH="$(( $(date +%s) + BUDGET_SECONDS ))"

# A0: audit and generator-path adaptation.  --explore only bypasses a failed
# numerical A0 threshold; real execution/model errors still stop the run.
"$PY" scripts/audit_open_vocab_v3_audio.py --config "$CFG"
"$PY" scripts/denoise_open_vocab_v3.py --config "$CFG" --device "$EVAL_DEVICE"
"$PY" scripts/prepare_open_vocab_v3.py --config "$CFG" --device "$EVAL_DEVICE" "${PREPARE_FLAG[@]}"
"$PY" scripts/download_open_vocab_v3_models.py --config "$CFG"
"$PY" scripts/finetune_open_vocab_v3_encodec_audio_models.py --config "$CFG" --scope fit --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --explore
"$PY" scripts/prepare_open_vocab_v3.py --config "$CFG" --with-speaker --device "$EVAL_DEVICE" --force

# T0--T3: audio-token, content, variational, and voice-swap gates.
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t0 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t0b --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/build_open_vocab_v3_encodec_cache.py --config "$CFG" --device "$EVAL_DEVICE" --force
"$PY" scripts/train_open_vocab_v3_encodec_clip.py --config "$CFG" --phase audio_content --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t1 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t1d --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/train_open_vocab_v3_encodec_clip.py --config "$CFG" --phase cvae --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t2 --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t2v --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase t3 --device "$EVAL_DEVICE" --no-fail --explore

# C/D: train-pair micro and full-fit EEG experiments, then export their WAVs
# even if they fail their stated engineering gates.
"$PY" scripts/train_open_vocab_v3_encodec_clip.py --config "$CFG" --phase micro --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase micro --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/export_open_vocab_v3_encodec_clip_pairs.py --config "$CFG" --stage micro --device "$EXPORT_DEVICE" --resume --explore
"$PY" scripts/train_open_vocab_v3_encodec_clip.py --config "$CFG" --phase fit --device "$TRAIN_DEVICE" --deadline-epoch "$DEADLINE_EPOCH" --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase fit --device "$EVAL_DEVICE" --no-fail --explore
"$PY" scripts/export_open_vocab_v3_encodec_clip_pairs.py --config "$CFG" --stage fit --device "$EXPORT_DEVICE" --resume --explore

# Exploratory held-out access: bypasses E by the user's explicit request and
# writes the bypass marker into reports, metadata, and the final manifest.
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase validation --device "$EVAL_DEVICE" --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase locked --device "$EVAL_DEVICE" --explore
"$PY" scripts/evaluate_open_vocab_v3_encodec_clip.py --config "$CFG" --phase locked_unseen --device "$EVAL_DEVICE" --explore
"$PY" scripts/export_open_vocab_v3_encodec_clip_pairs.py --config "$CFG" --stage final --device "$EXPORT_DEVICE" --resume --explore

echo "[v3 explore] complete"
echo "[v3 explore] gates=$OUTPUT_ROOT/gates"
echo "[v3 explore] fit_preview=$OUTPUT_ROOT/pairs/full_fit_preview"
echo "[v3 explore] final_pairs=$OUTPUT_ROOT/pairs/encodec_clip_mfcc_training_fit_v1/manifest.csv"
echo "[v3 explore] heldout reports are exploratory_gate_bypass=true."
