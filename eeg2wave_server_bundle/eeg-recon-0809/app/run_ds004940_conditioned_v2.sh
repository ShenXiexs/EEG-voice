#!/usr/bin/env bash
# Resumable, exploratory v2 repair run.  No command here modifies v3 outputs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export DATA_CONFIG="${DATA_CONFIG:-$PROJECT_ROOT/configs/training_data_v4_ds004940_fixed.yaml}"
export PILOT_CONFIG="${PILOT_CONFIG:-$PROJECT_ROOT/configs/ds004940_conditioned_v2.yaml}"
source "$SCRIPT_DIR/lib/joint_pilot_common.sh"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-ds004940_conditioned_v2}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$RUN_ROOT/$EXPERIMENT_NAME}"
V2_FROM="${V2_FROM:-all}" # all | a0 | m0 | m1 | resume (M0 through final export)
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
M0_STEPS="${M0_STEPS:-5000}"
M0_BATCH_SIZE="${M0_BATCH_SIZE:-2}"
M1_EPOCHS="${M1_EPOCHS:-50}"
M1_BATCH_SIZE="${M1_BATCH_SIZE:-4}"
M1_CONTENTS_PER_BATCH="${M1_CONTENTS_PER_BATCH:-2}"
M1_SUBJECTS_PER_CONTENT="${M1_SUBJECTS_PER_CONTENT:-2}"
THERMAL_MODE="${THERMAL_MODE:-1}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-90}"
NATIVE_RENDERER_STEPS="${NATIVE_RENDERER_STEPS:-2000}"
DIFFUSION_DENOISE="${DIFFUSION_DENOISE:-0}"
DIFFUSION_TRAIN_STEPS="${DIFFUSION_TRAIN_STEPS:-2000}"
V2_BYPASS_M0="${V2_BYPASS_M0:-0}"
A0_MAX_PAIRS="${A0_MAX_PAIRS:-402}"
case "$V2_FROM" in all|a0|m0|m1|resume) ;; *) echo "V2_FROM must be all, a0, m0, m1, or resume" >&2; exit 2 ;; esac
case "$DIFFUSION_DENOISE" in 0|1) ;; *) echo "DIFFUSION_DENOISE must be 0 or 1" >&2; exit 2 ;; esac
case "$THERMAL_MODE" in 0|1) ;; *) echo "THERMAL_MODE must be 0 or 1" >&2; exit 2 ;; esac

# Conservative defaults for a laptop: cap CPU helper threads, encourage MPS
# garbage collection, and leave headroom for macOS.  They do not alter EEG,
# audio targets, optimization semantics, or reported metrics.
if [[ "$THERMAL_MODE" == "1" ]]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
  export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.75}"
  export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.40}"
fi

start_joint_log "$EXPERIMENT_NAME"
require_joint_runtime
require_local_hubert
cd "$PROJECT_ROOT"

echo "WARNING: v2 is exploratory until pairing/listening gates are completed."
echo "experiment_root=$EXPERIMENT_ROOT"
echo "resume_from=$V2_FROM"
echo "thermal_mode=$THERMAL_MODE m0_batch=$M0_BATCH_SIZE m1_batch=$M1_BATCH_SIZE (${M1_CONTENTS_PER_BATCH} contents x ${M1_SUBJECTS_PER_CONTENT} subjects)"

if [[ "$V2_FROM" == "all" || "$V2_FROM" == "a0" ]]; then
  joint_run "$PYTHON_BIN" scripts/prepare_training_data.py --config "$DATA_CONFIG" audit
  joint_run "$PYTHON_BIN" scripts/prepare_training_data.py --config "$DATA_CONFIG" make-splits
  # A0 uses every unique DS004940 source waveform; it does not need EEG shards.
  joint_run "$PYTHON_BIN" scripts/cache_speech_targets.py --config "$DATA_CONFIG" --dataset ds004940 \
    --manifest all --target-name speech_targets_ds004940_a0 --include-hubert --hubert-local-path "$HUBERT_LOCAL_PATH"
  joint_run "$PYTHON_BIN" app/audit_speecht5_oracle.py --config "$PILOT_CONFIG" --data-config "$DATA_CONFIG" \
    --hubert-local-path "$HUBERT_LOCAL_PATH" \
    --manifest artifacts/training_data/v4_ds004940_fixed/manifests/manifest_all.csv \
    --targets artifacts/training_data/v4_ds004940_fixed/speech_targets/speech_targets_ds004940_a0.h5 \
    --output "$EXPERIMENT_ROOT/audio_a0" --max-pairs "$A0_MAX_PAIRS"
fi

if [[ "$V2_FROM" == "all" || "$V2_FROM" == "m0" || "$V2_FROM" == "resume" ]]; then
  joint_run "$PYTHON_BIN" scripts/prepare_m0_artifacts.py --data-config "$DATA_CONFIG" \
    --pilot-config "$PILOT_CONFIG" --hubert-local-path "$HUBERT_LOCAL_PATH" --artifact-set explore_m0
  joint_run "$PYTHON_BIN" app/audit_fixed_window.py \
    --manifest artifacts/training_data/v4_ds004940_fixed/manifests/manifest_explore_m0.csv \
    --output "$EXPERIMENT_ROOT/d0_fixed_window_m0.json"
  joint_run "$PYTHON_BIN" app/train_joint.py --config "$PILOT_CONFIG" --mode ds004940 --stage overfit \
    --seed 31 --explore --max-steps "$M0_STEPS" --batch-size "$M0_BATCH_SIZE" \
    --checkpoint-every "$CHECKPOINT_EVERY" --output-root "$EXPERIMENT_ROOT"
  joint_run "$PYTHON_BIN" app/evaluate_joint.py \
    --checkpoint "$EXPERIMENT_ROOT/overfit/ds004940/seed-31/checkpoint.pt" --dataset ds004940 --role train
fi

if [[ "$V2_FROM" == "all" || "$V2_FROM" == "m1" || "$V2_FROM" == "resume" ]]; then
  M0_EVALUATION="$EXPERIMENT_ROOT/overfit/ds004940/seed-31/evaluation_ds004940_train.json"
  if [[ "$V2_BYPASS_M0" != "1" ]]; then
    joint_run "$PYTHON_BIN" - "$M0_EVALUATION" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists(): raise SystemExit(f"M1 blocked: missing M0 evaluation {path}")
result = json.loads(path.read_text())
if not result.get("gate", {}).get("passed", False):
    raise SystemExit("M1 blocked: corrected M0 gate failed: " + json.dumps(result.get("gate", {}).get("checks", {})))
print("corrected_m0_gate=pass")
PY
  else
    echo "WARNING: V2_BYPASS_M0=1; M1 is engineering exploration only even if M0 failed."
  fi
  joint_run "$PYTHON_BIN" scripts/prepare_stage2_split.py --data-config "$DATA_CONFIG" --pilot-config "$PILOT_CONFIG" \
    --explore --materialize --hubert-local-path "$HUBERT_LOCAL_PATH"
  for seed in $(pilot_seeds); do
    joint_run "$PYTHON_BIN" app/train_joint.py --config "$PILOT_CONFIG" --mode ds004940 --stage generalization \
      --seed "$seed" --explore --max-epochs "$M1_EPOCHS" --batch-size "$M1_BATCH_SIZE" \
      --contents-per-batch "$M1_CONTENTS_PER_BATCH" --subjects-per-content "$M1_SUBJECTS_PER_CONTENT" \
      --checkpoint-every "$CHECKPOINT_EVERY" --output-root "$EXPERIMENT_ROOT"
    checkpoint="$EXPERIMENT_ROOT/generalization/ds004940/seed-$seed/checkpoint.pt"
    joint_run "$PYTHON_BIN" app/evaluate_joint.py --checkpoint "$checkpoint" --dataset ds004940 --role validation
    joint_run "$PYTHON_BIN" app/evaluate_joint.py --checkpoint "$checkpoint" --dataset ds004940 --role test
    if [[ "$COOLDOWN_SECONDS" -gt 0 ]]; then
      echo "cooldown_seconds=$COOLDOWN_SECONDS after seed=$seed"
      sleep "$COOLDOWN_SECONDS"
    fi
  done
  ARTIFACT="artifacts/training_data/v4_ds004940_fixed"
  joint_run "$PYTHON_BIN" app/train_native_audio_renderer.py --config "$PILOT_CONFIG" \
    --manifest "$ARTIFACT/manifests/explore_stage2_ds004940_conditioned_v2.csv" \
    --split "$ARTIFACT/splits/stage2_ds004940_conditioned_v2_fold-0.csv" \
    --targets "$ARTIFACT/speech_targets/speech_targets_explore_stage2_ds004940_conditioned_v2.h5" \
    --normalizer "$ARTIFACT/normalizers/explore_stage2_ds004940_conditioned_v2_fold-0.json" \
    --output "$EXPERIMENT_ROOT/native_audio_renderer" --max-steps "$NATIVE_RENDERER_STEPS" \
    --checkpoint-every "$CHECKPOINT_EVERY"
  if [[ "$DIFFUSION_DENOISE" == "1" ]]; then
    joint_run "$PYTHON_BIN" app/train_native_mel_diffusion.py --config "$PILOT_CONFIG" \
      --manifest "$ARTIFACT/manifests/explore_stage2_ds004940_conditioned_v2.csv" \
      --split "$ARTIFACT/splits/stage2_ds004940_conditioned_v2_fold-0.csv" \
      --targets "$ARTIFACT/speech_targets/speech_targets_explore_stage2_ds004940_conditioned_v2.h5" \
      --normalizer "$ARTIFACT/normalizers/explore_stage2_ds004940_conditioned_v2_fold-0.json" \
      --renderer "$EXPERIMENT_ROOT/native_audio_renderer/checkpoint.pt" \
      --output "$EXPERIMENT_ROOT/native_mel_diffusion" --max-steps "$DIFFUSION_TRAIN_STEPS" \
      --checkpoint-every "$CHECKPOINT_EVERY"
  fi
  for seed in $(pilot_seeds); do
    checkpoint="$EXPERIMENT_ROOT/generalization/ds004940/seed-$seed/checkpoint.pt"
    EXPORT_ARGS=(--checkpoint "$checkpoint" --renderer "$EXPERIMENT_ROOT/native_audio_renderer/checkpoint.pt"
      --role test --max-pairs 32 --output "$EXPERIMENT_ROOT/native_audio_pairs/seed-$seed/test"
      --diffusion-mode off)
    if [[ "$DIFFUSION_DENOISE" == "1" ]]; then
      EXPORT_ARGS=(--checkpoint "$checkpoint" --renderer "$EXPERIMENT_ROOT/native_audio_renderer/checkpoint.pt"
        --role test --max-pairs 32 --output "$EXPERIMENT_ROOT/native_audio_pairs/seed-$seed/test"
        --diffusion-mode on --diffusion-checkpoint "$EXPERIMENT_ROOT/native_mel_diffusion/checkpoint.pt")
    fi
    joint_run "$PYTHON_BIN" app/export_conditioned_audio_pairs.py "${EXPORT_ARGS[@]}"
  done
fi

echo "v2 exploratory pipeline complete.  Do not treat these as registered results."
