#!/usr/bin/env bash
# Shared fail-closed runtime helpers for the joint EEG-to-speech pilot.

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$APP_DIR/.." && pwd)"
DATA_CONFIG="${DATA_CONFIG:-$PROJECT_ROOT/configs/training_data_v3.yaml}"
PILOT_CONFIG="${PILOT_CONFIG:-$PROJECT_ROOT/configs/joint_pilot_v1.yaml}"
ARTIFACT_ROOT="$PROJECT_ROOT/artifacts/training_data/v3"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/joint_pilot_v1}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
elif [[ -x "/opt/anaconda3/envs/eegvoice/bin/python" ]]; then
  PYTHON_BIN="/opt/anaconda3/envs/eegvoice/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

DEFAULT_HUBERT="$PROJECT_ROOT/../feis_karaone_ds004306_bundle/artifacts/open_vocab_0722_project_hubert_v1/xdg_cache/huggingface/hub/models--facebook--hubert-base-ls960/snapshots/dba3bb02fda4248b6e082697eee756de8fe8aa8a"
HUBERT_LOCAL_PATH="${HUBERT_LOCAL_PATH:-$DEFAULT_HUBERT}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

start_joint_log() {
  local stage="$1"
  local timestamp
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$RUN_ROOT/logs"
  JOINT_LOG="$RUN_ROOT/logs/${stage}_${timestamp}.log"
  export JOINT_LOG
  {
    echo "[$(date -u +%FT%TZ)] stage=$stage"
    echo "project_root=$PROJECT_ROOT"
    echo "python=$PYTHON_BIN"
    echo "data_config=$DATA_CONFIG"
    echo "pilot_config=$PILOT_CONFIG"
    echo "log=$JOINT_LOG"
  } | tee -a "$JOINT_LOG"
}

joint_run() {
  local restore_errexit=0
  [[ $- == *e* ]] && restore_errexit=1
  set +e
  "$@" 2>&1 | tee -a "$JOINT_LOG"
  local status="${PIPESTATUS[0]}"
  if [[ "$restore_errexit" -eq 1 ]]; then set -e; else set +e; fi
  return "$status"
}

require_joint_runtime() {
  [[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; return 2; }
  [[ -f "$DATA_CONFIG" ]] || { echo "Missing data config: $DATA_CONFIG" >&2; return 2; }
  [[ -f "$PILOT_CONFIG" ]] || { echo "Missing pilot config: $PILOT_CONFIG" >&2; return 2; }
  "$PYTHON_BIN" -c 'import h5py, mne, numpy, pandas, scipy, torch, transformers, yaml'
}

require_local_hubert() {
  for required in config.json preprocessor_config.json pytorch_model.bin; do
    [[ -e "$HUBERT_LOCAL_PATH/$required" ]] || {
      echo "Incomplete local HuBERT snapshot: missing $HUBERT_LOCAL_PATH/$required" >&2
      echo "Set HUBERT_LOCAL_PATH to a complete local facebook/hubert-base-ls960 snapshot." >&2
      return 2
    }
  done
}

pilot_seeds() {
  "$PYTHON_BIN" - "$PILOT_CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1]) as handle:
    cfg = yaml.safe_load(handle)
print(" ".join(str(value) for value in cfg["training"]["seeds"]))
PY
}

require_formal_stage0() {
  "$PYTHON_BIN" - "$ARTIFACT_ROOT/qc/validate.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print(f"Stage-0 validation report is missing: {path}", file=sys.stderr)
    raise SystemExit(3)
payload = json.loads(path.read_text())
if not payload.get("formal_m0_ready", False):
    print("Formal M0 is blocked: " + ", ".join(payload.get("formal_m0_blockers", [])), file=sys.stderr)
    raise SystemExit(3)
print("formal_m0_ready=true")
PY
}

require_evaluation_gate() {
  local path="$1"
  "$PYTHON_BIN" - "$path" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print(f"Missing registered evaluation: {path}", file=sys.stderr)
    raise SystemExit(2)
payload = json.loads(path.read_text())
if payload.get("run_kind") != "pilot" or not payload.get("gate", {}).get("passed", False):
    print(f"Registered M0 gate failed: {path}", file=sys.stderr)
    print(json.dumps(payload.get("gate", {}), indent=2), file=sys.stderr)
    raise SystemExit(2)
print(f"registered_m0_gate=pass file={path}")
PY
}
