#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/joint_pilot_common.sh"
start_joint_log status
require_joint_runtime
joint_run "$PYTHON_BIN" - "$PROJECT_ROOT" "$ARTIFACT_ROOT" "$PILOT_CONFIG" <<'PY'
import json, sys, yaml
from pathlib import Path
project, artifact, pilot_path = map(Path, sys.argv[1:])
sys.path.insert(0, str(project / "app" / "src"))
from eeg2speech.gates import registered_m0_gate_status
pilot = yaml.safe_load(pilot_path.read_text())
validation_path = artifact / "qc" / "validate.json"
validation = json.loads(validation_path.read_text()) if validation_path.exists() else {"status": "missing"}
renderer_path = project / "outputs" / "joint_pilot_v1" / "audio_renderer" / "metrics.json"
renderer = json.loads(renderer_path.read_text()) if renderer_path.exists() else {"status": "missing"}
stage2_path = artifact / "qc" / "stage2_artifacts.json"
stage2 = json.loads(stage2_path.read_text()) if stage2_path.exists() else {"status": "not_materialized"}
print(json.dumps({
    "stage0": validation,
    "registered_m0": registered_m0_gate_status(project, pilot),
    "audio_renderer": renderer.get("gate", renderer),
    "stage2_artifacts": stage2,
}, indent=2))
PY
