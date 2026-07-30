#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$APP_DIR/.venv_0730/bin/python" ]]; then
  PY="$APP_DIR/.venv_0730/bin/python"
else
  PY=python3
fi

cd "$APP_DIR"
echo "[v3 bootstrap] python=$PY"
"$PY" -m pip install -r requirements_v3.txt
"$PY" -c 'import speechbrain, torch, transformers; print({"speechbrain": speechbrain.__version__, "torch": torch.__version__, "transformers": transformers.__version__})'
echo "[v3 bootstrap] complete"
