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
# The workstation's configured Tsinghua mirror does not currently expose
# SpeechBrain 1.0.3.  Pin the v3 extras to official PyPI while retaining that
# mirror as a fallback for ordinary wheels.
"$PY" -m pip install --prefer-binary \
  --index-url https://pypi.org/simple \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements_v3.txt
"$PY" -c 'import librosa, speechbrain, torch, transformers; print({"speechbrain": speechbrain.__version__, "librosa": librosa.__version__, "torch": torch.__version__, "transformers": transformers.__version__})'
echo "[v3 bootstrap] complete"
