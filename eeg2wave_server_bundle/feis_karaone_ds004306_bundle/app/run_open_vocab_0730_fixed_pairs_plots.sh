#!/usr/bin/env bash
# Render PNG comparisons for existing v0730-fixed WAV pairs.  No training or inference.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PAIR_DIR="${1:-$APP_DIR/../artifacts/open_vocab_0730_explicit_cp_fixed_v2/pairs/all_1341}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$APP_DIR/.venv_0730/bin/python" ]]; then
  PY="$APP_DIR/.venv_0730/bin/python"
else
  PY=python3
fi

export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/open_vocab_0730_plot_pycache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/open_vocab_0730_pair_plots_matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/open_vocab_0730_pair_plots_cache}"

cd "$APP_DIR"
echo "[0730-fixed plots] python=$PY"
echo "[0730-fixed plots] source=$PAIR_DIR/manifest.csv"
echo "[0730-fixed plots] output=$PAIR_DIR/comparison_pairs"
"$PY" scripts/plot_open_vocab_0730_fixed_pairs.py \
  --manifest "$PAIR_DIR/manifest.csv" \
  --output "$PAIR_DIR/comparison_pairs" \
  --dpi "${PLOT_DPI:-120}" \
  --resume-existing
