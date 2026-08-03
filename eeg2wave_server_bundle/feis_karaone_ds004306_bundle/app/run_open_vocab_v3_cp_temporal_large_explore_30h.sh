#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BUDGET_HOURS="${BUDGET_HOURS:-30}"
exec "$APP_DIR/run_open_vocab_v3_cp_temporal_large_explore_20h.sh" "$@"
