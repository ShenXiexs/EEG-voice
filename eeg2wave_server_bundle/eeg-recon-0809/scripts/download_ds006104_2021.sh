#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$BUNDLE_DIR/data/ds006104}"
S3_URI="s3://openneuro.org/ds006104"

command -v aws >/dev/null 2>&1 || {
  echo "ERROR: aws CLI is required. Install/configure it, then rerun." >&2
  exit 127
}

mkdir -p "$DATA_DIR"
mkdir -p "$BUNDLE_DIR/reports"
echo "[ds006104] destination: $DATA_DIR"
echo "[ds006104] downloading 2021 subjects S01-S16, ses-02 only"

aws s3 sync --no-sign-request "$S3_URI" "$DATA_DIR" \
  --exclude '*' \
  --include 'README*' \
  --include 'CHANGES*' \
  --include 'dataset_description.json' \
  --include 'participants.*' \
  --include 'task-*.json' \
  --include '.bidsignore' \
  --include 'sourcedata/**' \
  --include 'sub-S*/ses-02/**' \
  --only-show-errors

date -u +%Y-%m-%dT%H:%M:%SZ > "$BUNDLE_DIR/reports/download_completed_utc.txt"
echo "[ds006104] download complete"
echo "[ds006104] internal audio is not part of this command; place it in $DATA_DIR/audio_internal/"
