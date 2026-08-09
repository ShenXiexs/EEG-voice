#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$BUNDLE_DIR/data/ds006104}"

[[ -d "$DATA_DIR" ]] || { echo "ERROR: missing $DATA_DIR; run download first." >&2; exit 1; }
[[ -f "$DATA_DIR/dataset_description.json" ]] || { echo "ERROR: missing dataset_description.json" >&2; exit 1; }

missing=0
for i in $(seq -w 1 16); do
  eeg_dir="$DATA_DIR/sub-S${i}/ses-02/eeg"
  if [[ ! -d "$eeg_dir" ]]; then
    echo "MISSING: $eeg_dir"
    missing=1
    continue
  fi
  count=$(find "$eeg_dir" -maxdepth 1 -type f \( -iname '*.edf' -o -iname '*.bdf' -o -iname '*.set' -o -iname '*.vhdr' -o -iname '*.cnt' \) | wc -l | tr -d ' ')
  if [[ "$count" -eq 0 ]]; then
    echo "MISSING EEG FILE: $eeg_dir"
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "ERROR: the 2021/session-02 subset is incomplete." >&2
  exit 1
fi

echo "OK: all S01-S16/ses-02 EEG directories contain a raw EEG file."
echo "INFO: internal audio directory: $DATA_DIR/audio_internal"
if [[ -d "$DATA_DIR/audio_internal" ]]; then
  find "$DATA_DIR/audio_internal" -type f | sed -n '1,5p'
fi
