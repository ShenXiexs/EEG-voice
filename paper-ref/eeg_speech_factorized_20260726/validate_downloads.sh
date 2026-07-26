#!/usr/bin/env bash
set -u

pdf_dir="/Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/paper-ref/eeg_speech_factorized_20260726/pdf"
expected=20
valid=0
invalid=0
missing=0

for index in $(seq -w 1 "$expected"); do
  matches=("$pdf_dir"/S"$index"_*.pdf)
  if [ ! -e "${matches[0]}" ]; then
    printf 'MISSING  S%s\n' "$index"
    missing=$((missing + 1))
  elif file -b --mime-type "${matches[0]}" | grep -qx 'application/pdf'; then
    printf 'VALID    %s\n' "$(basename "${matches[0]}")"
    valid=$((valid + 1))
  else
    printf 'INVALID  %s (%s)\n' "$(basename "${matches[0]}")" "$(file -b --mime-type "${matches[0]}")"
    invalid=$((invalid + 1))
  fi
done

printf '\nSummary: valid=%d invalid=%d missing=%d expected=%d\n' "$valid" "$invalid" "$missing" "$expected"
