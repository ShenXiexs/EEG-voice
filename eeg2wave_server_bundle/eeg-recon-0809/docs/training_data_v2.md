# DS004940/DS006104 training-data build (v2)

`scripts/prepare_training_data.py` implements the `harmonized_v2` project
profile in `configs/training_data_v2.yaml`.  It is not a claim about either
dataset's official preprocessing.  EEG is written as `float32` **Volts**;
there is no ICA, clipping, per-trial z-score, or global normalization.

Run from this directory with the existing Conda environment:

```bash
conda activate eegvoice
python -m pip install -r requirements-preprocess.txt

# Hash raw inputs, BIDS events, paired WAV files and pinned official tables.
# It always writes QC first; --strict exits non-zero after writing warnings.
python scripts/prepare_training_data.py audit --fetch-aux --strict

# Only after reviewing audit.json.  This also audits each WAV's semantics and
# writes deduplicated 16 kHz waveform/log-Mel records without normalization.
python scripts/prepare_training_data.py build-audio-bank
python scripts/prepare_training_data.py make-splits

# Start with a bounded pilot.  Remove the two filters for the all-data build.
python scripts/prepare_training_data.py build \
  --dataset ds006104 --subjects S02 --allow-audit-warnings
python scripts/prepare_training_data.py validate --strict
```

The intentional `--allow-audit-warnings` flag is captured in shard provenance.
Do not use it to bypass a source-file hash mismatch; correct the input or the
pinned configuration first.  `audit`'s expected trial counts are release
expectations only: it records actual counts and exclusions without truncating
or renumbering any trial.  Pinned GitHub downloads try Conda-Python TLS first
and then system `curl`; a network failure is written to `qc/audit.json` and
causes `--strict` to exit non-zero instead of producing a traceback.

The frozen split CSVs in `artifacts/training_data/v2/splits/` are the only
permitted split definition.  For example, fit a normalizer from train rows
only:

```bash
python scripts/prepare_training_data.py fit-normalizer \
  --split-csv artifacts/training_data/v2/splits/joint_ood_fold-0.csv --fold 0
```

The default normalizer excludes `singlephoneme` (`neural_task=mixed`).  A
consumer that wants a full singlephoneme epoch must explicitly opt in to mixed
production; otherwise it must use `clean_perception_mask`, which is exactly
`[64, 140)` (76 output samples) for those trials.  The raw full 384-point
epoch is retained for research.

Each shard also contains `audio_loss_mask`: DS004940 keeps a fixed EEG tensor
but limits sentence-waveform loss to the observed audio duration plus the
configured 0.5 s tail.  This never changes the `[C, 1178]` tensor shape.

`--resume` only reuses an audio bank or shard if configuration SHA256,
source-lock SHA256, and channel-order hash match.  Shards are first written as
`.partial` and atomically renamed only after their internal HDF5 write closes.
