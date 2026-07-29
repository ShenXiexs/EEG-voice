# Open-vocabulary 0730 fixed-v2

This is a fail-closed correction of `open_vocab_0730_explicit_cp_v1`. It does
not overwrite v1, 0724, or 0728 artifacts.

## What changed

- Electrode coordinates are fused with their matching channel features before
  channel pooling; channel permutation is therefore a real negative control.
- Learned temporal positions are added before the EEG Transformer.
- EEG/audio alignment remains an OpenAI-style dual-tower CLIP objective. The
  audio tower is frozen, and same-label trials are multi-positive rather than
  false negatives. Token and utterance CLIP losses are both retained.
- Label text/phoneme spelling remains a 5% train-only auxiliary anchor. It is
  never an inference input, and `pot` is absent from its anchors.
- Prosody activity uses fit-only positive weights. Envelope regression is
  dominated by active bins, retains a small inactive-bin constraint, and has a
  weighted shape-correlation term.
- Renderer-facing duration, loudness, and envelope values are bounded to the
  audio training support.
- Direct EEG-token/audio-token CLIP retrieval is now evaluated for correct,
  zero, time-shuffled, and channel-shuffled EEG.
- Content, activity, duration, envelope, active-frame Mel error, non-silence,
  and between-trial variance all have independent controls.
- The 275 physically isolated `locked_test` records and 297 P02/MM21 records
  are final-test-only roles. They are never used for fitting or early stopping.
- The original 1,341 reference/reconstruction pairs are still all exported.
- SpeechT5 receives `mel_db / 10`, matching the source cache's
  `10*log10(power)` representation.

## One-shot command

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/feis_karaone_ds004306_bundle/app
./run_open_vocab_0730_fixed_all.sh
```

Training may use MPS. Evaluation and all-pair export default to CPU because
PyTorch 2.8 MPS can abort during repeated counterfactual Transformer forwards.
The total configured training budget is capped below ten hours; pair export is
post-training artifact generation.

To deliberately discard only fixed-v2 training outputs and retrain:

```bash
FORCE_RETRAIN=1 ./run_open_vocab_0730_fixed_all.sh
```

## Interpretation gate

`evaluation/generated_gate.json` is the scientific gate. WAV files are marked
`diagnostic_waveform_only` unless every registered correct-over-control,
non-silence, and variance check passes. `pairs_audit.json` only verifies file
integrity and must not be used as evidence of decoding quality.
