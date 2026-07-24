# OpenVoice-EEG v0724: factorized content and vocal realization

## Scientific contract

v0724 tests whether imagined-speech EEG carries separable information about
linguistic content and the acoustic realization of an utterance. It does not
assume that a word has one canonical waveform, and it does not define success
as sample-wise waveform correlation alone.

The public inference contract is label-free:

```python
encode(eeg, channel_xyz, channel_mask, time_mask)
generate(eeg, channel_xyz, channel_mask, time_mask)
```

No label, transcript, subject ID, dataset ID, or reference audio is accepted by
these methods. Labels and speaker IDs are training-only masks/probes.

## Representation and decoder

- The audio content branch uses frozen HuBERT layer-9 frames and a dedicated
  content projector.
- The independent realization branch consumes numeric log-mel, F0, voicing,
  log-RMS and activity features. A frozen WavLM x-vector supplies a global
  timbre target when available.
- The EEG trunk retains the variable-montage coordinate-aware patch encoder and
  Adapter-MoE, then uses separate content and realization query banks. Masked
  patch content keeps its coordinate/time identity and is reconstructed only
  after a cross-token contextual Transformer, so the pretext task cannot reduce
  to a token-wise global mean.
- A factor fusion block predicts a numeric 80-by-400 log-mel map and conditions
  a 300-step MaskGIT/EnCodec decoder. Predicted active duration controls the
  valid EnCodec steps, up to four seconds.
- Content is subject-adversarial; timbre is label-adversarial; the two global
  factors receive a cross-covariance penalty. EEG subject identity is never
  used as a speaker lookup.

The output bundle for each synthesis condition contains a WAV, the float
log-mel tensor (`.mel.npy`), a PNG for inspection only, and JSON metrics. PNG
pixels, axes and colormaps never enter training or evaluation.

## Pairing routes

| Dataset | Content | Realization/timbre | Exact code/energy |
|---|---|---|---|
| KaraOne | Exact and same-label multi-positive | Same-trial overt target; same-label utterances are hard negatives | Enabled at low weight |
| FEIS | Same-label weak positive | Unique subject-label timbre prototype only | Disabled |
| ds004306 | EEG self-supervision only | Disabled | Disabled |

KaraOne overt speech is the strongest available trial pairing, but it is not
treated as a deterministically synchronized rendering of the preceding imagined
speech. Exact code loss is therefore auxiliary. Samples with more than four
seconds of active speech participate only in content learning.

## Energy morphology evaluation

All spectral comparisons operate on float 80-bin log-mel matrices computed at
16 kHz with a 25-ms window and 10-ms hop. Active speech is detected from RMS,
with a threshold of `max(p10 noise floor + 6 dB, peak - 40 dB)`, gaps up to 50
ms closed, and 100 ms context retained.

Two views are always reported:

1. Raw-time fidelity: log-mel MAE, multi-resolution STFT, duration error,
   activity overlap, envelope correlation, F0 and RMS.
2. Rendition-normalized morphology: active regions are resized along time only
   to 128 frames; the 80-bin frequency axis is never resized. Foreground-weighted
   SSIM/soft-IoU and self-cost-corrected soft-DTW divergence are reported.

The normalized morphology score cannot pass the gate by itself. The report also
records stretch factor and raw duration error, and content is evaluated in a
frozen speech-representation space using pooled HuBERT cosine and frame-level
SpeechBERTScore. WavLM x-vector cosine is reported as an optional secondary
timbre metric when the model is available locally.

## Counterfactuals and gates

Every KaraOne validation trial synthesizes:

- correct content + correct realization;
- correct content + same-label wrong realization;
- wrong-label content + correct realization;
- wrong content + wrong realization;
- content-only and realization-only;
- shuffled EEG and zero EEG.

The audio prior must first satisfy the inherited absolute oracle thresholds,
including median log-mel MAE at or below 12 dB, and the new factor-swap gains.
The EEG gate then requires the preregistered morphology, soft-DTW, content,
subject-bootstrap and trial-win criteria in
`app/configs/open_vocab_0724_factorized_v1.yaml`. A passed validation gate is
cryptographically bound to the config, cache lineage, audio checkpoint, EEG
checkpoint and synthesis manifest before locked-test access is authorized.
It also checks factor selectivity: same-label realization swaps must preserve
content better than wrong-content swaps, while realization swaps must affect
duration more than content swaps.

## Reproducibility and ablations

The primary seed is 15; seeds 31 and 47, development subject-LOSO folds, and
g2/g3 held-label runs are written to distinct checkpoint namespaces. Only the
primary g1 validation gate can authorize the locked test. The ablation config
generator supports dual-token/no-energy-map, dual-token/no-disentanglement,
content-only, realization-only, full HuBERT v0724 and full ContentVec. Branches
are disabled by masks/zero loss weights rather than deleting modules, so the
parameter count remains identical. Non-ContentVec ablations reuse the immutable
passed audio prior; ContentVec receives its own cache, audio checkpoint and
audio-oracle gate. The existing 0722 implementation remains the external
single-condition baseline.

## Evidence boundary

The unified track contains imagined-speech epochs, not auditory-perception EEG.
The present vocabulary is finite and does not include “Hi”. Label-free inference
therefore does not itself establish open-vocabulary generalization. Held-out
utterance, subject and label results are reported separately, and
“open-vocabulary” is reserved for a successful held-out-label experiment.
