"""One native SpeechT5 Mel definition for adaptation, CVAE and export.

SpeechT5 HiFi-GAN consumes a time-major, 80-bin log-Mel spectrum.  v3 stores
the model-facing representation in this module's feature space and forbids the
old v2 power-dB/10 conversion or a learned adapter.  Keeping the contract in
one module makes the gate and training paths mechanically identical.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

CONTRACT = "speecht5_native_log_mel_v1"
_FEATURE_EXTRACTOR = None


def _mel_filter(*, sample_rate: int, n_fft: int, bins: int, fmin: float, fmax: float, device: torch.device) -> torch.Tensor:
    try:
        import librosa
        values = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=bins, fmin=fmin, fmax=fmax, htk=False, norm="slaney")
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("librosa is required to construct the SpeechT5 Mel contract") from error
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def native_speecht5_mel(waveform: torch.Tensor, cfg: dict, *, frames: int | None = None) -> torch.Tensor:
    """Return exactly the official SpeechT5 audio-target log10 Mel."""
    if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2: raise ValueError("waveform must be [B,S]")
    global _FEATURE_EXTRACTOR
    if _FEATURE_EXTRACTOR is None:
        from transformers import SpeechT5FeatureExtractor
        _FEATURE_EXTRACTOR=SpeechT5FeatureExtractor(sampling_rate=16000,num_mel_bins=80,hop_length=16,win_length=64,win_function="hann_window",fmin=80,fmax=7600,mel_floor=1e-10,do_normalize=False)
    extractor=_FEATURE_EXTRACTOR
    values=[]
    for row in waveform.detach().cpu().numpy():
        # _extract_mel_features is the same method used by
        # SpeechT5FeatureExtractor(audio_target=...).
        values.append(torch.from_numpy(extractor._extract_mel_features(np.asarray(row,dtype=np.float32))).float().T)
    width=max(x.shape[-1] for x in values);padded=torch.stack([F.pad(x,(0,width-x.shape[-1]),value=float(x.min())) for x in values]).to(waveform.device)
    return F.interpolate(padded,size=int(frames),mode="linear",align_corners=False) if frames is not None else padded


def native_mel_mask(valid_samples: np.ndarray | torch.Tensor, cfg: dict, *, frames: int) -> torch.Tensor:
    values = torch.as_tensor(valid_samples, dtype=torch.float32)
    steps = torch.ceil(values / int(cfg["vocoder"]["hop_length"])).long().clamp(1, int(frames))
    return torch.arange(int(frames)).unsqueeze(0) < steps.unsqueeze(1)
