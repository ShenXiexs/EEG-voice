from __future__ import annotations

import math

import torch


def mel_filterbank(*, sample_rate: int = 16000, n_fft: int = 512, mel_bins: int = 80, fmin: float = 0.0, fmax: float = 8000.0, device: torch.device | None = None) -> torch.Tensor:
    """Triangular Slaney-style filterbank implemented without external vocoders."""
    def hz_to_mel(value: torch.Tensor) -> torch.Tensor: return 2595.0 * torch.log10(1.0 + value / 700.0)
    def mel_to_hz(value: torch.Tensor) -> torch.Tensor: return 700.0 * (torch.pow(10.0, value / 2595.0) - 1.0)
    target = device or torch.device("cpu")
    frequencies=torch.linspace(0,sample_rate/2,n_fft//2+1,device=target)
    points=mel_to_hz(torch.linspace(hz_to_mel(torch.tensor(fmin,device=target)),hz_to_mel(torch.tensor(fmax,device=target)),mel_bins+2,device=target))
    filters=[]
    for left,center,right in zip(points[:-2],points[1:-1],points[2:]):
        filters.append(torch.clamp(torch.minimum((frequencies-left)/(center-left).clamp_min(1e-6),(right-frequencies)/(right-center).clamp_min(1e-6)),min=0))
    return torch.stack(filters)


def griffin_lim_from_log_mel(log_mel: torch.Tensor, *, iterations: int = 64, seed: int = 15, sample_rate: int = 16000, n_fft: int = 512, win_length: int = 400, hop_length: int = 160) -> torch.Tensor:
    """Deterministic inverse-mel Griffin–Lim for a single [80,400] dB mel."""
    if log_mel.shape != (80,400): raise ValueError(f"expected [80,400] mel, got {tuple(log_mel.shape)}")
    device=log_mel.device; filt=mel_filterbank(sample_rate=sample_rate,n_fft=n_fft,device=device)
    power=torch.pow(10.0,log_mel/10.0).clamp_min(1e-10)
    magnitude=torch.sqrt(torch.clamp(torch.linalg.pinv(filt)@power,min=0.0))
    generator=torch.Generator(device=device); generator.manual_seed(int(seed))
    phase=torch.exp(2j*math.pi*torch.rand(magnitude.shape,generator=generator,device=device))
    window=torch.hann_window(win_length,device=device)
    for _ in range(int(iterations)):
        wav=torch.istft(magnitude*phase,n_fft=n_fft,hop_length=hop_length,win_length=win_length,window=window,center=True)
        estimate=torch.stft(wav,n_fft=n_fft,hop_length=hop_length,win_length=win_length,window=window,return_complex=True,center=True)
        phase=estimate/(estimate.abs().clamp_min(1e-8))
    wav=torch.istft(magnitude*phase,n_fft=n_fft,hop_length=hop_length,win_length=win_length,window=window,center=True)
    return wav.clamp(-1,1)
