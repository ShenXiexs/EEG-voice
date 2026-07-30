from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import idct


def librosa_mfcc_to_mel_reference(
    mfcc: np.ndarray, cepstral_mean: np.ndarray, cepstral_std: np.ndarray, *, mel_bins: int = 80
) -> np.ndarray:
    """Authoritative library reference for the fixed analytic backend."""
    try:
        import librosa
    except ImportError as error:  # pragma: no cover - dependency bootstrap path
        raise RuntimeError("librosa is required for the MFCC inversion conformance gate") from error
    values=[]
    for normalized,mean,std in zip(np.asarray(mfcc),np.asarray(cepstral_mean),np.asarray(cepstral_std)):
        restored=normalized*np.maximum(std[:,None],1.0e-4)+mean[:,None]
        power=librosa.feature.inverse.mfcc_to_mel(restored,n_mels=int(mel_bins),dct_type=2,norm="ortho",ref=1.0)
        values.append(np.clip(librosa.power_to_db(power,ref=1.0,top_db=None),-80.0,0.0))
    return np.stack(values).astype(np.float32)


def _encoder(dimension: int, heads: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dimension,
        nhead=heads,
        dim_feedforward=dimension * 4,
        activation="gelu",
        dropout=dropout,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


def _position(steps: int, dimension: int) -> torch.Tensor:
    position = torch.arange(steps, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(torch.arange(0, dimension, 2) * (-math.log(10_000.0) / dimension))
    output = torch.zeros(steps, dimension, dtype=torch.float32)
    output[:, 0::2] = torch.sin(position * divisor)
    output[:, 1::2] = torch.cos(position * divisor[: output[:, 1::2].shape[1]])
    return output


class AnalyticMFCCToMel(nn.Module):
    """Fixed orthonormal inverse-DCT MFCC-to-log-Mel backend.

    It is the differentiable equivalent of the standard librosa/scipy MFCC
    inversion.  The omitted high-order cepstra are zero-filled.  Utterance
    CMVN is restored with statistics from a non-target speaker reference (or
    the fit-only canonical voice), never with statistics from the target test
    waveform.
    """

    def __init__(self, *, mfcc_bins: int = 40, mel_bins: int = 80):
        super().__init__()
        basis = idct(np.eye(mel_bins), type=2, axis=0, norm="ortho")[:, :mfcc_bins]
        self.register_buffer("inverse_dct_basis", torch.from_numpy(basis.astype(np.float32)))
        self.mfcc_bins = int(mfcc_bins)
        self.mel_bins = int(mel_bins)

    def forward(
        self, mfcc: torch.Tensor, cepstral_mean: torch.Tensor, cepstral_std: torch.Tensor
    ) -> torch.Tensor:
        if mfcc.ndim != 3 or mfcc.shape[1] != self.mfcc_bins:
            raise ValueError(f"analytic backend expects MFCC [B,{self.mfcc_bins},T]")
        expected = (mfcc.shape[0], self.mfcc_bins)
        if cepstral_mean.shape != expected or cepstral_std.shape != expected:
            raise ValueError(f"cepstral mean/std must both be {expected}")
        restored = mfcc * cepstral_std.clamp_min(1.0e-4).unsqueeze(-1) + cepstral_mean.unsqueeze(-1)
        mel_db = torch.einsum("mk,bkt->bmt", self.inverse_dct_basis, restored)
        return mel_db.clamp(-80.0, 0.0)


class MFCCMelDecoder(nn.Module):
    """Conditional variational MFCC-to-Mel residual decoder.

    The fixed analytic inverse-DCT backend carries the content path.  A CVAE
    models only its bounded Mel residual.  The posterior sees real Mel during
    audio-only training/oracle evaluation; EEG synthesis uses the conditional
    prior and therefore cannot access target audio.
    """

    def __init__(
        self,
        *,
        mfcc_bins: int = 40,
        mel_bins: int = 80,
        dimension: int = 128,
        voice_dim: int = 192,
        latent_dim: int = 32,
        residual_limit_db: float = 24.0,
    ):
        super().__init__()
        self.mfcc_bins = int(mfcc_bins)
        self.mel_bins = int(mel_bins)
        self.voice_dim = int(voice_dim)
        self.latent_dim = int(latent_dim)
        self.residual_limit_db = float(residual_limit_db)
        self.analytic_backend = AnalyticMFCCToMel(mfcc_bins=mfcc_bins, mel_bins=mel_bins)
        self.content = nn.Sequential(
            nn.Conv1d(mfcc_bins, dimension, 5, padding=2), nn.GELU(),
            nn.Conv1d(dimension, dimension, 5, padding=2), nn.GELU(),
        )
        self.voice = nn.Sequential(
            nn.Linear(voice_dim, dimension), nn.GELU(), nn.Linear(dimension, dimension)
        )
        self.prior = nn.Sequential(
            nn.Linear(dimension * 2, dimension), nn.GELU(), nn.Linear(dimension, latent_dim * 2)
        )
        self.posterior_audio = nn.Sequential(
            nn.Conv1d(mel_bins * 2, dimension, 5, padding=2), nn.GELU(),
            nn.Conv1d(dimension, dimension, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.posterior = nn.Sequential(
            nn.Linear(dimension * 3, dimension), nn.GELU(), nn.Linear(dimension, latent_dim * 2)
        )
        self.latent = nn.Linear(latent_dim, dimension)
        self.film = nn.Linear(dimension, dimension * 2)
        self.decoder = nn.Sequential(
            nn.Conv1d(dimension, dimension, 5, padding=2), nn.GELU(),
            nn.Conv1d(dimension, dimension, 5, padding=2), nn.GELU(),
            nn.Conv1d(dimension, dimension, 5, padding=2), nn.GELU(),
            nn.Conv1d(dimension, mel_bins, 1),
        )

    @staticmethod
    def _stats(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = value.chunk(2, dim=-1)
        return mean, logvar.clamp(-8.0, 6.0)

    @staticmethod
    def _sample(mean: torch.Tensor, logvar: torch.Tensor, stochastic: bool) -> torch.Tensor:
        if not stochastic:
            return mean
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    def distributions(
        self,
        mfcc: torch.Tensor,
        voice: torch.Tensor,
        cepstral_mean: torch.Tensor,
        cepstral_std: torch.Tensor,
        target_mel: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if voice.shape != (mfcc.shape[0], self.voice_dim):
            raise ValueError(f"voice must be [B,{self.voice_dim}]")
        analytic = self.analytic_backend(mfcc, cepstral_mean, cepstral_std)
        content = self.content(mfcc)
        content_pool = content.mean(-1)
        voice_hidden = self.voice(voice)
        prior_mean, prior_logvar = self._stats(self.prior(torch.cat((content_pool, voice_hidden), dim=-1)))
        result = {
            "analytic_mel": analytic,
            "content_hidden": content,
            "voice_hidden": voice_hidden,
            "prior_mean": prior_mean,
            "prior_logvar": prior_logvar,
        }
        if target_mel is not None:
            if target_mel.shape != analytic.shape:
                raise ValueError(f"target Mel must have shape {tuple(analytic.shape)}")
            normalized_target = target_mel / 40.0 + 1.0
            normalized_analytic = analytic / 40.0 + 1.0
            audio_hidden = self.posterior_audio(torch.cat((normalized_target, normalized_analytic), dim=1))
            posterior_mean, posterior_logvar = self._stats(
                self.posterior(torch.cat((content_pool, voice_hidden, audio_hidden), dim=-1))
            )
            result.update({"posterior_mean": posterior_mean, "posterior_logvar": posterior_logvar})
        return result

    def decode(
        self,
        analytic: torch.Tensor,
        content: torch.Tensor,
        voice_hidden: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        hidden = content + self.latent(latent).unsqueeze(-1)
        scale, bias = self.film(voice_hidden).chunk(2, dim=-1)
        hidden = hidden * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(-1)) + 0.1 * bias.unsqueeze(-1)
        residual = self.residual_limit_db * torch.tanh(self.decoder(hidden))
        return (analytic + residual).clamp(-80.0, 0.0)

    def generate(
        self,
        mfcc: torch.Tensor,
        voice: torch.Tensor,
        cepstral_mean: torch.Tensor,
        cepstral_std: torch.Tensor,
        *,
        stochastic: bool = False,
    ) -> dict[str, torch.Tensor]:
        values = self.distributions(mfcc, voice, cepstral_mean, cepstral_std)
        latent = self._sample(values["prior_mean"], values["prior_logvar"], stochastic)
        values["latent"] = latent
        values["mel"] = self.decode(
            values["analytic_mel"], values["content_hidden"], values["voice_hidden"], latent
        )
        return values

    def reconstruct(
        self,
        mfcc: torch.Tensor,
        voice: torch.Tensor,
        cepstral_mean: torch.Tensor,
        cepstral_std: torch.Tensor,
        target_mel: torch.Tensor,
        *,
        stochastic: bool = True,
    ) -> dict[str, torch.Tensor]:
        values = self.distributions(mfcc, voice, cepstral_mean, cepstral_std, target_mel)
        latent = self._sample(values["posterior_mean"], values["posterior_logvar"], stochastic)
        values["latent"] = latent
        values["mel"] = self.decode(
            values["analytic_mel"], values["content_hidden"], values["voice_hidden"], latent
        )
        return values

    def forward(
        self,
        mfcc: torch.Tensor,
        voice: torch.Tensor,
        cepstral_mean: torch.Tensor,
        cepstral_std: torch.Tensor,
    ) -> torch.Tensor:
        return self.generate(mfcc, voice, cepstral_mean, cepstral_std, stochastic=False)["mel"]


class EEGMFCCEncoder(nn.Module):
    """Spatial-temporal EEG encoder with direct canonical MFCC output.

    It has no label, text, speaker, duration, energy, F0, or prosody input.
    Coordinate/channel fusion happens before pooling so reversing signal
    channels genuinely invalidates the spatial correspondence.
    """

    def __init__(self, *, mfcc_bins: int = 40, dimension: int = 128, heads: int = 4, layers: int = 2, dropout: float = 0.10, token_steps: int = 16):
        super().__init__()
        self.mfcc_bins = mfcc_bins
        self.token_steps = token_steps
        self.temporal = nn.Sequential(
            nn.Conv1d(1, 64, 15, padding=7), nn.GELU(),
            nn.Conv1d(64, dimension, 9, padding=4), nn.GELU(),
        )
        self.coordinate = nn.Sequential(nn.Linear(3, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.fusion = nn.Sequential(
            nn.Linear(dimension * 2, dimension), nn.GELU(), nn.Linear(dimension, dimension), nn.LayerNorm(dimension)
        )
        self.position = nn.Parameter(_position(32, dimension))
        self.trunk = _encoder(dimension, heads, layers, dropout)
        self.mfcc_head = nn.Sequential(
            nn.Conv1d(dimension, dimension, 5, padding=2), nn.GELU(), nn.Conv1d(dimension, mfcc_bins, 1)
        )
        self.token_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, mfcc_bins, bias=False))
        self.clip_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def forward(
        self, eeg: torch.Tensor, channel_xyz: torch.Tensor, channel_mask: torch.Tensor, time_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if eeg.ndim != 3 or channel_xyz.shape != (*eeg.shape[:2], 3):
            raise ValueError("eeg must be [B,C,T] and xyz [B,C,3]")
        if channel_mask.shape != eeg.shape[:2] or time_mask.shape != (eeg.shape[0], eeg.shape[2]):
            raise ValueError("invalid channel/time mask")
        if not torch.isfinite(eeg).all() or not torch.isfinite(channel_xyz).all():
            raise ValueError("EEG and coordinates must be finite")
        batch, channels, samples = eeg.shape
        temporal = self.temporal(eeg.reshape(batch * channels, 1, samples))
        temporal = F.adaptive_avg_pool1d(temporal, 32).transpose(1, 2).reshape(batch, channels, 32, -1)
        coordinate = self.coordinate(channel_xyz).unsqueeze(2).expand(-1, -1, 32, -1)
        fused = self.fusion(torch.cat((temporal, coordinate), dim=-1))
        weight = channel_mask.to(fused.dtype).view(batch, channels, 1, 1)
        pooled = (fused * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        pooled_mask = F.interpolate(time_mask.float().unsqueeze(1), size=32, mode="nearest").squeeze(1).bool()
        latent = self.trunk(pooled + self.position.unsqueeze(0), src_key_padding_mask=~pooled_mask)
        mfcc = F.interpolate(self.mfcc_head(latent.transpose(1, 2)), size=256, mode="linear", align_corners=False)
        tokens = F.adaptive_avg_pool1d(latent.transpose(1, 2), self.token_steps).transpose(1, 2)
        return mfcc, self.token_head(tokens)
