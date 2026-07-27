"""Factorized, label-free EEG-to-speech model for the v0724 track.

The module deliberately keeps three concepts separate:

* content: the linguistic sequence represented by frozen HuBERT features;
* realization: frame-level energy/prosody represented by log-mel features;
* timbre: a global speaker-style vector represented by a frozen WavLM x-vector.

Labels, subjects, and dataset identifiers are training targets for adversarial
heads only.  They are never accepted by the public inference facade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def _all_finite(value: torch.Tensor) -> bool:
    """Return a reliable finite check, including after rare MPS false alarms.

    The normal device-side reduction remains the fast path.  Only when MPS
    reports a failure do we synchronize and repeat the check on CPU, so real
    NaN/Inf values are still rejected without slowing normal training.
    """

    device_result = torch.isfinite(value).all()
    if bool(device_result.detach().cpu().item()):
        return True
    if value.device.type == "mps":
        return bool(torch.isfinite(value.detach().cpu()).all().item())
    return False


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, strength: float) -> torch.Tensor:  # type: ignore[override]
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        return -ctx.strength * gradient, None


def grad_reverse(value: torch.Tensor, strength: float) -> torch.Tensor:
    """Identity in the forward pass and gradient negation in the backward pass."""

    return _GradientReverse.apply(value, float(strength))


def _transformer(
    d_model: int,
    heads: int,
    layers: int,
    dropout: float,
    *,
    expansion: int = 4,
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=heads,
        dim_feedforward=d_model * expansion,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


def sinusoidal_positions(
    length: int,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if length < 1:
        raise ValueError("position sequence must be non-empty")
    if dimension < 2:
        raise ValueError("position dimension must be at least 2")
    half = dimension // 2
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    scale = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / max(1, half - 1)
    )
    encoded = torch.cat(
        (torch.sin(positions * scale), torch.cos(positions * scale)), dim=1
    )
    if encoded.shape[1] < dimension:
        encoded = F.pad(encoded, (0, dimension - encoded.shape[1]))
    return encoded[:, :dimension].to(dtype=dtype)


def _masked_mean(tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or valid_mask.shape != tokens.shape[:2]:
        raise ValueError("masked mean expects tokens [B,T,D] and valid_mask [B,T]")
    weights = valid_mask.to(tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _normalize_sequence_mask(
    valid_mask: torch.Tensor | None,
    *,
    batch: int,
    steps: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if valid_mask is None:
        valid = torch.ones(batch, steps, device=device, dtype=torch.bool)
    else:
        if valid_mask.shape != (batch, steps):
            raise ValueError(f"{name} must be [B,T], got {tuple(valid_mask.shape)}")
        valid = valid_mask.to(device=device, dtype=torch.bool)
    if not valid.any(dim=1).all():
        raise ValueError(
            f"every sample must contain at least one valid {name} position"
        )
    return valid


def _resample_valid_sequence(
    tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    output_steps: int,
) -> torch.Tensor:
    """Resample only valid frames so right padding cannot affect a teacher."""

    if tokens.ndim != 3 or valid_mask.shape != tokens.shape[:2]:
        raise ValueError("tokens and valid_mask must have shapes [B,T,D] and [B,T]")
    if output_steps < 1:
        raise ValueError("output_steps must be positive")
    output: list[torch.Tensor] = []
    for item in range(tokens.shape[0]):
        value = tokens[item, valid_mask[item]]
        if value.shape[0] == 0:
            raise ValueError("cannot resample an empty sequence")
        resized = (
            F.interpolate(
                value.transpose(0, 1).unsqueeze(0),
                size=output_steps,
                mode="linear",
                align_corners=False,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
        output.append(resized)
    return torch.stack(output, dim=0)


def _as_code_mask(codes: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    batch, codebooks, steps = codes.shape
    if valid_mask is None:
        return torch.ones(batch, steps, device=codes.device, dtype=torch.bool)
    if valid_mask.shape == (batch, steps):
        return valid_mask.to(device=codes.device, dtype=torch.bool)
    if valid_mask.shape == (batch, codebooks, steps):
        # A time step is usable only when every residual stream is present.
        return valid_mask.to(device=codes.device, dtype=torch.bool).all(dim=1)
    raise ValueError("code_valid_mask must be [B,T] or [B,Q,T]")


@dataclass(frozen=True)
class FactorizedAudioConfig:
    codebooks: int = 8
    code_steps: int = 300
    code_rate_hz: float = 75.0
    vocab_size: int = 1024
    d_model: int = 192
    condition_steps: int = 50
    mel_bins: int = 80
    energy_frames: int = 400
    content_input_dimension: int = 768
    timbre_input_dimension: int = 512
    realization_input_dimension: int = 84
    audio_encoder_layers: int = 2
    fusion_layers: int = 2
    decoder_layers: int = 4
    heads: int = 6
    dropout: float = 0.10
    branch_dropout_probability: float = 0.10
    mel_db_min: float = -80.0
    mel_db_max: float = 0.0
    min_duration_seconds: float = 0.04
    max_duration_sec: float = 4.0
    generation_steps: int = 12
    generation_temperature: float = 0.0
    use_content_condition: bool = True
    use_realization_condition: bool = True
    use_energy_feedback: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if self.condition_steps < 1 or self.mel_bins < 1 or self.energy_frames < 1:
            raise ValueError(
                "condition_steps, mel_bins, and mel_frames must be positive"
            )
        if self.code_steps < 1 or self.code_rate_hz <= 0:
            raise ValueError("codec settings must be positive")
        if (
            self.min_duration_seconds <= 0
            or self.max_duration_sec < self.min_duration_seconds
        ):
            raise ValueError("duration bounds are invalid")
        if self.mel_db_max <= self.mel_db_min:
            raise ValueError("mel_max_db must exceed mel_min_db")
        if self.realization_input_dimension < self.mel_bins:
            raise ValueError("realization_input_dimension must include all mel bins")
        if not 0.0 <= self.branch_dropout_probability < 1.0:
            raise ValueError("branch_dropout_probability must be in [0,1)")
        if math.ceil(self.max_duration_sec * self.code_rate_hz) > self.code_steps:
            raise ValueError(
                "code_steps cannot represent max_duration_sec at code_rate_hz"
            )
        if not (self.use_content_condition or self.use_realization_condition):
            raise ValueError("at least one factor condition must be enabled")

    # Read-only aliases keep the implementation vocabulary explicit while the
    # constructor mirrors keys in open_vocab_0724_factorized_v1.yaml.
    @property
    def max_code_steps(self) -> int:
        return self.code_steps

    @property
    def codec_frame_rate(self) -> float:
        return self.code_rate_hz

    @property
    def mel_frames(self) -> int:
        return self.energy_frames

    @property
    def hubert_dimension(self) -> int:
        return self.content_input_dimension

    @property
    def timbre_dimension(self) -> int:
        return self.timbre_input_dimension

    @property
    def encoder_layers(self) -> int:
        return self.audio_encoder_layers

    @property
    def mel_min_db(self) -> float:
        return self.mel_db_min

    @property
    def mel_max_db(self) -> float:
        return self.mel_db_max

    @property
    def max_duration_seconds(self) -> float:
        return self.max_duration_sec


@dataclass(frozen=True)
class FactorizedEEGConfig:
    eeg_samples: int = 1280
    patch_size: int = 64
    patch_hop: int = 32
    d_model: int = 192
    condition_steps: int = 50
    mel_bins: int = 80
    mel_frames: int = 400
    heads: int = 6
    latent_layers: int = 3
    fusion_layers: int = 2
    dropout: float = 0.15
    specialists: int = 4
    specialist_bottleneck: int = 48
    soft_routing_epochs: int = 5
    top_k_specialists: int = 2
    expert_dropout: float = 0.10
    num_datasets: int = 3
    num_train_subjects: int = 38
    num_content_labels: int = 30
    adapter_moe_enabled: bool = True
    branch_dropout_probability: float = 0.10
    mel_min_db: float = -80.0
    mel_max_db: float = 0.0
    min_duration_seconds: float = 0.04
    max_duration_seconds: float = 4.0
    use_content_condition: bool = True
    use_realization_condition: bool = True
    use_energy_feedback: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if self.patch_size < 1 or self.patch_hop < 1:
            raise ValueError("patch_size and patch_hop must be positive")
        if self.condition_steps < 1 or self.mel_bins < 1 or self.mel_frames < 1:
            raise ValueError(
                "condition_steps, mel_bins, and mel_frames must be positive"
            )
        if self.top_k_specialists < 1 or self.top_k_specialists > self.specialists:
            raise ValueError("top_k_specialists must be within the specialist count")
        if min(self.num_datasets, self.num_train_subjects, self.num_content_labels) < 1:
            raise ValueError("all adversary class counts must be positive")
        if not 0.0 <= self.branch_dropout_probability < 1.0:
            raise ValueError("branch_dropout_probability must be in [0,1)")
        if self.mel_max_db <= self.mel_min_db:
            raise ValueError("mel_max_db must exceed mel_min_db")
        if (
            self.min_duration_seconds <= 0
            or self.max_duration_seconds < self.min_duration_seconds
        ):
            raise ValueError("duration bounds are invalid")
        if not (self.use_content_condition or self.use_realization_condition):
            raise ValueError("at least one factor condition must be enabled")


@dataclass(frozen=True)
class FactorizedAudioState:
    content_tokens: torch.Tensor
    realization_tokens: torch.Tensor
    content_global: torch.Tensor
    realization_global: torch.Tensor
    timbre_global: torch.Tensor
    fused_condition: torch.Tensor
    log_mel_energy: torch.Tensor
    log_f0_hz: torch.Tensor
    voicing_logits: torch.Tensor
    log_rms_dbfs: torch.Tensor
    activity_logits: torch.Tensor
    duration_seconds: torch.Tensor
    content_valid_mask: torch.Tensor
    realization_valid_mask: torch.Tensor


@dataclass(frozen=True)
class FactorizedAudioOutput:
    state: FactorizedAudioState
    code_logits: torch.Tensor


@dataclass(frozen=True)
class FactorizedConditionState:
    fused_condition: torch.Tensor
    log_mel_energy: torch.Tensor
    log_f0_hz: torch.Tensor
    voicing_logits: torch.Tensor
    log_rms_dbfs: torch.Tensor
    activity_logits: torch.Tensor
    duration_seconds: torch.Tensor


@dataclass(frozen=True)
class FactorizedEEGState:
    content_tokens: torch.Tensor
    realization_tokens: torch.Tensor
    content_global: torch.Tensor
    realization_global: torch.Tensor
    timbre_global: torch.Tensor
    fused_condition: torch.Tensor
    log_mel_energy: torch.Tensor
    log_f0_hz: torch.Tensor
    voicing_logits: torch.Tensor
    log_rms_dbfs: torch.Tensor
    activity_logits: torch.Tensor
    duration_seconds: torch.Tensor
    subject_logits: torch.Tensor
    dataset_logits: torch.Tensor
    timbre_label_logits: torch.Tensor
    router_dataset_logits: torch.Tensor
    patch_reconstruction: torch.Tensor
    patch_target: torch.Tensor
    patch_valid_mask: torch.Tensor
    patch_mask: torch.Tensor
    router: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class FactorizedGeneration:
    codes: torch.Tensor
    code_valid_mask: torch.Tensor
    content_tokens: torch.Tensor
    realization_tokens: torch.Tensor
    content_global: torch.Tensor
    realization_global: torch.Tensor
    timbre_global: torch.Tensor
    fused_condition: torch.Tensor
    log_mel_energy: torch.Tensor
    log_f0_hz: torch.Tensor
    voicing_logits: torch.Tensor
    log_rms_dbfs: torch.Tensor
    activity_logits: torch.Tensor
    duration_seconds: torch.Tensor

    @property
    def code_lengths(self) -> torch.Tensor:
        return self.code_valid_mask.sum(dim=-1)


class AudioContentProjector(nn.Module):
    """Project frozen HuBERT frame features into content-only tokens."""

    def __init__(self, cfg: FactorizedAudioConfig):
        super().__init__()
        self.cfg = cfg
        self.input = nn.Sequential(
            nn.LayerNorm(cfg.hubert_dimension),
            nn.Linear(cfg.hubert_dimension, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.encoder = _transformer(
            cfg.d_model, cfg.heads, cfg.encoder_layers, cfg.dropout
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.global_projection = nn.Sequential(
            nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model)
        )

    def forward(
        self,
        hubert_features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            hubert_features.ndim != 3
            or hubert_features.shape[-1] != self.cfg.hubert_dimension
        ):
            raise ValueError(
                f"hubert_features must be [B,T,{self.cfg.hubert_dimension}], "
                f"got {tuple(hubert_features.shape)}"
            )
        if not _all_finite(hubert_features):
            raise ValueError("hubert_features must be finite")
        valid = _normalize_sequence_mask(
            valid_mask,
            batch=hubert_features.shape[0],
            steps=hubert_features.shape[1],
            device=hubert_features.device,
            name="content_valid_mask",
        )
        value = self.input(hubert_features)
        value = _resample_valid_sequence(value, valid, self.cfg.condition_steps)
        value = self.norm(self.encoder(value))
        output_mask = torch.ones(value.shape[:2], device=value.device, dtype=torch.bool)
        pooled = _masked_mean(value, output_mask)
        return value, F.normalize(self.global_projection(pooled), dim=-1), output_mask


class AudioRealizationEncoder(nn.Module):
    """Encode frame-level log-mel energy and prosody independently of content."""

    def __init__(self, cfg: FactorizedAudioConfig):
        super().__init__()
        self.cfg = cfg
        self.input = nn.Sequential(
            nn.LayerNorm(cfg.realization_input_dimension),
            nn.Linear(cfg.realization_input_dimension, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.encoder = _transformer(
            cfg.d_model, cfg.heads, cfg.encoder_layers, cfg.dropout
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.global_projection = nn.Sequential(
            nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model)
        )

    def forward(
        self,
        realization_features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            realization_features.ndim != 3
            or realization_features.shape[-1] != self.cfg.realization_input_dimension
        ):
            raise ValueError(
                "realization_features must be "
                f"[B,T,{self.cfg.realization_input_dimension}]"
            )
        batch, steps, _ = realization_features.shape
        if not _all_finite(realization_features):
            raise ValueError("realization features must be finite")
        valid = _normalize_sequence_mask(
            valid_mask,
            batch=batch,
            steps=steps,
            device=realization_features.device,
            name="realization_valid_mask",
        )
        mel_scale = self.cfg.mel_max_db - self.cfg.mel_min_db
        normalized_parts = [
            (realization_features[..., : self.cfg.mel_bins] - self.cfg.mel_min_db)
            / mel_scale
        ]
        prosody = realization_features[..., self.cfg.mel_bins :]
        # Cache-v2 orders the four standard channels as log-F0, voicing,
        # log-RMS dBFS, and activity. Put their numerical ranges near [0,1]
        # before the joint LayerNorm; otherwise raw RMS dominates the vector.
        if prosody.shape[-1] >= 1:
            normalized_parts.append(prosody[..., 0:1] / math.log(1_000.0))
        if prosody.shape[-1] >= 2:
            normalized_parts.append(prosody[..., 1:2])
        if prosody.shape[-1] >= 3:
            normalized_parts.append(
                (prosody[..., 2:3] - self.cfg.mel_min_db) / mel_scale
            )
        if prosody.shape[-1] >= 4:
            normalized_parts.append(prosody[..., 3:4])
        if prosody.shape[-1] > 4:
            normalized_parts.append(prosody[..., 4:])
        features = torch.cat(normalized_parts, dim=-1)
        value = self.input(features)
        value = _resample_valid_sequence(value, valid, self.cfg.condition_steps)
        value = self.norm(self.encoder(value))
        output_mask = torch.ones(value.shape[:2], device=value.device, dtype=torch.bool)
        pooled = _masked_mean(value, output_mask)
        return value, F.normalize(self.global_projection(pooled), dim=-1), output_mask


class TimbreProjector(nn.Module):
    """Project a frozen WavLM speaker x-vector into the shared latent space."""

    def __init__(self, input_dimension: int, d_model: int):
        super().__init__()
        self.input_dimension = int(input_dimension)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, timbre_embedding: torch.Tensor) -> torch.Tensor:
        if (
            timbre_embedding.ndim != 2
            or timbre_embedding.shape[-1] != self.input_dimension
        ):
            raise ValueError(f"timbre_embedding must be [B,{self.input_dimension}]")
        if not _all_finite(timbre_embedding):
            raise ValueError("timbre_embedding must be finite")
        return F.normalize(self.net(timbre_embedding), dim=-1)


class FactorizedConditionFusion(nn.Module):
    """Fuse factorized tokens, predict energy, then feed energy back to the decoder memory."""

    def __init__(
        self,
        *,
        d_model: int,
        condition_steps: int,
        mel_bins: int,
        mel_frames: int,
        heads: int,
        layers: int,
        dropout: float,
        mel_min_db: float,
        mel_max_db: float,
        min_duration_seconds: float,
        max_duration_seconds: float,
        branch_dropout_probability: float = 0.0,
        use_content_condition: bool = True,
        use_realization_condition: bool = True,
        use_energy_feedback: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.condition_steps = int(condition_steps)
        self.mel_bins = int(mel_bins)
        self.mel_frames = int(mel_frames)
        self.mel_min_db = float(mel_min_db)
        self.mel_max_db = float(mel_max_db)
        self.min_duration_seconds = float(min_duration_seconds)
        self.max_duration_seconds = float(max_duration_seconds)
        self.branch_dropout_probability = float(branch_dropout_probability)
        self.use_content_condition = bool(use_content_condition)
        self.use_realization_condition = bool(use_realization_condition)
        self.use_energy_feedback = bool(use_energy_feedback)
        if not (self.use_content_condition or self.use_realization_condition):
            raise ValueError("at least one factor condition must be enabled")
        self.factor_projection = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.factor_refiner = _transformer(d_model, heads, layers, dropout)
        self.factor_norm = nn.LayerNorm(d_model)
        self.energy_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, mel_bins)
        )
        self.activity_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.prosody_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))
        self.duration_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.energy_input = nn.Sequential(
            nn.LayerNorm(mel_bins), nn.Linear(mel_bins, d_model), nn.GELU()
        )
        self.energy_refiner = _transformer(d_model, heads, 1, dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        content_tokens: torch.Tensor,
        realization_tokens: torch.Tensor,
        timbre_global: torch.Tensor,
        *,
        content_mask: torch.Tensor | None = None,
        realization_mask: torch.Tensor | None = None,
    ) -> FactorizedConditionState:
        expected = content_tokens.shape
        if content_tokens.ndim != 3 or expected[1:] != (
            self.condition_steps,
            self.d_model,
        ):
            raise ValueError("content_tokens have the wrong shape")
        if realization_tokens.shape != expected:
            raise ValueError(
                "content and realization tokens must have identical shapes"
            )
        if timbre_global.shape != (expected[0], self.d_model):
            raise ValueError("timbre_global must be [B,d_model]")
        batch = expected[0]
        if content_mask is None:
            content_valid = torch.ones(
                batch,
                self.condition_steps,
                device=content_tokens.device,
                dtype=torch.bool,
            )
        else:
            if content_mask.shape != (batch, self.condition_steps):
                raise ValueError("content_mask must be [B,condition_steps]")
            content_valid = content_mask.to(
                device=content_tokens.device, dtype=torch.bool
            )
        if realization_mask is None:
            realization_valid = torch.ones_like(content_valid)
        else:
            if realization_mask.shape != (batch, self.condition_steps):
                raise ValueError("realization_mask must be [B,condition_steps]")
            realization_valid = realization_mask.to(
                device=content_tokens.device, dtype=torch.bool
            )

        if not self.use_content_condition:
            content_valid = torch.zeros_like(content_valid)
        if not self.use_realization_condition:
            realization_valid = torch.zeros_like(realization_valid)

        if (
            self.training
            and self.branch_dropout_probability > 0
            and self.use_content_condition
            and self.use_realization_condition
        ):
            dropped = (
                torch.rand(batch, 2, device=content_tokens.device)
                < self.branch_dropout_probability
            )
            # Preserve at least one branch for every example.
            both = dropped.all(dim=1)
            dropped[both, 1] = False
            content_valid = content_valid & ~dropped[:, :1]
            realization_valid = realization_valid & ~dropped[:, 1:]
        fused_valid = content_valid | realization_valid
        if not fused_valid.any(dim=1).all():
            raise ValueError(
                "at least one factor branch must be valid for every sample"
            )
        content_value = content_tokens * content_valid.to(
            content_tokens.dtype
        ).unsqueeze(-1)
        realization_value = realization_tokens * realization_valid.to(
            realization_tokens.dtype
        ).unsqueeze(-1)
        timbre_active = (
            realization_valid.any(dim=1).to(timbre_global.dtype).view(batch, 1, 1)
        )
        timbre = (
            timbre_global.unsqueeze(1).expand(-1, self.condition_steps, -1)
            * timbre_active
        )
        base = self.factor_projection(
            torch.cat((content_value, realization_value, timbre), dim=-1)
        )
        base = self.factor_norm(
            self.factor_refiner(base, src_key_padding_mask=~fused_valid)
        )
        base = torch.where(fused_valid.unsqueeze(-1), base, torch.zeros_like(base))

        frame_tokens = F.interpolate(
            base.transpose(1, 2),
            size=self.mel_frames,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        raw_mel = self.energy_head(frame_tokens).transpose(1, 2)
        mel_range = self.mel_max_db - self.mel_min_db
        log_mel = self.mel_min_db + mel_range * torch.sigmoid(raw_mel)
        activity_logits = self.activity_head(frame_tokens).squeeze(-1)
        raw_prosody = self.prosody_head(frame_tokens)
        # F0 is represented as natural-log Hz and is non-negative. RMS uses
        # the same calibrated dBFS interval as the cached acoustic features.
        log_f0_hz = F.softplus(raw_prosody[..., 0])
        voicing_logits = raw_prosody[..., 1]
        log_rms_dbfs = self.mel_min_db + mel_range * torch.sigmoid(raw_prosody[..., 2])
        duration_fraction = torch.sigmoid(
            self.duration_head(_masked_mean(base, fused_valid)).squeeze(-1)
        )
        duration = (
            self.min_duration_seconds
            + (self.max_duration_seconds - self.min_duration_seconds)
            * duration_fraction
        )

        normalized_mel = (log_mel - self.mel_min_db) / mel_range
        energy_tokens = self.energy_input(normalized_mel.transpose(1, 2))
        energy_tokens = F.adaptive_avg_pool1d(
            energy_tokens.transpose(1, 2), self.condition_steps
        ).transpose(1, 2)
        energy_tokens = self.energy_refiner(energy_tokens)
        energy_scale = 1.0 if self.use_energy_feedback else 0.0
        fused = self.output_norm(base + energy_scale * energy_tokens)
        return FactorizedConditionState(
            fused_condition=fused,
            log_mel_energy=log_mel,
            log_f0_hz=log_f0_hz,
            voicing_logits=voicing_logits,
            log_rms_dbfs=log_rms_dbfs,
            activity_logits=activity_logits,
            duration_seconds=duration,
        )


class FactorizedAudioEncoder(nn.Module):
    """Audio-side factorizer with independent content, realization, and timbre inputs."""

    def __init__(self, cfg: FactorizedAudioConfig):
        super().__init__()
        self.cfg = cfg
        self.content_projector = AudioContentProjector(cfg)
        self.realization_encoder = AudioRealizationEncoder(cfg)
        self.timbre_projector = TimbreProjector(cfg.timbre_dimension, cfg.d_model)
        self.fusion = FactorizedConditionFusion(
            d_model=cfg.d_model,
            condition_steps=cfg.condition_steps,
            mel_bins=cfg.mel_bins,
            mel_frames=cfg.mel_frames,
            heads=cfg.heads,
            layers=cfg.fusion_layers,
            dropout=cfg.dropout,
            mel_min_db=cfg.mel_min_db,
            mel_max_db=cfg.mel_max_db,
            min_duration_seconds=cfg.min_duration_seconds,
            max_duration_seconds=cfg.max_duration_seconds,
            branch_dropout_probability=cfg.branch_dropout_probability,
            use_content_condition=cfg.use_content_condition,
            use_realization_condition=cfg.use_realization_condition,
            use_energy_feedback=cfg.use_energy_feedback,
        )

    def forward(
        self,
        content_tokens: torch.Tensor,
        content_token_mask: torch.Tensor,
        realization_features: torch.Tensor,
        realization_frame_mask: torch.Tensor,
        timbre_embedding: torch.Tensor,
    ) -> FactorizedAudioState:
        content, content_global, content_mask = self.content_projector(
            content_tokens, content_token_mask
        )
        realization, realization_global, realization_mask = self.realization_encoder(
            realization_features, realization_frame_mask
        )
        if (
            content.shape[0] != realization.shape[0]
            or content.shape[0] != timbre_embedding.shape[0]
        ):
            raise ValueError(
                "all factorized audio inputs must share the batch dimension"
            )
        timbre_global = self.timbre_projector(timbre_embedding)
        fused = self.fusion(content, realization, timbre_global)
        return FactorizedAudioState(
            content_tokens=content,
            realization_tokens=realization,
            content_global=content_global,
            realization_global=realization_global,
            timbre_global=timbre_global,
            fused_condition=fused.fused_condition,
            log_mel_energy=fused.log_mel_energy,
            log_f0_hz=fused.log_f0_hz,
            voicing_logits=fused.voicing_logits,
            log_rms_dbfs=fused.log_rms_dbfs,
            activity_logits=fused.activity_logits,
            duration_seconds=fused.duration_seconds,
            content_valid_mask=content_mask,
            realization_valid_mask=realization_mask,
        )

    def fuse(
        self,
        content_tokens: torch.Tensor,
        realization_tokens: torch.Tensor,
        timbre_global: torch.Tensor,
        *,
        content_mask: torch.Tensor | None = None,
        realization_mask: torch.Tensor | None = None,
    ) -> FactorizedConditionState:
        """Fuse arbitrary factors for counterfactual branch-swap synthesis."""

        return self.fusion(
            content_tokens,
            realization_tokens,
            timbre_global,
            content_mask=content_mask,
            realization_mask=realization_mask,
        )


class VariableLengthMaskedCodeDecoder(nn.Module):
    """MaskGIT EnCodec-token decoder with duration-controlled valid positions."""

    def __init__(self, cfg: FactorizedAudioConfig):
        super().__init__()
        self.cfg = cfg
        self.mask_id = int(cfg.vocab_size)
        self.code_embeddings = nn.ModuleList(
            [
                nn.Embedding(cfg.vocab_size + 1, cfg.d_model)
                for _ in range(cfg.codebooks)
            ]
        )
        self.codebook_embedding = nn.Parameter(
            torch.zeros(1, cfg.codebooks, 1, cfg.d_model)
        )
        self.position = nn.Parameter(torch.zeros(1, cfg.max_code_steps, cfg.d_model))
        layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.decoder_layers)
        self.output_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.vocab_size)
                )
                for _ in range(cfg.codebooks)
            ]
        )
        nn.init.normal_(self.codebook_embedding, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(
        self,
        codes: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor,
        code_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if codes.ndim != 3 or mask.shape != codes.shape:
            raise ValueError("codes and mask must share [B,Q,T]")
        batch, codebooks, steps = codes.shape
        if (
            codebooks != self.cfg.codebooks
            or steps > self.cfg.max_code_steps
            or steps < 1
        ):
            raise ValueError(
                f"codes must be [B,{self.cfg.codebooks},T<= {self.cfg.max_code_steps}]"
            )
        if condition.shape != (batch, self.cfg.condition_steps, self.cfg.d_model):
            raise ValueError("condition must be [B,condition_steps,d_model]")
        valid = _as_code_mask(codes, code_valid_mask)
        if not valid.any(dim=1).all():
            raise ValueError("every item must contain at least one valid code step")
        valid_stream = valid.unsqueeze(1).expand(-1, codebooks, -1)
        safe_codes = torch.where(valid_stream, codes, torch.zeros_like(codes))
        if ((safe_codes < 0) | (safe_codes >= self.cfg.vocab_size)).any():
            raise ValueError("valid codes must be within the codec vocabulary")
        effective_mask = mask.bool() & valid_stream
        masked = torch.where(
            effective_mask, torch.full_like(safe_codes, self.mask_id), safe_codes
        ).long()
        streams = [
            embedding(masked[:, index]) + self.codebook_embedding[:, index]
            for index, embedding in enumerate(self.code_embeddings)
        ]
        target = torch.stack(streams, dim=0).mean(dim=0) * math.sqrt(float(codebooks))
        hidden = self.decoder(
            target + self.position[:, :steps],
            condition,
            tgt_key_padding_mask=~valid,
        )
        logits = torch.stack([head(hidden) for head in self.output_heads], dim=1)
        return torch.where(valid_stream.unsqueeze(-1), logits, torch.zeros_like(logits))

    def duration_mask(self, duration_seconds: torch.Tensor) -> torch.Tensor:
        if duration_seconds.ndim != 1:
            raise ValueError("duration_seconds must be [B]")
        if not _all_finite(duration_seconds):
            raise ValueError("duration_seconds must be finite")
        lengths = torch.ceil(
            duration_seconds.clamp(
                min=self.cfg.min_duration_seconds,
                max=self.cfg.max_duration_seconds,
            )
            * self.cfg.codec_frame_rate
        ).long()
        lengths = lengths.clamp(min=1, max=self.cfg.max_code_steps)
        positions = torch.arange(
            self.cfg.max_code_steps, device=duration_seconds.device
        ).unsqueeze(0)
        return positions < lengths.unsqueeze(1)

    def valid_steps_mask(
        self,
        valid_code_steps: torch.Tensor,
        *,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert explicit code lengths or a time mask to ``[B,max_steps]``."""

        if valid_code_steps.ndim == 1:
            if valid_code_steps.shape != (batch,):
                raise ValueError("valid_code_steps lengths must be [B]")
            lengths = valid_code_steps.to(device=device).long()
            if (lengths < 1).any() or (lengths > self.cfg.max_code_steps).any():
                raise ValueError("valid code lengths must be within [1,max_code_steps]")
            positions = torch.arange(self.cfg.max_code_steps, device=device).unsqueeze(
                0
            )
            return positions < lengths.unsqueeze(1)
        if valid_code_steps.shape == (batch, self.cfg.max_code_steps):
            valid = valid_code_steps.to(device=device, dtype=torch.bool)
            if not valid.any(dim=1).all():
                raise ValueError(
                    "every sample must contain at least one valid code step"
                )
            return valid
        raise ValueError(
            "valid_code_steps must be lengths [B] or mask [B,max_code_steps]"
        )

    @torch.no_grad()
    def generate(
        self,
        condition: torch.Tensor,
        duration_seconds: torch.Tensor | None = None,
        *,
        valid_code_steps: torch.Tensor | None = None,
        steps: int | None = None,
        temperature: float | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if condition.ndim != 3 or condition.shape[1:] != (
            self.cfg.condition_steps,
            self.cfg.d_model,
        ):
            raise ValueError("condition must be [B,condition_steps,d_model]")
        if (duration_seconds is None) == (valid_code_steps is None):
            raise ValueError(
                "provide exactly one of duration_seconds or valid_code_steps"
            )
        if duration_seconds is not None and duration_seconds.shape != (
            condition.shape[0],
        ):
            raise ValueError("duration_seconds must match the condition batch")
        iterations = self.cfg.generation_steps if steps is None else int(steps)
        sampling_temperature = (
            self.cfg.generation_temperature
            if temperature is None
            else float(temperature)
        )
        if iterations < 1 or sampling_temperature < 0:
            raise ValueError(
                "generation steps must be positive and temperature non-negative"
            )
        batch = condition.shape[0]
        valid = (
            self.duration_mask(duration_seconds)
            if duration_seconds is not None
            else self.valid_steps_mask(
                valid_code_steps, batch=batch, device=condition.device  # type: ignore[arg-type]
            )
        )
        codes = torch.zeros(
            batch,
            self.cfg.codebooks,
            self.cfg.max_code_steps,
            device=condition.device,
            dtype=torch.long,
        )
        mask = valid.unsqueeze(1).expand(-1, self.cfg.codebooks, -1).clone()
        for step in range(iterations):
            logits = self(codes, mask, condition, valid)
            if sampling_temperature > 0:
                probabilities = torch.softmax(
                    logits.float() / sampling_temperature, dim=-1
                )
                proposed = torch.multinomial(
                    probabilities.reshape(-1, self.cfg.vocab_size),
                    1,
                    generator=generator,
                ).reshape_as(codes)
                confidence = probabilities.gather(-1, proposed.unsqueeze(-1)).squeeze(
                    -1
                )
            else:
                confidence, proposed = torch.softmax(logits.float(), dim=-1).max(dim=-1)
            remaining_iterations = max(1, iterations - step)
            for item in range(batch):
                remaining = torch.nonzero(
                    mask[item].reshape(-1), as_tuple=False
                ).flatten()
                if remaining.numel() == 0:
                    continue
                count = (
                    remaining.numel()
                    if remaining_iterations == 1
                    else max(1, math.ceil(remaining.numel() / remaining_iterations))
                )
                selected = remaining[
                    torch.topk(
                        confidence[item].reshape(-1)[remaining],
                        k=min(count, remaining.numel()),
                    ).indices
                ]
                flat_codes = codes[item].reshape(-1)
                flat_mask = mask[item].reshape(-1)
                flat_proposed = proposed[item].reshape(-1)
                flat_codes[selected] = flat_proposed[selected]
                flat_mask[selected] = False
        if mask.any():
            final = self(codes, mask, condition, valid).argmax(dim=-1)
            codes = torch.where(mask, final, codes)
        codes = torch.where(valid.unsqueeze(1), codes, torch.zeros_like(codes))
        return codes, valid


class FactorizedAudioModel(nn.Module):
    """Trainable audio prior: factorized teacher projection plus code decoder."""

    def __init__(self, cfg: FactorizedAudioConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = FactorizedAudioEncoder(cfg)
        self.decoder = VariableLengthMaskedCodeDecoder(cfg)

    def encode(
        self,
        content_tokens: torch.Tensor,
        content_token_mask: torch.Tensor,
        realization_features: torch.Tensor,
        realization_frame_mask: torch.Tensor,
        timbre_embedding: torch.Tensor,
    ) -> FactorizedAudioState:
        return self.encoder(
            content_tokens,
            content_token_mask,
            realization_features,
            realization_frame_mask,
            timbre_embedding,
        )

    def fuse(
        self,
        content_tokens: torch.Tensor,
        realization_tokens: torch.Tensor,
        timbre_global: torch.Tensor,
        *,
        content_mask: torch.Tensor | None = None,
        realization_mask: torch.Tensor | None = None,
    ) -> FactorizedConditionState:
        return self.encoder.fuse(
            content_tokens,
            realization_tokens,
            timbre_global,
            content_mask=content_mask,
            realization_mask=realization_mask,
        )

    def forward(
        self,
        content_tokens: torch.Tensor,
        content_token_mask: torch.Tensor,
        realization_features: torch.Tensor,
        realization_frame_mask: torch.Tensor,
        timbre_embedding: torch.Tensor,
        codes: torch.Tensor,
        code_mask: torch.Tensor,
        code_valid_mask: torch.Tensor,
        *,
        condition_dropout: torch.Tensor | None = None,
    ) -> FactorizedAudioOutput:
        state = self.encode(
            content_tokens,
            content_token_mask,
            realization_features,
            realization_frame_mask,
            timbre_embedding,
        )
        condition = state.fused_condition
        if condition_dropout is not None:
            if condition_dropout.shape != (condition.shape[0],):
                raise ValueError("condition_dropout must be [B]")
            condition = condition * (~condition_dropout.bool()).to(
                condition.dtype
            ).view(-1, 1, 1)
        logits = self.decoder(codes, code_mask, condition, code_valid_mask)
        return FactorizedAudioOutput(state=state, code_logits=logits)


class _Adapter(nn.Module):
    def __init__(self, dimension: int, bottleneck: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, bottleneck, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dimension, bias=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class AntiCollapseAdapterMoE(nn.Module):
    """Universal FFN plus soft/top-k low-rank specialists."""

    def __init__(self, cfg: FactorizedEEGConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.d_model)
        self.universal = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
        )
        self.specialists = nn.ModuleList(
            [
                _Adapter(cfg.d_model, cfg.specialist_bottleneck, cfg.dropout)
                for _ in range(cfg.specialists)
            ]
        )
        self.router = nn.Linear(cfg.d_model, cfg.specialists)

    def forward(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        epoch: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if tokens.ndim != 3 or valid_mask.shape != tokens.shape[:2]:
            raise ValueError("valid_mask must match tokens [B,T,D]")
        normalized = self.norm(tokens)
        logits = self.router(normalized)
        weights = torch.sigmoid(logits)
        if not self.cfg.adapter_moe_enabled:
            weights = torch.full_like(weights, 1.0 / float(self.cfg.specialists))
        elif self.training and self.cfg.expert_dropout > 0:
            keep = torch.rand_like(weights) >= float(self.cfg.expert_dropout)
            keep = keep | (~keep.any(dim=-1, keepdim=True)).expand_as(keep)
            weights = weights * keep.to(weights.dtype)
        if self.cfg.adapter_moe_enabled and int(epoch) >= int(
            self.cfg.soft_routing_epochs
        ):
            top = torch.topk(weights, k=self.cfg.top_k_specialists, dim=-1).indices
            active = torch.zeros_like(weights, dtype=torch.bool).scatter_(-1, top, True)
            weights = weights * active.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        specialist_outputs = torch.stack(
            [expert(normalized) for expert in self.specialists], dim=-2
        )
        specialist = (specialist_outputs * weights.unsqueeze(-1)).sum(dim=-2)
        output = tokens + self.universal(normalized) + specialist
        output = torch.where(valid_mask.unsqueeze(-1), output, torch.zeros_like(output))

        valid_weight = valid_mask.to(weights.dtype).unsqueeze(-1)
        denominator = valid_weight.sum().clamp_min(1.0)
        mass = (weights * valid_weight).sum(dim=(0, 1)) / denominator
        sample_denominator = valid_weight.sum(dim=1).clamp_min(1.0)
        sample_mass = (weights * valid_weight).sum(dim=1) / sample_denominator
        target = torch.full_like(mass, 1.0 / float(self.cfg.specialists))
        balance = (mass - target).square().mean() * self.cfg.specialists
        z_loss = (
            torch.logsumexp(logits.float(), dim=-1).square() * valid_mask
        ).sum() / valid_mask.sum().clamp_min(1)
        entropy_per_token = -(
            weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()
        ).sum(dim=-1)
        entropy = (entropy_per_token * valid_mask).sum() / valid_mask.sum().clamp_min(1)
        load = ((weights > 0).to(weights.dtype) * valid_weight).sum(
            dim=(0, 1)
        ) / denominator
        return output, {
            "specialist_mass": mass,
            "sample_specialist_mass": sample_mass,
            "specialist_load": load,
            "balance_loss": balance,
            "z_loss": z_loss.to(tokens.dtype),
            "entropy": entropy,
            "router_logits": logits,
            "router_weights": weights,
        }


class FactorizedEEGEncoder(nn.Module):
    """Variable-channel EEG encoder with independent content/realization queries."""

    def __init__(self, cfg: FactorizedEEGConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = nn.Sequential(
            nn.LayerNorm(cfg.patch_size),
            nn.Linear(cfg.patch_size, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(3, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model)
        )
        self.quality_embedding = nn.Sequential(nn.Linear(1, cfg.d_model), nn.Tanh())
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, cfg.d_model))
        self.input_norm = nn.LayerNorm(cfg.d_model)
        self.moe = AntiCollapseAdapterMoE(cfg)
        # Patch reconstruction must see other channels/times.  The MoE above
        # is token-wise, so it cannot by itself infer a masked patch from
        # unmasked context.
        self.patch_context = _transformer(cfg.d_model, cfg.heads, 1, cfg.dropout)
        self.patch_context_norm = nn.LayerNorm(cfg.d_model)

        self.content_queries = nn.Parameter(
            torch.zeros(1, cfg.condition_steps, cfg.d_model)
        )
        self.realization_queries = nn.Parameter(
            torch.zeros(1, cfg.condition_steps, cfg.d_model)
        )
        self.content_attention = nn.MultiheadAttention(
            cfg.d_model, cfg.heads, dropout=cfg.dropout, batch_first=True
        )
        self.realization_attention = nn.MultiheadAttention(
            cfg.d_model, cfg.heads, dropout=cfg.dropout, batch_first=True
        )
        self.content_refiner = _transformer(
            cfg.d_model, cfg.heads, cfg.latent_layers, cfg.dropout
        )
        self.realization_refiner = _transformer(
            cfg.d_model, cfg.heads, cfg.latent_layers, cfg.dropout
        )
        self.content_norm = nn.LayerNorm(cfg.d_model)
        self.realization_norm = nn.LayerNorm(cfg.d_model)
        self.content_projection = nn.Sequential(
            nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model)
        )
        self.realization_projection = nn.Sequential(
            nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model)
        )
        self.timbre_projection = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.fusion = FactorizedConditionFusion(
            d_model=cfg.d_model,
            condition_steps=cfg.condition_steps,
            mel_bins=cfg.mel_bins,
            mel_frames=cfg.mel_frames,
            heads=cfg.heads,
            layers=cfg.fusion_layers,
            dropout=cfg.dropout,
            mel_min_db=cfg.mel_min_db,
            mel_max_db=cfg.mel_max_db,
            min_duration_seconds=cfg.min_duration_seconds,
            max_duration_seconds=cfg.max_duration_seconds,
            branch_dropout_probability=cfg.branch_dropout_probability,
            use_content_condition=cfg.use_content_condition,
            use_realization_condition=cfg.use_realization_condition,
            use_energy_feedback=cfg.use_energy_feedback,
        )

        # Adversaries receive representations, never identifiers as inputs.
        self.subject_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.num_train_subjects),
        )
        self.dataset_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.num_datasets),
        )
        self.timbre_label_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.num_content_labels),
        )
        self.router_dataset_head = nn.Linear(cfg.specialists, cfg.num_datasets)
        self.patch_reconstruction = nn.Sequential(
            nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.patch_size)
        )
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.content_queries, std=0.02)
        nn.init.normal_(self.realization_queries, std=0.02)

    def _patches(
        self,
        eeg: torch.Tensor,
        channel_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if eeg.ndim != 3:
            raise ValueError("eeg must be [B,C,T]")
        if channel_mask.shape != eeg.shape[:2]:
            raise ValueError("channel_mask must be [B,C]")
        if time_mask.shape != (eeg.shape[0], eeg.shape[2]):
            raise ValueError("time_mask must be [B,T]")
        if eeg.shape[-1] < self.cfg.patch_size:
            raise ValueError("EEG sequence is shorter than one patch")
        masked_eeg = eeg * time_mask.to(eeg.dtype).unsqueeze(1)
        patches = masked_eeg.unfold(-1, self.cfg.patch_size, self.cfg.patch_hop)
        time_windows = time_mask.to(eeg.dtype).unfold(
            -1, self.cfg.patch_size, self.cfg.patch_hop
        )
        patch_time_valid = time_windows.mean(dim=-1) >= 0.5
        valid = channel_mask.bool().unsqueeze(-1) & patch_time_valid.bool().unsqueeze(1)
        if not valid.flatten(1).any(dim=1).all():
            raise ValueError(
                "each EEG sample needs at least one valid channel-time patch"
            )
        return patches, valid

    def fuse(
        self,
        content_tokens: torch.Tensor,
        realization_tokens: torch.Tensor,
        timbre_global: torch.Tensor,
        *,
        content_mask: torch.Tensor | None = None,
        realization_mask: torch.Tensor | None = None,
    ) -> FactorizedConditionState:
        """Fuse native or swapped factors without using identity metadata."""

        return self.fusion(
            content_tokens,
            realization_tokens,
            timbre_global,
            content_mask=content_mask,
            realization_mask=realization_mask,
        )

    def forward(
        self,
        eeg: torch.Tensor,
        channel_xyz: torch.Tensor,
        channel_mask: torch.Tensor,
        time_mask: torch.Tensor,
        *,
        epoch: int = 0,
        patch_mask: torch.Tensor | None = None,
        adversary_strength: float = 0.0,
    ) -> FactorizedEEGState:
        if channel_xyz.shape != (*eeg.shape[:2], 3):
            raise ValueError("channel_xyz must be [B,C,3]")
        if not _all_finite(eeg) or not _all_finite(channel_xyz):
            raise ValueError("EEG and channel coordinates must be finite")
        patches, patch_valid = self._patches(eeg, channel_mask, time_mask)
        patch_content = self.patch_embedding(patches)
        coordinate = self.coordinate_embedding(channel_xyz).unsqueeze(2)
        temporal = sinusoidal_positions(
            patch_content.shape[2], self.cfg.d_model, eeg.device, patch_content.dtype
        ).view(1, 1, patch_content.shape[2], self.cfg.d_model)
        quality = torch.log1p(
            torch.sqrt(patches.square().mean(dim=-1, keepdim=True) + 1e-8)
        )
        quality_tokens = self.quality_embedding(quality)
        if patch_mask is not None:
            if patch_mask.shape != patch_valid.shape:
                raise ValueError("patch_mask must be [B,C,P]")
            active_mask = patch_mask.bool() & patch_valid
            # Hide only signal-derived content.  Coordinate and temporal
            # embeddings remain available so every mask token retains its
            # channel/time identity.  Quality is also signal-derived and must
            # be removed to avoid leaking target-patch energy.
            patch_content = torch.where(
                active_mask.unsqueeze(-1),
                self.mask_token.expand_as(patch_content),
                patch_content,
            )
            quality_tokens = torch.where(
                active_mask.unsqueeze(-1),
                torch.zeros_like(quality_tokens),
                quality_tokens,
            )
        else:
            active_mask = torch.zeros_like(patch_valid)
        tokens = patch_content + coordinate + temporal + quality_tokens
        tokens = self.input_norm(tokens)
        batch, channels, patch_steps, dimension = tokens.shape
        flat = tokens.reshape(batch, channels * patch_steps, dimension)
        flat_valid = patch_valid.reshape(batch, channels * patch_steps)
        routed, router = self.moe(flat, flat_valid, epoch=epoch)
        contextual = self.patch_context(routed, src_key_padding_mask=~flat_valid)
        contextual = self.patch_context_norm(contextual)
        contextual = torch.where(
            flat_valid.unsqueeze(-1), contextual, torch.zeros_like(contextual)
        )
        shared_pool = _masked_mean(contextual, flat_valid)

        content_queries = self.content_queries.expand(batch, -1, -1)
        realization_queries = self.realization_queries.expand(batch, -1, -1)
        content, _ = self.content_attention(
            content_queries,
            contextual,
            contextual,
            key_padding_mask=~flat_valid,
            need_weights=False,
        )
        realization, _ = self.realization_attention(
            realization_queries,
            contextual,
            contextual,
            key_padding_mask=~flat_valid,
            need_weights=False,
        )
        content = self.content_norm(self.content_refiner(content))
        realization = self.realization_norm(self.realization_refiner(realization))
        content_pooled = content.mean(dim=1)
        realization_pooled = realization.mean(dim=1)
        content_global = F.normalize(self.content_projection(content_pooled), dim=-1)
        realization_global = F.normalize(
            self.realization_projection(realization_pooled), dim=-1
        )
        timbre_global = F.normalize(self.timbre_projection(realization_pooled), dim=-1)
        fused = self.fusion(content, realization, timbre_global)

        router_summary = router["sample_specialist_mass"]
        reconstruction = self.patch_reconstruction(contextual).reshape(
            batch, channels, patch_steps, self.cfg.patch_size
        )
        return FactorizedEEGState(
            content_tokens=content,
            realization_tokens=realization,
            content_global=content_global,
            realization_global=realization_global,
            timbre_global=timbre_global,
            fused_condition=fused.fused_condition,
            log_mel_energy=fused.log_mel_energy,
            log_f0_hz=fused.log_f0_hz,
            voicing_logits=fused.voicing_logits,
            log_rms_dbfs=fused.log_rms_dbfs,
            activity_logits=fused.activity_logits,
            duration_seconds=fused.duration_seconds,
            # Subject removal is intentionally restricted to content.
            subject_logits=self.subject_head(
                grad_reverse(content_global, adversary_strength)
            ),
            # Dataset removal acts on the shared trunk and router.
            dataset_logits=self.dataset_head(
                grad_reverse(shared_pool, adversary_strength)
            ),
            # Label removal is restricted to the global timbre subspace.
            timbre_label_logits=self.timbre_label_head(
                grad_reverse(timbre_global, adversary_strength)
            ),
            router_dataset_logits=self.router_dataset_head(
                grad_reverse(router_summary, adversary_strength)
            ),
            patch_reconstruction=reconstruction,
            patch_target=patches,
            patch_valid_mask=patch_valid,
            patch_mask=active_mask,
            router=router,
        )


class FactorizedEEGToSpeech(nn.Module):
    """Strict label-free inference facade with exactly four tensor inputs."""

    def __init__(
        self,
        eeg_encoder: FactorizedEEGEncoder,
        code_decoder: VariableLengthMaskedCodeDecoder,
    ):
        super().__init__()
        if eeg_encoder.cfg.d_model != code_decoder.cfg.d_model:
            raise ValueError("EEG encoder and code decoder d_model values must match")
        if eeg_encoder.cfg.condition_steps != code_decoder.cfg.condition_steps:
            raise ValueError("EEG encoder and code decoder condition_steps must match")
        if (
            eeg_encoder.cfg.max_duration_seconds
            != code_decoder.cfg.max_duration_seconds
        ):
            raise ValueError("EEG encoder and code decoder duration bounds must match")
        self.eeg_encoder = eeg_encoder
        self.code_decoder = code_decoder

    def encode(
        self,
        eeg: torch.Tensor,
        channel_xyz: torch.Tensor,
        channel_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> FactorizedEEGState:
        return self.eeg_encoder(eeg, channel_xyz, channel_mask, time_mask)

    def fuse(
        self,
        content_tokens: torch.Tensor,
        realization_tokens: torch.Tensor,
        timbre_global: torch.Tensor,
        *,
        content_mask: torch.Tensor | None = None,
        realization_mask: torch.Tensor | None = None,
    ) -> FactorizedConditionState:
        """Counterfactual factor-swap hook; no label/subject lookup is involved."""

        return self.eeg_encoder.fuse(
            content_tokens,
            realization_tokens,
            timbre_global,
            content_mask=content_mask,
            realization_mask=realization_mask,
        )

    @torch.no_grad()
    def generate(
        self,
        eeg: torch.Tensor,
        channel_xyz: torch.Tensor,
        channel_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> FactorizedGeneration:
        state = self.encode(eeg, channel_xyz, channel_mask, time_mask)
        codes, code_valid_mask = self.code_decoder.generate(
            state.fused_condition, state.duration_seconds
        )
        return FactorizedGeneration(
            codes=codes,
            code_valid_mask=code_valid_mask,
            content_tokens=state.content_tokens,
            realization_tokens=state.realization_tokens,
            content_global=state.content_global,
            realization_global=state.realization_global,
            timbre_global=state.timbre_global,
            fused_condition=state.fused_condition,
            log_mel_energy=state.log_mel_energy,
            log_f0_hz=state.log_f0_hz,
            voicing_logits=state.voicing_logits,
            log_rms_dbfs=state.log_rms_dbfs,
            activity_logits=state.activity_logits,
            duration_seconds=state.duration_seconds,
        )

    def forward(
        self,
        eeg: torch.Tensor,
        channel_xyz: torch.Tensor,
        channel_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> FactorizedEEGState:
        """Training-friendly forward; stochastic code generation stays in ``generate``."""

        return self.encode(eeg, channel_xyz, channel_mask, time_mask)


# A concise alias for scripts that name the public facade by its output role.
FactorizedGenerator = FactorizedEEGToSpeech


def random_code_mask(
    codes: torch.Tensor,
    *,
    min_ratio: float,
    max_ratio: float,
    full_mask_probability: float,
    code_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create a MaskGIT training mask without selecting padded code positions."""

    if codes.ndim != 3:
        raise ValueError("codes must be [B,Q,T]")
    if not 0.0 <= min_ratio <= max_ratio <= 1.0:
        raise ValueError("mask ratios must satisfy 0 <= min <= max <= 1")
    if not 0.0 <= full_mask_probability <= 1.0:
        raise ValueError("full_mask_probability must be in [0,1]")
    valid = _as_code_mask(codes, code_valid_mask)
    ratios = torch.empty(len(codes), 1, 1, device=codes.device).uniform_(
        min_ratio, max_ratio
    )
    if full_mask_probability > 0:
        full = torch.rand(len(codes), 1, 1, device=codes.device) < float(
            full_mask_probability
        )
        ratios = torch.where(full, torch.ones_like(ratios), ratios)
    mask = torch.rand(codes.shape, device=codes.device) < ratios
    mask &= valid.unsqueeze(1)
    for item in range(len(mask)):
        available = torch.nonzero(valid[item], as_tuple=False).flatten()
        if available.numel() and not mask[item].any():
            mask[item, :, available[0]] = True
    return mask


def random_patch_mask(valid_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    if valid_mask.ndim != 3:
        raise ValueError("valid_mask must be [B,C,P]")
    if not 0.0 <= float(ratio) <= 1.0:
        raise ValueError("patch mask ratio must be in [0,1]")
    mask = (
        torch.rand(valid_mask.shape, device=valid_mask.device) < float(ratio)
    ) & valid_mask.bool()
    for item in range(len(mask)):
        available = torch.nonzero(valid_mask[item], as_tuple=False)
        if available.numel() and not mask[item].any():
            first = available[0]
            mask[item, first[0], first[1]] = True
    return mask


__all__ = [
    "AntiCollapseAdapterMoE",
    "AudioContentProjector",
    "AudioRealizationEncoder",
    "FactorizedAudioConfig",
    "FactorizedAudioEncoder",
    "FactorizedAudioModel",
    "FactorizedAudioOutput",
    "FactorizedAudioState",
    "FactorizedConditionState",
    "FactorizedConditionFusion",
    "FactorizedEEGConfig",
    "FactorizedEEGEncoder",
    "FactorizedEEGState",
    "FactorizedEEGToSpeech",
    "FactorizedGeneration",
    "FactorizedGenerator",
    "TimbreProjector",
    "VariableLengthMaskedCodeDecoder",
    "grad_reverse",
    "random_code_mask",
    "random_patch_mask",
    "sinusoidal_positions",
]
