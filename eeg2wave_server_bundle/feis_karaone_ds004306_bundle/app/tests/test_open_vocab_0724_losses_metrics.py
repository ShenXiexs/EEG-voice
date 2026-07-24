from __future__ import annotations

import numpy as np
import pytest
import torch

from src.open_vocab_0724.audio_features import (
    AcousticFeatureConfig,
    extract_acoustic_features,
)
from src.open_vocab_0724.losses import (
    content_positive_weights,
    cross_covariance_loss,
    energy_structure_loss,
    exact_realization_clip_loss,
    masked_symmetric_multi_positive_clip_loss,
    monotonic_local_alignment_loss,
    prosody_activity_duration_loss,
    soft_dtw_divergence_torch,
    supervision_routing,
)
from src.open_vocab_0724.metrics import (
    crop_active_energy,
    detect_active_region,
    energy_structure_metrics,
    foreground_weighted_ssim,
    log_mel,
    reconstruction_metrics,
    soft_dtw_divergence,
    soft_iou,
    time_normalize_energy,
)


def test_dataset_supervision_is_conservative_and_horizon_aware() -> None:
    routing = supervision_routing(
        ["karaone", "karaone", "feis", "ds004306"],
        [
            "karaone_same_trial_overt",
            "karaone_same_trial_overt",
            "feis_subject_label",
            "weak_category_level",
        ],
        duration_seconds=torch.tensor([2.0, 4.5, 1.0, 1.0]),
    )
    # Long KaraOne audio still teaches content, but cannot supervise a truncated
    # realization/code target.
    assert routing.content.tolist() == [True, True, True, False]
    assert routing.exact_realization.tolist() == [True, False, False, False]
    assert routing.energy.tolist() == [True, False, False, False]
    assert routing.codec.tolist() == [True, False, False, False]
    assert routing.weak_timbre.tolist() == [False, False, True, False]
    assert routing.timbre.tolist() == [True, False, True, False]
    assert routing.eeg_self_supervised.tolist() == [True, True, True, True]
    assert routing.audio_generation_eligible.tolist() == [True, False, False, False]


def test_masked_multi_positive_content_clip_uses_all_same_label_positives() -> None:
    audio = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    good_eeg = audio.clone().requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 9])
    eligible = torch.tensor([True, True, True, False])
    positives = content_positive_weights(labels, eligible)
    good = masked_symmetric_multi_positive_clip_loss(
        good_eeg,
        audio,
        positives,
        eeg_eligible=eligible,
        audio_eligible=eligible,
    )["total"]
    collapsed = masked_symmetric_multi_positive_clip_loss(
        torch.ones_like(audio),
        audio,
        positives,
        eeg_eligible=eligible,
        audio_eligible=eligible,
    )["total"]
    assert good < collapsed
    good.backward()
    assert good_eeg.grad is not None
    assert torch.isfinite(good_eeg.grad).all()

    # An arbitrarily corrupted ineligible row must not change the objective.
    corrupted_audio = audio.clone()
    corrupted_audio[-1] = 1e8
    observed = masked_symmetric_multi_positive_clip_loss(
        audio,
        corrupted_audio,
        positives,
        eeg_eligible=eligible,
        audio_eligible=eligible,
    )["total"]
    expected = masked_symmetric_multi_positive_clip_loss(
        audio[:3],
        audio[:3],
        positives[:3, :3],
    )["total"]
    assert observed == pytest.approx(expected.item(), abs=1e-6)


def test_exact_realization_clip_keeps_same_content_utterances_as_negatives() -> None:
    audio = torch.eye(3)
    eligible = torch.ones(3, dtype=torch.bool)
    paired = exact_realization_clip_loss(audio.clone(), audio, eligible)["total"]
    collapsed_to_content = exact_realization_clip_loss(
        torch.ones_like(audio), audio, eligible
    )["total"]
    assert paired < collapsed_to_content
    masked = exact_realization_clip_loss(
        audio, audio, torch.tensor([True, True, False])
    )
    assert masked["active_eeg_rows"].item() == 2


def test_local_alignment_supports_different_lengths_and_token_masks() -> None:
    generator = torch.Generator().manual_seed(7)
    audio = torch.randn(2, 7, 8, generator=generator)
    eeg = torch.nn.functional.interpolate(
        audio.transpose(1, 2), size=5, mode="linear", align_corners=True
    ).transpose(1, 2)
    result = monotonic_local_alignment_loss(
        eeg,
        audio,
        torch.tensor([True, False]),
        eeg_token_mask=torch.tensor([[True] * 5, [True] * 5]),
        audio_token_mask=torch.tensor([[True] * 6 + [False], [True] * 7]),
    )
    assert torch.isfinite(result["total"])
    assert result["active_samples"].item() == 1


def test_soft_dtw_divergence_is_nonnegative_self_corrected_and_differentiable() -> None:
    x = torch.linspace(0, 1, 24).view(1, 24, 1).requires_grad_(True)
    same = soft_dtw_divergence_torch(x, x, band_ratio=0.25)
    different = soft_dtw_divergence_torch(x, 1.0 - x.detach(), band_ratio=0.25)
    assert same.item() == pytest.approx(0.0, abs=1e-7)
    assert different.item() > same.item()
    different.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    numpy_x = x.detach().numpy()[0, :, 0]
    assert soft_dtw_divergence(numpy_x, numpy_x) == pytest.approx(0.0, abs=1e-10)
    assert soft_dtw_divergence(numpy_x, 1.0 - numpy_x) > 0.0


def test_cross_covariance_penalty_and_factorized_structure_helpers_backpropagate() -> (
    None
):
    generator = torch.Generator().manual_seed(11)
    content = torch.randn(32, 6, generator=generator, requires_grad=True)
    dependent = content + 0.01 * torch.randn(32, 6, generator=generator)
    permuted = dependent[torch.randperm(32, generator=generator)]
    assert cross_covariance_loss(content, dependent) > cross_covariance_loss(
        content, permuted
    )

    target = torch.linspace(-80, 0, 24).view(1, 1, 24).expand(2, 8, 24)
    prediction = target.clone().requires_grad_(True)
    frame_mask = torch.ones(2, 24, dtype=torch.bool)
    frame_mask[:, -4:] = False
    prediction_with_masked_error = (
        prediction + (~frame_mask[:, None, :]).to(prediction) * 1000.0
    )
    energy = energy_structure_loss(
        prediction_with_masked_error, target, frame_mask, soft_dtw_weight=0.1
    )
    assert energy["log_mel_l1"].item() == pytest.approx(0.0, abs=1e-7)
    assert energy["soft_dtw"].item() == pytest.approx(0.0, abs=1e-7)
    energy["total"].backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_prosody_activity_duration_helper_masks_ineligible_samples() -> None:
    shape = (2, 6)
    zeros = torch.zeros(shape)
    activity_logits = torch.zeros(shape, requires_grad=True)
    result = prosody_activity_duration_loss(
        zeros,
        zeros,
        zeros,
        torch.ones(shape, dtype=torch.bool),
        zeros,
        zeros,
        activity_logits,
        torch.ones(shape),
        torch.tensor([1.0, 99.0]),
        torch.tensor([1.0, 1.0]),
        torch.ones(shape, dtype=torch.bool),
        torch.tensor([True, False]),
    )
    assert result["duration"].item() == pytest.approx(0.0)
    result["total"].backward()
    assert activity_logits.grad is not None
    assert activity_logits.grad[0].abs().sum() > 0
    assert activity_logits.grad[1].abs().sum() == 0


def _two_burst_waveform(sample_rate: int = 16000) -> np.ndarray:
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    first = ((time >= 0.20) & (time < 0.48)) * np.sin(2 * np.pi * 220 * time)
    second = ((time >= 0.58) & (time < 0.82)) * 0.65 * np.sin(2 * np.pi * 330 * time)
    return (first + second).astype(np.float32)


def _time_stretch(value: np.ndarray, factor: float) -> np.ndarray:
    target_length = round(len(value) * factor)
    return np.interp(
        np.linspace(0.0, 1.0, target_length),
        np.linspace(0.0, 1.0, len(value)),
        value,
    ).astype(np.float32)


def test_log_mel_active_crop_and_time_only_normalization() -> None:
    sample_rate = 16000
    waveform = _two_burst_waveform(sample_rate)
    energy = log_mel(waveform, sample_rate)
    assert energy.shape == (80, 100)
    assert float(energy.min()) >= -80.0
    assert float(energy.max()) <= 0.0
    start, end = detect_active_region(waveform, sample_rate)
    assert 0 < start < round(0.20 * sample_rate)
    assert round(0.82 * sample_rate) < end < len(waveform)
    active = crop_active_energy(energy)
    normalized = time_normalize_energy(active, 128)
    assert normalized.shape == (80, 128)  # frequency bins are untouched


def test_metric_log_mel_matches_cache_v2_acoustic_teacher() -> None:
    sample_rate = 16000
    waveform = _two_burst_waveform(sample_rate)
    metric_energy = log_mel(waveform, sample_rate)
    teacher_energy = extract_acoustic_features(
        waveform,
        valid_samples=len(waveform),
        config=AcousticFeatureConfig(max_frames=100),
    ).log_mel_energy
    assert metric_energy.shape == teacher_energy.shape == (80, 100)
    assert np.allclose(metric_energy, teacher_energy, atol=3e-4)


def test_morphology_is_time_scale_robust_but_frequency_sensitive() -> None:
    sample_rate = 16000
    reference_audio = _two_burst_waveform(sample_rate)
    stretched_audio = _time_stretch(reference_audio, 1.25)
    shifted_audio = np.pad(reference_audio, (4000, 0))
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    wrong_frequency = (
        ((time >= 0.20) & (time < 0.48)) * np.sin(2 * np.pi * 660 * time)
        + ((time >= 0.58) & (time < 0.82)) * 0.65 * np.sin(2 * np.pi * 880 * time)
    ).astype(np.float32)
    silence = np.zeros_like(reference_audio)

    reference = crop_active_energy(log_mel(reference_audio, sample_rate))
    stretched = crop_active_energy(log_mel(stretched_audio, sample_rate))
    shifted = crop_active_energy(log_mel(shifted_audio, sample_rate))
    shifted_frequency = crop_active_energy(log_mel(wrong_frequency, sample_rate))
    silent_energy = crop_active_energy(log_mel(silence, sample_rate))
    compressed_frames = round(reference.shape[1] * 0.75)
    compressed = np.stack(
        [
            np.interp(
                np.linspace(0.0, 1.0, compressed_frames),
                np.linspace(0.0, 1.0, reference.shape[1]),
                row,
            )
            for row in reference
        ]
    )
    stretch_ssim = foreground_weighted_ssim(reference, stretched)
    frequency_ssim = foreground_weighted_ssim(reference, shifted_frequency)
    assert stretch_ssim > 0.80
    assert foreground_weighted_ssim(reference, compressed) > 0.80
    assert foreground_weighted_ssim(reference, shifted) > 0.80
    assert stretch_ssim > frequency_ssim
    assert soft_iou(reference, stretched) > soft_iou(reference, shifted_frequency)
    assert foreground_weighted_ssim(reference, silent_energy) < 0.10


def test_reconstruction_metrics_keep_raw_duration_and_exact_identity() -> None:
    sample_rate = 16000
    waveform = _two_burst_waveform(sample_rate)
    identical = reconstruction_metrics(waveform, waveform.copy(), sample_rate)
    assert identical["waveform_correlation"] == pytest.approx(1.0, abs=1e-6)
    assert identical["energy_morphology_ssim"] == pytest.approx(1.0, abs=1e-6)
    assert identical["energy_soft_dtw_divergence"] == pytest.approx(0.0, abs=1e-9)
    assert identical["soft_dtw_divergence"] == pytest.approx(0.0, abs=1e-9)
    assert identical["log_mel_mae_db"] == pytest.approx(0.0, abs=1e-6)
    assert identical["raw_duration_error_seconds"] == pytest.approx(0.0)

    stretched = reconstruction_metrics(
        waveform, _time_stretch(waveform, 1.25), sample_rate
    )
    assert stretched["raw_duration_error_seconds"] == pytest.approx(0.25, abs=1e-4)
    assert stretched["energy_stretch_factor"] > 1.15
    assert stretched["energy_morphology_ssim"] > 0.80


def test_energy_structure_metric_reports_native_and_scaled_views() -> None:
    base = np.full((80, 20), -80.0)
    base[10:20, 3:17] = -10.0
    stretched = np.repeat(base, 2, axis=1)
    result = energy_structure_metrics(base, stretched, hop_seconds=0.01)
    assert result["stretch_factor"] == pytest.approx(2.0)
    assert result["active_duration_error_seconds"] == pytest.approx(0.14)
    assert result["time_normalized_log_mel_mae_db"] == pytest.approx(0.0, abs=1e-6)
    assert result["morphology_ssim"] == pytest.approx(1.0, abs=1e-6)
    assert result["native_log_mel_mae_db"] > 0.0
