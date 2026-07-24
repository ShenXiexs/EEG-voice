from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from src.open_vocab_0724.model import (
    FactorizedAudioConfig,
    FactorizedAudioModel,
    FactorizedEEGConfig,
    FactorizedEEGEncoder,
    FactorizedEEGToSpeech,
    FactorizedGeneration,
)


def _tiny_audio_model() -> FactorizedAudioModel:
    return FactorizedAudioModel(
        FactorizedAudioConfig(
            codebooks=2,
            code_steps=12,
            code_rate_hz=3.0,
            vocab_size=16,
            d_model=24,
            condition_steps=5,
            mel_bins=4,
            energy_frames=10,
            content_input_dimension=6,
            timbre_input_dimension=7,
            realization_input_dimension=8,
            audio_encoder_layers=1,
            fusion_layers=1,
            decoder_layers=1,
            heads=4,
            dropout=0.0,
            branch_dropout_probability=0.0,
            min_duration_seconds=0.1,
            max_duration_sec=4.0,
            generation_steps=2,
        )
    ).eval()


def _tiny_eeg_encoder() -> FactorizedEEGEncoder:
    return FactorizedEEGEncoder(
        FactorizedEEGConfig(
            eeg_samples=24,
            patch_size=8,
            patch_hop=8,
            d_model=24,
            condition_steps=5,
            mel_bins=4,
            mel_frames=10,
            heads=4,
            latent_layers=1,
            fusion_layers=1,
            dropout=0.0,
            specialists=2,
            specialist_bottleneck=6,
            soft_routing_epochs=2,
            top_k_specialists=1,
            expert_dropout=0.0,
            num_datasets=3,
            num_train_subjects=2,
            num_content_labels=2,
            adapter_moe_enabled=False,
            branch_dropout_probability=0.0,
        )
    ).eval()


def _eeg_inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(23)
    eeg = torch.randn(2, 3, 24, generator=generator)
    xyz = torch.randn(2, 3, 3, generator=generator)
    channel_mask = torch.ones(2, 3, dtype=torch.bool)
    time_mask = torch.ones(2, 24, dtype=torch.bool)
    return eeg, xyz, channel_mask, time_mask


def _has_nonzero_grad(value: torch.Tensor | None) -> bool:
    return value is not None and bool(
        torch.isfinite(value).all() and value.abs().sum() > 0
    )


def test_masked_patch_reconstruction_depends_on_unmasked_context() -> None:
    """The masked target is hidden while surrounding EEG changes its prediction."""

    torch.manual_seed(17)
    model = _tiny_eeg_encoder()
    eeg = torch.randn(1, 1, 24)
    channel_xyz = torch.tensor([[[0.1, -0.2, 0.3]]])
    channel_mask = torch.ones(1, 1, dtype=torch.bool)
    time_mask = torch.ones(1, 24, dtype=torch.bool)
    patch_mask = torch.tensor([[[False, True, False]]])

    changed_context = eeg.clone()
    changed_context[..., :8] = torch.linspace(-4.0, 4.0, 8)
    with torch.no_grad():
        first = model(
            eeg, channel_xyz, channel_mask, time_mask, patch_mask=patch_mask
        ).patch_reconstruction[0, 0, 1]
        second = model(
            changed_context,
            channel_xyz,
            channel_mask,
            time_mask,
            patch_mask=patch_mask,
        ).patch_reconstruction[0, 0, 1]
    assert not torch.allclose(first, second, atol=1e-6, rtol=1e-6)

    differentiable_eeg = eeg.clone().requires_grad_(True)
    prediction = model(
        differentiable_eeg,
        channel_xyz,
        channel_mask,
        time_mask,
        patch_mask=patch_mask,
    ).patch_reconstruction[0, 0, 1]
    prediction.square().sum().backward()
    assert differentiable_eeg.grad is not None
    # The prediction uses an unmasked neighbouring patch ...
    assert differentiable_eeg.grad[..., :8].abs().sum() > 1e-8
    # ... but cannot inspect the masked target patch itself.
    assert torch.count_nonzero(differentiable_eeg.grad[..., 8:16]) == 0


def test_audio_and_eeg_output_shapes_and_padding_masks() -> None:
    torch.manual_seed(29)
    audio = _tiny_audio_model()
    content = torch.randn(2, 5, 6)
    content_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    realization = torch.randn(2, 10, 8)
    realization[..., :4] = -80.0 + 80.0 * torch.rand(2, 10, 4)
    realization_mask = torch.tensor(
        [
            [True] * 10,
            [True, True, True, True, True, True, True, False, False, False],
        ]
    )
    timbre = torch.randn(2, 7)
    codes = torch.randint(0, 16, (2, 2, 12))
    code_valid = torch.tensor(
        [
            [[True] * 12, [True] * 12],
            [[True] * 8 + [False] * 4, [True] * 8 + [False] * 4],
        ]
    )
    codes = torch.where(code_valid, codes, torch.full_like(codes, -1))

    with torch.no_grad():
        output = audio(
            content,
            content_mask,
            realization,
            realization_mask,
            timbre,
            codes,
            code_valid,
            code_valid,
        )
    state = output.state
    assert state.content_tokens.shape == (2, 5, 24)
    assert state.realization_tokens.shape == (2, 5, 24)
    assert state.content_global.shape == (2, 24)
    assert state.realization_global.shape == (2, 24)
    assert state.timbre_global.shape == (2, 24)
    assert state.fused_condition.shape == (2, 5, 24)
    assert state.log_mel_energy.shape == (2, 4, 10)
    for name in (
        "log_f0_hz",
        "voicing_logits",
        "log_rms_dbfs",
        "activity_logits",
    ):
        assert getattr(state, name).shape == (2, 10)
    assert state.duration_seconds.shape == (2,)
    assert state.content_valid_mask.shape == (2, 5)
    assert state.realization_valid_mask.shape == (2, 5)
    assert output.code_logits.shape == (2, 2, 12, 16)
    assert torch.count_nonzero(output.code_logits[1, :, 8:]) == 0

    # Teacher padding cannot affect the fixed-length projected tokens.
    corrupted_content = content.clone()
    corrupted_content[1, 3:] = 1.0e6
    corrupted_realization = realization.clone()
    corrupted_realization[1, 7:] = -1.0e6
    with torch.no_grad():
        baseline = audio.encode(
            content,
            content_mask,
            realization,
            realization_mask,
            timbre,
        )
        corrupted = audio.encode(
            corrupted_content,
            content_mask,
            corrupted_realization,
            realization_mask,
            timbre,
        )
    torch.testing.assert_close(baseline.content_tokens[1], corrupted.content_tokens[1])
    torch.testing.assert_close(
        baseline.realization_tokens[1], corrupted.realization_tokens[1]
    )

    eeg_model = _tiny_eeg_encoder()
    eeg, xyz, channel_mask, time_mask = _eeg_inputs()
    channel_mask[1, 2] = False
    time_mask[1, 16:] = False
    patch_mask = torch.zeros(2, 3, 3, dtype=torch.bool)
    patch_mask[:, 0, 1] = True
    with torch.no_grad():
        eeg_state = eeg_model(
            eeg,
            xyz,
            channel_mask,
            time_mask,
            patch_mask=patch_mask,
        )
    assert eeg_state.content_tokens.shape == (2, 5, 24)
    assert eeg_state.realization_tokens.shape == (2, 5, 24)
    assert eeg_state.fused_condition.shape == (2, 5, 24)
    assert eeg_state.log_mel_energy.shape == (2, 4, 10)
    assert eeg_state.patch_target.shape == (2, 3, 3, 8)
    assert eeg_state.patch_reconstruction.shape == (2, 3, 3, 8)
    assert eeg_state.patch_valid_mask.shape == (2, 3, 3)
    assert eeg_state.patch_valid_mask.sum(dim=(1, 2)).tolist() == [9, 4]
    assert torch.equal(eeg_state.patch_mask, patch_mask & eeg_state.patch_valid_mask)


def test_duration_controls_generated_code_valid_lengths() -> None:
    torch.manual_seed(31)
    decoder = _tiny_audio_model().decoder
    condition = torch.randn(2, 5, 24)
    codes, valid = decoder.generate(
        condition,
        torch.tensor([1.0, 2.0]),
        steps=1,
        temperature=0.0,
    )
    assert codes.shape == (2, 2, 12)
    assert valid.shape == (2, 12)
    assert valid.sum(dim=1).tolist() == [3, 6]
    assert torch.count_nonzero(codes[0, :, 3:]) == 0
    assert torch.count_nonzero(codes[1, :, 6:]) == 0


def test_public_encode_and_generate_are_strictly_label_free() -> None:
    audio = _tiny_audio_model()
    eeg_model = _tiny_eeg_encoder()
    facade = FactorizedEEGToSpeech(eeg_model, audio.decoder).eval()
    expected = {"self", "eeg", "channel_xyz", "channel_mask", "time_mask"}
    for method in (FactorizedEEGToSpeech.encode, FactorizedEEGToSpeech.generate):
        signature = inspect.signature(method)
        assert set(signature.parameters) == expected
        for name, parameter in signature.parameters.items():
            if name != "self":
                assert parameter.default is inspect.Parameter.empty
    assert not expected & {"label", "subject", "dataset"}

    inputs = _eeg_inputs()
    with torch.no_grad():
        encoded = facade.encode(*inputs)
        generated = facade.generate(*inputs)
    assert encoded.fused_condition.shape == (2, 5, 24)
    assert isinstance(generated, FactorizedGeneration)
    assert generated.codes.shape == (2, 2, 12)
    with pytest.raises(TypeError):
        facade.encode(*inputs, label=torch.ones(2))  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        facade.generate(*inputs, subject=torch.ones(2))  # type: ignore[call-arg]


def test_channel_permutation_and_masked_values_are_invariant() -> None:
    torch.manual_seed(37)
    model = _tiny_eeg_encoder()
    eeg, xyz, channel_mask, time_mask = _eeg_inputs()
    with torch.no_grad():
        expected = model(eeg, xyz, channel_mask, time_mask)
        permutation = torch.tensor([2, 0, 1])
        permuted = model(
            eeg[:, permutation],
            xyz[:, permutation],
            channel_mask[:, permutation],
            time_mask,
        )
    for name in ("content_tokens", "realization_tokens", "fused_condition"):
        torch.testing.assert_close(
            getattr(expected, name), getattr(permuted, name), atol=3e-5, rtol=3e-5
        )

    masked_channels = channel_mask.clone()
    masked_channels[:, -1] = False
    truncated_time = time_mask.clone()
    truncated_time[:, 16:] = False
    corrupted_eeg = eeg.clone()
    corrupted_eeg[:, -1] = 1.0e6
    corrupted_eeg[:, :, 16:] = -1.0e6
    corrupted_xyz = xyz.clone()
    corrupted_xyz[:, -1] = 1.0e6
    with torch.no_grad():
        baseline = model(eeg, xyz, masked_channels, truncated_time)
        corrupted = model(
            corrupted_eeg,
            corrupted_xyz,
            masked_channels,
            truncated_time,
        )
    for name in (
        "content_tokens",
        "realization_tokens",
        "fused_condition",
        "log_mel_energy",
    ):
        torch.testing.assert_close(
            getattr(baseline, name), getattr(corrupted, name), atol=2e-5, rtol=2e-5
        )


def test_adversary_gradients_are_restricted_to_the_intended_paths() -> None:
    torch.manual_seed(41)
    model = _tiny_eeg_encoder().train()
    eeg, xyz, channel_mask, time_mask = _eeg_inputs()
    targets = torch.tensor([0, 1])

    model.zero_grad(set_to_none=True)
    state = model(
        eeg,
        xyz,
        channel_mask,
        time_mask,
        adversary_strength=1.0,
    )
    F.cross_entropy(state.subject_logits, targets).backward()
    assert _has_nonzero_grad(model.content_queries.grad)
    assert _has_nonzero_grad(model.content_projection[1].weight.grad)
    assert model.realization_queries.grad is None
    assert all(
        parameter.grad is None for parameter in model.timbre_projection.parameters()
    )

    model.zero_grad(set_to_none=True)
    state = model(
        eeg,
        xyz,
        channel_mask,
        time_mask,
        adversary_strength=1.0,
    )
    F.cross_entropy(state.timbre_label_logits, targets).backward()
    assert _has_nonzero_grad(model.realization_queries.grad)
    assert any(
        _has_nonzero_grad(parameter.grad)
        for parameter in model.timbre_projection.parameters()
    )
    assert model.content_queries.grad is None
    assert all(
        parameter.grad is None for parameter in model.content_projection.parameters()
    )

    model.zero_grad(set_to_none=True)
    state = model(
        eeg,
        xyz,
        channel_mask,
        time_mask,
        adversary_strength=1.0,
    )
    F.cross_entropy(state.dataset_logits, targets).backward()
    assert _has_nonzero_grad(model.patch_embedding[1].weight.grad)
    assert any(
        _has_nonzero_grad(parameter.grad)
        for parameter in model.patch_context.parameters()
    )
    assert model.content_queries.grad is None
    assert model.realization_queries.grad is None


def test_same_parameter_ablation_masks_remove_disabled_generation_branch() -> None:
    torch.manual_seed(43)
    base = _tiny_audio_model().cfg
    content = torch.randn(2, 5, 6)
    content_mask = torch.ones(2, 5, dtype=torch.bool)
    realization = torch.randn(2, 10, 8)
    realization[..., :4] = -80.0 + 80.0 * torch.rand(2, 10, 4)
    realization_mask = torch.ones(2, 10, dtype=torch.bool)
    timbre = torch.randn(2, 7)

    content_only = FactorizedAudioModel(
        replace(base, use_realization_condition=False)
    ).eval()
    with torch.no_grad():
        first = content_only.encode(
            content, content_mask, realization, realization_mask, timbre
        )
        second = content_only.encode(
            content,
            content_mask,
            realization + 100.0,
            realization_mask,
            timbre - 100.0,
        )
    torch.testing.assert_close(first.fused_condition, second.fused_condition)
    torch.testing.assert_close(first.log_mel_energy, second.log_mel_energy)

    realization_only = FactorizedAudioModel(
        replace(base, use_content_condition=False)
    ).eval()
    with torch.no_grad():
        first = realization_only.encode(
            content, content_mask, realization, realization_mask, timbre
        )
        second = realization_only.encode(
            content + 100.0,
            content_mask,
            realization,
            realization_mask,
            timbre,
        )
    torch.testing.assert_close(first.fused_condition, second.fused_condition)
