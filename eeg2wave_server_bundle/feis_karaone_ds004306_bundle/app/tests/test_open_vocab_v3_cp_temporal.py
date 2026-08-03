from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from src.open_vocab_v3.cp_temporal import (
    AudioCPEncoder, ContentMFCCDecoder, DeterministicAcousticBackbone,
    EEGCPEncoder, ResidualCVAE, global_clip_loss, horizontal_diagnostics,
    local_ot_clip_loss,
)
from src.open_vocab_v3.data import PreparedRecords, _fit_internal_dev_mask
from src.open_vocab_v3.runtime import checkpoint_schema, load_config, output_path


APP = Path(__file__).resolve().parents[1]


def small_audio() -> AudioCPEncoder:
    return AudioCPEncoder(
        embedding_dimension=8, dimension=32, heads=4, stem_layers=1,
        branch_layers=1, token_steps=96, acoustic_frames=161, dropout=0.0,
    )


def test_cp_temporal_config_has_an_independent_161_frame_schema():
    path, cfg = load_config(APP / "configs" / "open_vocab_v3_cp_temporal_large_v1.yaml")
    assert cfg["version"] == "openvoice-v3-cp-temporal-large-v1"
    assert cfg["audio"]["canonical_frames"] == 161
    assert cfg["audio"]["native_mel_frames"] == 161
    assert cfg["audio"]["content_tokens"] == 96
    assert "open_vocab_v3_cp_temporal_large_v1" in str(output_path(path, cfg, "prepared_cache"))
    assert checkpoint_schema(cfg, "content").endswith("content-v1")
    assert checkpoint_schema(cfg, "fit").endswith("fit-v1")


def test_codebook_order_is_preserved_and_192_becomes_96():
    torch.manual_seed(1)
    model = small_audio().eval()
    codes = torch.randint(0, 1024, (2, 8, 192))
    mask = torch.ones(2, 192, dtype=torch.bool)
    first = model(codes, mask)
    swapped = model(codes[:, torch.tensor([1, 0, 2, 3, 4, 5, 6, 7])], mask)
    assert first.local.shape == (2, 96, 32)
    assert first.p_base.shape == (2, 161, 3)
    assert first.p_plus.shape == (2, 161, 2)
    assert not torch.allclose(first.local, swapped.local)


def test_global_loss_does_not_update_local_head():
    model = small_audio()
    codes = torch.randint(0, 1024, (3, 8, 192))
    state = model(codes, torch.ones(3, 192, dtype=torch.bool))
    right = torch.randn_like(state.global_embedding)
    loss = global_clip_loss(state.global_embedding, right, ["a", "a", "b"], torch.tensor(0.0))
    loss.backward()
    assert all(parameter.grad is None for parameter in model.local_head.parameters())
    assert any(parameter.grad is not None for parameter in model.stem.parameters())


def test_local_ot_ignores_padding_and_keeps_trial_diagonal():
    torch.manual_seed(4)
    left = torch.randn(3, 8, 12)
    right = left.clone()
    mask = torch.zeros(3, 8, dtype=torch.bool); mask[:, :5] = True
    first, scores = local_ot_clip_loss(left, right, torch.tensor(0.0), mask, mask)
    corrupted = right.clone(); corrupted[:, 5:] = 1000.0
    second, changed_scores = local_ot_clip_loss(left, corrupted, torch.tensor(0.0), mask, mask)
    assert torch.allclose(first, second)
    assert torch.allclose(scores, changed_scores)
    assert torch.equal(scores.argmax(1), torch.arange(3))


def test_c0_depends_only_on_p_and_attention_is_auditable():
    decoder = ContentMFCCDecoder(dimension=32, heads=4, layers=1, token_steps=96, frames=161, dropout=0.0).eval()
    local = torch.randn(2, 96, 32)
    mask = torch.ones(2, 96, dtype=torch.bool)
    p1 = torch.randn(2, 161, 3)
    p2 = p1 + 1.0
    content1, full1, diagnostics = decoder(local, mask, p1, torch.tensor([0.8, 1.0]))
    content2, full2, _ = decoder(local, mask, p2, torch.tensor([0.8, 1.0]))
    assert content1.shape == (2, 39, 161)
    assert full1.shape == (2, 40, 161)
    assert torch.allclose(content1, content2)
    assert not torch.allclose(full1[:, 0], full2[:, 0])
    assert diagnostics["attention"].shape == (2, 161, 96)
    assert torch.isfinite(diagnostics["coverage"]).all()
    assert torch.isfinite(diagnostics["entropy"]).all()


def test_eeg_forward_has_no_label_text_audio_or_voice_input():
    signature = inspect.signature(EEGCPEncoder.forward)
    assert tuple(signature.parameters) == ("self", "eeg", "channel_xyz", "channel_mask", "time_mask")
    model = EEGCPEncoder(dimension=32, heads=4, layers=1, token_steps=96, acoustic_frames=161, dropout=0.0)
    state = model(torch.randn(2, 4, 128), torch.randn(2, 4, 3), torch.ones(2, 4, dtype=torch.bool), torch.ones(2, 128, dtype=torch.bool))
    assert state.local.shape == (2, 96, 32)
    assert state.p_base.shape == (2, 161, 3)


def test_residual_cvae_is_zero_initialized_and_capped():
    backbone = DeterministicAcousticBackbone(dimension=32, blocks=2, include_p_plus=True, dropout=0.0)
    model = ResidualCVAE(backbone, dimension=32, global_latent=8, local_latent=4, local_steps=8, residual_limit=0.8)
    content = torch.randn(2, 39, 161)
    p = torch.randn(2, 161, 3)
    voice = torch.randn(2, 192)
    plus = torch.randn(2, 161, 2)
    output = model(content, p, voice, plus, stochastic=True)
    assert torch.count_nonzero(output["residual"]) == 0
    with torch.no_grad():
        model.decoder[-1].weight.fill_(10.0)
    output = model(content, p, voice, plus, stochastic=True)
    assert float(output["residual"].detach().abs().max()) <= 0.800001


def test_horizontal_detector_rejects_static_stripes():
    rng = np.random.default_rng(3)
    target = rng.normal(size=(4, 80, 161))
    stripes = np.repeat(target.mean(-1, keepdims=True), 161, axis=-1)
    collapsed = horizontal_diagnostics(stripes, target)
    healthy = horizontal_diagnostics(target, target)
    assert collapsed["collapsed"]
    assert not healthy["collapsed"]


def test_fit_internal_dev_is_deterministic_and_stays_inside_fit():
    roles = np.asarray(["fit"] * 20 + ["subject_holdout_seen"] * 5)
    eligible = np.ones(25, dtype=bool)
    subjects = np.asarray(["s1"] * 10 + ["s2"] * 10 + ["s3"] * 5)
    labels = np.asarray(["a"] * 5 + ["b"] * 5 + ["a"] * 5 + ["b"] * 5 + ["a"] * 5)
    keys = np.asarray([f"k{i}" for i in range(25)])
    one = _fit_internal_dev_mask(roles, eligible, subjects, labels, keys, seed=31)
    two = _fit_internal_dev_mask(roles, eligible, subjects, labels, keys, seed=31)
    assert np.array_equal(one, two)
    assert one[:20].sum() == 4
    assert not one[20:].any()
