import torch

from src.eeg_voice_model.heads import AudioContrastiveHead, ProbeHead
from src.eeg_voice_model.tokenizer import BrainStyleEEGTokenizerConfig, BrainStyleEEGTokenizerV0


def test_tokenizer_and_heads_synthetic_forward():
    cfg = BrainStyleEEGTokenizerConfig(
        sample_rate=250,
        window_sec=2.0,
        dim=64,
        latent_queries=4,
        codebook_size=32,
        num_quantizers=2,
        encoder_channels=16,
        n_heads=4,
        dropout=0.0,
    )
    model = BrainStyleEEGTokenizerV0(cfg)
    eeg = torch.randn(2, 8, cfg.window_samples)
    sensor_pos = torch.randn(2, 8, 3)
    mask = torch.ones(2, 8, dtype=torch.bool)
    out = model(eeg, sensor_pos, mask)

    assert out["z"].shape[:2] == (2, 4)
    assert out["tokens"].shape[-1] == 2
    assert int(out["tokens"].max()) < 32
    assert out["x_rec"].shape == eeg.shape
    assert torch.isfinite(out["losses"]["loss"])

    probe = ProbeHead(cfg.dim, num_classes=3)
    probe_out = probe(out["z_q"], torch.tensor([0, 1]))
    assert probe_out["logits"].shape == (2, 3)
    assert torch.isfinite(probe_out["loss"])

    contrast = AudioContrastiveHead(cfg.dim, audio_dim=11, proj_dim=32)
    contrast_out = contrast(out["z_q"], torch.randn(2, 11))
    assert contrast_out["logits"].shape == (2, 2)
    assert torch.isfinite(contrast_out["loss"])
