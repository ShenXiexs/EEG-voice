from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch
from scipy.fft import idct

from src.open_vocab_v3.metrics import (
    cvae_audio_loss,
    fit_loss,
    overfit_loss,
    paired_r_at_1_above_chance,
    retrieval,
    variance_ratio,
)
from src.open_vocab_v3.data import channel_shuffled_eeg, time_shuffled_eeg
from src.open_vocab_v3.denoise import envelope_lag_ms, resample_waveform
from src.open_vocab_v3.model import AnalyticMFCCToMel, EEGMFCCEncoder, MFCCMelDecoder
from src.open_vocab_v3.runtime import load_config


APP = Path(__file__).resolve().parents[1]


def test_v3_config_has_a_new_artifact_firewall_and_fixed_content_contract() -> None:
    _, cfg = load_config(APP / "configs" / "open_vocab_v3_mfcc_training_first.yaml")
    assert cfg["paths"]["output_root"].endswith("open_vocab_v3_mfcc_training_first")
    assert cfg["audio"]["mfcc_bins"] == 40
    assert cfg["audio"]["canonical_frames"] == 256
    assert cfg["audio"]["max_active_seconds"] == 2.56
    assert cfg["paths"]["micro_checkpoint"] != cfg["paths"]["fit_checkpoint"]
    assert "locked_unseen_report" in cfg["paths"]
    assert cfg["model"]["audio_latent_dimension"] > 0
    assert cfg["training"]["canonical_voice_dropout"] == 0.0
    assert cfg["denoise"]["processing_sample_rate"] == 48000
    assert "training_review" in cfg["paths"]


def test_v3_eeg_path_has_no_label_text_or_speaker_forward_input() -> None:
    signature = inspect.signature(EEGMFCCEncoder.forward)
    assert tuple(signature.parameters) == ("self", "eeg", "channel_xyz", "channel_mask", "time_mask")
    model = EEGMFCCEncoder(dimension=32, heads=4, layers=1, dropout=0.0)
    eeg = torch.randn(2, 3, 64)
    xyz = torch.randn(2, 3, 3)
    channel_mask = torch.tensor([[True, True, True], [True, True, False]])
    time_mask = torch.ones(2, 64, dtype=torch.bool)
    mfcc, tokens = model(eeg, xyz, channel_mask, time_mask)
    assert mfcc.shape == (2, 40, 256)
    assert tokens.shape == (2, 16, 40)


def test_v3_losses_match_the_preregistered_weights_and_backpropagate() -> None:
    predicted = torch.randn(4, 40, 256, requires_grad=True)
    target = torch.randn(4, 40, 256)
    tokens = torch.randn(4, 16, 40, requires_grad=True)
    scale = torch.tensor(0.0, requires_grad=True)
    micro, _ = overfit_loss(predicted, target)
    full, components = fit_loss(predicted, target, tokens, ["a", "a", "b", "b"], scale)
    assert torch.isfinite(micro)
    assert torch.isfinite(full)
    assert set(components) == {"mfcc_l1", "delta_l1", "token_clip", "global_clip"}
    (micro + full).backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()


def test_strict_trial_retrieval_does_not_treat_same_label_trials_as_the_same_answer() -> None:
    target = np.zeros((4, 40, 8), dtype=np.float32)
    target[1] += 1.0
    target[2] += 2.0
    target[3] += 3.0
    labels = ["a", "a", "b", "b"]
    result = retrieval(target.copy(), target, labels, ["a0", "a1", "b0", "b1"])
    assert result["label_top1"] == 1.0
    assert result["paired_r_at_1"] == 1.0
    assert result["paired_rank_per_trial"] == [1, 1, 1, 1]
    assert result["chance_within_label"] == 0.5
    bootstrap = paired_r_at_1_above_chance(target, target, labels, samples=100, seed=7)
    assert bootstrap["mean_gain_over_chance"] == 0.5
    assert bootstrap["ci_low"] > 0.0


def test_between_trial_variance_rejects_a_repeated_rich_template() -> None:
    rng = np.random.default_rng(11)
    target = rng.normal(size=(6, 40, 12)).astype(np.float32)
    labels = ["a", "a", "a", "b", "b", "b"]
    collapsed = target.copy()
    collapsed[:3] = collapsed[0]
    collapsed[3:] = collapsed[3]
    assert variance_ratio(collapsed, target, labels) < 1.0e-10
    assert np.isclose(variance_ratio(target, target, labels), 1.0)


def test_audio_oracle_accepts_mfcc_and_an_explicit_voice_only() -> None:
    model = MFCCMelDecoder(dimension=32, voice_dim=192, latent_dim=8)
    mfcc=torch.randn(2,40,256);voice=torch.randn(2,192);mean=torch.randn(2,40);std=torch.rand(2,40)+0.1
    mel = model(mfcc,voice,mean,std)
    assert mel.shape == (2, 80, 256)
    assert float(mel.detach().min()) >= -80.0
    assert float(mel.detach().max()) <= 0.0
    posterior=model.reconstruct(mfcc,voice,mean,std,torch.empty_like(mel).uniform_(-80,0),stochastic=False)
    assert posterior["posterior_mean"].shape==(2,8)
    assert posterior["mel"].shape==(2,80,256)


def test_fixed_analytic_mfcc_backend_matches_scipy_inverse_dct() -> None:
    rng=np.random.default_rng(12);mfcc=rng.normal(size=(2,40,17)).astype(np.float32);mean=rng.normal(size=(2,40)).astype(np.float32);std=(rng.random((2,40))+0.1).astype(np.float32)
    backend=AnalyticMFCCToMel(mfcc_bins=40,mel_bins=80)
    actual=backend(torch.from_numpy(mfcc),torch.from_numpy(mean),torch.from_numpy(std)).numpy()
    expected=[]
    for index in range(2):
        restored=mfcc[index]*std[index,:,None]+mean[index,:,None]
        padded=np.pad(restored,((0,40),(0,0)))
        expected.append(np.clip(idct(padded,type=2,axis=0,norm="ortho"),-80,0))
    assert np.allclose(actual,np.stack(expected),atol=1.0e-5)


def test_cvae_audio_loss_trains_posterior_and_audio_free_prior() -> None:
    model=MFCCMelDecoder(dimension=32,voice_dim=192,latent_dim=8)
    mfcc=torch.randn(3,40,32);voice=torch.randn(3,192);mean=torch.randn(3,40);std=torch.rand(3,40)+0.1;target=torch.empty(3,80,32).uniform_(-80,0)
    values=model.distributions(mfcc,voice,mean,std,target)
    posterior=model.decode(values["analytic_mel"],values["content_hidden"],values["voice_hidden"],model._sample(values["posterior_mean"],values["posterior_logvar"],True))
    prior=model.decode(values["analytic_mel"],values["content_hidden"],values["voice_hidden"],values["prior_mean"])
    loss,parts=cvae_audio_loss(posterior,prior,values["analytic_mel"],target,values["posterior_mean"],values["posterior_logvar"],values["prior_mean"],values["prior_logvar"],kl_beta=.01,free_bits=.05,prior_weight=.35,analytic_consistency_weight=.05)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts)=={"posterior_mel","prior_mel","kl","kl_beta","analytic_residual"}
    assert model.prior[-1].weight.grad is not None


def test_shuffled_controls_preserve_padding_but_change_valid_eeg() -> None:
    eeg = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    time_mask = torch.tensor([[True, True, True, True, False], [True, True, False, False, False]])
    channel_mask = torch.tensor([[True, True, True], [True, True, False]])
    shuffled_time = time_shuffled_eeg(eeg, time_mask)
    shuffled_channel = channel_shuffled_eeg(eeg, channel_mask)
    assert not torch.equal(shuffled_time[0, :, :4], eeg[0, :, :4])
    assert torch.equal(shuffled_time[0, :, 4], eeg[0, :, 4])
    assert not torch.equal(shuffled_channel[0], eeg[0])
    assert torch.equal(shuffled_channel[1, 2], eeg[1, 2])


def test_denoise_alignment_utilities_preserve_duration_and_detect_delay() -> None:
    rate=16000;t=np.arange(rate,dtype=np.float32)/rate;wave=(0.2*np.sin(2*np.pi*180*t)*(1+0.5*np.sin(2*np.pi*3*t))).astype(np.float32)
    roundtrip=resample_waveform(resample_waveform(wave,rate,48000),48000,rate)
    assert abs(len(roundtrip)-len(wave))<=2
    shifted=np.pad(wave,(160,0))[:len(wave)]
    assert 0.0<=envelope_lag_ms(wave,shifted,rate)<=20.0
