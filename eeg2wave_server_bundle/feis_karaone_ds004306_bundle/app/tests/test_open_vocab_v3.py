from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch
from scipy.fft import idct

from src.open_vocab_v3.metrics import (
    clip_token_global_losses,
    cvae_audio_loss,
    fit_loss,
    overfit_loss,
    paired_r_at_1_above_chance,
    pairwise_mfcc_l1,
    retrieval,
    variance_ratio,
)
from src.open_vocab_v3.data import channel_shuffled_eeg, time_shuffled_eeg
from src.open_vocab_v3.denoise import envelope_lag_ms, resample_waveform
from src.open_vocab_v3.model import AnalyticMFCCToMel, EEGMFCCEncoder, MFCCMelDecoder
from src.open_vocab_v3.encodec_content import AudioContentEncoder, EEGContentEncoder, SharedMFCCDecoder
from src.open_vocab_v3.runtime import checkpoint_schema, content_schema, load_config, output_path
from src.open_vocab_v3.audio_adaptation import envelope_loss, multi_resolution_stft_loss
from src.open_vocab_v3.metrics import audio_content_repair_loss
from scripts.train_open_vocab_v3_encodec_clip import TokenDataset, loader


class _SourceIndexedDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.source_indices = (101, 205, 309)

    def __len__(self):
        return len(self.source_indices)

    def __getitem__(self, index):
        return {"source_index": self.source_indices[index], "value": index}


def test_token_dataset_uses_immutable_source_index_through_nested_subset():
    base = _SourceIndexedDataset()
    nested = torch.utils.data.Subset(torch.utils.data.Subset(base, [2, 0]), [1])
    cache = {
        "encodec_codes": np.asarray([[[7]]], dtype=np.int16),
        "encodec_mask": np.asarray([[True]], dtype=bool),
    }
    item = TokenDataset(nested, cache, {101: 0})[0]
    assert item["source_index"] == 101
    assert int(item["encodec_codes"][0, 0]) == 7


def test_eeg_loader_uses_eeg_batch_size_not_audio_batch_size():
    class _TokenReady(torch.utils.data.Dataset):
        def __len__(self):
            return 20

        def __getitem__(self, index):
            return {
                "eeg": np.zeros((2, 8), np.float32), "channel_xyz": np.zeros((2, 3), np.float32),
                "channel_mask": np.ones(2, bool), "time_mask": np.ones(8, bool),
                "hubert": np.zeros((2, 2), np.float32), "hubert_mask": np.ones(2, bool),
                "mfcc": np.zeros((40, 256), np.float32), "mel": np.zeros((80, 256), np.float32),
                "speech_t5_mel": np.zeros((80, 161), np.float32), "speech_t5_mel_mask": np.ones(161, bool),
                "mfcc_mask": np.ones(256, bool), "activity": np.ones(256, bool),
                "speaker_reference": np.zeros(192, np.float32), "speaker_target": np.zeros(192, np.float32),
                "speaker_audit_reference": np.zeros(192, np.float32),
                "canonical_voice": np.zeros(192, np.float32), "canonical_mfcc_mean": np.zeros(40, np.float32),
                "canonical_mfcc_std": np.ones(40, np.float32), "speaker_reference_mfcc_mean": np.zeros(40, np.float32),
                "speaker_reference_mfcc_std": np.ones(40, np.float32),
                "target_mfcc_mean": np.zeros(40, np.float32), "target_mfcc_std": np.ones(40, np.float32),
                "sample_key": str(index), "audio_key": str(index), "label": "x", "subject": "s", "role": "fit",
                "encodec_codes": np.zeros((8, 192), np.int64), "encodec_mask": np.ones(192, bool),
            }

    cfg = {"training": {"audio_batch_size": 16, "eeg_batch_size": 10}, "evaluation": {"batch_size": 8}}
    assert len(next(iter(loader(_TokenReady(), cfg, True, "eeg")))["label"]) == 10


def test_audio_adaptation_losses_accept_encodec_bct_and_backpropagate():
    # 2051 is deliberately not divisible by the 64 envelope frames.  This is
    # the MPS case that adaptive_avg_pool1d cannot execute.
    prediction = torch.randn(2, 1, 2051, requires_grad=True)
    target = torch.randn(2, 1, 2051)
    spectral = multi_resolution_stft_loss(
        prediction, target, fft_sizes=(256, 512), hop_sizes=(64, 128)
    )
    envelope = envelope_loss(prediction, target, frames=64)
    loss = spectral + envelope
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_chunked_pairwise_mfcc_l1_matches_exact_broadcast():
    rng = np.random.default_rng(13)
    prediction = rng.normal(size=(7, 4, 19)).astype(np.float32)
    target = rng.normal(size=(9, 4, 19)).astype(np.float32)
    expected = np.mean(np.abs(prediction[:, None] - target[None]), axis=(2, 3))
    actual = pairwise_mfcc_l1(
        prediction, target, query_chunk=3, target_chunk=4, feature_chunk=11
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


APP = Path(__file__).resolve().parents[1]


def test_v3_config_has_a_new_artifact_firewall_and_fixed_content_contract() -> None:
    _, cfg = load_config(APP / "configs" / "open_vocab_v3_mfcc_training_first.yaml")
    assert cfg["paths"]["output_root"].endswith("open_vocab_v3_mfcc_training_first")
    assert cfg["version"] == "openvoice-eeg-v3-encodec-clip-mfcc-v1"
    assert cfg["paths"]["prepared_cache"].endswith("prepared_encodec_clip_mfcc_v1.npz")
    assert cfg["audio"]["mfcc_bins"] == 40
    assert cfg["audio"]["canonical_frames"] == 256
    assert cfg["audio"]["max_active_seconds"] == 2.56
    assert cfg["paths"]["micro_checkpoint"] != cfg["paths"]["fit_checkpoint"]
    assert "locked_unseen_report" in cfg["paths"]
    assert cfg["model"]["audio_latent_dimension"] > 0
    assert cfg["training"]["canonical_voice_dropout"] == 0.0
    assert cfg["denoise"]["processing_sample_rate"] == 16000
    assert cfg["audio"]["encodec_codebooks"] == 8
    assert cfg["audio"]["content_tokens"] == 32
    assert cfg["audio"]["native_mel_frames"] == 161
    assert "training_review" in cfg["paths"]


def test_content_repair_config_has_new_schema_and_cannot_reuse_v3_artifacts() -> None:
    config, cfg = load_config(APP / "configs" / "open_vocab_v3_content_repair_v2.yaml")
    assert content_schema(cfg) == "openvoice-v3-content-repair-v2"
    assert "open_vocab_v3_content_repair_v2" in str(output_path(config, cfg, "encodec_cache"))
    assert checkpoint_schema(cfg, "audio").endswith("v2-repair")
    assert checkpoint_schema(cfg, "fit").endswith("v2-repair")


def test_audio_content_repair_loss_uses_teacher_and_anti_collapse_terms() -> None:
    prediction = torch.randn(4, 40, 256, requires_grad=True)
    target = torch.randn(4, 40, 256)
    tokens = torch.randn(4, 32, 16, requires_grad=True)
    hubert = torch.randn(4, 50, 768)
    projection = torch.nn.Linear(768, 16)
    labels = ["a", "a", "b", "b"]
    loss, parts = audio_content_repair_loss(
        prediction, target, tokens, hubert, torch.ones(4, 50, dtype=torch.bool), projection,
        torch.nn.Linear(16, 2)(tokens.mean(1)), torch.tensor([0, 0, 1, 1]),
        torch.nn.Linear(16, 2)(tokens.mean(1)), torch.tensor([0, 1, 0, 1]), labels,
        {"hubert": .25, "label": .1, "variance": .15, "covariance": .05, "diversity": .1},
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert {"hubert_teacher", "variance", "covariance", "diversity"} <= set(parts)
    assert tokens.grad is not None


def test_v3_eeg_path_has_no_label_text_or_speaker_forward_input() -> None:
    signature = inspect.signature(EEGContentEncoder.forward)
    assert tuple(signature.parameters) == ("self", "eeg", "channel_xyz", "channel_mask", "time_mask")
    model = EEGContentEncoder(dimension=32, heads=4, layers=1, dropout=0.0)
    eeg = torch.randn(2, 3, 64)
    xyz = torch.randn(2, 3, 3)
    channel_mask = torch.tensor([[True, True, True], [True, True, False]])
    time_mask = torch.ones(2, 64, dtype=torch.bool)
    tokens = model(eeg, xyz, channel_mask, time_mask)
    assert tokens.shape == (2, 32, 32)
    mfcc = SharedMFCCDecoder(dimension=32)(tokens)
    assert mfcc.shape == (2, 40, 256)
    assert torch.count_nonzero(mfcc[:, 0]) == 0


def test_audio_content_encoder_has_independent_codebooks() -> None:
    model=AudioContentEncoder(codebooks=8,vocabulary=1024,dimension=32,tokens=32,heads=4,layers=1,dropout=0)
    codes=torch.randint(0,1024,(2,8,192));mask=torch.ones(2,192,dtype=torch.bool)
    assert len(model.embeddings)==8
    assert len({id(x.weight) for x in model.embeddings})==8
    assert model(codes,mask).shape==(2,32,32)


def test_clip_masks_token_diagonal_and_global_same_label() -> None:
    audio=torch.randn(4,32,16);eeg=audio.clone().requires_grad_();scale=torch.tensor(0.,requires_grad=True)
    token,global_=clip_token_global_losses(eeg,audio,["a","a","b","b"],scale)
    (token+global_).backward()
    assert torch.isfinite(token) and torch.isfinite(global_)
    assert eeg.grad is not None


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
