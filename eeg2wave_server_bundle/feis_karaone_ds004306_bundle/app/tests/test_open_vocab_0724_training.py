from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from scripts.train_open_vocab_0724 import (
    PairAwareEEGBatchSampler,
    deterministic_patch_mask,
    eeg_objective,
    expand_code_valid,
    train_eeg,
)
from scripts.make_open_vocab_0724_ablation_config import make_ablation_config
from scripts.gate_open_vocab_0724 import (
    REQUIRED_COUNTERFACTUAL_MODES,
    manifest_coverage,
    records_have_required_modes,
    records_have_valid_controls,
)
from src.open_vocab_0724.model import (
    FactorizedAudioConfig,
    FactorizedAudioModel,
    FactorizedEEGConfig,
    FactorizedEEGEncoder,
)
from src.open_vocab_0724.lineage import (
    VALIDATION_GATE_SCHEMA_VERSION,
    VALIDATION_REPORT_SCHEMA_VERSION,
    authorize_locked_test_metadata,
    claim_locked_test_access,
    file_sha256,
)
from src.open_vocab_0724.runtime import resolve_run_checkpoint, run_identifier


def _rows() -> list[dict[str, str]]:
    return [
        {"dataset": "karaone", "label": "iy", "audio_key": "k1"},
        {"dataset": "karaone", "label": "iy", "audio_key": "k2"},
        {"dataset": "karaone", "label": "uw", "audio_key": "k3"},
        {"dataset": "feis", "label": "goose", "audio_key": "f1"},
        {"dataset": "feis", "label": "goose", "audio_key": "f1"},
        {"dataset": "feis", "label": "goose", "audio_key": "f1"},
        {"dataset": "feis", "label": "thought", "audio_key": "f2"},
        {"dataset": "ds004306", "label": "up", "audio_key": "d1"},
        {"dataset": "ds004306", "label": "down", "audio_key": "d2"},
    ]


def test_eeg_sampler_balances_datasets_and_injects_realization_hard_negative() -> None:
    rows = _rows()
    sampler = PairAwareEEGBatchSampler(rows, batch_size=6, seed=15)
    mass: defaultdict[str, float] = defaultdict(float)
    for row, weight in zip(rows, sampler.weights):
        mass[row["dataset"]] += weight
    assert mass == pytest.approx({"karaone": 1.0, "feis": 1.0, "ds004306": 1.0})

    batch = next(iter(sampler))
    karaone = [rows[index] for index in batch if rows[index]["dataset"] == "karaone"]
    assert any(
        first["label"] == second["label"] and first["audio_key"] != second["audio_key"]
        for first in karaone
        for second in karaone
    )


def test_validation_masks_and_code_mask_shapes_are_deterministic() -> None:
    valid = torch.tensor(
        [
            [[True, True, True], [True, False, False]],
            [[True, True, False], [True, True, False]],
        ]
    )
    first = deterministic_patch_mask(valid, 0.5)
    second = deterministic_patch_mask(valid, 0.5)
    assert torch.equal(first, second)
    assert torch.all(first <= valid)
    assert first.flatten(1).any(dim=1).all()

    codes = torch.zeros(2, 3, 4, dtype=torch.long)
    valid_2d = torch.tensor([[True, True, False, False], [False, False, False, False]])
    expanded = expand_code_valid(valid_2d, codes)
    assert expanded.shape == codes.shape
    assert expanded[0, :, :2].all()
    # Empty codec rows receive one safe placeholder step; loss routing still
    # excludes them through has_codec/code_supervision.
    assert expanded[1, :, 0].all()
    assert not expanded[1, :, 1:].any()


def test_seed_and_loso_artifacts_cannot_overwrite_primary_checkpoint(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "configs" / "v0724.yaml"
    config_path.parent.mkdir()
    config_path.write_text("version: unit\n", encoding="utf-8")
    cfg = {
        "training": {"seed": 15},
        "paths": {"eeg_checkpoint": "../artifacts/eeg/checkpoints/best.pt"},
    }
    primary = resolve_run_checkpoint(config_path, cfg, "eeg_checkpoint", seed=15)
    secondary = resolve_run_checkpoint(config_path, cfg, "eeg_checkpoint", seed=31)
    loso = resolve_run_checkpoint(
        config_path,
        cfg,
        "eeg_checkpoint",
        seed=15,
        loso_subject="karaone:S1",
    )
    held_label = resolve_run_checkpoint(
        config_path,
        cfg,
        "eeg_checkpoint",
        seed=15,
        generalization="g3",
        holdout_label="/IY/",
    )
    assert run_identifier(cfg, seed=15) is None
    assert primary.name == secondary.name == loso.name == "best.pt"
    assert len({primary, secondary, loso, held_label}) == 4
    assert "seed_31" in str(secondary)
    assert "loso_karaone_S1_seed_15" in str(loso)
    assert "g3_label_IY_seed_15" in str(held_label)


def test_eeg_pretraining_requires_the_frozen_audio_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, Path]] = []

    def reject_unfrozen_audio(
        config_path: Path,
        _cfg: dict[str, object],
        _lineage: dict[str, object],
        checkpoint: Path,
    ) -> None:
        calls.append((config_path, checkpoint))
        raise PermissionError("audio oracle has not passed")

    monkeypatch.setattr(
        "scripts.train_open_vocab_0724.require_frozen_audio_checkpoint",
        reject_unfrozen_audio,
    )
    config_path = tmp_path / "configs" / "v0724.yaml"
    context = SimpleNamespace(
        config={
            "training": {},
            "paths": {"audio_checkpoint": "../artifacts/audio.pt"},
        },
        config_path=config_path,
    )
    with pytest.raises(PermissionError, match="audio oracle"):
        train_eeg(
            cast(Any, SimpleNamespace()),
            cast(Any, context),
            cast(Any, object()),
            {},
            torch.device("cpu"),
            pretrain=True,
        )
    assert calls == [(config_path, tmp_path / "artifacts" / "audio.pt")]


def test_locked_test_gate_binds_report_and_access_is_single_session(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    audio = tmp_path / "audio.pt"
    eeg = tmp_path / "eeg.pt"
    config.write_text("version: test\n", encoding="utf-8")
    audio.write_bytes(b"audio")
    eeg.write_bytes(b"eeg")
    lineage = {"config_sha256": file_sha256(config)}
    synthesis = tmp_path / "synthesis_manifest.json"
    synthesis.write_text("{}", encoding="utf-8")
    report_path = tmp_path / "validation_report.json"
    report = {
        "schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
        "passed": True,
        "split": "validation",
        "test_accessed": False,
        "lineage": lineage,
        "audio_checkpoint_sha256": file_sha256(audio),
        "eeg_checkpoint_sha256": file_sha256(eeg),
        "synthesis_manifest": str(synthesis),
        "synthesis_manifest_sha256": file_sha256(synthesis),
        "loso_manifests": {},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    gate_path = tmp_path / "validation_gate.json"
    gate = {
        "schema_version": VALIDATION_GATE_SCHEMA_VERSION,
        "passed": True,
        "failed_checks": [],
        "lineage": lineage,
        "audio_checkpoint_sha256": file_sha256(audio),
        "eeg_checkpoint_sha256": file_sha256(eeg),
        "validation_report": str(report_path),
        "validation_report_sha256": file_sha256(report_path),
    }
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    authorize_locked_test_metadata(
        gate_path,
        config_path=config,
        audio_checkpoint=audio,
        eeg_checkpoint=eeg,
    )

    access_id = "unit_test_access"
    claim_locked_test_access(
        gate_path, purpose="latent_evaluation", access_id=access_id
    )
    claim_locked_test_access(
        gate_path, purpose="reconstruction_karaone", access_id=access_id
    )
    with pytest.raises(PermissionError, match="already accessed"):
        claim_locked_test_access(
            gate_path, purpose="latent_evaluation", access_id=access_id
        )
    with pytest.raises(PermissionError, match="another final-test session"):
        claim_locked_test_access(
            gate_path,
            purpose="reconstruction_feis",
            access_id="different_access",
        )

    report_path.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="SHA256"):
        authorize_locked_test_metadata(
            gate_path,
            config_path=config,
            audio_checkpoint=audio,
            eeg_checkpoint=eeg,
        )


def test_formal_gate_rejects_diagnostic_or_incomplete_synthesis() -> None:
    records = [
        {
            "sample_key": key,
            "metrics": {name: {} for name in REQUIRED_COUNTERFACTUAL_MODES},
            "controls": {
                "same_label_control_available": True,
                "wrong_label_control_available": True,
                "shuffled_control_available": True,
            },
        }
        for key in ("a", "b")
    ]
    source = {
        "diagnostic_limit": -1,
        "full_dataset_record_count": 2,
        "records": records,
    }
    complete, missing = manifest_coverage(source, {"a", "b"})
    assert complete and not missing
    assert records_have_required_modes(records)
    assert records_have_valid_controls(records)

    diagnostic = dict(source, diagnostic_limit=1)
    assert not manifest_coverage(diagnostic, {"a", "b"})[0]
    incomplete = dict(source, records=records[:1])
    assert not manifest_coverage(incomplete, {"a", "b"})[0]
    assert manifest_coverage(incomplete, {"a", "b"})[1] == ["b"]


def test_ablation_switches_retain_parameter_count_and_isolate_artifacts() -> None:
    base = FactorizedAudioConfig(
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
    )
    variants = (
        base,
        replace(base, use_energy_feedback=False),
        replace(base, use_realization_condition=False),
        replace(base, use_content_condition=False),
    )
    counts = [
        sum(parameter.numel() for parameter in FactorizedAudioModel(cfg).parameters())
        for cfg in variants
    ]
    assert len(set(counts)) == 1

    source = {
        "paths": {
            "output_root": "../../artifacts/open_vocab_0724_factorized_v1",
            "teacher_cache": "../../artifacts/open_vocab_0724_factorized_v1/cache/teacher_v2",
            "audio_checkpoint": "../../artifacts/open_vocab_0724_factorized_v1/audio/checkpoints/best.pt",
            "audio_oracle_gate": "../../artifacts/open_vocab_0724_factorized_v1/audio/metrics/audio_oracle_gate.json",
            "audio_freeze_manifest": "../../artifacts/open_vocab_0724_factorized_v1/audio/frozen_checkpoint.json",
        },
        "model": {},
        "teachers": {"hubert_model": "hubert-local"},
    }
    content_only = make_ablation_config(source, "content_only")
    assert content_only["model"]["eeg_use_content_condition"]
    assert not content_only["model"]["eeg_use_realization_condition"]
    assert content_only["paths"]["teacher_cache"] == source["paths"]["teacher_cache"]
    assert (
        content_only["paths"]["audio_checkpoint"] == source["paths"]["audio_checkpoint"]
    )
    assert content_only["paths"]["output_root"] != source["paths"]["output_root"]
    contentvec = make_ablation_config(
        source, "full_contentvec", contentvec_model="contentvec-local"
    )
    assert contentvec["teachers"]["hubert_model"] == "contentvec-local"
    assert contentvec["paths"]["teacher_cache"] != source["paths"]["teacher_cache"]


def test_paired_objective_backpropagates_and_ds_audio_targets_are_inert() -> None:
    torch.manual_seed(71)
    audio_cfg = FactorizedAudioConfig(
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
    )
    eeg_cfg = FactorizedEEGConfig(
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
        num_train_subjects=3,
        num_content_labels=3,
        adapter_moe_enabled=False,
        branch_dropout_probability=0.0,
        min_duration_seconds=0.1,
        max_duration_seconds=4.0,
    )
    audio = FactorizedAudioModel(audio_cfg).eval()
    eeg = FactorizedEEGEncoder(eeg_cfg).eval()
    batch_size = 3
    realization = torch.randn(batch_size, 10, 8)
    realization[..., :4] = -80.0 + 80.0 * torch.rand(batch_size, 10, 4)
    realization[..., 4] = 5.0
    realization[..., 5] = 1.0
    realization[..., 6] = -20.0
    realization[..., 7] = 1.0
    code_valid = torch.zeros(batch_size, 2, 12, dtype=torch.bool)
    code_valid[0, :, :6] = True
    batch = {
        "eeg": torch.randn(batch_size, 2, 24),
        "channel_xyz": torch.randn(batch_size, 2, 3),
        "channel_mask": torch.ones(batch_size, 2, dtype=torch.bool),
        "time_mask": torch.ones(batch_size, 24, dtype=torch.bool),
        "subject_idx": torch.tensor([0, 1, 2]),
        "dataset_idx": torch.tensor([0, 1, 2]),
        "label_idx": torch.tensor([0, 1, 2]),
        "content_tokens": torch.randn(batch_size, 5, 6),
        "content_token_mask": torch.ones(batch_size, 5, dtype=torch.bool),
        "realization_features": realization,
        "realization_frame_mask": torch.ones(batch_size, 10, dtype=torch.bool),
        "timbre_global": torch.randn(batch_size, 7),
        "has_audio_teacher": torch.tensor([True, True, True]),
        "has_timbre": torch.tensor([True, True, False]),
        "has_codec": torch.tensor([True, False, False]),
        "content_supervision": torch.tensor([True, True, False]),
        "realization_supervision": torch.tensor([True, False, False]),
        "timbre_supervision": torch.tensor([True, True, False]),
        "energy_supervision": torch.tensor([True, False, False]),
        "code_supervision": torch.tensor([True, False, False]),
        "exact_pair_supervision": torch.tensor([True, False, False]),
        "feis_prototype_supervision": torch.tensor([False, True, False]),
        "log_mel_energy": realization[..., :4].transpose(1, 2).clone(),
        "f0_log_hz": realization[..., 4].clone(),
        "voicing": realization[..., 5].clone(),
        "log_rms_dbfs": realization[..., 6].clone(),
        "activity_mask": realization[..., 7].bool().clone(),
        "duration_seconds": torch.tensor([2.0, 1.0, 1.0]),
        "codes": torch.randint(0, 16, (batch_size, 2, 12)),
        "code_valid_mask": code_valid,
        "feis_audio_weight": torch.tensor([1.0, 0.5, 1.0]),
    }
    cfg = {
        "experiment": {"ablation": "full_v0724"},
        "training": {
            "patch_mask_ratio": 0.3,
            "channel_drop_probabilities": [0.1],
            "coordinate_noise_std": 0.0,
            "signal_noise_std": 0.0,
            "mask_ratio_min": 0.5,
            "mask_ratio_max": 0.5,
            "full_mask_probability": 0.0,
        },
        "loss": {
            "eeg_masked_pretraining": 1.0,
            "channel_consistency": 0.2,
            "subject_adversarial": 0.05,
            "dataset_adversarial": 0.05,
            "timbre_label_adversarial": 0.05,
            "moe": 0.01,
            "contrastive_temperature": 0.08,
            "content_clip": 1.0,
            "realization_clip": 1.0,
            "realization_local": 0.5,
            "mel_l1": 0.5,
            "mel_soft_dtw": 0.5,
            "soft_dtw_gamma": 0.1,
            "soft_dtw_band_fraction": 0.25,
            "soft_dtw_train_frames": 8,
            "activity_duration_prosody": 0.25,
            "codebook_weights": [1.0, 1.0],
            "eeg_code": 0.25,
            "feis_timbre_prototype": 0.25,
            "cross_covariance": 0.05,
        },
    }
    loss, _, _, targets = eeg_objective(
        eeg,
        audio,
        batch,
        cfg,
        epoch=0,
        adversary_strength=0.1,
        pretrain=False,
        augment=False,
        stochastic_mask=False,
    )
    assert targets is not None
    assert targets["eligible"].tolist() == [True, True, False]
    assert torch.isfinite(loss)

    corrupted = dict(batch)
    corrupted["content_tokens"] = batch["content_tokens"].clone()
    corrupted["realization_features"] = batch["realization_features"].clone()
    corrupted["timbre_global"] = batch["timbre_global"].clone()
    corrupted["content_tokens"][2] = 1.0e5
    corrupted["realization_features"][2] = -1.0e5
    corrupted["timbre_global"][2] = 1.0e5
    second_loss, _, _, _ = eeg_objective(
        eeg,
        audio,
        corrupted,
        cfg,
        epoch=0,
        adversary_strength=0.1,
        pretrain=False,
        augment=False,
        stochastic_mask=False,
    )
    torch.testing.assert_close(loss, second_loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in eeg.parameters()
    )
