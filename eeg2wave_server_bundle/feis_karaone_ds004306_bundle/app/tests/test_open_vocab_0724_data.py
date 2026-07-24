from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from src.open_vocab_0724.audio_features import (
    AcousticFeatureConfig,
    ActiveSpeechConfig,
    AudioPreparationConfig,
    detect_active_speech,
    extract_acoustic_features,
    prepare_waveform_segment,
)
from src.open_vocab_0724.data import (
    TEACHER_CACHE_SCHEMA_VERSION,
    FactorizedAudioDataset,
    TeacherCacheV2,
    _row_selected,
    build_project_records,
    collate_factorized,
    supervision_for_dataset,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_active_speech_rule_gap_closing_and_context() -> None:
    sample_rate = 16_000
    audio = np.zeros(sample_rate * 2, dtype=np.float32)
    time = np.arange(int(0.30 * sample_rate), dtype=np.float32) / sample_rate
    tone = 0.20 * np.sin(2.0 * np.pi * 220.0 * time)
    audio[8_000:12_800] = tone
    # A 40 ms zero gap must be closed by the registered <=50 ms rule.
    audio[10_000:10_640] = 0.0
    cfg = ActiveSpeechConfig(sample_rate=sample_rate)
    bounds = detect_active_speech(audio, cfg)
    expected_threshold = max(
        float(np.percentile(bounds.frame_rms_dbfs, 10.0) + 6.0),
        float(bounds.frame_rms_dbfs.max() - 40.0),
    )
    assert bounds.threshold_dbfs == pytest.approx(expected_threshold)
    assert bounds.has_activity
    assert bounds.speech_start_sample <= 8_000
    assert bounds.speech_end_sample >= 12_800
    assert bounds.context_start_sample == max(0, bounds.speech_start_sample - 1_600)
    assert bounds.context_end_sample == min(
        len(audio), bounds.speech_end_sample + 1_600
    )
    gap_frames = bounds.frame_activity[63:66]
    assert gap_frames.all()


def test_shared_segment_features_and_overlong_content_only() -> None:
    sample_rate = 16_000
    one_second = np.arange(sample_rate, dtype=np.float32) / sample_rate
    tone = (0.1 * np.sin(2.0 * np.pi * 220.0 * one_second)).astype(np.float32)
    prep_cfg = AudioPreparationConfig(
        sample_rate=sample_rate, max_active_seconds=4.0, target_rms=0.08
    )
    prepared = prepare_waveform_segment(tone, sample_rate, prep_cfg)
    assert prepared.waveform.shape == (64_000,)
    assert prepared.valid_samples == sample_rate
    assert prepared.reconstruction_eligible
    assert prepared.normalization_gain > 0.0
    assert len(prepared.pcm_sha256) == 64

    features = extract_acoustic_features(
        prepared.waveform,
        valid_samples=prepared.valid_samples,
        config=AcousticFeatureConfig(sample_rate=sample_rate),
    )
    assert features.log_mel_energy.shape == (80, 400)
    assert features.realization_features.shape == (400, 84)
    assert features.frame_valid_mask.sum() == 100
    assert np.isfinite(features.realization_features).all()
    assert features.log_mel_energy.min() >= -80.0
    assert features.log_mel_energy.max() <= 0.0
    assert features.voicing.sum() > 0
    assert np.all(features.log_mel_energy[:, 100:] == -80.0)

    long_time = np.arange(sample_rate * 5, dtype=np.float32) / sample_rate
    long_tone = (0.1 * np.sin(2.0 * np.pi * 180.0 * long_time)).astype(np.float32)
    overlong = prepare_waveform_segment(long_tone, sample_rate, prep_cfg)
    assert overlong.waveform.shape == (64_000,)
    assert overlong.exceeds_max_active_seconds
    assert not overlong.reconstruction_eligible
    silence = prepare_waveform_segment(
        np.zeros(sample_rate, dtype=np.float32), sample_rate, prep_cfg
    )
    assert not silence.has_activity
    assert not silence.reconstruction_eligible


def test_dataset_specific_supervision_routes() -> None:
    karaone = supervision_for_dataset("karaone", reconstruction_eligible=True)
    assert (
        karaone.content
        and karaone.realization
        and karaone.energy
        and karaone.code
        and karaone.exact_pair
    )
    feis = supervision_for_dataset("feis", reconstruction_eligible=True)
    assert feis.content and feis.timbre and feis.feis_prototype
    assert (
        not feis.realization
        and not feis.energy
        and not feis.code
        and not feis.exact_pair
    )
    ds = supervision_for_dataset("ds004306", reconstruction_eligible=True)
    assert not any(vars(ds).values())
    overlong = supervision_for_dataset("karaone", reconstruction_eligible=False)
    assert overlong.content
    assert not overlong.realization and not overlong.audio_generation_eligible


def test_subject_loso_uses_only_locked_training_rows() -> None:
    context = cast(Any, SimpleNamespace(split_for=lambda row: row["official_split"]))
    held_out_train = {
        "official_split": "train",
        "subject_group_id": "karaone:S1",
        "label": "iy",
    }
    other_train = {
        "official_split": "train",
        "subject_group_id": "karaone:S2",
        "label": "iy",
    }
    official_validation = {
        "official_split": "validation",
        "subject_group_id": "karaone:S1",
        "label": "iy",
    }
    assert _row_selected(
        context, held_out_train, "validation", "g1", None, "karaone:S1"
    )
    assert not _row_selected(context, held_out_train, "train", "g1", None, "karaone:S1")
    assert _row_selected(context, other_train, "train", "g1", None, "karaone:S1")
    assert not _row_selected(
        context, official_validation, "validation", "g1", None, "karaone:S1"
    )
    with pytest.raises(PermissionError):
        _row_selected(context, held_out_train, "test", "g1", None, "karaone:S1")


def test_record_ids_split_integrity_and_feis_deduplication(tmp_path: Path) -> None:
    rows = (
        {
            "audio_key": "k1",
            "audio_relpath": "a.wav",
            "audio_valid_samples": "16000",
            "dataset": "karaone",
            "label": "/IY/",
            "pairing_confidence": "karaone_same_trial_overt",
            "subject_group_id": "karaone:S1",
            "split": "train",
        },
        {
            "audio_key": "f1",
            "audio_relpath": "b.wav",
            "audio_valid_samples": "16000",
            "dataset": "feis",
            "label": "Goose",
            "pairing_confidence": "feis_subject_label",
            "subject_group_id": "feis:01",
            "split": "train",
        },
        {
            "audio_key": "f1",
            "audio_relpath": "b.wav",
            "audio_valid_samples": "16000",
            "dataset": "feis",
            "label": "Goose",
            "pairing_confidence": "feis_subject_label",
            "subject_group_id": "feis:01",
            "split": "train",
        },
        {
            "audio_key": "d1",
            "audio_relpath": "c.wav",
            "audio_valid_samples": "16000",
            "dataset": "ds004306",
            "label": "up",
            "pairing_confidence": "weak_category_level",
            "subject_group_id": "ds004306:01",
            "split": "train",
        },
        {
            "audio_key": "d1",
            "audio_relpath": "c.wav",
            "audio_valid_samples": "16000",
            "dataset": "ds004306",
            "label": "up",
            "pairing_confidence": "weak_category_level",
            "subject_group_id": "ds004306:02",
            "split": "validation",
        },
    )
    context = cast(
        Any,
        SimpleNamespace(
            rows=rows, audio_root=tmp_path, split_for=lambda row: row["split"]
        ),
    )
    records = {record.audio_key: record for record in build_project_records(context)}
    assert records["k1"].content_id == "iy"
    assert records["k1"].audio_utterance_id == "k1"
    assert records["f1"].audio_speaker_id == "feis:01"
    assert records["f1"].row_count == 2
    assert records["f1"].pairing_scope == "unique_subject_label_prototype"
    assert records["d1"].eeg_subject_group_id == "multiple"
    assert records["d1"].split_names == ("train", "validation")
    assert not records["d1"].audio_generation_eligible


def _make_teacher_cache(root: Path) -> TeacherCacheV2:
    root.mkdir()
    keys = np.asarray(["k1", "f1", "d1"])
    count = len(keys)
    shard = {
        "schema_version": np.asarray(TEACHER_CACHE_SCHEMA_VERSION),
        "build_fingerprint": np.asarray("unit-test"),
        "keys": keys,
        "content_tokens": np.ones((count, 50, 768), dtype=np.float16),
        "content_token_mask": np.ones((count, 50), dtype=bool),
        "realization_features": np.ones((count, 400, 84), dtype=np.float16),
        "realization_frame_mask": np.ones((count, 400), dtype=bool),
        "log_mel_energy": np.zeros((count, 80, 400), dtype=np.float16),
        "f0_log_hz": np.ones((count, 400), dtype=np.float16),
        "voicing": np.ones((count, 400), dtype=np.float16),
        "log_rms_dbfs": np.full((count, 400), -20.0, dtype=np.float16),
        "activity_mask": np.ones((count, 400), dtype=bool),
        "timbre_global": np.ones((count, 512), dtype=np.float16),
        "has_timbre": np.asarray([True, True, False]),
        "encodec_codes": np.zeros((count, 8, 300), dtype=np.int16),
        "encodec_scale": np.ones((count, 1), dtype=np.float32),
        "encodec_scale_valid": np.zeros(count, dtype=bool),
        "code_valid_mask": np.ones((count, 8, 300), dtype=bool),
        "has_codec": np.asarray([True, True, False]),
    }
    shard_path = root / "records_00000.npz"
    np.savez_compressed(shard_path, **shard)
    statistics_path = root / "train_statistics.npz"
    np.savez_compressed(
        statistics_path,
        realization_mean=np.zeros(84, dtype=np.float32),
        realization_std=np.ones(84, dtype=np.float32),
        realization_frame_count=np.asarray(800, dtype=np.int64),
        feis_prototype_ids=np.asarray(["feis:01::goose"]),
        feis_timbre_prototypes=np.ones((1, 512), dtype=np.float32) / np.sqrt(512.0),
        fit_split_only=np.asarray(True),
    )
    records = {
        "k1": {
            "audio_key": "k1",
            "audio_path": "/tmp/k1.wav",
            "audio_relpath": "k1.wav",
            "dataset": "karaone",
            "label": "iy",
            "content_id": "iy",
            "audio_utterance_id": "k1",
            "audio_speaker_id": "karaone:S1",
            "eeg_subject_group_id": "karaone:S1",
            "pairing_scope": "exact_trial",
            "pairing_confidence": "karaone_same_trial_overt",
            "split_names": ["train"],
            "fit_split": True,
            "reconstruction_eligible": True,
            "active_duration_seconds": 1.2,
            "segment_valid_samples": 19200,
            "native_sample_count": 19200,
            "native_rms": 0.1,
            "normalization_gain": 0.8,
            "active_start_sample": 100,
            "active_end_sample": 19000,
            "segment_source_start_sample": 0,
            "segment_source_end_sample": 19200,
            "segment_pcm_sha256": "a" * 64,
            "source_audio_sha256": "b" * 64,
        },
        "f1": {
            "audio_key": "f1",
            "audio_path": "/tmp/f1.wav",
            "audio_relpath": "f1.wav",
            "dataset": "feis",
            "label": "goose",
            "content_id": "goose",
            "audio_utterance_id": "f1",
            "audio_speaker_id": "feis:01",
            "eeg_subject_group_id": "feis:01",
            "pairing_scope": "unique_subject_label_prototype",
            "pairing_confidence": "feis_subject_label",
            "split_names": ["train"],
            "fit_split": True,
            "reconstruction_eligible": True,
            "active_duration_seconds": 0.8,
        },
        "d1": {
            "audio_key": "d1",
            "audio_path": "/tmp/d1.wav",
            "audio_relpath": "d1.wav",
            "dataset": "ds004306",
            "label": "up",
            "content_id": "up",
            "audio_utterance_id": "d1",
            "audio_speaker_id": "unavailable",
            "eeg_subject_group_id": "multiple",
            "pairing_scope": "none",
            "pairing_confidence": "weak_category_level",
            "split_names": ["train", "validation"],
            "fit_split": False,
            "reconstruction_eligible": False,
            "active_duration_seconds": 1.0,
        },
    }
    index = {
        "schema_version": TEACHER_CACHE_SCHEMA_VERSION,
        "content_steps": 50,
        "content_dimension": 768,
        "realization_frames": 400,
        "realization_dimension": 84,
        "mel_bins": 80,
        "timbre_dimension": 512,
        "codebooks": 8,
        "code_steps": 300,
        "sample_rate": 16000,
        "record_index": {
            key: ["records_00000.npz", i] for i, key in enumerate(keys.tolist())
        },
        "records": records,
        "statistics_file": "train_statistics.npz",
        "file_sha256": {
            "records_00000.npz": _sha256(shard_path),
            "train_statistics.npz": _sha256(statistics_path),
        },
    }
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return TeacherCacheV2(root, verify_hashes=True)


def test_teacher_cache_lookup_train_stats_and_audio_collation(tmp_path: Path) -> None:
    cache = _make_teacher_cache(tmp_path / "cache")
    assert cache.audit()["passed"]
    record = cache.lookup("k1")
    assert record["content_tokens"].shape == (50, 768)
    assert record["realization_features"].shape == (400, 84)
    assert record["log_mel_energy"].shape == (80, 400)
    assert record["encodec_codes"].shape == (8, 300)
    assert "feis:01::goose" in cache.feis_timbre_prototypes

    context = cast(Any, SimpleNamespace(label_to_index={"goose": 0, "iy": 1, "up": 2}))
    dataset = FactorizedAudioDataset(context, cache, split="train")
    samples = [dataset[index] for index in range(len(dataset))]
    assert {sample["dataset"] for sample in samples} == {"karaone", "feis"}
    assert all("duration_seconds" in sample for sample in samples)
    assert all("audio_path" in sample for sample in samples)
    batch = collate_factorized(samples)
    assert batch["content_tokens"].shape == (2, 50, 768)
    assert batch["codes"].shape == (2, 8, 300)
    assert batch["duration_seconds"].shape == (2,)
