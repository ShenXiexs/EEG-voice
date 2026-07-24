#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
KARA_APP = APP.parents[1] / "karaone_overt_recon_bundle" / "app"
if str(KARA_APP) not in sys.path:
    sys.path.insert(0, str(KARA_APP))

from src.open_vocab_0722.audio_io import read_wav, wav_info  # noqa: E402
from src.open_vocab_0724.audio_features import (  # noqa: E402
    AcousticFeatureConfig,
    AcousticFeatures,
    ActiveSpeechConfig,
    AudioPreparationConfig,
    PreparedWaveform,
    extract_acoustic_features,
    fallback_timbre_embedding,
    prepare_waveform_segment,
)
from src.open_vocab_0724.data import (  # noqa: E402
    AudioRecordSpec,
    TEACHER_CACHE_SCHEMA_VERSION,
    TeacherCacheV2,
    build_project_records,
    load_context,
    resolve_config_path,
)
from src.open_vocab_0724.runtime import load_config  # noqa: E402


BUILD_STATE_SCHEMA_VERSION = "openvoice-0724-teacher-build-state-v2"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_artifact_identity(reference: str | Path) -> dict[str, str]:
    """Bind local model bytes; retain an explicit unresolved HF identifier otherwise."""

    path = Path(reference)
    if path.is_file():
        return {"reference": str(path.resolve()), "sha256": file_sha256(path)}
    if path.is_dir():
        digest = hashlib.sha256(b"openvoice-0724-model-directory-v1\0")
        files = sorted(item for item in path.rglob("*") if item.is_file())
        for child in files:
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_sha256(child).encode("ascii"))
            digest.update(b"\0")
        return {"reference": str(path.resolve()), "sha256": digest.hexdigest()}
    return {"reference": str(reference), "sha256": "unresolved-huggingface-reference"}


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_reference(config_path: Path, value: str | Path) -> str:
    candidate = resolve_config_path(config_path, value)
    return str(candidate) if candidate.exists() else str(value)


def preparation_config(cfg: dict[str, Any]) -> AudioPreparationConfig:
    audio = cfg["audio"]
    sample_rate = int(audio.get("sample_rate", 16_000))
    active = ActiveSpeechConfig(
        sample_rate=sample_rate,
        window_ms=float(audio.get("active_window_ms", 25.0)),
        hop_ms=float(audio.get("active_hop_ms", 10.0)),
        noise_margin_db=float(audio.get("active_noise_margin_db", 6.0)),
        peak_margin_db=float(audio.get("active_peak_margin_db", 40.0)),
        close_gap_ms=float(audio.get("active_close_gap_ms", 50.0)),
        context_ms=float(audio.get("active_context_ms", 100.0)),
    )
    return AudioPreparationConfig(
        sample_rate=sample_rate,
        max_active_seconds=float(audio.get("max_active_seconds", 4.0)),
        target_rms=(
            float(audio["target_rms"]) if audio.get("target_rms") is not None else None
        ),
        active=active,
    )


def acoustic_config(cfg: dict[str, Any]) -> AcousticFeatureConfig:
    audio = cfg["audio"]
    return AcousticFeatureConfig(
        sample_rate=int(audio.get("sample_rate", 16_000)),
        window_ms=float(audio.get("active_window_ms", 25.0)),
        hop_ms=float(audio.get("active_hop_ms", 10.0)),
        mel_bins=int(audio.get("mel_bins", 80)),
        max_frames=int(audio.get("mel_frames", 400)),
        fmin_hz=float(audio.get("mel_fmin_hz", 0.0)),
        fmax_hz=float(audio.get("mel_fmax_hz", 8_000.0)),
        min_db=float(audio.get("mel_db_min", -80.0)),
        max_db=float(audio.get("mel_db_max", 0.0)),
    )


def masked_pool_hidden_tokens(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
    token_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool valid teacher frames without allowing padded frames into a token."""

    if hidden.ndim != 3:
        raise ValueError(f"Expected hidden [B,T,D], got {tuple(hidden.shape)}")
    batch, frames, dimension = hidden.shape
    if attention_mask is None:
        attention_mask = torch.ones(
            batch, frames, dtype=torch.bool, device=hidden.device
        )
    if attention_mask.shape != (batch, frames):
        raise ValueError(
            f"Teacher attention mask mismatch: {tuple(attention_mask.shape)}"
        )
    values = hidden.new_zeros((batch, int(token_steps), dimension))
    masks = torch.zeros(batch, int(token_steps), dtype=torch.bool, device=hidden.device)
    for index in range(batch):
        valid = hidden[index, attention_mask[index].bool()]
        if len(valid) == 0:
            continue
        if len(valid) >= int(token_steps):
            pooled_input = valid.transpose(0, 1).unsqueeze(0)
            # Adaptive pooling on MPS has a divisibility limitation.
            if hidden.device.type == "mps" and len(valid) % int(token_steps) != 0:
                pooled = F.adaptive_avg_pool1d(pooled_input.cpu(), int(token_steps)).to(
                    hidden.device
                )
            else:
                pooled = F.adaptive_avg_pool1d(pooled_input, int(token_steps))
            values[index] = pooled.squeeze(0).transpose(0, 1)
            masks[index] = True
        else:
            values[index, : len(valid)] = valid
            masks[index, : len(valid)] = True
    return values, masks


class HubertLayerTeacher:
    def __init__(
        self,
        model_name: str,
        *,
        layer: int,
        token_steps: int,
        device: torch.device,
        local_files_only: bool,
    ):
        from transformers import AutoFeatureExtractor, AutoModel

        self.model_name = str(model_name)
        self.layer = int(layer)
        self.token_steps = int(token_steps)
        self.device = device
        self.extractor = AutoFeatureExtractor.from_pretrained(
            self.model_name, local_files_only=bool(local_files_only)
        )
        self.model = (
            AutoModel.from_pretrained(
                self.model_name, local_files_only=bool(local_files_only)
            )
            .to(device)
            .eval()
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.dimension = int(getattr(self.model.config, "hidden_size"))

    @torch.inference_mode()
    def encode(
        self, prepared: Sequence[PreparedWaveform]
    ) -> tuple[np.ndarray, np.ndarray]:
        waveforms = [item.waveform[: item.valid_samples] for item in prepared]
        encoded = self.extractor(
            waveforms,
            sampling_rate=int(prepared[0].sample_rate),
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        output = self.model(**encoded, output_hidden_states=True, return_dict=True)
        if output.hidden_states is None or not 0 <= self.layer < len(
            output.hidden_states
        ):
            raise ValueError(f"HuBERT layer {self.layer} is unavailable")
        hidden = output.hidden_states[self.layer]
        input_mask = encoded.get("attention_mask")
        if input_mask is None:
            feature_mask = None
        elif hasattr(self.model, "_get_feature_vector_attention_mask"):
            feature_mask = self.model._get_feature_vector_attention_mask(
                hidden.shape[1], input_mask
            ).bool()
        else:
            lengths = input_mask.sum(dim=1).float()
            feature_lengths = torch.ceil(
                lengths / input_mask.shape[1] * hidden.shape[1]
            ).long()
            feature_mask = (
                torch.arange(hidden.shape[1], device=hidden.device)[None, :]
                < feature_lengths[:, None]
            )
        tokens, mask = masked_pool_hidden_tokens(hidden, feature_mask, self.token_steps)
        return tokens.float().cpu().numpy(), mask.cpu().numpy()


class SpeakerTeacher:
    def __init__(
        self,
        model_name: str | None,
        *,
        expected_dimension: int,
        device: torch.device,
        local_files_only: bool,
        required: bool,
        disabled: bool,
    ):
        self.expected_dimension = int(expected_dimension)
        self.device = device
        self.model_name = str(model_name) if model_name else "disabled"
        self.extractor: Any = None
        self.model: Any = None
        self.backend = "acoustic_statistics_fallback"
        self.revision = "fallback-v1"
        self.load_error: str | None = None
        if disabled or not model_name:
            return
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

            self.extractor = AutoFeatureExtractor.from_pretrained(
                str(model_name), local_files_only=bool(local_files_only)
            )
            self.model = (
                AutoModelForAudioXVector.from_pretrained(
                    str(model_name), local_files_only=bool(local_files_only)
                )
                .to(device)
                .eval()
            )
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            self.backend = "wavlm_xvector"
            self.revision = str(
                getattr(self.model.config, "_commit_hash", None) or self.model_name
            )
        except Exception as error:  # optional checkpoint may be absent offline
            self.load_error = f"{type(error).__name__}: {error}"
            if required:
                raise RuntimeError(
                    f"Required WavLM speaker teacher could not be loaded: {self.load_error}"
                ) from error

    @torch.inference_mode()
    def encode(
        self,
        prepared: Sequence[PreparedWaveform],
        acoustic: Sequence[AcousticFeatures],
    ) -> np.ndarray:
        if self.model is None or self.extractor is None:
            return np.stack(
                [
                    fallback_timbre_embedding(item, self.expected_dimension)
                    for item in acoustic
                ]
            ).astype(np.float32)
        waveforms = [item.waveform[: item.valid_samples] for item in prepared]
        encoded = self.extractor(
            waveforms,
            sampling_rate=int(prepared[0].sample_rate),
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        output = self.model(**encoded, return_dict=True)
        embedding = getattr(output, "embeddings", None)
        if embedding is None:
            raise ValueError("WavLM x-vector model returned no embeddings")
        values = F.normalize(embedding.float(), dim=-1).cpu().numpy()
        if values.shape[1] != self.expected_dimension:
            raise ValueError(
                f"WavLM embedding dimension {values.shape[1]} != configured {self.expected_dimension}"
            )
        return values.astype(np.float32)


class CodecTeacher:
    def __init__(
        self,
        cfg: dict[str, Any],
        config_path: Path,
        device: torch.device,
        *,
        disabled: bool,
    ):
        self.disabled = bool(disabled)
        self.codebooks = int(cfg["codec"]["codebooks"])
        self.code_steps = int(cfg["codec"]["code_steps"])
        self.backend: Any = None
        self.codec_sample_rate = 0
        if self.disabled:
            return
        from src.karaone_0715.codec import DiscreteEncodec, DiscreteEncodecConfig

        codec = cfg["codec"]
        self.backend = DiscreteEncodec(
            DiscreteEncodecConfig(
                model_path=str(
                    resolve_config_path(config_path, cfg["paths"]["encodec_model"])
                ),
                sample_rate=int(codec["sample_rate"]),
                duration_sec=float(codec.get("max_duration_sec", 4.0)),
                bandwidth=float(codec["bandwidth"]),
            ),
            device,
        )
        self.codec_sample_rate = int(self.backend.codec_sample_rate)

    def encode(self, prepared: Sequence[PreparedWaveform]) -> dict[str, np.ndarray]:
        batch = len(prepared)
        codes = np.zeros((batch, self.codebooks, self.code_steps), dtype=np.int16)
        scales = np.ones((batch, 1), dtype=np.float32)
        scale_valid = np.zeros(batch, dtype=bool)
        code_mask = np.zeros((batch, self.codebooks, self.code_steps), dtype=bool)
        has_codec = np.zeros(batch, dtype=bool)
        if self.backend is None or not batch:
            return {
                "encodec_codes": codes,
                "encodec_scale": scales,
                "encodec_scale_valid": scale_valid,
                "code_valid_mask": code_mask,
                "has_codec": has_codec,
            }
        encoded = self.backend.encode(np.stack([item.waveform for item in prepared]))
        observed = np.asarray(encoded["codes"])
        if observed.shape != codes.shape:
            raise ValueError(
                f"Unexpected v0724 EnCodec shape {observed.shape}; expected {codes.shape}"
            )
        codes[:] = observed
        scales[:] = np.asarray(encoded["scale"], dtype=np.float32).reshape(batch, -1)[
            :, :1
        ]
        scale_valid[:] = np.asarray(encoded["scale_valid"], dtype=bool)
        maximum_samples = int(prepared[0].waveform.shape[0])
        for index, item in enumerate(prepared):
            steps = max(
                1,
                min(
                    self.code_steps,
                    int(
                        np.ceil(item.valid_samples / maximum_samples * self.code_steps)
                    ),
                ),
            )
            code_mask[index, :, :steps] = True
        has_codec[:] = True
        return {
            "encodec_codes": codes,
            "encodec_scale": scales,
            "encodec_scale_valid": scale_valid,
            "code_valid_mask": code_mask,
            "has_codec": has_codec,
        }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.stem}.partial.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _read_record_audio(spec: AudioRecordSpec) -> tuple[np.ndarray, int, int]:
    rate, total_frames = wav_info(spec.audio_path)
    frames = (
        total_frames
        if spec.source_valid_samples is None
        else min(total_frames, int(spec.source_valid_samples))
    )
    audio, observed_rate = read_wav(spec.audio_path, frames=frames)
    if int(observed_rate) != int(rate):
        raise ValueError(f"WAV metadata changed while reading: {spec.audio_path}")
    return audio, int(rate), int(total_frames)


def _build_fingerprint(
    config_path: Path,
    context: Any,
    records: Sequence[AudioRecordSpec],
    prep_cfg: AudioPreparationConfig,
    acoustic_cfg: AcousticFeatureConfig,
    args: argparse.Namespace,
    model_lineage: dict[str, Any],
) -> str:
    return object_sha256(
        {
            "schema_version": TEACHER_CACHE_SCHEMA_VERSION,
            "config_sha256": file_sha256(config_path),
            "manifest_sha256": file_sha256(context.manifest_path),
            "subject_split_sha256": file_sha256(context.split_path),
            "records": [record.index_metadata() for record in records],
            "preparation": asdict(prep_cfg),
            "acoustics": asdict(acoustic_cfg),
            "model_lineage": model_lineage,
            "hubert_model": str(context.config["teachers"]["hubert_model"]),
            "hubert_layer": int(context.config["teachers"]["hubert_layer"]),
            "wavlm_model": str(
                context.config["teachers"].get("wavlm_model") or "disabled"
            ),
            "skip_codec": bool(args.skip_codec),
            "skip_speaker": bool(args.skip_speaker),
        }
    )


def _empty_shard_arrays(
    size: int,
    *,
    content_steps: int,
    content_dimension: int,
    feature_cfg: AcousticFeatureConfig,
    timbre_dimension: int,
    codebooks: int,
    code_steps: int,
) -> dict[str, np.ndarray]:
    return {
        "content_tokens": np.zeros(
            (size, content_steps, content_dimension), dtype=np.float32
        ),
        "content_token_mask": np.zeros((size, content_steps), dtype=bool),
        "realization_features": np.zeros(
            (size, feature_cfg.max_frames, feature_cfg.mel_bins + 4), dtype=np.float32
        ),
        "realization_frame_mask": np.zeros((size, feature_cfg.max_frames), dtype=bool),
        "log_mel_energy": np.full(
            (size, feature_cfg.mel_bins, feature_cfg.max_frames),
            feature_cfg.min_db,
            dtype=np.float32,
        ),
        "f0_log_hz": np.zeros((size, feature_cfg.max_frames), dtype=np.float32),
        "voicing": np.zeros((size, feature_cfg.max_frames), dtype=np.float32),
        "log_rms_dbfs": np.full(
            (size, feature_cfg.max_frames), feature_cfg.min_db, dtype=np.float32
        ),
        "activity_mask": np.zeros((size, feature_cfg.max_frames), dtype=bool),
        "timbre_global": np.zeros((size, timbre_dimension), dtype=np.float32),
        "has_timbre": np.zeros(size, dtype=bool),
        "encodec_codes": np.zeros((size, codebooks, code_steps), dtype=np.int16),
        "encodec_scale": np.ones((size, 1), dtype=np.float32),
        "encodec_scale_valid": np.zeros(size, dtype=bool),
        "code_valid_mask": np.zeros((size, codebooks, code_steps), dtype=bool),
        "has_codec": np.zeros(size, dtype=bool),
    }


def _valid_existing_shard(
    path: Path, expected_keys: Sequence[str], digest: str | None
) -> bool:
    if digest is None or not path.is_file() or file_sha256(path) != digest:
        return False
    try:
        with np.load(path, allow_pickle=False) as raw:
            return np.asarray(raw["keys"]).astype(str).tolist() == list(expected_keys)
    except Exception:
        return False


def _training_statistics(
    output: Path,
    shard_names: Sequence[str],
    records_by_key: dict[str, dict[str, Any]],
    realization_dimension: int,
    timbre_dimension: int,
) -> dict[str, np.ndarray]:
    total = np.zeros(realization_dimension, dtype=np.float64)
    squares = np.zeros(realization_dimension, dtype=np.float64)
    count = 0
    prototype_values: dict[str, list[np.ndarray]] = {}
    for shard_name in shard_names:
        with np.load(output / shard_name, allow_pickle=False) as raw:
            keys = np.asarray(raw["keys"]).astype(str)
            features = np.asarray(raw["realization_features"], dtype=np.float64)
            masks = np.asarray(raw["realization_frame_mask"], dtype=bool)
            timbre = np.asarray(raw["timbre_global"], dtype=np.float32)
            has_timbre = np.asarray(raw["has_timbre"], dtype=bool)
            for row, key in enumerate(keys):
                metadata = records_by_key[str(key)]
                if not bool(metadata["fit_split"]):
                    continue
                valid = features[row, masks[row]]
                if len(valid):
                    total += valid.sum(axis=0)
                    squares += np.square(valid).sum(axis=0)
                    count += len(valid)
                if metadata["dataset"] == "feis" and has_timbre[row]:
                    prototype_id = (
                        f"{metadata['eeg_subject_group_id']}::{metadata['content_id']}"
                    )
                    prototype_values.setdefault(prototype_id, []).append(timbre[row])
    if count:
        mean = total / count
        variance = np.maximum(squares / count - np.square(mean), 1.0e-8)
        std = np.sqrt(variance)
    else:
        mean = np.zeros(realization_dimension, dtype=np.float64)
        std = np.ones(realization_dimension, dtype=np.float64)
    prototype_ids = sorted(prototype_values)
    if prototype_ids:
        prototypes = np.stack(
            [np.mean(np.stack(prototype_values[key]), axis=0) for key in prototype_ids]
        ).astype(np.float32)
        norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
        prototypes = prototypes / np.maximum(norms, 1.0e-8)
    else:
        prototypes = np.zeros((0, timbre_dimension), dtype=np.float32)
    return {
        "realization_mean": mean.astype(np.float32),
        "realization_std": std.astype(np.float32),
        "realization_frame_count": np.asarray(count, dtype=np.int64),
        "feis_prototype_ids": np.asarray(prototype_ids),
        "feis_timbre_prototypes": prototypes,
        "fit_split_only": np.asarray(True),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the v0724 factorized teacher/cache schema v2"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Explicitly permit Hugging Face downloads",
    )
    parser.add_argument(
        "--skip-speaker",
        action="store_true",
        help="Use deterministic acoustic-statistics fallback",
    )
    parser.add_argument(
        "--skip-codec", action="store_true", help="Feature/cache smoke test only"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.shard_size < 1:
        raise ValueError("batch-size and shard-size must be positive")
    config_path, cfg = load_config(args.config)
    context = load_context(config_path)
    if (
        str(cfg["data"].get("pairing_policy_version"))
        != "openvoice-factorized-pairing-v2"
    ):
        raise ValueError("v0724 cache requires openvoice-factorized-pairing-v2")
    output = (
        args.output.resolve()
        if args.output is not None
        else resolve_config_path(config_path, cfg["paths"]["teacher_cache"])
    )
    if output.exists() and args.rebuild:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    records = build_project_records(context)
    prep_cfg = preparation_config(cfg)
    feature_cfg = acoustic_config(cfg)
    if prep_cfg.max_samples != int(
        cfg["audio"].get("max_samples", prep_cfg.max_samples)
    ):
        raise ValueError(
            "audio.max_samples disagrees with sample_rate * max_active_seconds"
        )
    if feature_cfg.max_frames != int(cfg["model"]["energy_frames"]):
        raise ValueError("audio.mel_frames and model.energy_frames must match")
    hubert_reference = model_reference(config_path, cfg["teachers"]["hubert_model"])
    wavlm_value = cfg["teachers"].get("wavlm_model")
    wavlm_reference = (
        model_reference(config_path, wavlm_value) if wavlm_value else "disabled"
    )
    codec_reference = str(
        resolve_config_path(config_path, cfg["paths"]["encodec_model"])
    )
    model_lineage = {
        "hubert": model_artifact_identity(hubert_reference),
        "wavlm": model_artifact_identity(wavlm_reference),
        "encodec": model_artifact_identity(codec_reference),
    }
    fingerprint = _build_fingerprint(
        config_path, context, records, prep_cfg, feature_cfg, args, model_lineage
    )
    index_path = output / "index.json"
    if index_path.is_file() and not args.rebuild:
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if existing.get("build_fingerprint") != fingerprint:
            raise ValueError(
                "Existing v0724 cache lineage differs; rerun with --rebuild"
            )
        audit = TeacherCacheV2(output, verify_hashes=True).audit()
        print(
            json.dumps(
                {"output": str(output), "status": "already-complete", **audit}, indent=2
            )
        )
        return

    state_path = output / ".build_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("schema_version") != BUILD_STATE_SCHEMA_VERSION
            or state.get("build_fingerprint") != fingerprint
        ):
            raise ValueError(
                "Partial v0724 cache has incompatible lineage; rerun with --rebuild"
            )
    else:
        state = {
            "schema_version": BUILD_STATE_SCHEMA_VERSION,
            "build_fingerprint": fingerprint,
            "completed_shards": {},
        }
        _atomic_json(state_path, state)

    # Keep locked-test tensors in physically separate shards. Training rows can
    # therefore never cause a test array to enter the shard LRU as a side
    # effect of reading a neighbouring train record.
    records_by_split: dict[str, list[AudioRecordSpec]] = {
        "train": [],
        "validation": [],
        "test": [],
        "shared": [],
    }
    for record in records:
        bucket = (
            record.split_names[0]
            if len(record.split_names) == 1
            and record.split_names[0] in {"train", "validation", "test"}
            else "shared"
        )
        records_by_split[bucket].append(record)
    shard_specs: list[tuple[str, list[AudioRecordSpec]]] = []
    for bucket in ("train", "validation", "test", "shared"):
        bucket_records = records_by_split[bucket]
        for start in range(0, len(bucket_records), args.shard_size):
            name = f"records_{bucket}_{start // args.shard_size:05d}.npz"
            shard_specs.append((name, bucket_records[start : start + args.shard_size]))
    pending = [
        (name, items)
        for name, items in shard_specs
        if not _valid_existing_shard(
            output / name,
            [item.audio_key for item in items],
            state.get("completed_shards", {}).get(name),
        )
    ]

    device = torch.device(args.device) if args.device else default_device()
    teachers_cfg = cfg["teachers"]
    hubert = None
    speaker = None
    codec = None
    if pending:
        hubert = HubertLayerTeacher(
            hubert_reference,
            layer=int(teachers_cfg.get("hubert_layer", 9)),
            token_steps=int(teachers_cfg.get("content_steps", 50)),
            device=device,
            local_files_only=not bool(args.allow_network),
        )
        expected_content_dimension = int(
            teachers_cfg.get("hubert_dimension", hubert.dimension)
        )
        if hubert.dimension != expected_content_dimension:
            raise ValueError(
                f"HuBERT dimension {hubert.dimension} != configured {expected_content_dimension}"
            )
        wavlm_name = teachers_cfg.get("wavlm_model")
        speaker = SpeakerTeacher(
            wavlm_reference if wavlm_name else None,
            expected_dimension=int(teachers_cfg.get("wavlm_dimension", 512)),
            device=device,
            local_files_only=not bool(args.allow_network),
            required=bool(teachers_cfg.get("wavlm_required", False)),
            disabled=bool(args.skip_speaker),
        )
        codec = CodecTeacher(cfg, config_path, device, disabled=bool(args.skip_codec))
        effective = {
            "speaker_backend": speaker.backend,
            "speaker_revision": speaker.revision,
            "codec_enabled": not bool(args.skip_codec),
        }
        saved_effective = state.get("effective_teachers")
        if saved_effective is not None and saved_effective != effective:
            raise ValueError(
                f"Resumed v0724 cache teacher backend changed: {saved_effective} != {effective}; use --rebuild"
            )
        state["effective_teachers"] = effective
        state["speaker_load_error"] = speaker.load_error
        _atomic_json(state_path, state)

    content_steps = int(teachers_cfg.get("content_steps", 50))
    content_dimension = int(teachers_cfg.get("hubert_dimension", 768))
    timbre_dimension = int(teachers_cfg.get("wavlm_dimension", 512))
    codebooks = int(cfg["codec"]["codebooks"])
    code_steps = int(cfg["codec"]["code_steps"])
    metadata_by_key: dict[str, dict[str, Any]] = {
        record.audio_key: record.index_metadata() for record in records
    }

    for shard_name, shard_records in tqdm(
        pending, desc="[0724 teacher] shards", unit="shard"
    ):
        arrays = _empty_shard_arrays(
            len(shard_records),
            content_steps=content_steps,
            content_dimension=content_dimension,
            feature_cfg=feature_cfg,
            timbre_dimension=timbre_dimension,
            codebooks=codebooks,
            code_steps=code_steps,
        )
        for start in range(0, len(shard_records), args.batch_size):
            chunk = shard_records[start : start + args.batch_size]
            prepared: list[PreparedWaveform] = []
            acoustics: list[AcousticFeatures] = []
            file_counts: list[int] = []
            for record in chunk:
                audio, source_rate, total_frames = _read_record_audio(record)
                item = prepare_waveform_segment(audio, source_rate, prep_cfg)
                prepared.append(item)
                acoustics.append(
                    extract_acoustic_features(
                        item.waveform,
                        valid_samples=item.valid_samples,
                        config=feature_cfg,
                    )
                )
                file_counts.append(total_frames)
            chunk_offset = start
            for local, (record, item, features, total_frames) in enumerate(
                zip(chunk, prepared, acoustics, file_counts)
            ):
                position = chunk_offset + local
                arrays["realization_features"][position] = features.realization_features
                arrays["realization_frame_mask"][position] = features.frame_valid_mask
                arrays["log_mel_energy"][position] = features.log_mel_energy
                arrays["f0_log_hz"][position] = features.log_f0_hz
                arrays["voicing"][position] = features.voicing
                arrays["log_rms_dbfs"][position] = features.log_rms_dbfs
                arrays["activity_mask"][position] = features.activity_mask
                reconstruction_eligible = bool(
                    record.audio_generation_eligible and item.reconstruction_eligible
                )
                metadata_by_key[record.audio_key].update(
                    {
                        "native_file_sample_count": int(total_frames),
                        "native_sample_count": int(item.native_sample_count),
                        "native_sample_rate": int(item.source_sample_rate),
                        "resampled_sample_count": int(item.resampled_sample_count),
                        "sample_rate": int(item.sample_rate),
                        "native_rms": float(item.native_rms),
                        "normalization_gain": float(item.normalization_gain),
                        "active_start_sample": int(item.active_start_sample),
                        "active_end_sample": int(item.active_end_sample),
                        "context_start_sample": int(item.context_start_sample),
                        "context_end_sample": int(item.context_end_sample),
                        "segment_source_start_sample": int(
                            item.segment_source_start_sample
                        ),
                        "segment_source_end_sample": int(
                            item.segment_source_end_sample
                        ),
                        "segment_valid_samples": int(item.valid_samples),
                        "segment_pcm_sha256": str(item.pcm_sha256),
                        "source_audio_sha256": file_sha256(record.audio_path),
                        "active_duration_seconds": float(item.active_duration_seconds),
                        "has_activity": bool(item.has_activity),
                        "exceeds_max_active_seconds": bool(
                            item.exceeds_max_active_seconds
                        ),
                        "reconstruction_eligible": reconstruction_eligible,
                        "audio_generation_eligible": reconstruction_eligible,
                        "realization_valid_frames": int(
                            features.frame_valid_mask.sum()
                        ),
                    }
                )

            # ds004306 intentionally receives no frozen audio supervision.
            content_positions = [
                i for i, record in enumerate(chunk) if record.dataset != "ds004306"
            ]
            if content_positions:
                assert hubert is not None
                values, masks = hubert.encode([prepared[i] for i in content_positions])
                for source_position, values_row, mask_row in zip(
                    content_positions, values, masks
                ):
                    destination = chunk_offset + source_position
                    arrays["content_tokens"][destination] = values_row
                    arrays["content_token_mask"][destination] = mask_row
                    metadata_by_key[chunk[source_position].audio_key][
                        "content_token_valid_steps"
                    ] = int(mask_row.sum())

            feature_positions = [
                i
                for i, record in enumerate(chunk)
                if bool(metadata_by_key[record.audio_key]["reconstruction_eligible"])
            ]
            if feature_positions:
                assert speaker is not None and codec is not None
                timbre = speaker.encode(
                    [prepared[i] for i in feature_positions],
                    [acoustics[i] for i in feature_positions],
                )
                encoded = codec.encode([prepared[i] for i in feature_positions])
                for selected_row, source_position in enumerate(feature_positions):
                    destination = chunk_offset + source_position
                    arrays["timbre_global"][destination] = timbre[selected_row]
                    arrays["has_timbre"][destination] = True
                    for name in (
                        "encodec_codes",
                        "encodec_scale",
                        "encodec_scale_valid",
                        "code_valid_mask",
                        "has_codec",
                    ):
                        arrays[name][destination] = encoded[name][selected_row]
                    metadata = metadata_by_key[chunk[source_position].audio_key]
                    metadata["code_valid_steps"] = int(
                        encoded["code_valid_mask"][selected_row, 0].sum()
                    )
                    metadata["timbre_available"] = True
                    metadata["codec_available"] = bool(
                        encoded["has_codec"][selected_row]
                    )
            for record in chunk:
                metadata = metadata_by_key[record.audio_key]
                metadata.setdefault("content_token_valid_steps", 0)
                metadata.setdefault("code_valid_steps", 0)
                metadata.setdefault("timbre_available", False)
                metadata.setdefault("codec_available", False)

        shard_payload = {
            "schema_version": np.asarray(TEACHER_CACHE_SCHEMA_VERSION),
            "build_fingerprint": np.asarray(fingerprint),
            "keys": np.asarray([record.audio_key for record in shard_records]),
            **{
                name: (
                    value.astype(np.float16)
                    if name
                    in {
                        "content_tokens",
                        "realization_features",
                        "log_mel_energy",
                        "f0_log_hz",
                        "voicing",
                        "log_rms_dbfs",
                        "timbre_global",
                    }
                    else value
                )
                for name, value in arrays.items()
            },
        }
        shard_path = output / shard_name
        _write_npz_atomic(shard_path, shard_payload)
        digest = file_sha256(shard_path)
        state.setdefault("completed_shards", {})[shard_name] = digest
        state.setdefault("record_metadata", {}).update(
            {
                record.audio_key: metadata_by_key[record.audio_key]
                for record in shard_records
            }
        )
        _atomic_json(state_path, state)

    # Metadata for resumed shards is recovered from the state written only
    # after each shard was atomically completed.
    metadata_by_key.update(
        {
            str(key): dict(value)
            for key, value in state.get("record_metadata", {}).items()
        }
    )
    missing_metadata = [
        key
        for key in metadata_by_key
        if "segment_pcm_sha256" not in metadata_by_key[key]
    ]
    if missing_metadata:
        raise RuntimeError(
            f"Completed shards lack resumable record metadata: {missing_metadata[:3]}"
        )

    shard_names = [name for name, _ in shard_specs]
    statistics_shards = [
        name for name, items in shard_specs if any(item.fit_split for item in items)
    ]
    statistics = _training_statistics(
        output,
        statistics_shards,
        metadata_by_key,
        feature_cfg.mel_bins + 4,
        timbre_dimension,
    )
    statistics_path = output / "train_statistics.npz"
    _write_npz_atomic(statistics_path, statistics)
    file_digests = {name: file_sha256(output / name) for name in shard_names}
    file_digests[statistics_path.name] = file_sha256(statistics_path)
    record_index: dict[str, list[Any]] = {}
    for shard_name, shard_records in shard_specs:
        for row, record in enumerate(shard_records):
            record_index[record.audio_key] = [shard_name, row]
    aggregate = hashlib.sha256()
    for name, digest in sorted(file_digests.items()):
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")

    effective_teachers = state.get("effective_teachers", {})
    speaker_backend = str(
        effective_teachers.get(
            "speaker_backend",
            speaker.backend if speaker is not None else "resumed_from_cache",
        )
    )
    index = {
        "schema_version": TEACHER_CACHE_SCHEMA_VERSION,
        "build_fingerprint": fingerprint,
        "config_sha256": file_sha256(config_path),
        "manifest_sha256": file_sha256(context.manifest_path),
        "subject_split_sha256": file_sha256(context.split_path),
        "subject_split_version": str(context.split.get("version", "unknown")),
        "pairing_policy_version": str(cfg["data"]["pairing_policy_version"]),
        "model_lineage": model_lineage,
        "sample_rate": int(prep_cfg.sample_rate),
        "max_samples": int(prep_cfg.max_samples),
        "max_active_seconds": float(prep_cfg.max_active_seconds),
        "content_steps": content_steps,
        "content_dimension": content_dimension,
        "hubert_model": str(teachers_cfg["hubert_model"]),
        "hubert_layer": int(teachers_cfg.get("hubert_layer", 9)),
        "realization_frames": int(feature_cfg.max_frames),
        "realization_dimension": int(feature_cfg.mel_bins + 4),
        "mel_bins": int(feature_cfg.mel_bins),
        "timbre_dimension": timbre_dimension,
        "wavlm_model": str(teachers_cfg.get("wavlm_model") or "disabled"),
        "speaker_backend": speaker_backend,
        "speaker_revision": effective_teachers.get("speaker_revision", "unknown"),
        "speaker_load_error": state.get("speaker_load_error"),
        "codebooks": codebooks,
        "code_steps": code_steps,
        "codec_enabled": not bool(args.skip_codec),
        "codec_sample_rate": codec.codec_sample_rate if codec is not None else 0,
        "feature_config": asdict(feature_cfg),
        "preparation_config": asdict(prep_cfg),
        "record_index": record_index,
        "records": metadata_by_key,
        "statistics_file": statistics_path.name,
        "statistics_fit_split_only": True,
        "split_isolated_shards": True,
        "statistics_shards": statistics_shards,
        "file_sha256": file_digests,
        "content_sha256": aggregate.hexdigest(),
        "resumable_build": True,
        "locked_test_encoded_but_not_fit": True,
        "locked_test_physically_isolated": True,
        "ds004306_audio_generation_eligible": False,
    }
    _atomic_json(index_path, index)
    audit = TeacherCacheV2(output, verify_hashes=True).audit()
    summary = {
        "schema_version": "openvoice-0724-cache-audit-v1",
        "output": str(output),
        "records": len(records),
        "shards": len(shard_names),
        "speaker_backend": speaker_backend,
        "codec_enabled": not bool(args.skip_codec),
        "reconstruction_eligible": sum(
            bool(value["reconstruction_eligible"]) for value in metadata_by_key.values()
        ),
        "content_only_over_4s": sum(
            bool(value["exceeds_max_active_seconds"]) and value["dataset"] != "ds004306"
            for value in metadata_by_key.values()
        ),
        "device": str(device),
        **audit,
    }
    _atomic_json(output / "audit.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
