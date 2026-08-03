from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from src.open_vocab_0724.audio_features import AudioPreparationConfig

from .data import PreparedRecords, _accepted_denoise_paths, _read_waveform, light_prepare_waveform
from .runtime import output_path


class ECAPAEncoder:
    """SpeechBrain ECAPA wrapper with an optional KaraOne-adapted backbone."""

    def __init__(
        self,
        *,
        source: str,
        savedir: Path,
        device: torch.device,
        adapted_checkpoint: Path | None = None,
    ):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as error:  # pragma: no cover - environment specific
            raise RuntimeError(
                "Gate V2 requires SpeechBrain ECAPA. Install requirements_v3.txt and cache the configured model."
            ) from error
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source=source, savedir=str(savedir), run_opts={"device": str(device)}
        )
        self.adapted_checkpoint = adapted_checkpoint
        if adapted_checkpoint is not None:
            if not adapted_checkpoint.is_file():
                raise FileNotFoundError(
                    f"KaraOne ECAPA adaptation is missing: {adapted_checkpoint}. "
                    "Run the configured v3 audio-adaptation stage first."
                )
            payload = torch.load(adapted_checkpoint, map_location="cpu", weights_only=False)
            if str(payload.get("source")) != str(source):
                raise ValueError("ECAPA adaptation source does not match the configured pretrained model")
            states = payload.get("module_state_dicts")
            if not isinstance(states, dict) or "embedding_model" not in states:
                raise ValueError("invalid KaraOne ECAPA adaptation checkpoint")
            for name, state in states.items():
                if not hasattr(self.model.mods, name):
                    raise ValueError(f"ECAPA adaptation references missing module {name}")
                getattr(self.model.mods, name).load_state_dict(state, strict=True)
        for module in self.model.mods.values():
            module.eval()

    @torch.no_grad()
    def encode(self, waveform: np.ndarray) -> np.ndarray:
        value = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).view(1, -1).to(self.device)
        embedding = self.model.encode_batch(value)
        result = embedding.detach().cpu().numpy().reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(result))
        return result / norm if norm > 1.0e-8 else result


def _audio_paths(manifest: Path) -> dict[str, str]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        return {
            str(row["sample_key"]): str(row["audio_relpath"])
            for row in csv.DictReader(handle)
            if row.get("dataset") == "karaone"
        }


def attach_speaker_embeddings(
    records: PreparedRecords, *, config_path: Path, cfg: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Attach non-target same-subject reference embeddings to the v3 cache.

    These fields are only consumed by V1/V2 audio oracles.  The EEG primary
    path always uses the cached canonical voice, never a target-subject voice.
    """
    source = str(cfg["speaker"]["model_id"])
    # These are experiment artifacts, so they must pass through output_path.
    # In explore mode it redirects both paths into the isolated ``_explore``
    # namespace.  Resolving cfg["paths"] directly here previously made prepare
    # look for the primary checkpoint after adaptation had correctly written
    # the exploratory one.
    cache_dir = output_path(config_path, cfg, "speaker_model_root")
    use_adapted = str(cfg.get("speaker", {}).get("conditioning", "adapted")).lower() == "adapted"
    adapted_checkpoint = output_path(config_path, cfg, "speaker_adapted_checkpoint") if use_adapted else None
    encoder = ECAPAEncoder(
        source=source, savedir=cache_dir, device=device, adapted_checkpoint=adapted_checkpoint
    )
    audit_encoder = ECAPAEncoder(source=source, savedir=cache_dir, device=device) if use_adapted else encoder
    audio_root = (config_path.parent / cfg["data"]["audio_root"]).resolve()
    manifest = (config_path.parent / cfg["data"]["unified_manifest"]).resolve()
    paths = _audio_paths(manifest)
    denoised_paths = _accepted_denoise_paths(config_path, cfg)
    prep_cfg = AudioPreparationConfig(
        sample_rate=int(cfg["audio"]["sample_rate"]),
        max_active_seconds=float(cfg["audio"]["max_active_seconds"]),
        target_rms=float(cfg["audio"]["target_rms"]),
    )
    keys = records.arrays["sample_keys"].astype(str)
    subjects = records.arrays["subjects"].astype(str)
    cp_temporal = str(cfg.get("version", "")) == "openvoice-v3-cp-temporal-large-v1"
    fit = (records.roles == "fit") & records.arrays["fit_eligible"].astype(bool)
    if cp_temporal and "fit_internal_dev" in records.arrays:
        fit &= ~records.arrays["fit_internal_dev"].astype(bool)
    expected_dimension = int(cfg["speaker"]["embedding_dimension"])
    embeddings: list[np.ndarray] = []
    audit_embeddings: list[np.ndarray] = []
    for index, key in enumerate(tqdm(keys.tolist(), desc="[v3 prepare] ECAPA audio references", unit="trial", dynamic_ncols=True)):
        # CP-temporal primary synthesis uses a fit-only canonical voice. Do not
        # run the generation-side speaker model over held-out WAVs before the
        # listening gate merely to populate unused target-speaker fields.
        if cp_temporal and not fit[index]:
            embeddings.append(np.zeros(expected_dimension, dtype=np.float32))
            audit_embeddings.append(np.zeros(expected_dimension, dtype=np.float32))
            continue
        waveform, rate = _read_waveform(denoised_paths.get(key, audio_root / paths[key]))
        prepared, _ = light_prepare_waveform(waveform, rate, prep_cfg)
        active_wave = prepared.waveform[: max(1, prepared.valid_samples)]
        embedding = encoder.encode(active_wave)
        embeddings.append(embedding)
        audit_embeddings.append(audit_encoder.encode(active_wave) if audit_encoder is not encoder else embedding.copy())
    values = np.stack(embeddings).astype(np.float32)
    audit_values = np.stack(audit_embeddings).astype(np.float32)
    if values.shape[1] != expected_dimension:
        raise ValueError(
            f"configured ECAPA dimension is {expected_dimension}, but {source} returned {values.shape[1]}"
        )
    reference = np.zeros_like(values)
    audit_reference = np.zeros_like(audit_values)
    reference_mfcc_mean = np.zeros_like(records.arrays["mfcc_mean"], dtype=np.float32)
    reference_mfcc_std = np.zeros_like(records.arrays["mfcc_std"], dtype=np.float32)
    reference_keys: list[str] = []
    eligible = records.arrays["fit_eligible"].astype(bool)
    for index, subject in enumerate(subjects.tolist()):
        # Fit trials use only clean/eligible fit references, so validation,
        # locked, overlong, and pending-manual-review audio cannot leak into
        # audio-decoder training. Held-out rows get an eligible same-subject
        # fallback solely to keep the audio-only cache structurally complete.
        candidates = [candidate for candidate in np.flatnonzero((subjects == subject) & fit).tolist() if candidate != index]
        if cp_temporal and not candidates:
            reference_keys.append("fit_only_canonical_pending")
            continue
        if not candidates:
            candidates = [candidate for candidate in np.flatnonzero((subjects == subject) & eligible).tolist() if candidate != index]
        if not candidates:
            raise ValueError(f"no non-target speaker reference exists for {keys[index]}")
        selected = sorted(candidates, key=lambda item: keys[item])[: int(cfg["speaker"]["reference_trials"])]
        vector = values[selected].mean(0)
        reference[index] = vector / max(float(np.linalg.norm(vector)), 1.0e-8)
        audit_vector = audit_values[selected].mean(0)
        audit_reference[index] = audit_vector / max(float(np.linalg.norm(audit_vector)), 1.0e-8)
        reference_mfcc_mean[index] = records.arrays["mfcc_mean"][selected].mean(0)
        reference_mfcc_std[index] = records.arrays["mfcc_std"][selected].mean(0)
        reference_keys.append("|".join(keys[selected].tolist()))
    subject_centroids = []
    center_subjects = sorted(set(subjects[fit].tolist()))
    for subject in center_subjects:
        # Canonical voice is fit-only by construction.  Do not derive it via
        # reference vectors, because those may intentionally include a
        # same-subject non-target trial for the separate V2 audio-only oracle.
        vector = values[(subjects == subject) & fit].mean(0)
        subject_centroids.append(vector / max(float(np.linalg.norm(vector)), 1.0e-8))
    centers = np.stack(subject_centroids)
    similarity = centers @ centers.T
    medoid_index = int(np.argmax(similarity.mean(1)))
    medoid = centers[medoid_index]
    medoid_subject = center_subjects[medoid_index]
    medoid_trials = (subjects == medoid_subject) & fit
    if cp_temporal:
        audit_medoid = audit_values[medoid_trials].mean(0)
        audit_medoid = audit_medoid / max(float(np.linalg.norm(audit_medoid)), 1.0e-8)
        reference[~fit] = medoid
        audit_reference[~fit] = audit_medoid
        values[~fit] = medoid
        audit_values[~fit] = audit_medoid
    records.arrays["speaker_target_embedding"] = values
    records.arrays["speaker_reference_embedding"] = reference
    records.arrays["speaker_audit_target_embedding"] = audit_values
    records.arrays["speaker_audit_reference_embedding"] = audit_reference
    records.arrays["speaker_reference_mfcc_mean"] = reference_mfcc_mean
    records.arrays["speaker_reference_mfcc_std"] = reference_mfcc_std
    records.arrays["speaker_reference_keys"] = np.asarray(reference_keys)
    records.arrays["canonical_voice"] = medoid.astype(np.float32)
    records.arrays["canonical_mfcc_mean"] = records.arrays["mfcc_mean"][medoid_trials].mean(0).astype(np.float32)
    records.arrays["canonical_mfcc_std"] = records.arrays["mfcc_std"][medoid_trials].mean(0).astype(np.float32)
    return {
        "backend": "speechbrain_ecapa", "model_id": source, "model_cache": str(cache_dir),
        "conditioning_checkpoint": str(adapted_checkpoint) if adapted_checkpoint is not None else None,
        "conditioning_encoder": "KaraOne-fit-finetuned ECAPA backbone" if use_adapted else "frozen external ECAPA checkpoint",
        "audit_encoder": "untouched external ECAPA checkpoint",
        "embedding_dimension": int(values.shape[1]), "reference_trials": int(cfg["speaker"]["reference_trials"]),
        "canonical_voice_policy": "fit_subject_centroid_medoid",
        "canonical_voice_subject": medoid_subject,
        "canonical_voice_subject_count": len(subject_centroids),
        "accepted_denoised_trials": len(denoised_paths),
    }


def speaker_distribution(embeddings: np.ndarray, subjects: list[str], labels: list[str], *, max_pairs: int = 20000, seed: int = 31) -> dict[str, dict[str, float]]:
    values = np.asarray(embeddings, dtype=np.float32)
    subjects = list(map(str, subjects)); labels = [str(value).strip().strip("/").lower() for value in labels]
    pairs = [(left, right) for left in range(len(values)) for right in range(left + 1, len(values))]
    rng = np.random.default_rng(seed)
    if len(pairs) > max_pairs:
        pairs = [pairs[index] for index in rng.choice(len(pairs), max_pairs, replace=False)]
    buckets: dict[str, list[float]] = {"same_speaker": [], "different_speaker_same_label": [], "different_speaker_different_label": []}
    for left, right in pairs:
        score = float(values[left] @ values[right])
        if subjects[left] == subjects[right]:
            buckets["same_speaker"].append(score)
        elif labels[left] == labels[right]:
            buckets["different_speaker_same_label"].append(score)
        else:
            buckets["different_speaker_different_label"].append(score)
    return {
        name: {"n": len(score), "mean": float(np.mean(score)) if score else float("nan"), "p10": float(np.quantile(score, .10)) if score else float("nan"), "p90": float(np.quantile(score, .90)) if score else float("nan")}
        for name, score in buckets.items()
    }
