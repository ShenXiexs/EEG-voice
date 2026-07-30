from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from src.open_vocab_0724.audio_features import AudioPreparationConfig

from .data import PreparedRecords, _read_waveform, light_prepare_waveform


class ECAPAEncoder:
    """Thin, explicit wrapper around a pinned SpeechBrain ECAPA checkpoint."""

    def __init__(self, *, source: str, savedir: Path, device: torch.device):
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
    cache_dir = (config_path.parent / cfg["paths"]["speaker_model_root"]).resolve()
    encoder = ECAPAEncoder(source=source, savedir=cache_dir, device=device)
    audio_root = (config_path.parent / cfg["data"]["audio_root"]).resolve()
    manifest = (config_path.parent / cfg["data"]["unified_manifest"]).resolve()
    paths = _audio_paths(manifest)
    prep_cfg = AudioPreparationConfig(
        sample_rate=int(cfg["audio"]["sample_rate"]),
        max_active_seconds=float(cfg["audio"]["max_active_seconds"]),
        target_rms=float(cfg["audio"]["target_rms"]),
    )
    keys = records.arrays["sample_keys"].astype(str)
    subjects = records.arrays["subjects"].astype(str)
    embeddings: list[np.ndarray] = []
    for key in tqdm(keys.tolist(), desc="[v3 prepare] ECAPA audio references", unit="trial", dynamic_ncols=True):
        waveform, rate = _read_waveform(audio_root / paths[key])
        prepared, _ = light_prepare_waveform(waveform, rate, prep_cfg)
        embeddings.append(encoder.encode(prepared.waveform[: max(1, prepared.valid_samples)]))
    values = np.stack(embeddings).astype(np.float32)
    expected_dimension = int(cfg["speaker"]["embedding_dimension"])
    if values.shape[1] != expected_dimension:
        raise ValueError(
            f"configured ECAPA dimension is {expected_dimension}, but {source} returned {values.shape[1]}"
        )
    reference = np.zeros_like(values)
    reference_keys: list[str] = []
    fit = (records.roles == "fit") & records.arrays["fit_eligible"].astype(bool)
    eligible = records.arrays["fit_eligible"].astype(bool)
    for index, subject in enumerate(subjects.tolist()):
        # Fit trials use only clean/eligible fit references, so validation,
        # locked, overlong, and pending-manual-review audio cannot leak into
        # audio-decoder training. Held-out rows get an eligible same-subject
        # fallback solely to keep the audio-only cache structurally complete.
        candidates = [candidate for candidate in np.flatnonzero((subjects == subject) & fit).tolist() if candidate != index]
        if not candidates:
            candidates = [candidate for candidate in np.flatnonzero((subjects == subject) & eligible).tolist() if candidate != index]
        if not candidates:
            raise ValueError(f"no non-target speaker reference exists for {keys[index]}")
        selected = sorted(candidates, key=lambda item: keys[item])[: int(cfg["speaker"]["reference_trials"])]
        vector = values[selected].mean(0)
        reference[index] = vector / max(float(np.linalg.norm(vector)), 1.0e-8)
        reference_keys.append("|".join(keys[selected].tolist()))
    subject_centroids = []
    for subject in sorted(set(subjects[fit].tolist())):
        # Canonical voice is fit-only by construction.  Do not derive it via
        # reference vectors, because those may intentionally include a
        # same-subject non-target trial for the separate V2 audio-only oracle.
        vector = values[(subjects == subject) & fit].mean(0)
        subject_centroids.append(vector / max(float(np.linalg.norm(vector)), 1.0e-8))
    centers = np.stack(subject_centroids)
    similarity = centers @ centers.T
    medoid = centers[np.argmax(similarity.mean(1))]
    records.arrays["speaker_target_embedding"] = values
    records.arrays["speaker_reference_embedding"] = reference
    records.arrays["speaker_reference_keys"] = np.asarray(reference_keys)
    records.arrays["canonical_voice"] = medoid.astype(np.float32)
    return {
        "backend": "speechbrain_ecapa", "model_id": source, "model_cache": str(cache_dir),
        "embedding_dimension": int(values.shape[1]), "reference_trials": int(cfg["speaker"]["reference_trials"]),
        "canonical_voice_policy": "fit_subject_centroid_medoid", "canonical_voice_subject_count": len(subject_centroids),
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
