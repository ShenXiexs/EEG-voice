#!/usr/bin/env python3
"""Export per-trial audio comparison bundles for exploratory EEG-to-speech runs.

The layout deliberately follows the qualitative ``pairs/.../<trial>`` bundles
in the reference project: a source WAV, model/control WAVs, energy and
spectrogram comparisons, and a self-contained provenance record.  This
project has an MFCC-to-log-mel renderer but *no validated neural vocoder*.
Consequently all generated WAVs here use deterministic Griffin--Lim inversion
of the renderer's Slaney log-mel output.  They are diagnostic listening aids,
not waveform-reconstruction metrics or claims of vocoder quality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.io import wavfile
from tqdm.auto import tqdm

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(APP / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cache_speech_targets import load_wave, slaney_filterbank
from eeg2speech.data import JointManifestDataset, homogeneous_collate, phoneme_vocabulary_from_manifest, pilot_indices
from eeg2speech.losses import counterfactual_eeg
from eeg2speech.model import AudioMFCCRenderer, JointEEGContentModel, RendererState
from paper_plot_style import COLORS, configure, plt, save_figure


SAMPLE_RATE = 16_000
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400
MEL_FRAMES = 161
WAV_NAMES = {
    "target": "01_target_logmel_griffinlim_oracle.wav",
    "single": "02_single_eeg_mfcc_griffinlim.wav",
    "joint": "03_joint_eeg_mfcc_griffinlim.wav",
    "zero": "04_joint_zero_eeg_griffinlim.wav",
    "time_shuffle": "05_joint_time_shuffled_eeg_griffinlim.wav",
    "channel_shuffle": "06_joint_channel_shuffled_eeg_griffinlim.wav",
}
SINGLE_ONLY_WAV_NAMES = {
    "target": "01_target_logmel_griffinlim_oracle.wav",
    "single": "02_ds004940_eeg_mfcc_griffinlim.wav",
    "zero": "03_zero_eeg_griffinlim.wav",
    "time_shuffle": "04_time_shuffled_eeg_griffinlim.wav",
    "channel_shuffle": "05_channel_shuffled_eeg_griffinlim.wav",
}
DISPLAY = {
    "target": "Target log-mel oracle",
    "single": "Single-dataset EEG",
    "joint": "Joint EEG",
    "zero": "Joint zero EEG",
    "time_shuffle": "Joint time-shuffled EEG",
    "channel_shuffle": "Joint channel-shuffled EEG",
}
SINGLE_ONLY_DISPLAY = {
    "target": "Target log-mel oracle",
    "single": "DS004940 EEG",
    "zero": "Zero EEG",
    "time_shuffle": "Time-shuffled EEG",
    "channel_shuffle": "Channel-shuffled EEG",
}


def resolve(path: str | Path, base: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move(batch: dict[str, Any], target: torch.device) -> dict[str, Any]:
    return {key: value.to(target) if torch.is_tensor(value) else value for key, value in batch.items()}


def normalized_waveform(value: np.ndarray, peak: float = 0.95) -> np.ndarray:
    output = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    maximum = float(np.abs(output).max()) if len(output) else 0.0
    if maximum > 1e-8:
        output = output * (peak / maximum)
    return np.clip(output, -1.0, 1.0)


def write_pcm16(path: Path, value: np.ndarray, *, peak_normalize: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = normalized_waveform(value) if peak_normalize else np.clip(
        np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0,
    )
    pcm = np.round(value * 32767.0).astype(np.int16)
    wavfile.write(path, SAMPLE_RATE, pcm)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"failed to write WAV: {path}")


def inverse_log_mel(log_mel: torch.Tensor, *, iterations: int, seed: int) -> np.ndarray:
    """Invert this repository's Slaney log-power mel with deterministic GL.

    The forward acoustic cache uses ``center=False``.  For a stable inverse
    with arbitrary fixed-length 161-frame model output, this diagnostic uses
    ``center=True`` and reconstructs a 1.6-second signal.  It intentionally
    does not pretend to be an exact inverse of the cached waveform analysis.
    """
    if log_mel.shape != (80, MEL_FRAMES):
        raise ValueError(f"expected [80,{MEL_FRAMES}] log-mel, got {tuple(log_mel.shape)}")
    if iterations < 1:
        raise ValueError("Griffin--Lim iterations must be positive")
    target = log_mel.detach().float().cpu().clamp(-23.0, 10.0)
    bank = slaney_filterbank(SAMPLE_RATE, N_FFT, 80, 50.0, 7600.0).float()
    # Minimum-norm nonnegative least-squares approximation of linear power.
    power = torch.linalg.pinv(bank) @ target.exp()
    magnitude = power.clamp_min(0.0).sqrt()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    phase_angles = 2.0 * torch.pi * torch.rand(magnitude.shape, generator=generator)
    phase = torch.polar(torch.ones_like(phase_angles), phase_angles)
    window = torch.hann_window(WIN_LENGTH)
    length = HOP_LENGTH * (MEL_FRAMES - 1)
    spectrum = magnitude.to(torch.complex64) * phase
    for _ in range(iterations):
        waveform = torch.istft(spectrum, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                               window=window, center=True, length=length)
        estimate = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                              window=window, center=True, return_complex=True)
        if estimate.shape[1] != MEL_FRAMES:
            raise RuntimeError(f"unexpected Griffin--Lim STFT frame count {estimate.shape[1]}")
        spectrum = magnitude.to(torch.complex64) * estimate / estimate.abs().clamp_min(1e-8)
    return torch.istft(spectrum, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                       window=window, center=True, length=length).numpy().astype(np.float32)


def _renderer_state(renderer: AudioMFCCRenderer, mfcc: torch.Tensor) -> RendererState:
    state = renderer(mfcc)
    if not all(torch.isfinite(value).all() for value in (state.log_mel, state.rms, state.activity_logits)):
        raise RuntimeError("renderer produced non-finite acoustic output")
    return state


def _plot_matrix_rows(output: Path, stem: str, rows: list[tuple[str, np.ndarray]], *, cmap: str = "magma") -> None:
    values = np.concatenate([matrix.reshape(-1) for _, matrix in rows])
    lower, upper = np.quantile(values[np.isfinite(values)], [0.02, 0.98])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower, upper = float(np.nanmin(values)), float(np.nanmax(values) + 1e-6)
    figure, axes = plt.subplots(len(rows), 1, figsize=(7.2, 1.65 * len(rows)), sharex=True)
    if len(rows) == 1:
        axes = [axes]
    image = None
    for axis, (label, matrix) in zip(axes, rows):
        image = axis.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest", cmap=cmap,
                            vmin=lower, vmax=upper, extent=[0, 1, 0, matrix.shape[0]])
        axis.set_ylabel(label)
    axes[-1].set_xlabel("relative acoustic time")
    figure.colorbar(image, ax=axes, pad=0.01, label="value")
    save_figure(figure, output, stem, ("png", "pdf"), dpi=300)


def _plot_energy(output: Path, target_rms: np.ndarray, target_activity: np.ndarray,
                 renderer_states: dict[str, RendererState], display: dict[str, str]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 4.9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    x = np.linspace(0.0, 1.0, MEL_FRAMES)
    axes[0].plot(x, target_rms, color="#222222", linewidth=1.7, label="Target waveform RMS")
    colors = {"single": COLORS["single"], "joint": COLORS["joint"], "zero": COLORS["zero"],
              "time_shuffle": COLORS["time_shuffle"], "channel_shuffle": COLORS["channel_shuffle"]}
    style = {key: colors[key] for key in renderer_states if key in colors}
    for key, color in style.items():
        axes[0].plot(x, renderer_states[key].rms.squeeze().detach().cpu().numpy(), color=color,
                     linewidth=1.15, alpha=0.95, label=display[key])
    axes[0].set_ylabel("frame RMS")
    axes[0].legend(loc="upper right", ncol=2, frameon=False)
    axes[1].step(x, target_activity.astype(float), where="mid", color="#222222", linewidth=1.7, label="Target activity")
    for key, color in style.items():
        axes[1].plot(x, torch.sigmoid(renderer_states[key].activity_logits.squeeze()).detach().cpu().numpy(),
                     color=color, linewidth=1.15, alpha=0.95, label=display[key])
    axes[1].set_ylabel("activity")
    axes[1].set_xlabel("relative acoustic time")
    axes[1].set_ylim(-0.05, 1.05)
    save_figure(figure, output, "energy_envelope_comparison", ("png", "pdf"), dpi=300)


def _plot_bundle(output: Path, batch: dict[str, Any], predicted: dict[str, torch.Tensor],
                 renderer_states: dict[str, RendererState], *, single_only: bool) -> None:
    target_mel = batch["acoustic_log_mel"][0].detach().cpu().numpy()
    target_mfcc = batch["content_mfcc"][0].detach().cpu().numpy()
    display = SINGLE_ONLY_DISPLAY if single_only else DISPLAY
    primary = ("target", "single", "zero") if single_only else ("target", "single", "joint", "zero")
    _plot_matrix_rows(output, "mel_comparison", [(display[key], target_mel if key == "target" else renderer_states[key].log_mel.squeeze().detach().cpu().numpy()) for key in primary])
    _plot_matrix_rows(output, "mfcc_comparison", [(display[key], target_mfcc if key == "target" else predicted[key].squeeze().detach().cpu().numpy()) for key in primary], cmap="coolwarm")
    _plot_energy(output, batch["acoustic_rms"][0].detach().cpu().numpy(),
                 batch["acoustic_activity"][0].detach().cpu().numpy(), renderer_states, display)


@dataclass(frozen=True)
class RunContext:
    checkpoint: Path
    payload: dict[str, Any]
    cfg: dict[str, Any]
    manifest: Path
    split: Path
    targets: Path
    normalizer: Path
    vocabulary: dict[str, int]
    model: JointEEGContentModel


def load_context(checkpoint: Path, target_device: torch.device) -> RunContext:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for required in ("pilot_config", "model_config", "model"):
        if required not in payload:
            raise RuntimeError(f"checkpoint is missing {required}: {checkpoint}")
    cfg = payload["pilot_config"]
    data_cfg = yaml.safe_load(resolve(cfg["data_config"], ROOT / "configs").read_text())
    artifact_root = ROOT / data_cfg["output_root"]
    stage = str(payload.get("stage", "generalization"))
    split_protocol = payload.get("split_protocol", cfg["split"]["protocol"] if stage == "overfit" else "stage2_joint_ood")
    artifact_set = payload.get("artifact_set") or ("built" if stage == "overfit" else "stage2")
    target_name = payload.get("target_name") or ("speech_targets" if artifact_set == "built" else "speech_targets_stage2")
    normalizer_name = payload.get("normalizer_name") or f"{split_protocol}_fold-{cfg['split']['fold']}"
    split = artifact_root / "splits" / f"{split_protocol}_fold-{cfg['split']['fold']}.csv"
    manifest = artifact_root / "manifests" / f"manifest_{artifact_set}.csv"
    targets = artifact_root / "speech_targets" / f"{target_name}.h5"
    normalizer = artifact_root / "normalizers" / f"{normalizer_name}.json"
    sources = artifact_root / "source_lock.json"
    needed = (sources, split, manifest, targets, normalizer)
    absent = [str(path) for path in needed if not path.exists()]
    if absent:
        raise RuntimeError(f"checkpoint artifact files are missing: {absent}")
    expected = payload.get("artifact_hashes", {})
    current = {path.name: sha256_file(path) for path in needed}
    if expected and expected != current:
        changed = sorted(key for key in current if current.get(key) != expected.get(key))
        raise RuntimeError(f"checkpoint artifact provenance mismatch: {changed}")
    model = JointEEGContentModel(**payload["model_config"]).to(target_device)
    model.load_state_dict(payload["model"])
    model.eval()
    return RunContext(checkpoint, payload, cfg, manifest, split, targets, normalizer,
                      payload.get("phoneme_vocabulary") or phoneme_vocabulary_from_manifest(manifest), model)


def load_renderer(path: Path, target_device: torch.device) -> AudioMFCCRenderer:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not payload.get("gate") or not all(payload["gate"].values()):
        raise RuntimeError("audio renderer failed/misses its audio-only oracle gate; refusing to export diagnostic audio")
    renderer = AudioMFCCRenderer(**payload["model_config"]).to(target_device)
    renderer.load_state_dict(payload["model"])
    renderer.eval()
    return renderer


def selected_indices(context: RunContext, dataset_name: str, role: str) -> tuple[JointManifestDataset, list[int]]:
    dataset = JointManifestDataset(context.manifest, context.split, role, dataset_name,
                                   context.targets, context.normalizer,
                                   float(context.cfg["loss"]["weak_content_weight"]),
                                   supervision_types={"paired_audio", "weak_audio"},
                                   phoneme_vocabulary=context.vocabulary)
    stage = str(context.payload.get("stage", "generalization")) if role == "train" else "generalization"
    return dataset, pilot_indices(dataset, context.cfg, stage, role)


def one_record(dataset: JointManifestDataset, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    row = dataset.frame.iloc[index].to_dict()
    return homogeneous_collate([dataset[index]]), row


def export_trial(*, output: Path, single: RunContext, joint: RunContext | None, renderer: AudioMFCCRenderer,
                 renderer_checkpoint: Path, dataset_name: str, role: str, index: int, seed: int,
                 iterations: int, target_device: torch.device, selection_policy: str = "all_selected_pairs") -> dict[str, Any]:
    single_dataset, single_indices = selected_indices(single, dataset_name, role)
    joint_dataset = None
    try:
        batch, row = one_record(single_dataset, index)
        trial_id = str(row["trial_id"])
        joint_batch = None
        if joint is not None:
            joint_dataset, joint_indices = selected_indices(joint, dataset_name, role)
            trial_to_joint = {str(joint_dataset.frame.iloc[item].trial_id): item for item in joint_indices}
            if trial_id not in trial_to_joint:
                raise RuntimeError(f"joint checkpoint has no matching selected trial {trial_id}")
            joint_batch, _ = one_record(joint_dataset, trial_to_joint[trial_id])
            # ``audio_id`` is intentionally derived by JointManifestDataset rather
            # than stored as a redundant manifest column.  Compare the loader's
            # derived, provenance-checked identity instead of indexing the row.
            if str(joint_batch["audio_id"][0]) != str(batch["audio_id"][0]):
                raise RuntimeError(f"single/joint audio identity differs for {trial_id}")
        batch = move(batch, target_device)
        if joint_batch is not None:
            joint_batch = move(joint_batch, target_device)
        with torch.no_grad():
            single_state = single.model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], batch["time_mask"], batch["dataset_id"])
            predictions = {"single": single_state.mfcc}
            control_model, control_batch = single.model, batch
            if joint is not None and joint_batch is not None:
                joint_state = joint.model(joint_batch["eeg"], joint_batch["channel_xyz"], joint_batch["channel_mask"], joint_batch["time_mask"], joint_batch["dataset_id"])
                predictions["joint"] = joint_state.mfcc
                control_model, control_batch = joint.model, joint_batch
            for control in ("zero", "time_shuffle", "channel_shuffle"):
                controlled = counterfactual_eeg(control_batch["eeg"], control, time_mask=control_batch["time_mask"], channel_mask=control_batch["channel_mask"])
                predictions[control] = control_model(controlled, control_batch["channel_xyz"], control_batch["channel_mask"], control_batch["time_mask"], control_batch["dataset_id"]).mfcc
            renderer_states = {key: _renderer_state(renderer, value) for key, value in predictions.items()}
        source_path = ROOT / str(row["audio_path"])
        if not source_path.exists() or sha256_file(source_path) != str(row["audio_sha256"]):
            raise RuntimeError(f"source audio is missing or changed: {source_path}")
        source, source_rate, source_channels = load_wave(source_path)
        output.mkdir(parents=True, exist_ok=True)
        # Retain the reference's original gain.  Generated files are separately
        # peak-normalized only for listening, never for plotted target energy.
        write_pcm16(output / "00_source_reference_16k.wav", source, peak_normalize=False)
        target_state = RendererState(batch["acoustic_log_mel"], batch["acoustic_rms"], torch.zeros_like(batch["acoustic_rms"]))
        waveforms = {"target": inverse_log_mel(target_state.log_mel[0], iterations=iterations, seed=seed)}
        waveforms.update({key: inverse_log_mel(state.log_mel[0], iterations=iterations, seed=seed + offset)
                          for offset, (key, state) in enumerate(renderer_states.items(), start=1)})
        wav_names = SINGLE_ONLY_WAV_NAMES if joint is None else WAV_NAMES
        for key, waveform in waveforms.items():
            write_pcm16(output / wav_names[key], waveform, peak_normalize=True)
        _plot_bundle(output, control_batch, predictions, renderer_states, single_only=joint is None)
        files = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()}
        metadata = {
            "schema_version": "eeg2speech-audio-pair-comparison-v1",
            "trial_id": trial_id, "dataset": dataset_name, "role": role,
            "selection_policy": selection_policy,
            "subject": str(row["subject"]), "task": str(row["task"]), "condition": str(row["condition"]),
            "tms_applied": str(row.get("tms_applied", "")), "linguistic_content_id": str(row["linguistic_content_id"]),
            "pairing_level": str(row["pairing_level"]), "audio_semantics": str(row["audio_semantics"]),
            "reference_status": ("verified_presented_waveform" if str(row["pairing_level"]) == "verified_exact"
                                 else "candidate_audio_reference_not_acoustic_supervision"),
            "source_audio": {"path": str(row["audio_path"]), "sha256": str(row["audio_sha256"]),
                             "original_sample_rate_hz": source_rate, "original_channels": source_channels,
                             "duration_seconds_at_16khz": len(source) / SAMPLE_RATE},
            "checkpoints": {"single": str(single.checkpoint), **({"joint": str(joint.checkpoint)} if joint is not None else {})},
            "renderer_checkpoint": str(renderer_checkpoint),
            "generation": {"sample_rate_hz": SAMPLE_RATE, "mel_frames": MEL_FRAMES,
                           "griffin_lim_iterations": iterations,
                           "generated_duration_seconds": HOP_LENGTH * (MEL_FRAMES - 1) / SAMPLE_RATE,
                           "normalization": "source WAV gain-preserving; generated Griffin-Lim WAVs independently peak-normalized to 0.95"},
            "comparison_mode": "single_only" if joint is None else "single_vs_joint",
            "interpretation": "exploratory diagnostic audio only; Griffin-Lim is not a validated neural vocoder and generated WAVs are not waveform metrics.",
            "files_sha256": files,
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return {"seed": seed, "dataset": dataset_name, "role": role, "trial_id": trial_id,
                "subject": str(row["subject"]), "folder": str(output), "pairing_level": str(row["pairing_level"])}
    finally:
        single_dataset.close()
        if joint_dataset is not None:
            joint_dataset.close()


def parse_csv(value: str, choices: set[str], flag: str) -> list[str]:
    result = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(result) - choices)
    if not result or unknown:
        raise ValueError(f"{flag} must be comma-separated subset of {sorted(choices)}; unknown={unknown}")
    return result


def one_train_representative_per_content(dataset: JointManifestDataset, ordered: list[int]) -> list[int]:
    """Keep the first stable trial for every content after deterministic sorting."""
    seen: set[str] = set()
    result: list[int] = []
    for index in ordered:
        content = str(dataset.frame.iloc[index].linguistic_content_id)
        if content not in seen:
            seen.add(content)
            result.append(index)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True, help="output root containing generalization/<mode>/seed-N checkpoints")
    parser.add_argument("--renderer-checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", default="31")
    parser.add_argument("--datasets", default="ds004940,ds006104")
    parser.add_argument("--roles", default="validation,test")
    parser.add_argument("--max-pairs", type=int, default=3, help="deterministic pairs per dataset/role/seed; 0 exports all")
    parser.add_argument("--griffin-lim-iterations", type=int, default=32)
    parser.add_argument("--one-train-representative-per-content", action="store_true",
                        help="for role=train, export one deterministic subject×trial per linguistic content")
    parser.add_argument("--manifest-name", default="export_manifest",
                        help="safe basename for this invocation's CSV/JSON manifest")
    parser.add_argument("--single-only", action="store_true",
                        help="export one dataset checkpoint and its EEG controls; do not require a joint checkpoint")
    parser.add_argument("--overwrite", action="store_true", help="replace an already complete trial bundle")
    args = parser.parse_args()
    if args.max_pairs < 0:
        parser.error("--max-pairs must be nonnegative")
    if args.griffin_lim_iterations < 1:
        parser.error("--griffin-lim-iterations must be positive")
    if not args.manifest_name.replace("_", "").replace("-", "").isalnum():
        parser.error("--manifest-name must contain only letters, numbers, _ and -")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    datasets = parse_csv(args.datasets, {"ds004940", "ds006104"}, "--datasets")
    roles = parse_csv(args.roles, {"train", "validation", "test"}, "--roles")
    experiment_root = args.experiment_root.resolve()
    renderer_path = args.renderer_checkpoint.resolve()
    if not renderer_path.exists():
        raise RuntimeError(f"renderer checkpoint is missing: {renderer_path}")
    configure(); target_device = device(); renderer = load_renderer(renderer_path, target_device)
    # Resolve the complete deterministic export plan before rendering.  Griffin--
    # Lim is intentionally relatively slow, so a pair-level progress bar is
    # more useful than printing only after each comparison bundle finishes.
    # It includes already-complete folders: a resumed invocation therefore
    # immediately shows how much work will be skipped versus rendered.
    export_plan: list[tuple[int, str, str, int, str]] = []
    contexts_by_seed: dict[int, tuple[dict[str, RunContext], RunContext | None]] = {}
    for seed in seeds:
        single_contexts = {}
        for dataset_name in datasets:
            checkpoint = experiment_root / "generalization" / dataset_name / f"seed-{seed}" / "checkpoint.pt"
            if not checkpoint.exists():
                raise RuntimeError(f"single checkpoint is missing: {checkpoint}")
            single_contexts[dataset_name] = load_context(checkpoint, target_device)
        joint_context = None
        if not args.single_only:
            joint_checkpoint = experiment_root / "generalization" / "joint" / f"seed-{seed}" / "checkpoint.pt"
            if not joint_checkpoint.exists():
                raise RuntimeError(f"joint checkpoint is missing: {joint_checkpoint}")
            joint_context = load_context(joint_checkpoint, target_device)
        contexts_by_seed[seed] = (single_contexts, joint_context)
        for dataset_name in datasets:
            for role in roles:
                dataset, indices = selected_indices(single_contexts[dataset_name], dataset_name, role)
                try:
                    ordered = sorted(indices, key=lambda item: hashlib.sha256(
                        str(dataset.frame.iloc[item].trial_id).encode()).hexdigest())
                    if args.one_train_representative_per_content and role == "train":
                        ordered = one_train_representative_per_content(dataset, ordered)
                    trial_ids = {item: str(dataset.frame.iloc[item].trial_id) for item in ordered}
                finally:
                    dataset.close()
                if args.max_pairs:
                    ordered = ordered[:args.max_pairs]
                for index in ordered:
                    export_plan.append((seed, dataset_name, role, index, trial_ids[index]))

    selection_policy = ("one_train_representative_per_content"
                        if args.one_train_representative_per_content else "all_selected_pairs")
    print(json.dumps({"status": "export_plan_ready", "pairs_total": len(export_plan),
                      "griffin_lim_iterations": args.griffin_lim_iterations,
                      "selection_policy": selection_policy,
                      "overwrite": args.overwrite}, indent=2))
    exported: list[dict[str, Any]] = []
    skipped_complete = 0
    progress = tqdm(export_plan, desc="Audio comparison bundles", unit="pair", dynamic_ncols=True, mininterval=0.5)
    for seed, dataset_name, role, index, trial_id in progress:
        progress.set_postfix_str(f"seed={seed} {dataset_name} {role}", refresh=False)
        folder = experiment_root / "generalization" / "audio_pair_comparisons" / f"seed-{seed}" / dataset_name / role / trial_id
        if (folder / "metadata.json").exists() and not args.overwrite:
            skipped_complete += 1
            continue
        single_contexts, joint_context = contexts_by_seed[seed]
        exported.append(export_trial(output=folder, single=single_contexts[dataset_name], joint=joint_context,
                                     renderer=renderer, renderer_checkpoint=renderer_path,
                                     dataset_name=dataset_name, role=role, index=index, seed=seed,
                                     iterations=args.griffin_lim_iterations, target_device=target_device,
                                     selection_policy=selection_policy))
    output_root = experiment_root / "generalization" / "audio_pair_comparisons"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / f"{args.manifest_name}.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "dataset", "role", "trial_id", "subject", "pairing_level", "folder"])
        writer.writeheader(); writer.writerows(exported)
    summary = {"schema_version": "eeg2speech-audio-pair-export-v1", "experiment_root": str(experiment_root),
               "renderer_checkpoint": str(renderer_path), "renderer_sha256": sha256_file(renderer_path),
               "pairs_considered_this_invocation": len(export_plan),
               "exports_written_this_invocation": len(exported),
               "skipped_complete_this_invocation": skipped_complete, "manifest_csv": str(manifest),
               "selection_policy": selection_policy,
               "warning": "All generated WAVs are Griffin-Lim diagnostics from predicted log-mel, not outputs from a validated neural vocoder."}
    (output_root / f"{args.manifest_name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
