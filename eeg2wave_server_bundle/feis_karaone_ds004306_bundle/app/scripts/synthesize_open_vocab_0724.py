#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openvoice_0724_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/openvoice_0724_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

KARAONE_APP = APP.parents[1] / "karaone_overt_recon_bundle" / "app"
if str(KARAONE_APP) not in sys.path:
    sys.path.insert(0, str(KARAONE_APP))

from src.karaone_0715.codec import DiscreteEncodec, DiscreteEncodecConfig  # type: ignore[import-not-found]  # noqa: E402
from src.open_vocab_0722.audio_io import read_wav, write_wav  # noqa: E402
from src.open_vocab_0724.audio_features import (  # noqa: E402
    ActiveSpeechConfig,
    AudioPreparationConfig,
    prepare_waveform_segment,
    resample_audio,
)
from src.open_vocab_0724.audio_gate import require_frozen_audio_checkpoint  # noqa: E402
from src.open_vocab_0724.data import (  # noqa: E402
    FactorizedEEGDataset,
    TeacherCacheV2,
    collate_factorized,
    load_context,
)
from src.open_vocab_0724.lineage import (  # noqa: E402
    authorize_locked_test,
    authorize_locked_test_metadata,
    build_lineage,
    claim_locked_test_access,
    file_sha256,
    validate_checkpoint,
)
from src.open_vocab_0724.metrics import (  # noqa: E402
    energy_structure_metrics,
    log_mel,
    reconstruction_metrics,
    summarize,
)
from src.open_vocab_0724.model import (  # noqa: E402
    FactorizedAudioModel,
    FactorizedConditionState,
    FactorizedEEGEncoder,
)
from src.open_vocab_0724.runtime import (  # noqa: E402
    audio_model_config,
    default_device,
    eeg_model_config,
    load_config,
    move_batch,
    resolve_config_path,
    resolve_run_checkpoint,
    run_identifier,
    write_json,
)


MODES = (
    "correct_content_correct_realization",
    "correct_content_wrong_realization",
    "wrong_content_correct_realization",
    "wrong_content_wrong_realization",
    "content_only",
    "realization_only",
    "shuffled_eeg",
    "zero_eeg",
    "audio_condition_oracle",
    "codec_oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize v0724 factorized counterfactuals"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("karaone", "feis"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--generalization", choices=("g1", "g2", "g3"), default="g1")
    parser.add_argument("--holdout-label", default=None)
    parser.add_argument("--loso-subject", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument(
        "--exploratory-test",
        action="store_true",
        help=(
            "Diagnostic only: permit test synthesis without a passing "
            "validation gate. Results are not a formal locked-test result."
        ),
    )
    parser.add_argument("--test-access-id", default=None)
    parser.add_argument("--skip-decoded-content-metric", action="store_true")
    parser.add_argument("--skip-decoded-timbre-metric", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse completed per-trial records and continue an interrupted synthesis.",
    )
    return parser.parse_args()


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def expected_dependencies(
    payload: dict[str, Any],
    cfg: dict[str, Any],
    config_path: Path,
    *,
    seed: int,
    loso_subject: str | None,
    generalization: str,
    holdout_label: str | None,
) -> dict[str, str]:
    expected: dict[str, str] = {}
    declared = payload.get("dependencies") or {}
    if "audio_checkpoint_sha256" in declared:
        expected["audio_checkpoint_sha256"] = file_sha256(
            resolve_config_path(config_path, cfg["paths"]["audio_checkpoint"])
        )
    if "eeg_pretrain_checkpoint_sha256" in declared:
        expected["eeg_pretrain_checkpoint_sha256"] = file_sha256(
            resolve_run_checkpoint(
                config_path,
                cfg,
                "eeg_pretrain_checkpoint",
                seed=seed,
                loso_subject=loso_subject,
                generalization=generalization,
                holdout_label=holdout_label,
            )
        )
    return expected


def load_models(
    context: Any,
    lineage: dict[str, Any],
    device: torch.device,
    *,
    eeg_path: Path,
    seed: int,
    loso_subject: str | None,
    generalization: str,
    holdout_label: str | None,
) -> tuple[FactorizedAudioModel, FactorizedEEGEncoder, dict[str, Any]]:
    cfg = context.config
    audio_path = resolve_config_path(
        context.config_path, cfg["paths"]["audio_checkpoint"]
    )
    require_frozen_audio_checkpoint(context.config_path, cfg, lineage, audio_path)
    audio_payload = torch.load(audio_path, map_location="cpu", weights_only=False)
    validate_checkpoint(
        audio_payload, phase="audio", lineage=lineage, source=str(audio_path)
    )
    audio = FactorizedAudioModel(audio_model_config(cfg)).to(device)
    audio.load_state_dict(audio_payload["model_state"])
    audio.eval()
    eeg_payload = torch.load(eeg_path, map_location="cpu", weights_only=False)
    # The requested generalization fields are validated in ``main`` below,
    # where CLI values are available.  Here we at least enforce seed/fold.
    observed_run = eeg_payload.get("run") or {}
    if (
        int(observed_run.get("seed", -1)) != int(seed)
        or observed_run.get("loso_subject") != loso_subject
    ):
        raise ValueError(
            f"{eeg_path} run metadata does not match seed/LOSO request: "
            f"{observed_run!r}"
        )
    validate_checkpoint(
        eeg_payload,
        phase="eeg",
        lineage=lineage,
        dependencies=expected_dependencies(
            eeg_payload,
            cfg,
            context.config_path,
            seed=seed,
            loso_subject=loso_subject,
            generalization=generalization,
            holdout_label=holdout_label,
        ),
        source=str(eeg_path),
    )
    eeg = FactorizedEEGEncoder(
        eeg_model_config(
            cfg,
            num_train_subjects=len(context.subject_to_index),
            num_content_labels=len(context.label_to_index),
        )
    ).to(device)
    eeg.load_state_dict(eeg_payload["model_state"])
    eeg.eval()
    return audio, eeg, eeg_payload


def condition_from_state(state: Any, index: int) -> FactorizedConditionState:
    return FactorizedConditionState(
        fused_condition=state.fused_condition[index : index + 1],
        log_mel_energy=state.log_mel_energy[index : index + 1],
        log_f0_hz=state.log_f0_hz[index : index + 1],
        voicing_logits=state.voicing_logits[index : index + 1],
        log_rms_dbfs=state.log_rms_dbfs[index : index + 1],
        activity_logits=state.activity_logits[index : index + 1],
        duration_seconds=state.duration_seconds[index : index + 1],
    )


def counterfactuals(eeg: FactorizedEEGEncoder, state: Any, zero: Any) -> tuple[
    dict[str, FactorizedConditionState],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    ones = torch.ones(
        1,
        state.content_tokens.shape[1],
        dtype=torch.bool,
        device=state.content_tokens.device,
    )
    zeros = torch.zeros_like(ones)
    content_zero = torch.zeros_like(state.content_tokens[:1])
    realization_zero = torch.zeros_like(state.realization_tokens[:1])
    timbre_zero = torch.zeros_like(state.timbre_global[:1])
    modes = {
        "correct_content_correct_realization": condition_from_state(state, 0),
        "correct_content_wrong_realization": eeg.fuse(
            state.content_tokens[:1],
            state.realization_tokens[1:2],
            state.timbre_global[1:2],
        ),
        "wrong_content_correct_realization": eeg.fuse(
            state.content_tokens[2:3],
            state.realization_tokens[:1],
            state.timbre_global[:1],
        ),
        "wrong_content_wrong_realization": eeg.fuse(
            state.content_tokens[2:3],
            state.realization_tokens[2:3],
            state.timbre_global[2:3],
        ),
        "content_only": eeg.fuse(
            state.content_tokens[:1],
            realization_zero,
            timbre_zero,
            content_mask=ones,
            realization_mask=zeros,
        ),
        "realization_only": eeg.fuse(
            content_zero,
            state.realization_tokens[:1],
            state.timbre_global[:1],
            content_mask=zeros,
            realization_mask=ones,
        ),
        "shuffled_eeg": condition_from_state(state, 3),
        "zero_eeg": condition_from_state(zero, 0),
    }
    content_sources = {
        "correct_content_correct_realization": state.content_global[:1],
        "correct_content_wrong_realization": state.content_global[:1],
        "wrong_content_correct_realization": state.content_global[2:3],
        "wrong_content_wrong_realization": state.content_global[2:3],
        "content_only": state.content_global[:1],
        "realization_only": torch.zeros_like(state.content_global[:1]),
        "shuffled_eeg": state.content_global[3:4],
        "zero_eeg": zero.content_global[:1],
    }
    timbre_sources = {
        "correct_content_correct_realization": state.timbre_global[:1],
        "correct_content_wrong_realization": state.timbre_global[1:2],
        "wrong_content_correct_realization": state.timbre_global[:1],
        "wrong_content_wrong_realization": state.timbre_global[2:3],
        "content_only": timbre_zero,
        "realization_only": state.timbre_global[:1],
        "shuffled_eeg": state.timbre_global[3:4],
        "zero_eeg": zero.timbre_global[:1],
    }
    return modes, content_sources, timbre_sources


def control_indices(
    rows: Sequence[dict[str, str]], seed: int
) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_label[str(row["label"])].append(index)
    same, wrong = [], []
    for index, row in enumerate(rows):
        candidates = [
            candidate
            for candidate in by_label[str(row["label"])]
            if candidate != index and rows[candidate]["audio_key"] != row["audio_key"]
        ]
        same.append(candidates[0] if candidates else index)
        alternatives = [
            candidate
            for candidate, other in enumerate(rows)
            if other["label"] != row["label"]
        ]
        wrong.append(alternatives[index % len(alternatives)] if alternatives else index)
    shuffled = np.arange(len(rows))
    rng.shuffle(shuffled)
    if len(rows) > 1:
        for index in range(len(rows)):
            if shuffled[index] == index:
                shuffled[index], shuffled[(index + 1) % len(rows)] = (
                    shuffled[(index + 1) % len(rows)],
                    shuffled[index],
                )
    return same, wrong, shuffled.astype(int).tolist()


def preparation_config(cfg: dict[str, Any]) -> AudioPreparationConfig:
    audio = cfg["audio"]
    return AudioPreparationConfig(
        sample_rate=int(audio["sample_rate"]),
        max_active_seconds=float(audio["max_active_seconds"]),
        target_rms=float(audio["target_rms"]),
        active=ActiveSpeechConfig(
            sample_rate=int(audio["sample_rate"]),
            window_ms=float(audio["active_window_ms"]),
            hop_ms=float(audio["active_hop_ms"]),
            noise_margin_db=float(audio["active_noise_margin_db"]),
            peak_margin_db=float(audio["active_peak_margin_db"]),
            close_gap_ms=float(audio["active_close_gap_ms"]),
            context_ms=float(audio["active_context_ms"]),
        ),
    )


def reference_audio(
    metadata: dict[str, Any], cfg: dict[str, Any], codec_rate: int
) -> np.ndarray:
    audio, rate = read_wav(Path(str(metadata["audio_path"])))
    prepared = prepare_waveform_segment(audio, rate, preparation_config(cfg))
    expected = str(
        metadata.get("segment_pcm_sha256") or metadata.get("pcm_sha256") or ""
    )
    if expected and prepared.pcm_sha256 != expected:
        raise ValueError(
            f"Teacher/reference waveform mismatch for {metadata.get('audio_key')}"
        )
    value = prepared.waveform[: prepared.valid_samples]
    return resample_audio(value, prepared.sample_rate, codec_rate)


def plot_energy(path: Path, energy: np.ndarray, *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 3.2), constrained_layout=True)
    image = axis.imshow(
        energy, origin="lower", aspect="auto", vmin=-80.0, vmax=0.0, cmap="magma"
    )
    axis.set(title=title, xlabel="10-ms frame", ylabel="mel bin")
    figure.colorbar(image, ax=axis, label="dB")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def explicit_factor_metrics(
    condition: FactorizedConditionState,
    batch: dict[str, Any],
) -> dict[str, float]:
    valid = batch["realization_frame_mask"][0].bool()
    reference_activity = batch["activity_mask"][0].bool() & valid
    predicted_activity = condition.activity_logits[0].sigmoid() >= 0.5
    predicted_activity &= valid
    intersection = int((reference_activity & predicted_activity).sum().item())
    activity_dice = (
        2.0
        * intersection
        / max(
            1,
            int(reference_activity.sum().item() + predicted_activity.sum().item()),
        )
    )
    voiced = batch["voicing"][0].bool() & valid
    f0_mae = (
        float(
            (condition.log_f0_hz[0, voiced] - batch["f0_log_hz"][0, voiced])
            .abs()
            .mean()
        )
        if voiced.any()
        else float("nan")
    )
    rms_mae = (
        float(
            (condition.log_rms_dbfs[0, valid] - batch["log_rms_dbfs"][0, valid])
            .abs()
            .mean()
        )
        if valid.any()
        else float("nan")
    )
    return {
        "predicted_activity_dice": float(activity_dice),
        "predicted_f0_log_mae": f0_mae,
        "predicted_log_rms_mae_db": rms_mae,
        "predicted_duration_error_seconds": abs(
            float(condition.duration_seconds[0]) - float(batch["duration_seconds"][0])
        ),
    }


class ContentMetric:
    def __init__(
        self, model_name: str, layer: int, device: torch.device, disabled: bool
    ):
        self.disabled = bool(disabled)
        self.device = device
        self.layer = int(layer)
        self.extractor = self.model = None
        if not self.disabled:
            from transformers import AutoFeatureExtractor, AutoModel

            path = Path(model_name)
            kwargs = {"local_files_only": True} if path.exists() else {}
            self.extractor = AutoFeatureExtractor.from_pretrained(model_name, **kwargs)
            self.model = (
                AutoModel.from_pretrained(model_name, **kwargs).to(device).eval()
            )

    @torch.no_grad()
    def compare(
        self, waveforms: list[np.ndarray], source_rate: int
    ) -> tuple[torch.Tensor | None, list[float] | None]:
        if self.disabled or self.extractor is None or self.model is None:
            return None, None
        values = [
            resample_audio(waveform, source_rate, 16000) for waveform in waveforms
        ]
        encoded = self.extractor(
            values, sampling_rate=16000, padding=True, return_tensors="pt"
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        output = self.model(**encoded, output_hidden_states=True)
        hidden = output.hidden_states[self.layer]
        if "attention_mask" in encoded and hasattr(
            self.model, "_get_feature_vector_attention_mask"
        ):
            mask = self.model._get_feature_vector_attention_mask(
                hidden.shape[1], encoded["attention_mask"]
            ).bool()
        else:
            mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        weights = mask.to(hidden)
        pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        normalized = F.normalize(hidden, dim=-1)
        reference = normalized[0, mask[0]]
        frame_scores: list[float] = []
        for index in range(1, len(waveforms)):
            candidate = normalized[index, mask[index]]
            if not len(reference) or not len(candidate):
                frame_scores.append(float("nan"))
                continue
            similarity = (reference @ candidate.T).clamp(0.0, 1.0)
            precision = similarity.max(dim=0).values.mean()
            recall = similarity.max(dim=1).values.mean()
            score = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
            frame_scores.append(float(score))
        return F.normalize(pooled, dim=-1), frame_scores


class TimbreMetric:
    """Optional local-only WavLM x-vector comparison for decoded waveforms."""

    def __init__(self, model_name: str, device: torch.device, disabled: bool):
        self.device = device
        self.extractor = self.model = None
        self.load_error: str | None = None
        if disabled:
            return
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

            self.extractor = AutoFeatureExtractor.from_pretrained(
                model_name, local_files_only=True
            )
            self.model = (
                AutoModelForAudioXVector.from_pretrained(
                    model_name, local_files_only=True
                )
                .to(device)
                .eval()
            )
        except Exception as error:  # optional metric, never a generation input
            self.extractor = self.model = None
            self.load_error = f"{type(error).__name__}: {error}"

    @torch.no_grad()
    def embeddings(
        self, waveforms: list[np.ndarray], source_rate: int
    ) -> torch.Tensor | None:
        if self.extractor is None or self.model is None:
            return None
        values = [
            resample_audio(waveform, source_rate, 16000) for waveform in waveforms
        ]
        encoded = self.extractor(
            values, sampling_rate=16000, padding=True, return_tensors="pt"
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        output = self.model(**encoded)
        return F.normalize(output.embeddings, dim=-1)


def main() -> None:
    args = parse_args()
    if args.limit == 0 or args.limit < -1:
        raise ValueError("--limit must be -1 or a positive number of trials")
    if args.split == "test" and (args.limit >= 0 or args.skip_decoded_content_metric):
        raise PermissionError(
            "The one-shot locked test cannot use --limit or skip the decoded "
            "content metric"
        )
    config_path, raw_cfg = load_config(args.config)
    seed = int(raw_cfg["training"]["seed"] if args.seed is None else args.seed)
    args.seed = seed
    audio_path = resolve_config_path(config_path, raw_cfg["paths"]["audio_checkpoint"])
    eeg_path = resolve_run_checkpoint(
        config_path,
        raw_cfg,
        "eeg_checkpoint",
        seed=seed,
        loso_subject=args.loso_subject,
        generalization=args.generalization,
        holdout_label=args.holdout_label,
    )
    gate_path = resolve_config_path(config_path, raw_cfg["paths"]["validation_gate"])
    if args.split == "test":
        if (
            args.loso_subject is not None
            or seed != int(raw_cfg["training"]["seed"])
            or args.generalization != "g1"
            or args.holdout_label is not None
        ):
            raise PermissionError(
                "Locked-test synthesis is primary-g1 only; LOSO, held-label, "
                "and secondary-seed runs are development-only"
            )
        if not args.exploratory_test:
            if not args.allow_final_test:
                raise PermissionError("Locked test synthesis requires --allow-final-test")
            authorize_locked_test_metadata(
                gate_path,
                config_path=config_path,
                audio_checkpoint=audio_path,
                eeg_checkpoint=eeg_path,
            )
            claim_locked_test_access(
                gate_path,
                purpose=f"reconstruction_{args.dataset}",
                access_id=args.test_access_id,
            )
    context = load_context(config_path)
    teachers = TeacherCacheV2(
        resolve_config_path(config_path, raw_cfg["paths"]["teacher_cache"])
    )
    lineage = build_lineage(context)
    if args.split == "test" and not args.exploratory_test:
        authorize_locked_test(
            gate_path,
            lineage=lineage,
            audio_checkpoint=audio_path,
            eeg_checkpoint=eeg_path,
        )
    device = torch.device(args.device) if args.device else default_device()
    audio, eeg, eeg_payload = load_models(
        context,
        lineage,
        device,
        eeg_path=eeg_path,
        seed=seed,
        loso_subject=args.loso_subject,
        generalization=args.generalization,
        holdout_label=args.holdout_label,
    )
    requested_run = {
        "seed": seed,
        "generalization": args.generalization,
        "holdout_label": args.holdout_label,
        "loso_subject": args.loso_subject,
    }
    if eeg_payload.get("run") != requested_run:
        raise ValueError(
            f"{eeg_path} run metadata mismatch: observed={eeg_payload.get('run')!r}, "
            f"expected={requested_run!r}"
        )
    dataset = FactorizedEEGDataset(
        context,
        teachers,
        split=args.split,
        generalization=args.generalization,
        holdout_label=args.holdout_label,
        loso_subject=args.loso_subject,
        datasets=(args.dataset,),
        allow_locked_test=args.split == "test",
    )
    eligible_indices = [
        index
        for index, row in enumerate(dataset.rows)
        if bool(
            teachers.metadata(str(row["audio_key"])).get(
                "reconstruction_eligible", False
            )
        )
    ]
    if not eligible_indices:
        raise ValueError(
            f"No reconstruction-eligible {args.dataset}/{args.split} trials"
        )
    eligible_rows = [dataset.rows[index] for index in eligible_indices]
    same_local, wrong_local, shuffled_local = control_indices(eligible_rows, seed)
    same_indices = {
        original: eligible_indices[same_local[position]]
        for position, original in enumerate(eligible_indices)
    }
    wrong_indices = {
        original: eligible_indices[wrong_local[position]]
        for position, original in enumerate(eligible_indices)
    }
    shuffled_indices = {
        original: eligible_indices[shuffled_local[position]]
        for position, original in enumerate(eligible_indices)
    }
    full_dataset_record_count = len(eligible_indices)
    indices = list(eligible_indices)
    indices = indices if args.limit < 0 else indices[: args.limit]
    run_id = run_identifier(
        raw_cfg,
        seed=seed,
        loso_subject=args.loso_subject,
        generalization=args.generalization,
        holdout_label=args.holdout_label,
    )
    default_output = (
        resolve_config_path(config_path, raw_cfg["paths"]["output_root"])
        / "synthesis"
        / args.dataset
        / args.split
    )
    if run_id is not None:
        default_output = default_output / "runs" / run_id
    if args.limit >= 0:
        default_output = default_output / f"diagnostic_limit_{args.limit}"
    if args.skip_decoded_content_metric:
        default_output = default_output / "diagnostic_no_decoded_content_metric"
    output_root = args.output.resolve() if args.output else default_output
    output_root.mkdir(parents=True, exist_ok=True)
    codec = DiscreteEncodec(
        DiscreteEncodecConfig(
            model_path=str(
                resolve_config_path(config_path, raw_cfg["paths"]["encodec_model"])
            ),
            sample_rate=int(raw_cfg["codec"]["sample_rate"]),
            duration_sec=float(raw_cfg["codec"]["max_duration_sec"]),
            bandwidth=float(raw_cfg["codec"]["bandwidth"]),
        ),
        device,
    )
    content_metric = ContentMetric(
        str(resolve_config_path(config_path, raw_cfg["teachers"]["hubert_model"])),
        int(raw_cfg["teachers"]["hubert_layer"]),
        device,
        args.skip_decoded_content_metric,
    )
    timbre_metric = TimbreMetric(
        str(raw_cfg["teachers"]["wavlm_model"]),
        device,
        args.skip_decoded_timbre_metric,
    )
    records: list[dict[str, Any]] = []
    aggregate: dict[str, list[dict[str, float]]] = defaultdict(list)
    retrieval_eeg, retrieval_audio, retrieval_labels = [], [], []

    for output_index, index in enumerate(
        tqdm(indices, desc=f"[0724 synth] {args.dataset}/{args.split}", unit="trial")
    ):
        sample_indices = [
            index,
            same_indices[index],
            wrong_indices[index],
            shuffled_indices[index],
        ]
        raw_samples = [dataset[value] for value in sample_indices]
        batch = move_batch(collate_factorized(raw_samples), device)
        with torch.no_grad():
            state = eeg(
                batch["eeg"],
                batch["channel_xyz"],
                batch["channel_mask"],
                batch["time_mask"],
                epoch=int(eeg_payload["epoch"]),
            )
            audio_state = audio.encode(
                batch["content_tokens"][:1],
                batch["content_token_mask"][:1],
                batch["realization_features"][:1],
                batch["realization_frame_mask"][:1],
                batch["timbre_global"][:1],
            )

        stem = f"{output_index:04d}_{safe(str(batch['sample_key'][0]))}"
        record_path = output_root / "records" / f"{stem}.json"
        if args.resume_existing and record_path.exists():
            existing = json.loads(record_path.read_text(encoding="utf-8"))
            if (
                str(existing.get("sample_key")) != str(batch["sample_key"][0])
                or str(existing.get("audio_key")) != str(batch["audio_key"][0])
            ):
                raise ValueError(
                    f"Resume record does not match current dataset ordering: {record_path}"
                )
            records.append(existing)
            for name, result in dict(existing.get("metrics") or {}).items():
                aggregate[str(name)].append(dict(result))
            retrieval_eeg.append(state.content_global[0].detach().cpu())
            retrieval_audio.append(audio_state.content_global[0].detach().cpu())
            retrieval_labels.append(int(batch["label_idx"][0]))
            continue

        with torch.no_grad():
            zero = eeg(
                torch.zeros_like(batch["eeg"][:1]),
                batch["channel_xyz"][:1],
                batch["channel_mask"][:1],
                batch["time_mask"][:1],
                epoch=int(eeg_payload["epoch"]),
            )
        modes, content_sources, timbre_sources = counterfactuals(eeg, state, zero)
        modes["audio_condition_oracle"] = FactorizedConditionState(
            fused_condition=audio_state.fused_condition,
            log_mel_energy=audio_state.log_mel_energy,
            log_f0_hz=audio_state.log_f0_hz,
            voicing_logits=audio_state.voicing_logits,
            log_rms_dbfs=audio_state.log_rms_dbfs,
            activity_logits=audio_state.activity_logits,
            duration_seconds=audio_state.duration_seconds,
        )
        content_sources["audio_condition_oracle"] = audio_state.content_global
        timbre_sources["audio_condition_oracle"] = audio_state.timbre_global

        decoded: dict[str, np.ndarray] = {}
        predicted_maps: dict[str, np.ndarray] = {}
        factor_results: dict[str, dict[str, float]] = {}
        for name, condition in modes.items():
            codes, valid = audio.decoder.generate(
                condition.fused_condition, condition.duration_seconds
            )
            length = int(valid[0].sum().item())
            decoded[name] = codec.decode(codes[0, :, :length].cpu().numpy(), scale=None)
            predicted_maps[name] = condition.log_mel_energy[0].detach().cpu().numpy()
            factor_results[name] = explicit_factor_metrics(condition, batch)
        target_valid = batch["code_valid_mask"][0].any(dim=0)
        target_steps = int(target_valid.sum().item())
        decoded["codec_oracle"] = codec.decode(
            batch["codes"][0, :, :target_steps].cpu().numpy(), scale=None
        )
        predicted_maps["codec_oracle"] = batch["log_mel_energy"][0].cpu().numpy()
        factor_results["codec_oracle"] = {
            "predicted_activity_dice": 1.0,
            "predicted_f0_log_mae": 0.0,
            "predicted_log_rms_mae_db": 0.0,
            "predicted_duration_error_seconds": 0.0,
        }
        content_sources["codec_oracle"] = audio_state.content_global
        timbre_sources["codec_oracle"] = audio_state.timbre_global

        metadata = teachers.metadata(str(batch["audio_key"][0]))
        reference = reference_audio(metadata, raw_cfg, codec.codec_sample_rate)
        reference_map = batch["log_mel_energy"][0].cpu().numpy()
        reference_dir = output_root / "reference"
        reference_dir.mkdir(parents=True, exist_ok=True)
        write_wav(reference_dir / f"{stem}.wav", reference, codec.codec_sample_rate)
        np.save(reference_dir / f"{stem}.mel.npy", reference_map)
        plot_energy(
            reference_dir / f"{stem}.png",
            reference_map,
            title=f"reference: {batch['label'][0]}",
        )

        mode_metrics: dict[str, dict[str, float]] = {}
        candidate_names = list(decoded)
        decoded_embeddings, frame_scores = content_metric.compare(
            [reference] + [decoded[name] for name in candidate_names],
            codec.codec_sample_rate,
        )
        decoded_timbre = timbre_metric.embeddings(
            [reference] + [decoded[name] for name in candidate_names],
            codec.codec_sample_rate,
        )
        for candidate_index, name in enumerate(candidate_names):
            folder = output_root / name
            folder.mkdir(parents=True, exist_ok=True)
            waveform = decoded[name]
            write_wav(folder / f"{stem}.wav", waveform, codec.codec_sample_rate)
            np.save(folder / f"{stem}.mel.npy", predicted_maps[name])
            plot_energy(
                folder / f"{stem}.png",
                predicted_maps[name],
                title=f"{name}: {batch['label'][0]}",
            )
            direct = energy_structure_metrics(reference_map, predicted_maps[name])
            waveform_result = reconstruction_metrics(
                reference,
                waveform,
                codec.codec_sample_rate,
                max_lag_ms=float(raw_cfg["evaluation"]["max_envelope_lag_ms"]),
            )
            result = dict(direct)
            result.update(factor_results[name])
            result.update(
                {f"decoded_{key}": value for key, value in waveform_result.items()}
            )
            result["predicted_map_decoded_consistency"] = energy_structure_metrics(
                predicted_maps[name], log_mel(waveform, codec.codec_sample_rate)
            )["morphology_ssim"]
            result["latent_content_cosine"] = float(
                F.cosine_similarity(
                    content_sources[name], audio_state.content_global, dim=-1
                ).item()
            )
            result["timbre_cosine"] = float(
                F.cosine_similarity(
                    timbre_sources[name], audio_state.timbre_global, dim=-1
                ).item()
            )
            if decoded_timbre is not None:
                result["wavlm_xvector_cosine"] = float(
                    (decoded_timbre[0] * decoded_timbre[candidate_index + 1])
                    .sum()
                    .item()
                )
            if decoded_embeddings is not None:
                result["content_cosine"] = float(
                    (decoded_embeddings[0] * decoded_embeddings[candidate_index + 1])
                    .sum()
                    .item()
                )
                assert frame_scores is not None
                result["speech_bertscore"] = frame_scores[candidate_index]
                result["hubert_frame_matching_f1"] = frame_scores[candidate_index]
            else:
                result["content_cosine"] = result["latent_content_cosine"]
            mode_metrics[name] = result
            aggregate[name].append(result)
        retrieval_eeg.append(state.content_global[0].detach().cpu())
        retrieval_audio.append(audio_state.content_global[0].detach().cpu())
        retrieval_labels.append(int(batch["label_idx"][0]))
        records.append(
            {
                "sample_key": str(batch["sample_key"][0]),
                "audio_key": str(batch["audio_key"][0]),
                "subject_group_id": str(batch["subject_group_id"][0]),
                "label": str(batch["label"][0]),
                "pairing_scope": str(batch["pairing_scope"][0]),
                "stem": stem,
                "metrics": mode_metrics,
                "controls": {
                    "same_label_index": int(same_indices[index]),
                    "wrong_label_index": int(wrong_indices[index]),
                    "shuffled_index": int(shuffled_indices[index]),
                    "same_label_sample_key": str(raw_samples[1]["sample_key"]),
                    "same_label_audio_key": str(raw_samples[1]["audio_key"]),
                    "wrong_label_sample_key": str(raw_samples[2]["sample_key"]),
                    "wrong_label": str(raw_samples[2]["label"]),
                    "shuffled_sample_key": str(raw_samples[3]["sample_key"]),
                    "same_label_control_available": bool(
                        same_indices[index] != index
                        and str(raw_samples[1]["audio_key"])
                        != str(raw_samples[0]["audio_key"])
                        and str(raw_samples[1]["label"]) == str(raw_samples[0]["label"])
                    ),
                    "wrong_label_control_available": bool(
                        wrong_indices[index] != index
                        and str(raw_samples[2]["label"]) != str(raw_samples[0]["label"])
                    ),
                    "shuffled_control_available": bool(
                        shuffled_indices[index] != index
                    ),
                },
            }
        )
        write_json(record_path, records[-1])

    retrieval: dict[str, float] = {}
    if retrieval_eeg:
        eeg_values = F.normalize(torch.stack(retrieval_eeg), dim=-1)
        audio_values = F.normalize(torch.stack(retrieval_audio), dim=-1)
        labels = torch.tensor(retrieval_labels)
        predictions = labels[(eeg_values @ audio_values.T).argmax(dim=1)]
        per_label = [
            (predictions[labels == label] == label).float().mean()
            for label in labels.unique()
        ]
        retrieval = {
            "macro_top1": float(torch.stack(per_label).mean()),
            "balanced_chance": 1.0 / max(1, int(labels.unique().numel())),
        }
    manifest = {
        "schema_version": "openvoice-0724-synthesis-v1",
        "dataset": args.dataset,
        "split": args.split,
        "generalization": args.generalization,
        "holdout_label": args.holdout_label,
        "loso_subject": args.loso_subject,
        "seed": seed,
        "diagnostic_limit": int(args.limit),
        "full_dataset_record_count": int(full_dataset_record_count),
        "records": records,
        "aggregate": {name: summarize(values) for name, values in aggregate.items()},
        "retrieval": retrieval,
        "lineage": lineage,
        "audio_checkpoint_sha256": file_sha256(audio_path),
        "eeg_checkpoint_sha256": file_sha256(eeg_path),
        "decoded_timbre_metric_available": timbre_metric.model is not None,
        "decoded_timbre_metric_load_error": timbre_metric.load_error,
        "decoded_content_metric_available": bool(
            content_metric.model is not None and not content_metric.disabled
        ),
        "test_accessed": args.split == "test",
        "metrics_use_png_pixels": False,
        "frequency_axis_scaled": False,
    }
    write_json(output_root / "synthesis_manifest.json", manifest)
    print(
        json.dumps(
            {
                "output": str(output_root),
                "records": len(records),
                "retrieval": retrieval,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
