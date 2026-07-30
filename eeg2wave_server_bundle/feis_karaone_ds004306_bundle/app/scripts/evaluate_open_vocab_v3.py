#!/usr/bin/env python3
"""Run v3 fail-closed oracle and EEG-MFCC gates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from scripts.train_open_vocab_v3 import load_audio, load_eeg, micro_dataset
from src.open_vocab_v3.data import V3Dataset, _accepted_denoise_paths, canonical_mfcc_from_waveform, channel_shuffled_eeg, collate, light_prepare_waveform, load_prepared, time_shuffled_eeg
from src.open_vocab_v3.hubert import HubertMetric, dtw_cosine
from src.open_vocab_v3.metrics import bootstrap_mean_gain, mfcc_distance, paired_r_at_1_above_chance, paired_win_rate, retrieval, same_label_template, variance_ratio
from src.open_vocab_v3.model import librosa_mfcc_to_mel_reference
from src.open_vocab_v3.runtime import capture_lineage, default_device, load_config, move_batch, output_path, read_json, require_passed_gate, resolve_config_path, sha256_file, write_json
from src.open_vocab_v3.speaker import ECAPAEncoder, speaker_distribution
from src.open_vocab_v3.vocoder import SpeechT5PowerDbHiFiGan, pcm16
from src.open_vocab_0724.audio_features import AudioPreparationConfig


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v3 gates")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("v0", "v1", "v2", "micro", "fit", "validation", "locked", "locked_unseen"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-fail", action="store_true", help="write failed report without a nonzero exit")
    parser.add_argument("--limit-per-label", type=int, default=0)
    return parser.parse_args()


def audio_paths(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {str(row["sample_key"]): str(row["audio_relpath"]) for row in csv.DictReader(handle) if row.get("dataset") == "karaone"}


def read_wave(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    value, rate = sf.read(path, always_2d=False, dtype="float32")
    if value.ndim == 2:
        value = value.mean(1)
    return np.asarray(value, dtype=np.float32), int(rate)


def cleaned_reference(path: Path, cfg: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Use the v3 light-cleaned audio, never the raw waveform, for audio gates."""
    waveform, rate = read_wave(path)
    prepared, _ = light_prepare_waveform(
        waveform,
        rate,
        AudioPreparationConfig(
            sample_rate=int(cfg["audio"]["sample_rate"]),
            max_active_seconds=float(cfg["audio"]["max_active_seconds"]),
            target_rms=float(cfg["audio"]["target_rms"]),
        ),
    )
    return prepared.waveform[: max(1, prepared.valid_samples)], int(cfg["audio"]["sample_rate"])


def waveform_sha256(waveform: np.ndarray) -> str:
    pcm = np.asarray(np.clip(waveform, -1.0, 1.0) * 32767.0, dtype="<i2")
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def selected_dataset(records: Any, roles: tuple[str, ...], *, eligible: bool, max_per_label: int) -> Any:
    base = V3Dataset(records, roles, eligible_only=eligible)
    if not max_per_label:
        return base
    grouped: dict[str, list[int]] = {}
    for index, record_index in enumerate(base.indices):
        grouped.setdefault(str(records.arrays["labels"][record_index]), []).append(index)
    selected = []
    for label, indices in sorted(grouped.items()):
        selected.extend(sorted(indices, key=lambda item: str(records.arrays["sample_keys"][base.indices[item]]))[:max_per_label])
    return Subset(base, selected)


def batches(dataset: Any, *, batch_size: int, device: torch.device):
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0):
        yield move_batch(batch, device)


def hubert_retrieval(generated: list[np.ndarray], references: list[np.ndarray], rates: list[int], labels: list[str], metric: HubertMetric) -> dict[str, Any]:
    generated_h = [metric.encode(waveform, rate) for waveform, rate in zip(generated, rates)]
    reference_h = [metric.encode(waveform, rate) for waveform, rate in zip(references, rates)]
    similarity = np.asarray([[dtw_cosine(left, right) for right in reference_h] for left in generated_h], dtype=np.float32)
    nearest = similarity.argmax(1)
    canonical = [str(value).strip().strip("/").lower() for value in labels]
    top1 = float(np.mean([canonical[index] == canonical[candidate] for index, candidate in enumerate(nearest)]))
    gaps = []
    for index, label in enumerate(canonical):
        wrong = [similarity[index, candidate] for candidate, candidate_label in enumerate(canonical) if candidate_label != label]
        gaps.append(float(similarity[index, index] - np.median(wrong)) if wrong else 0.0)
    return {"label_top1": top1, "dtw_hubert_correct_minus_wrong_median": float(np.mean(gaps)), "dtw_hubert_diagonal_mean": float(np.diag(similarity).mean()), "nearest_index": nearest.tolist()}


def hubert_pair_metrics(
    generated: list[np.ndarray], references: list[np.ndarray], rates: list[int], labels: list[str], metric: HubertMetric
) -> dict[str, Any]:
    """Scalable held-out audio metrics: pool retrieval plus exact paired DTW.

    Full all-to-all DTW is used for the small V0/V1 oracle subsets.  The
    subject-holdout and locked reports instead use all real utterances as a
    pooled-embedding retrieval bank, while retaining DTW-aligned cosine for
    the paired generated/reference comparison that answers the primary question.
    """
    generated_h = [metric.encode(waveform, rate) for waveform, rate in zip(generated, rates)]
    reference_h = [metric.encode(waveform, rate) for waveform, rate in zip(references, rates)]
    def pooled(value: np.ndarray) -> np.ndarray:
        mean = np.asarray(value, dtype=np.float32).mean(0)
        return mean / max(float(np.linalg.norm(mean)), 1.0e-8)
    generated_pool = np.stack([pooled(value) for value in generated_h])
    reference_pool = np.stack([pooled(value) for value in reference_h])
    similarity = generated_pool @ reference_pool.T
    nearest = similarity.argmax(1)
    canonical = [str(value).strip().strip("/").lower() for value in labels]
    paired = np.asarray([dtw_cosine(left, right) for left, right in zip(generated_h, reference_h)], dtype=np.float32)
    return {
        "label_top1": float(np.mean([canonical[index] == canonical[candidate] for index, candidate in enumerate(nearest)])),
        "paired_dtw_hubert_cosine_mean": float(paired.mean()),
        "paired_dtw_hubert_cosine_median": float(np.median(paired)),
        "paired_dtw_hubert_cosine_per_trial": paired.tolist(),
        "nearest_index": nearest.tolist(),
        "retrieval_backend": "utterance-pooled-HuBERT cosine against real audio pool",
    }


def bootstrap_scalar_gain(correct: np.ndarray, control: np.ndarray, *, samples: int, seed: int) -> dict[str, float]:
    gains = np.asarray(correct, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.asarray([gains[rng.integers(0, len(gains), len(gains))].mean() for _ in range(samples)])
    return {"mean_gain": float(gains.mean()), "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975))}


def _vocoder(config_path: Path, cfg: dict[str, Any], device: torch.device) -> SpeechT5PowerDbHiFiGan:
    return SpeechT5PowerDbHiFiGan(output_path(config_path, cfg, "vocoder_root"), device=device)


def gate_v0(config_path: Path, cfg: dict[str, Any], records: Any, device: torch.device, limit: int) -> dict[str, Any]:
    dataset = selected_dataset(records, ("fit",), eligible=True, max_per_label=limit or int(cfg["evaluation"]["oracle_per_label"]))
    backend = _vocoder(config_path, cfg, device)
    teacher = HubertMetric(output_path(config_path, cfg, "hubert_root"), layer=int(cfg["teachers"]["hubert_layer"]), device=device)
    paths = audio_paths(output_path(config_path, cfg, "unified_manifest")); root = output_path(config_path, cfg, "audio_root")
    generated: list[np.ndarray] = []; references: list[np.ndarray] = []; rates: list[int] = []; labels: list[str] = []; selected_mel: list[np.ndarray] = []
    for batch in tqdm(batches(dataset, batch_size=1, device=device), total=len(dataset), desc="[v3 V0] vocoder oracle", unit="pair", dynamic_ncols=True):
        key = batch["sample_key"][0]
        source_index = int(np.flatnonzero(records.arrays["sample_keys"].astype(str) == key)[0])
        wave = pcm16(backend.synthesize(torch.from_numpy(records.arrays["vocoder_mel"][source_index:source_index + 1]).to(device))[0])
        reference, rate = cleaned_reference(root / paths[key], cfg)
        generated.append(wave); references.append(reference); rates.append(rate); labels.append(batch["label"][0]); selected_mel.append(records.arrays["vocoder_mel"][source_index])
    metrics = hubert_retrieval(generated, references, rates, labels, teacher)
    threshold = cfg["gates"]["v0"]
    checks = {"label_retrieval": metrics["label_top1"] >= float(threshold["label_top1_min"]), "dtw_gap": metrics["dtw_hubert_correct_minus_wrong_median"] >= float(threshold["dtw_gap_min"])}
    manifest = output_path(config_path, cfg, "vocoder_manifest")
    source_config = resolve_config_path(config_path, cfg["v0_source_mel"]["source_config"])
    mel_values = np.stack(selected_mel).astype(np.float32)
    return {
        "gate": "V0", "n": len(labels), "metrics": metrics, "thresholds": threshold, "checks": checks,
        "interface": {
            "source_mel": "immutable v0728 80-bin power-dB mel cache",
            "source_mel_shape": list(records.arrays["vocoder_mel"].shape[1:]),
            "source_mel_parameters": cfg["v0_source_mel"],
            "source_config": str(source_config),
            "source_config_sha256": sha256_file(source_config),
            "selected_input_numeric_range": {
                "minimum": float(mel_values.min()), "maximum": float(mel_values.max()),
                "p01": float(np.quantile(mel_values, .01)), "p99": float(np.quantile(mel_values, .99)),
            },
            "speecht5_input_transform": "power_dB / 10, transpose [B,80,T] -> [B,T,80]",
            "output_sample_rate": int(cfg["vocoder"]["sample_rate"]),
            "vocoder_manifest": str(manifest),
            "vocoder_manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
            "generated_wav_sha256": [waveform_sha256(wave) for wave in generated],
        },
        "passed": bool(all(checks.values())),
    }


@torch.no_grad()
def gate_v1(config_path: Path, cfg: dict[str, Any], records: Any, device: torch.device, limit: int) -> dict[str, Any]:
    dataset = selected_dataset(records, ("fit",), eligible=True, max_per_label=limit or int(cfg["evaluation"]["oracle_per_label"]))
    decoder, _ = load_audio(config_path, cfg, device); backend = _vocoder(config_path, cfg, device)
    teacher = HubertMetric(output_path(config_path, cfg, "hubert_root"), layer=int(cfg["teachers"]["hubert_layer"]), device=device)
    paths = audio_paths(output_path(config_path, cfg, "unified_manifest")); root = output_path(config_path, cfg, "audio_root"); denoised = _accepted_denoise_paths(config_path, cfg)
    generated_prior=[]; generated_posterior=[]; generated_analytic=[]
    references=[]; rates=[]; labels=[]; input_mfcc=[]; output_mfcc=[]; mel_gaps=[];analytic_values=[];analytic_means=[];analytic_stds=[]
    for batch in tqdm(batches(dataset, batch_size=1, device=device), total=len(dataset), desc="[v3 V1] content oracle", unit="pair", dynamic_ncols=True):
        voice = batch["canonical_voice"].float()
        mean=batch["canonical_mfcc_mean"].float();std=batch["canonical_mfcc_std"].float()
        prior=decoder.generate(batch["mfcc"].float(),voice,mean,std,stochastic=False)
        posterior=decoder.reconstruct(batch["mfcc"].float(),voice,mean,std,batch["mel"].float(),stochastic=False)
        analytic=prior["analytic_mel"]
        prior_wave=pcm16(backend.synthesize(prior["mel"])[0])
        posterior_wave=pcm16(backend.synthesize(posterior["mel"])[0])
        analytic_wave=pcm16(backend.synthesize(analytic)[0])
        key=batch["sample_key"][0]; reference,rate=cleaned_reference(denoised.get(key,root/paths[key]),cfg)
        generated_prior.append(prior_wave);generated_posterior.append(posterior_wave);generated_analytic.append(analytic_wave)
        references.append(reference);rates.append(rate);labels.append(batch["label"][0]);input_mfcc.append(batch["mfcc"][0].cpu().numpy());output_mfcc.append(canonical_mfcc_from_waveform(prior_wave,int(cfg["vocoder"]["sample_rate"]),cfg));mel_gaps.append(float(torch.mean(torch.abs(prior["mel"]-posterior["mel"])).cpu()));analytic_values.append(analytic[0].cpu().numpy());analytic_means.append(mean[0].cpu().numpy());analytic_stds.append(std[0].cpu().numpy())
    input_array=np.stack(input_mfcc); output_array=np.stack(output_mfcc); template=same_label_template(input_array,labels)
    feature_ratio=float(mfcc_distance(output_array,input_array).mean()/max(float(mfcc_distance(template,input_array).mean()),1e-8))
    content=hubert_retrieval(generated_prior,references,rates,labels,teacher)
    posterior_content=hubert_retrieval(generated_posterior,references,rates,labels,teacher)
    analytic_content=hubert_retrieval(generated_analytic,references,rates,labels,teacher)
    librosa_reference=librosa_mfcc_to_mel_reference(np.stack(input_mfcc),np.stack(analytic_means),np.stack(analytic_stds),mel_bins=int(cfg["audio"]["mel_bins"]));analytic_error=float(np.max(np.abs(np.stack(analytic_values)-librosa_reference)))
    prior_posterior_gap=float(np.mean(mel_gaps));threshold=cfg["gates"]["v1"]
    checks={"mfcc_relative_to_template":feature_ratio<=float(threshold["mfcc_template_ratio_max"]),"label_retrieval":content["label_top1"]>=float(threshold["label_top1_min"]),"dtw_gap":content["dtw_hubert_correct_minus_wrong_median"]>=float(threshold["dtw_gap_min"]),"prior_posterior_gap":prior_posterior_gap<=float(threshold["prior_posterior_mel_gap_max"]),"librosa_backend_conformance":analytic_error<=float(threshold["analytic_backend_max_abs_error"])}
    return {"gate":"V1","n":len(labels),"metrics":{"prior_mean":content,"posterior_oracle":posterior_content,"fixed_analytic_backend":analytic_content,"analytic_backend":"librosa.feature.inverse.mfcc_to_mel with differentiable torch equivalent","analytic_backend_max_abs_error":analytic_error,"generated_to_input_mfcc_template_ratio":feature_ratio,"prior_posterior_mel_l1":prior_posterior_gap},"thresholds":threshold,"checks":checks,"passed":bool(all(checks.values()))}


@torch.no_grad()
def gate_v2(config_path: Path, cfg: dict[str, Any], records: Any, device: torch.device, limit: int) -> dict[str, Any]:
    if "speaker_reference_embedding" not in records.arrays:
        raise RuntimeError("V2 requires a prepared cache with --with-speaker")
    dataset=selected_dataset(records,("fit",),eligible=True,max_per_label=limit or int(cfg["evaluation"]["oracle_per_label"]))
    decoder,_=load_audio(config_path,cfg,device);backend=_vocoder(config_path,cfg,device)
    encoder=ECAPAEncoder(source=str(cfg["speaker"]["model_id"]),savedir=output_path(config_path,cfg,"speaker_model_root"),device=device)
    teacher=HubertMetric(output_path(config_path,cfg,"hubert_root"),layer=int(cfg["teachers"]["hubert_layer"]),device=device)
    generated_target=[];generated_canonical=[];references=[];rates=[];labels=[];target_embeddings=[];denoised=_accepted_denoise_paths(config_path,cfg)
    for batch in tqdm(batches(dataset,batch_size=1,device=device),total=len(dataset),desc="[v3 V2] timbre oracle",unit="pair",dynamic_ncols=True):
        mel_target=decoder(batch["mfcc"].float(),batch["speaker_reference"].float(),batch["speaker_reference_mfcc_mean"].float(),batch["speaker_reference_mfcc_std"].float()); mel_canonical=decoder(batch["mfcc"].float(),batch["canonical_voice"].float(),batch["canonical_mfcc_mean"].float(),batch["canonical_mfcc_std"].float())
        generated_target.append(pcm16(backend.synthesize(mel_target)[0]));generated_canonical.append(pcm16(backend.synthesize(mel_canonical)[0]))
        key=batch["sample_key"][0];raw_path=output_path(config_path,cfg,"audio_root") / audio_paths(output_path(config_path,cfg,"unified_manifest"))[key];reference,rate=cleaned_reference(denoised.get(key,raw_path),cfg)
        references.append(reference);rates.append(rate);labels.append(batch["label"][0]);target_embeddings.append(batch["speaker_reference"][0].cpu().numpy())
    target_embeddings=np.stack(target_embeddings); predicted=np.stack([encoder.encode(wave) for wave in generated_target]); canonical=np.stack([encoder.encode(wave) for wave in generated_canonical])
    score=np.sum(predicted*target_embeddings,axis=1); swapped=np.sum(canonical*target_embeddings,axis=1)
    fit_selector=(records.roles == "fit") & records.arrays["fit_eligible"].astype(bool)
    distribution=speaker_distribution(records.arrays["speaker_target_embedding"][fit_selector],records.arrays["subjects"][fit_selector].astype(str).tolist(),records.arrays["labels"][fit_selector].astype(str).tolist(),seed=int(cfg["training"]["seed"]))
    p90=float(distribution["different_speaker_same_label"]["p90"])
    content_target=hubert_retrieval(generated_target,references,rates,labels,teacher)
    content_canonical=hubert_retrieval(generated_canonical,references,rates,labels,teacher)
    content_delta=abs(float(content_target["label_top1"])-float(content_canonical["label_top1"]))
    speaker_swap_bootstrap=bootstrap_scalar_gain(score,swapped,samples=int(cfg["evaluation"]["bootstrap_samples"]),seed=int(cfg["training"]["seed"])+60)
    over_p90=float(np.mean(score>p90));threshold=cfg["gates"]["v2"];checks={"target_over_different_speaker_p90":over_p90>=float(threshold["target_over_p90_fraction_min"]),"speaker_swap_effect":float((score-swapped).mean())>=float(threshold["speaker_swap_margin_min"]),"speaker_swap_ci":speaker_swap_bootstrap["ci_low"]>0.0,"content_preserved":content_delta<=float(threshold["content_change_max"])}
    return {"gate":"V2","n":len(labels),"metrics":{"target_speaker_similarity_mean":float(score.mean()),"target_over_different_speaker_same_label_p90_fraction":over_p90,"canonical_speaker_similarity_mean":float(swapped.mean()),"speaker_swap_margin":float((score-swapped).mean()),"speaker_swap_bootstrap":speaker_swap_bootstrap,"different_speaker_same_label_p90":p90,"content_retrieval_target_voice":content_target,"content_retrieval_canonical_voice":content_canonical,"content_retrieval_absolute_change":content_delta,"real_audio_distribution":distribution},"thresholds":threshold,"checks":checks,"passed":bool(all(checks.values()))}


@torch.no_grad()
def heldout_wav_metrics(
    config_path: Path,
    cfg: dict[str, Any],
    records: Any,
    predictions: dict[str, np.ndarray],
    keys: list[str],
    labels: list[str],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render held-out MFCC predictions with the fixed canonical voice only."""
    decoder, _ = load_audio(config_path, cfg, device)
    backend = _vocoder(config_path, cfg, device)
    teacher = HubertMetric(output_path(config_path, cfg, "hubert_root"), layer=int(cfg["teachers"]["hubert_layer"]), device=device)
    voice = np.asarray(records.arrays["canonical_voice"], dtype=np.float32)
    cepstral_mean = np.asarray(records.arrays["canonical_mfcc_mean"], dtype=np.float32)
    cepstral_std = np.asarray(records.arrays["canonical_mfcc_std"], dtype=np.float32)
    if voice.ndim != 1:
        raise ValueError("v3 canonical voice must be a single fit-only ECAPA medoid")
    generated: dict[str, list[np.ndarray]] = {name: [] for name in predictions}
    batch_size = int(cfg["evaluation"]["batch_size"])
    for name, value in predictions.items():
        for start in tqdm(range(0, len(value), batch_size), desc=f"[v3 WAV {name}]", unit="batch", dynamic_ncols=True):
            block = np.asarray(value[start : start + batch_size], dtype=np.float32)
            mfcc = torch.from_numpy(block).to(device)
            voices = torch.from_numpy(np.repeat(voice[None], len(block), axis=0)).to(device)
            means = torch.from_numpy(np.repeat(cepstral_mean[None], len(block), axis=0)).to(device)
            stds = torch.from_numpy(np.repeat(cepstral_std[None], len(block), axis=0)).to(device)
            rendered = pcm16(backend.synthesize(decoder(mfcc, voices, means, stds)))
            if rendered.ndim == 1:
                rendered = rendered[None]
            if len(rendered) != len(block):
                raise RuntimeError(f"v3 vocoder returned {len(rendered)} waveforms for a batch of {len(block)}")
            generated[name].extend([np.asarray(wave, dtype=np.float32) for wave in rendered])
    paths = audio_paths(output_path(config_path, cfg, "unified_manifest"))
    audio_root = output_path(config_path, cfg, "audio_root")
    denoised = _accepted_denoise_paths(config_path, cfg)
    references, rates = zip(*[cleaned_reference(denoised.get(key, audio_root / paths[key]), cfg) for key in keys])
    metrics = {
        name: hubert_pair_metrics(waves, list(references), list(rates), labels, teacher)
        for name, waves in generated.items()
    }
    correct = np.asarray(metrics["correct"]["paired_dtw_hubert_cosine_per_trial"], dtype=np.float32)
    bootstrap = {
        name: bootstrap_scalar_gain(
            correct,
            np.asarray(metrics[name]["paired_dtw_hubert_cosine_per_trial"], dtype=np.float32),
            samples=int(cfg["evaluation"]["bootstrap_samples"]), seed=int(cfg["training"]["seed"]) + 70 + index,
        )
        for index, name in enumerate(("zero", "time", "channel"))
    }
    return metrics, bootstrap


@torch.no_grad()
def eeg_stage(config_path: Path, cfg: dict[str, Any], records: Any, device: torch.device, stage: str) -> dict[str, Any]:
    expected_checkpoint_stage = "micro" if stage == "micro" else "fit"
    model,payload=load_eeg(config_path,cfg,device,stage=expected_checkpoint_stage)
    if payload["extra"].get("stage") != expected_checkpoint_stage:
        raise RuntimeError(
            f"v3 {stage} evaluation requires a {expected_checkpoint_stage} checkpoint, "
            f"not {payload['extra'].get('stage')!r}"
        )
    if stage=="micro": dataset=micro_dataset(records,str(cfg["micro_gate"]["subject"]),int(cfg["micro_gate"]["per_label"]))
    elif stage=="fit": dataset=V3Dataset(records,("fit",),eligible_only=True)
    elif stage=="validation": dataset=V3Dataset(records,("subject_holdout_seen",),eligible_only=False)
    elif stage=="locked": dataset=V3Dataset(records,("locked_test_seen_label",),eligible_only=False)
    elif stage=="locked_unseen": dataset=V3Dataset(records,("locked_test_unseen_label",),eligible_only=False)
    else: raise ValueError(stage)
    output={name:[] for name in ("correct","zero","time","channel")};target=[];labels=[];keys=[]
    for batch in tqdm(batches(dataset,batch_size=int(cfg["evaluation"]["batch_size"]),device=device),total=int(np.ceil(len(dataset)/int(cfg["evaluation"]["batch_size"]))),desc=f"[v3 {stage}] EEG",unit="batch",dynamic_ncols=True):
        kwargs=(batch["channel_xyz"].float(),batch["channel_mask"],batch["time_mask"])
        output["correct"].append(model(batch["eeg"].float(),*kwargs)[0].cpu().numpy())
        output["zero"].append(model(torch.zeros_like(batch["eeg"]).float(),*kwargs)[0].cpu().numpy())
        output["time"].append(model(time_shuffled_eeg(batch["eeg"].float(),batch["time_mask"]),*kwargs)[0].cpu().numpy())
        output["channel"].append(model(channel_shuffled_eeg(batch["eeg"].float(),batch["channel_mask"]),*kwargs)[0].cpu().numpy())
        target.append(batch["mfcc"].cpu().numpy());labels.extend(batch["label"]);keys.extend(batch["sample_key"])
    target=np.concatenate(target);output={name:np.concatenate(value) for name,value in output.items()}
    metrics={name:{**retrieval(value,target,labels,keys),"mfcc_mae":float(mfcc_distance(value,target).mean()),"masked_mfcc_mae":float(mfcc_distance(value,target).mean()),"mfcc_mask_policy":"all 256 canonical frames are VAD-active time-normalized support","between_trial_variance_ratio":variance_ratio(value,target,labels)} for name,value in output.items()}
    template=same_label_template(target,labels);relative=float(mfcc_distance(output["correct"],target).mean()/max(float(mfcc_distance(template,target).mean()),1e-8))
    wins={name:paired_win_rate(output["correct"],output[name],target) for name in ("zero","time","channel")}
    bootstrap={name:bootstrap_mean_gain(output["correct"],output[name],target,samples=int(cfg["evaluation"]["bootstrap_samples"]),seed=int(cfg["training"]["seed"])+i) for i,name in enumerate(("zero","time","channel"))}
    paired_bootstrap=paired_r_at_1_above_chance(output["correct"],target,labels,samples=int(cfg["evaluation"]["bootstrap_samples"]),seed=int(cfg["training"]["seed"])+40)
    result={"stage":stage,"checkpoint_stage":payload["extra"].get("stage"),"n":len(labels),"sample_keys":keys,"exploratory":stage=="locked_unseen","correct":metrics["correct"],"controls":{name:metrics[name] for name in ("zero","time","channel")},"correct_vs_control_win_rate":wins,"bootstrap":bootstrap,"paired_r_at_1_bootstrap":paired_bootstrap,"template_error_ratio":relative}
    if stage in {"micro","fit"}:
        threshold=cfg["gates"][stage];checks={"label_top1":metrics["correct"]["label_top1"]>=float(threshold["label_top1_min"]),"paired_r_at_1":metrics["correct"]["paired_r_at_1"]>=float(threshold["paired_r_at_1_min"]),"variance":metrics["correct"]["between_trial_variance_ratio"]>=float(threshold["variance_ratio_min"])}
        if "template_ratio_max" in threshold:
            checks["template_ratio"]=relative<=float(threshold["template_ratio_max"])
        checks.update({f"win_{name}":wins[name]>=float(threshold["paired_win_rate_min"]) for name in wins})
        if stage=="fit":
            checks["paired_r_at_1_above_chance_ci"] = paired_bootstrap["ci_low"] > 0.0
        result.update({"thresholds":threshold,"checks":checks,"passed":bool(all(checks.values()))})
    if stage in {"validation", "locked", "locked_unseen"}:
        wav, wav_bootstrap = heldout_wav_metrics(config_path, cfg, records, output, keys, labels, device)
        result["canonical_voice_wav"] = {
            "voice_policy": "fit_subject_centroid_medoid; no target-subject reference audio",
            "metrics": wav,
            "correct_minus_control_paired_bootstrap": wav_bootstrap,
        }
    return result


def save_gate(path: Path, report: dict[str, Any], no_fail: bool) -> None:
    write_json(path,report)
    print(f"[v3 {report.get('gate', report.get('stage'))}] passed={report.get('passed', True)} report={path}",flush=True)
    if "passed" in report and not report["passed"] and not no_fail: raise SystemExit(2)


def main() -> None:
    args=parse();config_path,cfg=load_config(args.config);device=default_device(args.device);records=load_prepared(output_path(config_path,cfg,"prepared_cache"))
    for phase in (args.phase,):
        if phase=="v0":
            report=gate_v0(config_path,cfg,records,device,args.limit_per_label)
            report["lineage"]=capture_lineage(config_path,cfg,artifact_keys=("vocoder_manifest",))
            save_gate(output_path(config_path,cfg,"v0_gate"),report,args.no_fail)
        elif phase=="v1":
            require_passed_gate(config_path,cfg,"v0_gate",lineage_artifact_keys=("vocoder_manifest",))
            report=gate_v1(config_path,cfg,records,device,args.limit_per_label)
            report["lineage"]=capture_lineage(config_path,cfg,artifact_keys=("audio_checkpoint","v0_gate"))
            save_gate(output_path(config_path,cfg,"v1_gate"),report,args.no_fail)
        elif phase=="v2":
            require_passed_gate(config_path,cfg,"v1_gate",lineage_artifact_keys=("audio_checkpoint","v0_gate"))
            report=gate_v2(config_path,cfg,records,device,args.limit_per_label)
            report["lineage"]=capture_lineage(config_path,cfg,artifact_keys=("audio_checkpoint","v1_gate"))
            save_gate(output_path(config_path,cfg,"v2_gate"),report,args.no_fail)
        else:
            if phase=="micro":
                require_passed_gate(config_path,cfg,"v2_gate",lineage_artifact_keys=("audio_checkpoint","v1_gate"))
                lineage_keys=("micro_checkpoint","v2_gate")
            elif phase=="fit":
                require_passed_gate(config_path,cfg,"micro_gate",lineage_artifact_keys=("micro_checkpoint","v2_gate"))
                lineage_keys=("fit_checkpoint","micro_gate")
            else:
                require_passed_gate(config_path,cfg,"fit_gate",lineage_artifact_keys=("fit_checkpoint","micro_gate"))
                review_path=output_path(config_path,cfg,"training_review")
                if not review_path.is_file():
                    raise RuntimeError("v3 fail-closed: listen to and approve the full-fit training WAV preview before held-out evaluation")
                review=read_json(review_path)
                expected_review=capture_lineage(config_path,cfg,artifact_keys=("fit_checkpoint","fit_gate","fit_preview_manifest"))
                if not review.get("passed",False) or review.get("lineage")!=expected_review:
                    raise RuntimeError("v3 fail-closed: training WAV human review is rejected or stale")
                lineage_keys=("fit_checkpoint","fit_gate","training_review")
            report=eeg_stage(config_path,cfg,records,device,phase)
            report["lineage"]=capture_lineage(config_path,cfg,artifact_keys=lineage_keys)
            if phase=="locked_unseen":
                report["interpretation_note"]="single-label pot retrieval is exploratory and label top-1 is not an informative success metric"
            path=output_path(config_path,cfg,{"micro":"micro_gate","fit":"fit_gate","validation":"validation_report","locked":"locked_report","locked_unseen":"locked_unseen_report"}[phase]);save_gate(path,report,args.no_fail)


if __name__=="__main__":main()
