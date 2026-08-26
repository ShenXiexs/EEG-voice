#!/usr/bin/env python3
"""Train the separately gated audio-only MFCC-to-acoustic renderer."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(APP / "src"))

from eeg2speech.data import JointManifestDataset, homogeneous_collate, phoneme_vocabulary_from_manifest
from eeg2speech.model import AudioMFCCRenderer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_holdout_indices(frame, validation_fraction: float = 0.2) -> tuple[list[int], list[int]]:
    """Create a deterministic content-disjoint audio-only renderer split."""
    contents = sorted(
        set(frame.linguistic_content_id.astype(str)),
        key=lambda value: hashlib.sha256(f"audio-renderer|{value}".encode()).hexdigest(),
    )
    if len(contents) < 2:
        raise RuntimeError("audio renderer needs at least two linguistic contents for its oracle split")
    validation_count = max(1, min(len(contents) - 1, int(round(len(contents) * validation_fraction))))
    validation_contents = set(contents[-validation_count:])
    train_indices = frame.index[~frame.linguistic_content_id.isin(validation_contents)].astype(int).tolist()
    validation_indices = frame.index[frame.linguistic_content_id.isin(validation_contents)].astype(int).tolist()
    if not train_indices or not validation_indices:
        raise RuntimeError("audio renderer content holdout produced an empty role")
    if set(frame.iloc[train_indices].linguistic_content_id) & set(frame.iloc[validation_indices].linguistic_content_id):
        raise RuntimeError("audio renderer content holdout leaked linguistic content")
    return train_indices, validation_indices


def renderer_loss(state, batch, cfg: dict) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["acoustic_supervision"].bool()
    if not mask.any():
        raise RuntimeError("audio renderer received no verified-exact acoustic targets")
    mel = F.smooth_l1_loss(state.log_mel[mask], batch["acoustic_log_mel"][mask])
    rms = F.smooth_l1_loss(state.rms[mask], batch["acoustic_rms"][mask])
    activity = F.binary_cross_entropy_with_logits(state.activity_logits[mask], batch["acoustic_activity"][mask].float())
    total = float(cfg["log_mel_weight"]) * mel + float(cfg["rms_weight"]) * rms + float(cfg["activity_weight"]) * activity
    return total, {"total": float(total.detach()), "log_mel": float(mel.detach()),
                   "rms": float(rms.detach()), "activity_bce": float(activity.detach())}


def evaluate(model, loader, device, train_template: torch.Tensor) -> dict[str, float]:
    predictions = []; targets = []; activities = []; activity_targets = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            mfcc = batch["content_mfcc"].to(device)
            state = model(mfcc)
            predictions.append(state.log_mel.cpu()); targets.append(batch["acoustic_log_mel"])
            activities.append(state.activity_logits.cpu()); activity_targets.append(batch["acoustic_activity"])
    prediction, target = torch.cat(predictions), torch.cat(targets)
    logits, truth = torch.cat(activities), torch.cat(activity_targets).bool()
    error = float((prediction - target).abs().mean())
    baseline = float((train_template.expand_as(target) - target).abs().mean())
    predicted = logits >= 0
    tp = (predicted & truth).sum().float(); fp = (predicted & ~truth).sum().float(); fn = (~predicted & truth).sum().float()
    f1 = float(2 * tp / (2 * tp + fp + fn).clamp_min(1))
    return {"log_mel_mae": error, "train_template_log_mel_mae": baseline,
            "log_mel_improvement": float(1.0 - error / max(baseline, 1e-8)), "activity_f1": f1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "joint_pilot_v1.yaml")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    data_cfg = yaml.safe_load((args.config.parent / cfg["data_config"]).resolve().read_text())
    artifact_root = ROOT / data_cfg["output_root"]
    manifest = artifact_root / "manifests" / "manifest_built.csv"
    split = artifact_root / "splits" / f"{cfg['split']['protocol']}_fold-{cfg['split']['fold']}.csv"
    targets = artifact_root / "speech_targets" / "speech_targets.h5"
    normalizer = artifact_root / "normalizers" / f"{split.stem}.json"
    vocabulary = phoneme_vocabulary_from_manifest(manifest)
    train = JointManifestDataset(manifest, split, "train", "ds004940", targets, normalizer,
                                 float(cfg["loss"]["weak_content_weight"]),
                                 supervision_types={"paired_audio"}, phoneme_vocabulary=vocabulary)
    renderer_cfg = cfg["audio_renderer"]
    train_indices, validation_indices = content_holdout_indices(train.frame)
    train_loader = DataLoader(Subset(train, train_indices), batch_size=int(renderer_cfg["batch_size"]),
                              shuffle=True, collate_fn=homogeneous_collate)
    validation_loader = DataLoader(Subset(train, validation_indices), batch_size=int(renderer_cfg["batch_size"]),
                                   shuffle=False, collate_fn=homogeneous_collate)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_config = {key: renderer_cfg[key] for key in ("hidden_dimension", "layers", "dropout")}
    model = AudioMFCCRenderer(**model_config).to(device)
    first = next(iter(train_loader)); state = model(first["content_mfcc"].to(device))
    loss, metrics = renderer_loss(state, {key: value.to(device) if torch.is_tensor(value) else value for key, value in first.items()}, renderer_cfg)
    loss.backward()
    if args.dry_run:
        finite = bool(torch.isfinite(loss)) and all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        print(json.dumps({"status": "pass" if finite else "fail", "metrics": metrics}, indent=2)); train.close(); return 0 if finite else 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(renderer_cfg["learning_rate"]))
    model.zero_grad(set_to_none=True)
    iterator = iter(train_loader); history = []
    for step in range(1, (args.max_steps or int(renderer_cfg["max_steps"])) + 1):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(train_loader); batch = next(iterator)
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True); state = model(batch["content_mfcc"])
        loss, metrics = renderer_loss(state, batch, renderer_cfg)
        if not torch.isfinite(loss): raise RuntimeError(f"nonfinite renderer loss at step {step}")
        loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0: history.append({"step": step, **metrics})
    train_template = torch.stack([train[index]["acoustic_log_mel"] for index in train_indices]).mean(0, keepdim=True)
    validation_metrics = evaluate(model, validation_loader, device, train_template)
    checks = {
        "log_mel": validation_metrics["log_mel_improvement"] >= float(renderer_cfg["gate_log_mel_improvement_min"]),
        "activity": validation_metrics["activity_f1"] >= float(renderer_cfg["gate_activity_f1_min"]),
    }
    required = [manifest, split, targets, normalizer, artifact_root / "source_lock.json"]
    artifact_hashes = {path.name: sha256_file(path) for path in required}
    output = ROOT / "outputs" / "joint_pilot_v1" / "audio_renderer"
    output.mkdir(parents=True, exist_ok=True)
    split_summary = {"protocol": "m0_train_fold_linguistic_content_holdout",
                     "train_pairs": len(train_indices), "validation_pairs": len(validation_indices),
                     "train_contents": int(train.frame.iloc[train_indices].linguistic_content_id.nunique()),
                     "validation_contents": int(train.frame.iloc[validation_indices].linguistic_content_id.nunique())}
    torch.save({"model": model.state_dict(), "model_config": model_config, "artifact_hashes": artifact_hashes,
                "gate": checks, "validation": validation_metrics, "split": split_summary}, output / "checkpoint.pt")
    (output / "metrics.json").write_text(json.dumps({"history": history, "validation": validation_metrics,
                                                       "split": split_summary,
                                                       "gate": {"checks": checks, "passed": all(checks.values())}}, indent=2) + "\n")
    print(json.dumps({"validation": validation_metrics, "split": split_summary, "gate": checks}, indent=2))
    train.close()
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
