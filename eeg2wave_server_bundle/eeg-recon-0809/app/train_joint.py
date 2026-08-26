#!/usr/bin/env python3
"""Run the gated DS004940/DS006104 content pilot; never launches full training."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(APP / "src"))

from eeg2speech.data import (AlternatingBatchIterator, JointManifestDataset, auxiliary_indices,
                             homogeneous_collate, phoneme_vocabulary_from_manifest, pilot_indices)
from eeg2speech.gates import require_registered_m0_gates
from eeg2speech.losses import counterfactual_eeg, joint_content_loss
from eeg2speech.model import JointEEGContentModel


def resolve(path: str | Path, base: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (base / value).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_code_sha256() -> str:
    digest = hashlib.sha256()
    paths = sorted([Path(__file__), *list((APP / "src").rglob("*.py"))], key=lambda path: str(path))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode()); digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def loader_for(dataset: JointManifestDataset, indices: list[int], batch_size: int, seed: int) -> DataLoader:
    if not indices:
        raise RuntimeError("pilot selection produced zero trials")
    subset = Subset(dataset, indices)
    weights = dataset.sampling_weights()[indices]
    sampler = WeightedRandomSampler(weights, num_samples=max(len(indices), batch_size), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    return DataLoader(subset, batch_size=batch_size, sampler=sampler, collate_fn=homogeneous_collate, drop_last=False)


def evaluation_loader_for(dataset: JointManifestDataset, indices: list[int], batch_size: int) -> DataLoader:
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                      collate_fn=homogeneous_collate, drop_last=False)


def move(batch: dict, target: torch.device) -> dict:
    return {key: value.to(target) if torch.is_tensor(value) else value for key, value in batch.items()}


def retrieval_r1(prediction: torch.Tensor, target: torch.Tensor, eligible: torch.Tensor,
                 labels: list[str]) -> float:
    if eligible.sum() < 2:
        return float("nan")
    left = torch.nn.functional.normalize(prediction[eligible].flatten(1), dim=-1)
    right = torch.nn.functional.normalize(target[eligible].flatten(1), dim=-1)
    names = [labels[index] for index in eligible.nonzero(as_tuple=False).flatten().tolist()]
    nearest = (left @ right.T).argmax(1).tolist()
    return float(np.mean([names[index] == names[target_index] for index, target_index in enumerate(nearest)]))


def evaluate_batch(model, batch: dict, target: torch.device) -> dict[str, float]:
    batch = move(batch, target)
    with torch.no_grad():
        state = model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], batch["time_mask"], batch["dataset_id"])
        eligible = batch["pairing_weight"] > 0
        correct = torch.nn.functional.l1_loss(state.mfcc[eligible], batch["content_mfcc"][eligible]) if eligible.any() else state.mfcc.new_tensor(float("nan"))
        metrics = {"content_retrieval_r1": retrieval_r1(state.mfcc, batch["content_mfcc"], eligible, batch["linguistic_content_id"]), "correct_mfcc_l1": float(correct)}
        for control in ("zero", "time_shuffle", "channel_shuffle"):
            controlled = counterfactual_eeg(batch["eeg"], control, time_mask=batch["time_mask"], channel_mask=batch["channel_mask"])
            output = model(controlled, batch["channel_xyz"], batch["channel_mask"], batch["time_mask"], batch["dataset_id"])
            value = torch.nn.functional.l1_loss(output.mfcc[eligible], batch["content_mfcc"][eligible]) if eligible.any() else output.mfcc.new_tensor(float("nan"))
            metrics[f"{control}_mfcc_l1"] = float(value)
    return metrics


def full_content_retrieval(model, loader: DataLoader, target: torch.device) -> float:
    predictions = []; teachers = []; labels = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, target)
            state = model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], batch["time_mask"], batch["dataset_id"])
            eligible = batch["pairing_weight"] > 0
            predictions.append(state.mfcc[eligible]); teachers.append(batch["content_mfcc"][eligible])
            labels.extend(batch["linguistic_content_id"][index] for index in eligible.nonzero(as_tuple=False).flatten().tolist())
    if was_training:
        model.train()
    prediction = torch.cat(predictions)
    teacher = torch.cat(teachers)
    return retrieval_r1(prediction, teacher, torch.ones(len(prediction), dtype=torch.bool, device=prediction.device), labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "joint_pilot_v1.yaml")
    parser.add_argument("--mode", choices=["ds004940", "ds006104", "joint"], required=True)
    parser.add_argument("--stage", choices=["overfit", "generalization"], default="overfit")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-model", action="store_true", help="use a 48-dim/1-layer engineering smoke model")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.smoke_model:
        cfg["model"].update({"dimension": 48, "heads": 4, "layers": 1, "local_layers": 1, "dropout": 0.0})
        cfg["training"]["batch_size"] = 2
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    data_cfg_path = resolve(cfg["data_config"], args.config.parent)
    data_cfg = yaml.safe_load(data_cfg_path.read_text())
    # v3 extends v2; output_root is explicitly overridden in the child.
    artifact_root = ROOT / data_cfg["output_root"]
    audit = json.loads((artifact_root / "qc" / "audit.json").read_text())
    if audit.get("included_counts", {}).get("ds004940") != 17489 or audit.get("included_counts", {}).get("ds006104") != 10888:
        raise RuntimeError("Stage 0 audit gate failed: expected 17,489 and 10,888 included trials")
    if args.stage == "generalization" and bool(cfg["training"].get("stage2_requires_all_m0_gates", True)):
        require_registered_m0_gates(ROOT, cfg)
    split_protocol = cfg["split"]["protocol"] if args.stage == "overfit" else "stage2_joint_ood"
    split_path = artifact_root / "splits" / f"{split_protocol}_fold-{cfg['split']['fold']}.csv"
    artifact_set = "built" if args.stage == "overfit" else "stage2"
    manifest_path = artifact_root / "manifests" / f"manifest_{artifact_set}.csv"
    target_name = "speech_targets" if args.stage == "overfit" else "speech_targets_stage2"
    target_path = artifact_root / "speech_targets" / f"{target_name}.h5"
    normalizer_path = artifact_root / "normalizers" / f"{split_path.stem}.json"
    source_lock_path = artifact_root / "source_lock.json"
    validation_path = artifact_root / "qc" / "validate.json"
    for required in (split_path, manifest_path, target_path, normalizer_path, source_lock_path, validation_path):
        if not required.exists():
            raise RuntimeError(f"required gated artifact is missing: {required}")
    validation = json.loads(validation_path.read_text())
    if validation.get("status") != "pass":
        raise RuntimeError("Stage 0 validation gate is not passing")
    if args.stage == "overfit" and not args.dry_run and not args.smoke_model:
        blockers = list(validation.get("formal_m0_blockers", []))
        if args.mode == "ds006104":
            blockers = [value for value in blockers if value != "ds004940_human_pair_review"]
        if blockers:
            raise RuntimeError(f"formal M0 is blocked by Stage-0 gates: {blockers}")
    source_lock = json.loads(source_lock_path.read_text())
    artifact_hashes = {path.name: sha256_file(path) for path in
                       (source_lock_path, split_path, manifest_path, target_path, normalizer_path)}

    names = [args.mode] if args.mode != "joint" else ["ds004940", "ds006104"]
    datasets = {}; loaders = {}; evaluation_loaders = {}
    selections = {}
    vocabulary = phoneme_vocabulary_from_manifest(manifest_path)
    if len(vocabulary) > int(cfg["model"]["phoneme_classes"]):
        raise RuntimeError(f"phoneme vocabulary has {len(vocabulary)} labels but model has {cfg['model']['phoneme_classes']} classes")
    for name in names:
        dataset = JointManifestDataset(manifest_path, split_path, "train", name, target_path, normalizer_path,
                                       float(cfg["loss"]["weak_content_weight"]),
                                       supervision_types={"paired_audio", "weak_audio"},
                                       phoneme_vocabulary=vocabulary)
        indices = pilot_indices(dataset, cfg, args.stage, "train")
        datasets[name] = dataset
        loaders[name] = loader_for(dataset, indices, int(cfg["training"]["batch_size"]), args.seed)
        evaluation_loaders[name] = evaluation_loader_for(dataset, indices, int(cfg["training"]["batch_size"]))
        selected = dataset.frame.iloc[indices]
        selections[name] = {"pairs": len(selected), "subjects": int(selected.subject.nunique()),
                            "contents": int(selected.linguistic_content_id.nunique()),
                            "subject_counts": selected.groupby("subject").size().astype(int).to_dict()}

    if args.mode in {"ds006104", "joint"}:
        auxiliary = JointManifestDataset(manifest_path, split_path, "train", "ds006104", target_path, normalizer_path,
                                         float(cfg["loss"]["weak_content_weight"]),
                                         supervision_types={"label_only"}, phoneme_vocabulary=vocabulary)
        aux_indices = auxiliary_indices(auxiliary, cfg, args.stage)
        aux_name = "ds006104_label_only"
        datasets[aux_name] = auxiliary
        loaders[aux_name] = loader_for(auxiliary, aux_indices, int(cfg["training"]["batch_size"]), args.seed + 1009)
        selected = auxiliary.frame.iloc[aux_indices]
        selections[aux_name] = {"pairs": len(selected), "subjects": int(selected.subject.nunique()),
                                "contents": int(selected.linguistic_content_id.nunique()),
                                "labels": int(selected.phoneme_label.nunique()),
                                "subject_counts": selected.groupby("subject").size().astype(int).to_dict()}

    model = JointEEGContentModel(**cfg["model"]).to(device())
    dry_batches = {name: move(next(iter(loader)), device()) for name, loader in loaders.items()}
    first_batch = dry_batches[names[0]]
    state = model(first_batch["eeg"], first_batch["channel_xyz"], first_batch["channel_mask"], first_batch["time_mask"], first_batch["dataset_id"])
    if state.mfcc.shape[1:] != (39, 161) or not torch.isfinite(state.mfcc).all(): raise RuntimeError("model forward contract failed")
    if args.dry_run:
        by_dataset = {}
        model.zero_grad(set_to_none=True)
        for name in loaders:
            batch = dry_batches[name]
            output = model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], batch["time_mask"], batch["dataset_id"])
            loss, metrics = joint_content_loss(output, batch, model, cfg["loss"])
            (loss / len(loaders)).backward()
            by_dataset[name] = {"batch_shape": list(batch["eeg"].shape), "metrics": metrics}
        gradient_finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        print(json.dumps({"status": "pass" if gradient_finite else "fail", "mode": args.mode,
                          "datasets": by_dataset,
                          "selections": selections,
                          "gradient_finite": gradient_finite}, indent=2))
        for dataset in datasets.values(): dataset.close()
        return 0 if gradient_finite else 2

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]),
                                  weight_decay=float(cfg["training"]["weight_decay"]))
    maximum = args.max_steps or int(cfg["training"]["max_steps"])
    schedule = list(names)
    if "ds006104_label_only" in loaders:
        interval = int(cfg["training"]["label_only_batch_interval"])
        if interval < 2:
            raise RuntimeError("label_only_batch_interval must be at least 2")
        schedule = [names[index % len(names)] for index in range(interval - 1)] + ["ds006104_label_only"]
    iterator = iter(AlternatingBatchIterator(loaders, schedule))
    history = []
    seen_batch_sources: set[str] = set()
    early_stopped = False
    step = 0
    for step in range(1, maximum + 1):
        name, batch = next(iterator)
        batch = move(batch, device())
        optimizer.zero_grad(set_to_none=True)
        state = model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], batch["time_mask"], batch["dataset_id"])
        loss, metrics = joint_content_loss(state, batch, model, cfg["loss"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip"]))
        optimizer.step()
        if name not in seen_batch_sources or step % 100 == 0:
            seen_batch_sources.add(name)
            history.append({"step": step, "dataset": name, **metrics})
            if step % 100 == 0:
                screen = {dataset_name: full_content_retrieval(model, evaluation_loaders[dataset_name], device())
                          for dataset_name in names}
                history[-1]["full_content_retrieval_r1"] = screen
                if args.stage == "overfit" and all(
                    value >= float(cfg["training"]["early_stop_pair_retrieval_r1"]) for value in screen.values()
                ):
                    early_stopped = True
            print(json.dumps(history[-1]))
            if early_stopped:
                break

    controls = {name: evaluate_batch(model, next(iter(loaders[name])), device()) for name in names}
    run_kind = "smoke" if args.smoke_model else "pilot"
    checkpoint_dir = ROOT / "outputs" / "joint_pilot_v1" / run_kind / args.stage / args.mode / f"seed-{args.seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": cfg["model"], "pilot_config": cfg,
                "mode": args.mode, "stage": args.stage, "seed": args.seed, "run_kind": run_kind,
                "split_protocol": split_protocol,
                "selections": selections, "source_lock_sha256": source_lock["source_lock_sha256"],
                "preprocess_config_sha256": source_lock["config_sha256"],
                "runtime_code_sha256": runtime_code_sha256(), "phoneme_vocabulary": vocabulary,
                "artifact_hashes": artifact_hashes}, checkpoint_dir / "checkpoint.pt")
    interpretation = "engineering_only_no_scientific_gate_claim" if args.smoke_model else "evaluate_against_registered_pilot_gates"
    (checkpoint_dir / "metrics.json").write_text(json.dumps({"run_kind": run_kind, "interpretation": interpretation,
                                                              "selections": selections, "history": history,
                                                              "controls": controls, "steps_completed": step,
                                                              "early_stopped": early_stopped,
                                                              "batch_schedule": schedule,
                                                              "runtime_code_sha256": runtime_code_sha256(),
                                                              "artifact_hashes": artifact_hashes}, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(checkpoint_dir / "checkpoint.pt"), "run_kind": run_kind,
                      "interpretation": interpretation, "controls": controls}, indent=2))
    for dataset in datasets.values(): dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
