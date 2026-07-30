#!/usr/bin/env python3
"""Train v3 audio oracle and the content-first EEG-to-MFCC stages."""
from __future__ import annotations

import argparse
import json
import math
import time
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_v3.data import V3Dataset, collate, load_prepared
from src.open_vocab_v3.metrics import cvae_audio_loss, fit_loss, overfit_loss
from src.open_vocab_v3.model import EEGMFCCEncoder, MFCCMelDecoder
from src.open_vocab_v3.runtime import capture_lineage, checkpoint_path, default_device, load_config, move_batch, output_path, read_json, require_passed_gate, seed_everything, sha256_file, write_json


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v3 content-first modules")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("audio", "micro", "fit"), required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wall-hours", type=float, default=None)
    parser.add_argument(
        "--deadline-epoch", type=float, default=0.0,
        help="optional Unix deadline shared across separately invoked v3 stages",
    )
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
    return parser.parse_args()


def loader(dataset: Any, *, batch_size: int, train: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, collate_fn=collate, num_workers=0)


def save_checkpoint(path: Path, *, schema: str, model: torch.nn.Module, epoch: int, score: float, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": schema, "epoch": epoch, "score": score, "state_dict": model.state_dict(), "extra": extra}, path)


def load_audio(config_path: Path, cfg: dict[str, Any], device: torch.device) -> tuple[MFCCMelDecoder, dict[str, Any]]:
    model = MFCCMelDecoder(
        mfcc_bins=int(cfg["audio"]["mfcc_bins"]), mel_bins=int(cfg["audio"]["mel_bins"]),
        dimension=int(cfg["model"]["audio_dimension"]), voice_dim=int(cfg["speaker"]["embedding_dimension"]),
        latent_dim=int(cfg["model"]["audio_latent_dimension"]),
        residual_limit_db=float(cfg["model"]["audio_residual_limit_db"]),
    ).to(device)
    raw = torch.load(checkpoint_path(config_path, cfg, "audio"), map_location=device, weights_only=False)
    if raw.get("schema_version") != "openvoice-v3-audio-cvae-v2":
        raise ValueError(f"not a v3 audio checkpoint: {raw.get('schema_version')}")
    model.load_state_dict(raw["state_dict"], strict=True)
    return model.eval(), raw


def load_eeg(
    config_path: Path, cfg: dict[str, Any], device: torch.device, *, stage: str = "fit"
) -> tuple[EEGMFCCEncoder, dict[str, Any]]:
    if stage not in {"micro", "fit"}:
        raise ValueError(f"unknown v3 EEG checkpoint stage: {stage}")
    model = EEGMFCCEncoder(
        mfcc_bins=int(cfg["audio"]["mfcc_bins"]), dimension=int(cfg["model"]["eeg_dimension"]),
        heads=int(cfg["model"]["heads"]), layers=int(cfg["model"]["layers"]), dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    raw = torch.load(checkpoint_path(config_path, cfg, stage), map_location=device, weights_only=False)
    expected_schema = f"openvoice-v3-eeg-{stage}-v1"
    if raw.get("schema_version") != expected_schema:
        raise ValueError(f"not a v3 EEG checkpoint: {raw.get('schema_version')}")
    model.load_state_dict(raw["state_dict"], strict=True)
    return model.eval(), raw


def _history(config_path: Path, cfg: dict[str, Any], name: str) -> Path:
    return output_path(config_path, cfg, "output_root") / name / "metrics" / "training.jsonl"


def _append(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def deadline_reached(args: argparse.Namespace) -> bool:
    """Use an externally supplied absolute deadline when the shell gates stages.

    The audio, 50-pair, and full-fit commands are intentionally separate so
    the gate runner can stop between them.  A common Unix deadline keeps that
    convenience from silently turning the 9.5-hour budget into three budgets.
    """
    return bool(args.deadline_epoch and time.time() >= float(args.deadline_epoch))


def train_audio(config_path: Path, cfg: dict[str, Any], records: Any, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    if "speaker_reference_embedding" not in records.arrays:
        raise RuntimeError("audio oracle requires --with-speaker prepared cache for V1/V2")
    train = V3Dataset(records, ("fit",), eligible_only=True)
    model = MFCCMelDecoder(
        mfcc_bins=int(cfg["audio"]["mfcc_bins"]), mel_bins=int(cfg["audio"]["mel_bins"]),
        dimension=int(cfg["model"]["audio_dimension"]), voice_dim=int(cfg["speaker"]["embedding_dimension"]),
        latent_dim=int(cfg["model"]["audio_latent_dimension"]),
        residual_limit_db=float(cfg["model"]["audio_residual_limit_db"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["audio_lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    best, stale, started = math.inf, 0, time.monotonic()
    total = int(cfg["training"]["audio_epochs"])
    history = _history(config_path, cfg, "audio")
    if args.fresh and history.exists():
        history.unlink()
    for epoch in range(total):
        model.train(); train_values = []
        for step, batch in enumerate(loader(train, batch_size=int(cfg["training"]["audio_batch_size"]), train=True)):
            if deadline_reached(args):
                break
            batch = move_batch(batch, device)
            voice = batch["speaker_reference"].float()
            mfcc = batch["mfcc"].float()
            mean = batch["speaker_reference_mfcc_mean"].float()
            std = batch["speaker_reference_mfcc_std"].float()
            target_mel = batch["mel"].float()
            values = model.distributions(mfcc, voice, mean, std, target_mel)
            posterior_latent = model._sample(
                values["posterior_mean"], values["posterior_logvar"], stochastic=True
            )
            posterior_mel = model.decode(
                values["analytic_mel"], values["content_hidden"], values["voice_hidden"], posterior_latent
            )
            prior_mel = model.decode(
                values["analytic_mel"], values["content_hidden"], values["voice_hidden"], values["prior_mean"]
            )
            warmup = max(1, int(cfg["training"]["cvae_kl_warmup_epochs"]))
            kl_beta = float(cfg["training"]["cvae_kl_beta_max"]) * min(1.0, float(epoch + 1) / warmup)
            loss, components = cvae_audio_loss(
                posterior_mel, prior_mel, values["analytic_mel"], target_mel,
                values["posterior_mean"], values["posterior_logvar"],
                values["prior_mean"], values["prior_logvar"],
                kl_beta=kl_beta,
                free_bits=float(cfg["training"]["cvae_free_bits"]),
                prior_weight=float(cfg["training"]["cvae_prior_reconstruction_weight"]),
                analytic_consistency_weight=float(cfg["training"]["cvae_analytic_consistency_weight"]),
            )
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip"])); optimizer.step()
            train_values.append(float(loss.detach()))
            if deadline_reached(args) or (args.smoke_steps and step + 1 >= args.smoke_steps):
                break
        if not train_values:
            raise RuntimeError("v3 audio stage reached the shared deadline before one optimizer step")
        # This stage is audio-only.  Do not use the held-out EEG validation role
        # to select an oracle checkpoint before the training-first gates.
        score = float(np.mean(train_values)); record = {
            "epoch": epoch + 1, "train_loss": score, "selection_loss": score,
            "last_components": components, "elapsed_seconds": time.monotonic() - started,
        }
        _append(history, record)
        print(f"[v3 audio] epoch={epoch + 1}/{total} train={score:.5f} elapsed={record['elapsed_seconds']:.1f}s", flush=True)
        if score < best:
            best, stale = score, 0
            save_checkpoint(
                checkpoint_path(config_path, cfg, "audio"), schema="openvoice-v3-audio-cvae-v2",
                model=model, epoch=epoch, score=score,
                extra={
                    "fit_role": "fit", "decoder": "conditional_variational_residual",
                    "analytic_backend": "orthonormal_inverse_dct_librosa_equivalent",
                    "voice_condition": "non_target_ecapa_reference",
                    "eeg_uses": "conditional_prior_mean_or_sample_only",
                },
            )
        else:
            stale += 1
        if stale >= int(cfg["training"]["audio_patience"]):
            break
        if deadline_reached(args) or (args.wall_hours and time.monotonic() - started >= args.wall_hours * 3600):
            break
    checkpoint = checkpoint_path(config_path, cfg, "audio")
    return {"elapsed_seconds": time.monotonic() - started, "epochs_completed": epoch + 1, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "best_loss": best}


def micro_dataset(records: Any, subject: str, per_label: int) -> Subset:
    source = V3Dataset(records, ("fit",), eligible_only=True)
    selected: list[int] = []
    by_label: dict[str, list[int]] = {}
    for item, record_index in enumerate(source.indices):
        if str(records.arrays["subjects"][record_index]) != subject:
            continue
        label = str(records.arrays["labels"][record_index])
        by_label.setdefault(label, []).append(item)
    if len(by_label) != 10:
        raise ValueError(f"v3 50-pair gate requires exactly 10 seen labels for {subject}, found {len(by_label)}")
    for label in sorted(by_label):
        ordered = sorted(by_label[label], key=lambda item: str(records.arrays["sample_keys"][source.indices[item]]))
        if len(ordered) < per_label:
            raise ValueError(f"micro subject {subject} has only {len(ordered)} samples for label {label}")
        selected.extend(ordered[:per_label])
    if len(selected) != 50 or len(selected) != int(cfg_value := per_label * len(by_label)):
        raise ValueError(f"micro dataset has {len(selected)}, expected exactly 50 ({cfg_value} from config)")
    return Subset(source, selected)


def _new_eeg(cfg: dict[str, Any], device: torch.device) -> EEGMFCCEncoder:
    return EEGMFCCEncoder(
        mfcc_bins=int(cfg["audio"]["mfcc_bins"]), dimension=int(cfg["model"]["eeg_dimension"]),
        heads=int(cfg["model"]["heads"]), layers=int(cfg["model"]["layers"]), dropout=float(cfg["model"]["dropout"]),
    ).to(device)


def train_micro(config_path: Path, cfg: dict[str, Any], records: Any, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    dataset = micro_dataset(records, str(cfg["micro_gate"]["subject"]), int(cfg["micro_gate"]["per_label"]))
    model = _new_eeg(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["eeg_lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    started = time.monotonic(); history = _history(config_path, cfg, "eeg_micro")
    if args.fresh and history.exists(): history.unlink()
    epoch = -1
    for epoch in range(int(cfg["training"]["micro_epochs"])):
        model.train(); values=[]; parts=[]
        for step, batch in enumerate(loader(dataset, batch_size=int(cfg["training"]["eeg_batch_size"]), train=True)):
            if deadline_reached(args): break
            batch=move_batch(batch,device); predicted,_=model(batch["eeg"].float(),batch["channel_xyz"].float(),batch["channel_mask"],batch["time_mask"])
            loss, part=overfit_loss(predicted,batch["mfcc"].float())
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["training"]["grad_clip"])); optimizer.step()
            values.append(float(loss.detach())); parts.append(part)
            if deadline_reached(args) or (args.smoke_steps and step+1>=args.smoke_steps): break
        if not values:
            raise RuntimeError("v3 50-pair stage reached the shared deadline before one optimizer step")
        record={"epoch":epoch+1,"train_loss":float(np.mean(values)),"components":{key:float(np.mean([part[key] for part in parts])) for key in parts[0]},"elapsed_seconds":time.monotonic()-started}
        _append(history,record); print(f"[v3 micro] epoch={epoch+1} train={record['train_loss']:.5f} elapsed={record['elapsed_seconds']:.1f}s",flush=True)
        if deadline_reached(args) or (args.wall_hours and time.monotonic()-started>=args.wall_hours*3600): break
    checkpoint=checkpoint_path(config_path,cfg,"micro")
    save_checkpoint(checkpoint,schema="openvoice-v3-eeg-micro-v1",model=model,epoch=epoch,score=record["train_loss"],extra={"stage":"micro","subject":str(cfg["micro_gate"]["subject"]),"per_label":int(cfg["micro_gate"]["per_label"])})
    return {"elapsed_seconds":time.monotonic()-started,"epochs_completed":epoch+1,"checkpoint":str(checkpoint),"checkpoint_sha256":sha256_file(checkpoint),"final_loss":record["train_loss"]}


def train_fit(config_path: Path, cfg: dict[str, Any], records: Any, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    train=V3Dataset(records,("fit",),eligible_only=True)
    model=_new_eeg(cfg,device); optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg["training"]["eeg_lr"]),weight_decay=float(cfg["training"]["weight_decay"]))
    best,stale,started=math.inf,0,time.monotonic(); history=_history(config_path,cfg,"eeg_fit")
    if args.fresh and history.exists(): history.unlink()
    for epoch in range(int(cfg["training"]["fit_epochs"])):
        model.train(); values=[]; parts=[]
        for step,batch in enumerate(loader(train,batch_size=int(cfg["training"]["eeg_batch_size"]),train=True)):
            if deadline_reached(args):break
            batch=move_batch(batch,device); predicted,tokens=model(batch["eeg"].float(),batch["channel_xyz"].float(),batch["channel_mask"],batch["time_mask"])
            loss,part=fit_loss(predicted,batch["mfcc"].float(),tokens,batch["label"],model.clip_logit_scale)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["training"]["grad_clip"]));optimizer.step()
            values.append(float(loss.detach()));parts.append(part)
            if deadline_reached(args) or (args.smoke_steps and step+1>=args.smoke_steps):break
        if not values:
            raise RuntimeError("v3 full-fit stage reached the shared deadline before one optimizer step")
        # Full-fit is intentionally selected on its own objective.  The
        # subject-holdout role is not touched until the full-fit gate passes.
        score=float(np.mean(values));record={"epoch":epoch+1,"train_loss":score,"selection_loss":score,"components":{key:float(np.mean([part[key] for part in parts])) for key in parts[0]},"elapsed_seconds":time.monotonic()-started}
        _append(history,record);print(f"[v3 fit] epoch={epoch+1} train={score:.5f} elapsed={record['elapsed_seconds']:.1f}s",flush=True)
        if score<best:
            best,stale=score,0;save_checkpoint(checkpoint_path(config_path,cfg,"fit"),schema="openvoice-v3-eeg-fit-v1",model=model,epoch=epoch,score=score,extra={"stage":"fit","fit_role":"fit","validation_role":"subject_holdout_seen","text_anchor":False,"loss":{"mfcc":.5,"delta":.2,"token_clip":.15,"global_clip":.15}})
        else: stale+=1
        if stale>=int(cfg["training"]["fit_patience"]):break
        if deadline_reached(args) or (args.wall_hours and time.monotonic()-started>=args.wall_hours*3600):break
    checkpoint=checkpoint_path(config_path,cfg,"fit")
    return {"elapsed_seconds":time.monotonic()-started,"epochs_completed":epoch+1,"checkpoint":str(checkpoint),"checkpoint_sha256":sha256_file(checkpoint),"best_loss":best}


def update_run_manifest(
    config_path: Path, cfg: dict[str, Any], args: argparse.Namespace, device: torch.device, phase_result: dict[str, Any]
) -> None:
    path=output_path(config_path,cfg,"run_manifest")
    if args.fresh and args.phase=="audio":
        payload={}
    elif path.is_file():
        payload=read_json(path)
    else:
        payload={}
    phases=dict(payload.get("training_phases",{}));phases[args.phase]=phase_result
    write_json(path,{"schema_version":"openvoice-v3-run-v3-cvae-denoise-training-review","prepared_lineage":capture_lineage(config_path,cfg),"audio_decoder":"fixed inverse-DCT baseline + conditional variational residual","requested_wall_hours":float(args.wall_hours or cfg["training"]["wall_hours"]),"shared_deadline_epoch":float(args.deadline_epoch),"training_phases":phases,"total_training_elapsed_seconds":sum(float(value.get("elapsed_seconds",0.0)) for value in phases.values()),"last_device":str(device)})


def main() -> None:
    args=parse();config_path,cfg=load_config(args.config);seed_everything(int(cfg["training"]["seed"]));device=default_device(args.device)
    records=load_prepared(output_path(config_path,cfg,"prepared_cache"))
    if args.phase=="audio":
        require_passed_gate(config_path,cfg,"v0_gate",lineage_artifact_keys=("vocoder_manifest",))
        result=train_audio(config_path,cfg,records,device,args)
    elif args.phase=="micro":
        require_passed_gate(config_path,cfg,"v0_gate",lineage_artifact_keys=("vocoder_manifest",))
        require_passed_gate(config_path,cfg,"v1_gate",lineage_artifact_keys=("audio_checkpoint","v0_gate"))
        require_passed_gate(config_path,cfg,"v2_gate",lineage_artifact_keys=("audio_checkpoint","v1_gate"))
        result=train_micro(config_path,cfg,records,device,args)
    else:
        require_passed_gate(config_path,cfg,"micro_gate",lineage_artifact_keys=("micro_checkpoint","v2_gate"))
        result=train_fit(config_path,cfg,records,device,args)
    update_run_manifest(config_path,cfg,args,device,result)


if __name__=="__main__": main()
