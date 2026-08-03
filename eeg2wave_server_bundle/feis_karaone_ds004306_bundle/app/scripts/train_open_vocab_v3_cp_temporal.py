#!/usr/bin/env python3
"""Train the staged CP-temporal-large v3 pipeline."""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_v3.cp_temporal import (
    PREPARATION_SCHEMA, SCHEMA, AudioCPEncoder, ContentMFCCDecoder,
    DeterministicAcousticBackbone, EEGCPEncoder, MelContentTeacher,
    ResidualCVAE, attention_regularization, envelope_correlation_loss,
    global_clip_loss, local_ot_clip_loss, masked_l1, parameter_count,
    reverse_gradient, soft_ssim_loss, temporal_cosine_loss,
    temporal_delta_loss,
)
from src.open_vocab_v3.data import V3Dataset, collate, load_prepared
from src.open_vocab_v3.runtime import (
    checkpoint_schema, default_device, load_config, move_batch, output_path,
    require_passed_gate, seed_everything, sha256_file,
)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("oracle", "prosody", "content", "cvae", "micro", "fit", "eeg_prosody"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deadline-epoch", type=float, default=0.0)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--explore", action="store_true")
    return parser.parse_args()


def expired(args: argparse.Namespace) -> bool:
    return bool(args.deadline_epoch and time.time() >= args.deadline_epoch)


def save_checkpoint(path: Path, schema: str, modules: dict[str, nn.Module], **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": schema, "modules": {name: module.state_dict() for name, module in modules.items()}, "extra": extra}, path)


def load_checkpoint(path: Path, schema: str, modules: dict[str, nn.Module], device: torch.device) -> dict[str, Any]:
    raw = torch.load(path, map_location=device, weights_only=False)
    if raw.get("schema_version") != schema:
        raise ValueError(f"stale CP-temporal checkpoint {path}: {raw.get('schema_version')!r}")
    for name, module in modules.items():
        if name not in raw.get("modules", {}):
            raise ValueError(f"checkpoint {path} lacks module {name}")
        module.load_state_dict(raw["modules"][name], strict=True)
    return raw


class TokenDataset(Dataset):
    def __init__(self, base: Dataset, cache: dict[str, np.ndarray], mapping: dict[int, int]):
        self.base, self.cache, self.mapping = base, cache, mapping

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int) -> dict[str, Any]:
        result = self.base[item]
        source = int(result["source_index"])
        if source not in self.mapping:
            raise RuntimeError(f"source {source} absent from fit-only frozen EnCodec cache")
        index = self.mapping[source]
        result["encodec_codes"] = self.cache["encodec_codes"][index]
        result["encodec_mask"] = self.cache["encodec_mask"][index]
        return result


def token_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    result = collate(items)
    result["encodec_codes"] = torch.as_tensor(np.stack([item["encodec_codes"] for item in items])).long()
    result["encodec_mask"] = torch.as_tensor(np.stack([item["encodec_mask"] for item in items])).bool()
    return result


def attach_codes(records, config_path, cfg) -> tuple[dict[str, np.ndarray], dict[int, int]]:
    path = output_path(config_path, cfg, "encodec_cache")
    raw = np.load(path, allow_pickle=False)
    if str(raw["schema"].item()) != SCHEMA:
        raise ValueError("stale non-CP EnCodec cache rejected")
    if str(raw["source_prepared_sha256"].item()) != sha256_file(output_path(config_path, cfg, "prepared_cache")):
        raise ValueError("EnCodec cache/prepared-cache lineage mismatch")
    cache = {name: np.asarray(raw[name]) for name in raw.files}
    return cache, {int(source): index for index, source in enumerate(cache["source_indices"].tolist())}


def make_modules(cfg: dict[str, Any], device: torch.device):
    model = cfg["model"]
    common = dict(
        dimension=int(model["content_dimension"]), heads=int(model["heads"]),
        stem_layers=int(model["content_stem_layers"]), branch_layers=int(model["content_branch_layers"]),
        token_steps=int(cfg["audio"]["content_tokens"]), acoustic_frames=int(cfg["audio"]["native_mel_frames"]),
        dropout=float(model["dropout"]), global_gradient_scale=float(model["global_gradient_scale"]),
        embedding_dimension=int(model["codebook_embedding_dimension"]),
    )
    audio = AudioCPEncoder(**common).to(device)
    decoder = ContentMFCCDecoder(
        dimension=int(model["content_dimension"]), heads=int(model["heads"]),
        layers=int(model["decoder_layers"]), token_steps=int(cfg["audio"]["content_tokens"]),
        frames=int(cfg["audio"]["native_mel_frames"]), dropout=float(model["dropout"]),
    ).to(device)
    eeg = EEGCPEncoder(
        dimension=int(model["eeg_dimension"]), heads=int(model["heads"]), layers=int(model["eeg_layers"]),
        token_steps=int(cfg["audio"]["content_tokens"]), acoustic_frames=int(cfg["audio"]["native_mel_frames"]),
        dropout=float(model["dropout"]),
    ).to(device)
    backbone = DeterministicAcousticBackbone(
        voice_dimension=int(cfg["speaker"]["embedding_dimension"]), dimension=int(model["audio_dimension"]),
        blocks=int(model["acoustic_blocks"]), include_p_plus=bool(model["include_p_plus"]), dropout=float(model["dropout"]),
    ).to(device)
    teacher = MelContentTeacher(dimension=int(model["audio_dimension"]), token_steps=int(cfg["audio"]["content_tokens"])).to(device)
    cvae = ResidualCVAE(
        backbone, dimension=int(model["audio_dimension"]), global_latent=int(model["audio_latent_dimension"]),
        local_latent=int(model["local_latent_dimension"]), local_steps=int(model["local_latent_steps"]),
        residual_limit=float(model["audio_residual_limit_log10"]),
    ).to(device)
    return audio, decoder, eeg, backbone, teacher, cvae


def _indices_dataset(records, indices: np.ndarray) -> Dataset:
    base = V3Dataset(records, ("fit",), eligible_only=True)
    position = {source: item for item, source in enumerate(base.indices)}
    return Subset(base, [position[int(index)] for index in indices])


def train_dev(records) -> tuple[Dataset, Dataset]:
    fit = (records.roles == "fit") & records.arrays["fit_eligible"].astype(bool)
    dev = fit & records.arrays["fit_internal_dev"].astype(bool)
    return _indices_dataset(records, np.flatnonzero(fit & ~dev)), _indices_dataset(records, np.flatnonzero(dev))


def micro_dataset(records, cfg) -> Dataset:
    base = V3Dataset(records, ("fit",), eligible_only=True)
    subject, per_label = str(cfg["micro_gate"]["subject"]), int(cfg["micro_gate"]["per_label"])
    groups: dict[str, list[int]] = {}
    for position, source in enumerate(base.indices):
        if str(records.arrays["subjects"][source]) == subject:
            groups.setdefault(str(records.arrays["labels"][source]), []).append(position)
    chosen = [position for label, positions in sorted(groups.items())
              for position in sorted(positions, key=lambda p: str(records.arrays["sample_keys"][base.indices[p]]))[:per_label]]
    if len(chosen) != 50:
        raise RuntimeError(f"CP-temporal micro set must contain 50 pairs, got {len(chosen)}")
    return Subset(base, chosen)


def loader(dataset: Dataset, cfg, *, train: bool, token: bool, eeg: bool = False) -> DataLoader:
    batch = int(cfg["training"]["eeg_batch_size" if eeg else "audio_batch_size"] if train else cfg["evaluation"]["batch_size"])
    collator = token_collate if token else collate
    sampler = None
    if train and eeg:
        pairs = [(str(dataset[index]["subject"]), str(dataset[index]["label"])) for index in range(len(dataset))]
        counts = {pair: pairs.count(pair) for pair in set(pairs)}
        sampler = WeightedRandomSampler(torch.as_tensor([1 / counts[pair] for pair in pairs], dtype=torch.double), len(pairs), replacement=True)
    return DataLoader(dataset, batch_size=batch, shuffle=train and sampler is None, sampler=sampler, collate_fn=collator, num_workers=0)


def labels_and_subjects(dataset: Dataset) -> tuple[list[str], list[str]]:
    labels = sorted({str(dataset[index]["label"]) for index in range(len(dataset))})
    subjects = sorted({str(dataset[index]["subject"]) for index in range(len(dataset))})
    return labels, subjects


def _label_target(names: list[str], mapping: dict[str, int], device) -> torch.Tensor:
    return torch.tensor([mapping[str(name)] for name in names], device=device)


def _text_anchor(names: list[str], dimension: int, device) -> torch.Tensor:
    rows = []
    for name in names:
        digest = hashlib.sha256(str(name).strip().lower().encode()).digest()
        value = torch.tensor(list(digest), dtype=torch.float32, device=device) / 127.5 - 1.0
        rows.append(value.repeat(math.ceil(dimension / len(value)))[:dimension])
    return F.normalize(torch.stack(rows), dim=-1)


def _teacher_loss(teacher: MelContentTeacher, mel: torch.Tensor, hubert: torch.Tensor,
                  label_target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    predicted, logits = teacher(mel)
    target = teacher.project_hubert(hubert)
    content = (1 - F.cosine_similarity(predicted, target.detach(), dim=-1)).mean()
    label = F.cross_entropy(logits, label_target)
    return content + 0.2 * label, {"teacher_content": float(content.detach()), "teacher_label": float(label.detach())}


def _oracle_loss(backbone, teacher, batch, *, use_plus: bool) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["speech_t5_mel"].float()
    generated = backbone(batch["content_mfcc"].float(), batch["p_base"].float(), batch["canonical_voice"].float(),
                         batch["p_plus"].float() if use_plus else None)
    mel = masked_l1(generated, target, batch["speech_t5_mel_mask"])
    delta = temporal_delta_loss(generated, target, batch["speech_t5_mel_mask"])
    temporal = temporal_cosine_loss(generated, target, batch["speech_t5_mel_mask"])
    with torch.no_grad():
        target_content = teacher.encode_mel(target)
    content = (1 - F.cosine_similarity(teacher.encode_mel(generated), target_content, dim=-1)).mean()
    envelope = envelope_correlation_loss(generated.mean(1), target.mean(1))
    ssim = soft_ssim_loss(generated, target)
    loss = .35 * mel + .15 * delta + .15 * temporal + .15 * content + .10 * envelope + .10 * ssim
    return loss, {"mel": float(mel.detach()), "delta": float(delta.detach()), "temporal": float(temporal.detach()),
                  "content": float(content.detach()), "envelope": float(envelope.detach()), "ssim": float(ssim.detach())}


def _run_dev_loss(model_loss, data_loader, device) -> float:
    values = []
    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch(batch, device); values.append(float(model_loss(batch)[0]))
    return float(np.mean(values)) if values else math.inf


def train_oracle(cp, cfg, records, device, args) -> None:
    if not args.explore:
        require_passed_gate(cp, cfg, "t0_gate")
    train_set, dev_set = train_dev(records); labels, _ = labels_and_subjects(train_set); label_map = {name: index for index, name in enumerate(labels)}
    _, _, _, backbone, teacher, _ = make_modules(cfg, device)
    teacher_opt = torch.optim.AdamW(teacher.parameters(), lr=float(cfg["training"]["audio_lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    teacher_epochs = min(20, int(cfg["training"]["oracle_epochs"]))
    for epoch in range(teacher_epochs):
        values = []
        for step, batch in enumerate(loader(train_set, cfg, train=True, token=False)):
            if expired(args):break
            batch = move_batch(batch, device); target = _label_target(batch["label"], label_map, device)
            loss, _ = _teacher_loss(teacher, batch["speech_t5_mel"].float(), batch["hubert"].float(), target)
            teacher_opt.zero_grad(set_to_none=True); loss.backward(); teacher_opt.step(); values.append(float(loss.detach()))
            if args.smoke_steps and step + 1 >= args.smoke_steps: break
        if not values:break
        print(f"[v3 CP teacher] epoch={epoch+1}/{teacher_epochs} loss={np.mean(values):.5f}", flush=True)
        if args.smoke_steps: break
    teacher.eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=float(cfg["training"]["audio_lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    best, stale, history = math.inf, 0, []
    for epoch in range(int(cfg["training"]["oracle_epochs"])):
        values = []
        backbone.train()
        for step, batch in enumerate(loader(train_set, cfg, train=True, token=False)):
            if expired(args): break
            batch = move_batch(batch, device); base_loss, base_parts = _oracle_loss(backbone, teacher, batch, use_plus=False); plus_loss, plus_parts = _oracle_loss(backbone, teacher, batch, use_plus=True); loss=.5*(base_loss+plus_loss);parts={f"base_{k}":v for k,v in base_parts.items()}|{f"plus_{k}":v for k,v in plus_parts.items()}
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(backbone.parameters(), float(cfg["training"]["grad_clip"])); optimizer.step(); values.append(float(loss.detach()))
            if args.smoke_steps and step + 1 >= args.smoke_steps: break
        if not values: break
        backbone.eval(); dev = _run_dev_loss(lambda b: (lambda x,y:(.5*(x[0]+y[0]),{}))(_oracle_loss(backbone,teacher,b,use_plus=False),_oracle_loss(backbone,teacher,b,use_plus=True)), loader(dev_set, cfg, train=False, token=False), device)
        history.append({"epoch": epoch + 1, "train": float(np.mean(values)), "dev": dev, "parts": parts})
        if dev < best:
            best, stale = dev, 0
            save_checkpoint(output_path(cp, cfg, "oracle_checkpoint"), checkpoint_schema(cfg, "oracle"), {"backbone": backbone, "teacher": teacher},
                            history=history, labels=labels, parameter_count=parameter_count(backbone, teacher), fit_internal_dev=True)
        else: stale += 1
        print(f"[v3 CP oracle] epoch={epoch+1} train={np.mean(values):.5f} dev={dev:.5f}", flush=True)
        if stale >= int(cfg["training"]["oracle_patience"]) or args.smoke_steps: break


def _prosody_loss(state, batch) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["p_base"].float();valid=batch["mfcc_mask"].to(target.dtype);denominator=valid.sum().clamp_min(1.0);activity=(F.binary_cross_entropy_with_logits(state.p_base[...,0],target[...,0],reduction="none")*valid).sum()/denominator
    probability = torch.sigmoid(state.p_base[..., 0]); intersection = (probability * target[..., 0]*valid).sum()
    dice = 1 - (2 * intersection + 1) / ((probability*valid).sum() + (target[...,0]*valid).sum() + 1)
    pred_energy=state.p_base[...,1];true_energy=target[...,1];pred_center=(pred_energy*valid).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1);true_center=(true_energy*valid).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1);left=(pred_energy-pred_center)*valid;right=(true_energy-true_center)*valid;corr=(left*right).sum(-1)/(left.square().sum(-1).sqrt()*right.square().sum(-1).sqrt()).clamp_min(1e-6)
    energy = (F.smooth_l1_loss(pred_energy,true_energy,reduction="none")*valid).sum()/denominator+(1-corr).mean()
    delta = (F.smooth_l1_loss(state.p_base[...,2],target[...,2],reduction="none")*valid).sum()/denominator
    duration = F.smooth_l1_loss(state.duration_fraction, batch["duration_fraction"].float())
    plus_target=batch["p_plus"].float();voicing=(F.binary_cross_entropy_with_logits(state.p_plus[...,0],plus_target[...,0],reduction="none")*valid).sum()/denominator;voiced=valid*plus_target[...,0];f0=(F.smooth_l1_loss(torch.sigmoid(state.p_plus[...,1]),plus_target[...,1],reduction="none")*voiced).sum()/voiced.sum().clamp_min(1);plus=voicing+f0
    loss = .35 * (activity + dice) + .35 * energy + .15 * delta + .15 * duration + .05 * plus
    return loss, {"activity": float(activity.detach()), "dice": float(dice.detach()), "energy": float(energy.detach()),
                  "delta": float(delta.detach()), "duration": float(duration.detach()), "p_plus": float(plus.detach())}


def train_prosody(cp, cfg, records, device, args) -> None:
    if not args.explore:
        require_passed_gate(cp, cfg, "oracle_gate", lineage_artifact_keys=("oracle_checkpoint",))
    cache, mapping = attach_codes(records, cp, cfg); train_set, dev_set = train_dev(records)
    train_set, dev_set = TokenDataset(train_set, cache, mapping), TokenDataset(dev_set, cache, mapping)
    audio, _, _, _, _, _ = make_modules(cfg, device); optimizer = torch.optim.AdamW(audio.parameters(), lr=float(cfg["training"]["audio_lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    best, stale, history = math.inf, 0, []
    for epoch in range(int(cfg["training"]["prosody_epochs"])):
        values=[]; audio.train()
        for step, batch in enumerate(loader(train_set, cfg, train=True, token=True)):
            if expired(args): break
            batch=move_batch(batch,device);loss,parts=_prosody_loss(audio(batch["encodec_codes"],batch["encodec_mask"]),batch)
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(audio.parameters(),float(cfg["training"]["grad_clip"]));optimizer.step();values.append(float(loss.detach()))
            if args.smoke_steps and step+1>=args.smoke_steps:break
        if not values:break
        audio.eval();dev=_run_dev_loss(lambda b:_prosody_loss(audio(b["encodec_codes"],b["encodec_mask"]),b),loader(dev_set,cfg,train=False,token=True),device)
        history.append({"epoch":epoch+1,"train":float(np.mean(values)),"dev":dev,"parts":parts})
        if dev<best:best,stale=dev,0;save_checkpoint(output_path(cp,cfg,"prosody_checkpoint"),checkpoint_schema(cfg,"prosody"),{"audio":audio},history=history,parameter_count=parameter_count(audio))
        else:stale+=1
        print(f"[v3 CP prosody] epoch={epoch+1} train={np.mean(values):.5f} dev={dev:.5f}",flush=True)
        if stale>=int(cfg["training"]["prosody_patience"]) or args.smoke_steps:break


def _content_encoder_loss(state,batch,teacher_projection,label_head,speaker_head,label_map,subject_map,scale):
    hubert=teacher_projection(batch["hubert"].float());hubert=F.interpolate(hubert.transpose(1,2),size=96,mode="linear",align_corners=False).transpose(1,2)
    hubert_mask=F.interpolate(batch["hubert_mask"].float().unsqueeze(1),size=96,mode="nearest").squeeze(1).bool();local_ot,scores=local_ot_clip_loss(state.local,hubert,scale,state.token_mask,hubert_mask);temporal_value=1-F.cosine_similarity(state.local[:,1:]-state.local[:,:-1],hubert[:,1:]-hubert[:,:-1],dim=-1);temporal_mask=state.token_mask[:,1:]&state.token_mask[:,:-1]&hubert_mask[:,1:]&hubert_mask[:,:-1];temporal=(temporal_value*temporal_mask).sum()/temporal_mask.sum().clamp_min(1)
    labels=_label_target(batch["label"],label_map,state.local.device);subjects=_label_target(batch["subject"],subject_map,state.local.device)
    label_probe=F.cross_entropy(label_head(state.global_embedding),labels);speaker=F.cross_entropy(speaker_head(reverse_gradient(state.global_embedding)),subjects)
    teacher_global=F.normalize(hubert.mean(1),dim=-1);global_clip=global_clip_loss(state.global_embedding,teacher_global,batch["label"],scale)
    label_supcon=global_clip_loss(state.global_embedding,state.global_embedding.detach(),batch["label"],scale);hubert_global=(1-F.cosine_similarity(state.global_embedding,teacher_global.detach(),dim=-1)).mean()
    local_std=state.local.reshape(-1,state.local.shape[-1]).std(0);variance=F.relu(1.0-local_std).mean()
    centered=state.local.reshape(-1,state.local.shape[-1])-state.local.reshape(-1,state.local.shape[-1]).mean(0);covariance=(centered.T@centered)/max(len(centered)-1,1);covariance=(covariance-torch.diag_embed(torch.diagonal(covariance))).square().mean()
    local_loss=.60*local_ot+.15*temporal+.15*variance+.10*covariance
    global_loss=.40*global_clip+.25*label_supcon+.20*hubert_global+.15*speaker
    loss=.60*local_loss+.40*global_loss
    return loss,{"local_ot":float(local_ot.detach()),"temporal":float(temporal.detach()),"variance":float(variance.detach()),"covariance":float(covariance.detach()),"global_clip":float(global_clip.detach()),"label_supcon":float(label_supcon.detach()),"label_probe_unoptimized":float(label_probe.detach()),"hubert_global":float(hubert_global.detach()),"speaker":float(speaker.detach()),"paired_score":float(scores.diag().mean().detach())}


def _decoder_loss(state,decoder,batch):
    # T1C isolates content: ground-truth P/duration condition the shared MFCC
    # decoder here. Predicted P is introduced only by the four-way Stage 4.
    content,full,diag=decoder(state.local,state.token_mask,batch["p_base"].float(),batch["duration_fraction"].float());target=batch["content_mfcc"].float()
    mask=batch["mfcc_mask"];l1=masked_l1(content,target,mask);delta=temporal_delta_loss(content,target,mask);temporal=temporal_cosine_loss(content,target,mask);c0=masked_l1(full[:,0:1],batch["mfcc_c0"].float().unsqueeze(1),mask);attention=attention_regularization(diag)
    variance=F.relu(target.std(0).mean()*.5-content.std(0).mean())
    pred_flat=content.transpose(1,2).reshape(-1,content.shape[1]);target_flat=target.transpose(1,2).reshape(-1,target.shape[1]);pred_flat=pred_flat-pred_flat.mean(0);target_flat=target_flat-target_flat.mean(0);pred_cov=pred_flat.T@pred_flat/max(len(pred_flat)-1,1);target_cov=target_flat.T@target_flat/max(len(target_flat)-1,1);covariance=F.smooth_l1_loss(pred_cov,target_cov)
    loss=.40*l1+.20*delta+.15*temporal+.10*c0+.075*variance+.025*covariance+.05*attention
    return loss,{"mfcc":float(l1.detach()),"delta":float(delta.detach()),"temporal":float(temporal.detach()),"c0":float(c0.detach()),"variance":float(variance.detach()),"covariance":float(covariance.detach()),"attention":float(attention.detach())}


def train_content(cp,cfg,records,device,args):
    if not args.explore:require_passed_gate(cp,cfg,"prosody_gate",lineage_artifact_keys=("prosody_checkpoint",))
    cache,mapping=attach_codes(records,cp,cfg);train_set,dev_set=train_dev(records);train_set,dev_set=TokenDataset(train_set,cache,mapping),TokenDataset(dev_set,cache,mapping)
    labels,subjects=labels_and_subjects(train_set);label_map={x:i for i,x in enumerate(labels)};subject_map={x:i for i,x in enumerate(subjects)}
    audio,decoder,_,_,_,_=make_modules(cfg,device);load_checkpoint(output_path(cp,cfg,"prosody_checkpoint"),checkpoint_schema(cfg,"prosody"),{"audio":audio},device)
    projection=nn.Linear(768,int(cfg["model"]["content_dimension"])).to(device);label_head=nn.Linear(int(cfg["model"]["content_dimension"]),len(labels)).to(device);speaker_head=nn.Linear(int(cfg["model"]["content_dimension"]),len(subjects)).to(device);scale=nn.Parameter(torch.tensor(math.log(1/.07),device=device))
    history=[]
    def run_stage(name,epochs,parameters,loss_function,lr_scale=1.0):
        optimizer=torch.optim.AdamW(parameters,lr=float(cfg["training"]["audio_lr"])*lr_scale,weight_decay=float(cfg["training"]["weight_decay"]));best=math.inf;stale=0
        for epoch in range(int(epochs)):
            values=[];audio.train(name!="decoder");decoder.train(name!="encoder")
            for step,batch in enumerate(loader(train_set,cfg,train=True,token=True)):
                if expired(args):break
                batch=move_batch(batch,device);state=audio(batch["encodec_codes"],batch["encodec_mask"]);loss,parts=loss_function(state,batch)
                # The shared stem is updated in encoder/joint stages. Preserve
                # the already-gated P solution while optimizing C.
                if name!="decoder":loss=loss+.10*_prosody_loss(state,batch)[0]
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(parameters,float(cfg["training"]["grad_clip"]));optimizer.step();values.append(float(loss.detach()))
                if args.smoke_steps and step+1>=args.smoke_steps:break
            if not values:break
            audio.eval();decoder.eval();dev_values=[]
            with torch.no_grad():
                for dev_batch in loader(dev_set,cfg,train=False,token=True):
                    dev_batch=move_batch(dev_batch,device);dev_state=audio(dev_batch["encodec_codes"],dev_batch["encodec_mask"]);dev_loss,_=loss_function(dev_state,dev_batch)
                    if name!="decoder":dev_loss=dev_loss+.10*_prosody_loss(dev_state,dev_batch)[0]
                    dev_values.append(float(dev_loss))
            dev=float(np.mean(dev_values)) if dev_values else math.inf
            history.append({"stage":name,"epoch":epoch+1,"train":float(np.mean(values)),"dev":dev,"parts":parts});print(f"[v3 CP content {name}] epoch={epoch+1}/{epochs} train={np.mean(values):.5f} dev={dev:.5f}",flush=True)
            if dev<best:best,stale=dev,0;save_checkpoint(output_path(cp,cfg,"content_checkpoint"),checkpoint_schema(cfg,"content"),{"audio":audio,"decoder":decoder,"teacher_projection":projection,"label_head":label_head,"speaker_head":speaker_head},history=history,labels=labels,subjects=subjects,logit_scale=float(scale.detach()),parameter_count=parameter_count(audio,decoder,projection,label_head,speaker_head),fit_internal_dev=True)
            else:stale+=1
            if stale>=int(cfg["training"]["content_patience"]) or args.smoke_steps:break
        best_path=output_path(cp,cfg,"content_checkpoint")
        if best_path.is_file():
            best_raw=load_checkpoint(best_path,checkpoint_schema(cfg,"content"),{"audio":audio,"decoder":decoder,"teacher_projection":projection,"label_head":label_head,"speaker_head":speaker_head},device)
            scale.data.fill_(float(best_raw.get("extra",{}).get("logit_scale",float(scale.detach()))))
    def encoder_with_mfcc(state,batch):
        content_parts=_content_encoder_loss(state,batch,projection,label_head,speaker_head,label_map,subject_map,scale);decoder_parts=_decoder_loss(state,decoder,batch)
        return .75*content_parts[0]+.25*decoder_parts[0],content_parts[1]|{f"mfcc_{key}":value for key,value in decoder_parts[1].items()}
    encoder_parameters=list(audio.parameters())+list(decoder.parameters())+list(projection.parameters())+list(label_head.parameters())+list(speaker_head.parameters())+[scale]
    run_stage("encoder",cfg["training"]["content_encoder_epochs"],encoder_parameters,encoder_with_mfcc)
    for p in audio.parameters():p.requires_grad_(False)
    run_stage("decoder",cfg["training"]["content_decoder_epochs"],list(decoder.parameters()),lambda state,batch:_decoder_loss(state,decoder,batch))
    for p in audio.parameters():p.requires_grad_(True)
    joint=list(audio.parameters())+list(decoder.parameters())+list(projection.parameters())+list(label_head.parameters())+list(speaker_head.parameters())+[scale]
    run_stage("joint",cfg["training"]["content_joint_epochs"],joint,lambda state,batch:(lambda a,b:(a[0]+b[0],a[1]|{f"decoder_{k}":v for k,v in b[1].items()}))(_content_encoder_loss(state,batch,projection,label_head,speaker_head,label_map,subject_map,scale),_decoder_loss(state,decoder,batch)),float(cfg["training"]["joint_lr_scale"]))


def _kl(mean,logvar):return .5*(mean.square()+logvar.exp()-1-logvar).mean()


def train_cvae(cp,cfg,records,device,args):
    if not args.explore:require_passed_gate(cp,cfg,"intervention_gate",lineage_artifact_keys=("content_checkpoint","oracle_checkpoint"))
    train_set,dev_set=train_dev(records);_,_,_,backbone,teacher,cvae=make_modules(cfg,device);load_checkpoint(output_path(cp,cfg,"oracle_checkpoint"),checkpoint_schema(cfg,"oracle"),{"backbone":backbone,"teacher":teacher},device);cvae.backbone=backbone
    teacher.eval();[p.requires_grad_(False) for p in teacher.parameters()]
    base_lr=float(cfg["training"]["cvae_lr"]);backbone_ids={id(p) for p in cvae.backbone.parameters()};residual_parameters=[p for p in cvae.parameters() if id(p) not in backbone_ids]
    optimizer=torch.optim.AdamW([
        {"params":residual_parameters,"lr":base_lr},
        {"params":list(cvae.backbone.parameters()),"lr":base_lr*float(cfg["training"]["joint_lr_scale"])},
    ],weight_decay=float(cfg["training"]["weight_decay"]));best=math.inf;stale=0;history=[]
    def cvae_loss(batch, *, stochastic):
        target=batch["speech_t5_mel"].float();post=cvae(batch["content_mfcc"].float(),batch["p_base"].float(),batch["canonical_voice"].float(),None,target=target,stochastic=stochastic);prior=cvae(batch["content_mfcc"].float(),batch["p_base"].float(),batch["canonical_voice"].float(),None,target=None,stochastic=False)
        post_l1=masked_l1(post["mel"],target,batch["speech_t5_mel_mask"]);prior_l1=masked_l1(prior["mel"],target,batch["speech_t5_mel_mask"]);kl=_kl(post["posterior_global_mean"],post["posterior_global_logvar"])+_kl(post["posterior_local_mean"],post["posterior_local_logvar"]);content=(1-F.cosine_similarity(teacher.encode_mel(prior["mel"]),teacher.encode_mel(target).detach(),dim=-1)).mean();budget=F.relu(prior["residual"].square().mean().sqrt()-.30*prior["deterministic"].square().mean().sqrt());inactive=(prior["residual"].abs().mean(1)*(1-batch["p_base"][...,0].float())).mean();loss=.45*post_l1+.20*prior_l1+.15*content+.05*current_beta*kl+.10*budget+.05*inactive
        return loss,{"posterior":float(post_l1.detach()),"prior":float(prior_l1.detach()),"content":float(content.detach()),"kl":float(kl.detach()),"beta":current_beta,"budget":float(budget.detach()),"inactive":float(inactive.detach())}
    current_beta=0.0
    for epoch in range(int(cfg["training"]["cvae_epochs"])):
        frozen=epoch<int(cfg["training"]["cvae_backbone_frozen_epochs"]);[p.requires_grad_(not frozen) for p in cvae.backbone.parameters()];values=[]
        current_beta=float(cfg["training"]["cvae_kl_beta_max"])*min(1.0,((epoch%int(cfg["training"]["cvae_kl_cycle_epochs"]))+1)/(int(cfg["training"]["cvae_kl_cycle_epochs"])*.5))
        for step,batch in enumerate(loader(train_set,cfg,train=True,token=False)):
            if expired(args):break
            batch=move_batch(batch,device);loss,parts=cvae_loss(batch,stochastic=True)
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(cvae.parameters(),float(cfg["training"]["grad_clip"]));optimizer.step();values.append(float(loss.detach()))
            if args.smoke_steps and step+1>=args.smoke_steps:break
        if not values:break
        cvae.eval();dev_values=[]
        with torch.no_grad():
            for dev_batch in loader(dev_set,cfg,train=False,token=False):
                dev_batch=move_batch(dev_batch,device);dev_loss,_=cvae_loss(dev_batch,stochastic=False);dev_values.append(float(dev_loss))
        val=float(np.mean(dev_values)) if dev_values else math.inf;history.append({"epoch":epoch+1,"train":float(np.mean(values)),"dev":val,"parts":parts})
        if val<best:best,stale=val,0;save_checkpoint(output_path(cp,cfg,"cvae_checkpoint"),checkpoint_schema(cfg,"cvae"),{"cvae":cvae,"teacher":teacher},history=history,parameter_count=parameter_count(cvae,teacher),residual_limit=float(cfg["model"]["audio_residual_limit_log10"]),fit_internal_dev=True)
        else:stale+=1
        print(f"[v3 CP CVAE] epoch={epoch+1} train={np.mean(values):.5f} dev={val:.5f} beta={current_beta:.6f}",flush=True)
        if stale>=int(cfg["training"]["cvae_patience"]) or args.smoke_steps:break


def _augment_eeg(eeg,channel_mask,cfg):
    if float(cfg["training"]["channel_dropout"])>0:
        keep=(torch.rand_like(channel_mask.float())>=float(cfg["training"]["channel_dropout"]))|~channel_mask;eeg=eeg*keep.unsqueeze(-1)
    eeg=eeg*(1+float(cfg["training"]["amplitude_jitter"])*(2*torch.rand(len(eeg),1,1,device=eeg.device)-1));shift=int(cfg["training"]["time_shift_samples"])
    if shift:eeg=torch.stack([torch.roll(row,int(torch.randint(-shift,shift+1,(1,),device=eeg.device)),dims=-1) for row in eeg])
    return eeg


def train_eeg(cp,cfg,records,device,args,phase):
    prerequisite="cvae_gate" if phase=="micro" else "micro_gate"
    if not args.explore:
        require_passed_gate(cp,cfg,prerequisite,lineage_artifact_keys=(("cvae_checkpoint",) if phase=="micro" else ("micro_checkpoint",)))
    cache,mapping=attach_codes(records,cp,cfg);train_base,dev_base=(micro_dataset(records,cfg),micro_dataset(records,cfg)) if phase=="micro" else train_dev(records);train_set,dev_set=TokenDataset(train_base,cache,mapping),TokenDataset(dev_base,cache,mapping)
    audio,decoder,eeg,_,_,_=make_modules(cfg,device);raw=load_checkpoint(output_path(cp,cfg,"content_checkpoint"),checkpoint_schema(cfg,"content"),{"audio":audio,"decoder":decoder},device);audio.eval();decoder.eval();[p.requires_grad_(False) for p in list(audio.parameters())+list(decoder.parameters())]
    lr=float(cfg["training"]["eeg_micro_lr" if phase=="micro" else "eeg_fit_lr"]);optimizer=torch.optim.AdamW(eeg.parameters(),lr=lr,weight_decay=float(cfg["training"]["weight_decay"]));epochs=int(cfg["training"]["micro_epochs" if phase=="micro" else "fit_epochs"]);patience=int(cfg["training"]["micro_patience" if phase=="micro" else "fit_patience"]);best=math.inf;stale=0;history=[];accum=int(cfg["training"]["gradient_accumulation"])
    train_loader=loader(train_set,cfg,train=True,token=True,eeg=True);updates_per_epoch=max(1,math.ceil(len(train_loader)/accum));total_updates=max(1,epochs*updates_per_epoch);warmup=max(1,int(total_updates*float(cfg["training"]["warmup_fraction"])))
    def schedule(step):
        if phase=="micro":return 1.0
        if step<warmup:return max((step+1)/warmup,1e-3)
        progress=min(1.0,(step-warmup)/max(total_updates-warmup,1));return .5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule);update_count=0
    scale=eeg.clip_logit_scale
    def eeg_loss(batch, *, augment):
        signal=_augment_eeg(batch["eeg"].float(),batch["channel_mask"],cfg) if augment else batch["eeg"].float();pred=eeg(signal,batch["channel_xyz"].float(),batch["channel_mask"],batch["time_mask"])
        with torch.no_grad():target=audio(batch["encodec_codes"],batch["encodec_mask"])
        pred_mfcc,_,_=decoder(pred.local,pred.token_mask,batch["canonical_p_base"].float(),batch["canonical_duration_fraction"].float());target_mfcc=batch["eeg_content_mfcc"].float();mask=batch["canonical_content_mask"];l1=masked_l1(pred_mfcc,target_mfcc,mask);delta=temporal_delta_loss(pred_mfcc,target_mfcc,mask);local,_=local_ot_clip_loss(pred.local,target.local,scale,pred.token_mask,target.token_mask);global_=global_clip_loss(pred.global_embedding,target.global_embedding,batch["label"],scale);text=pred.global_embedding.new_zeros(()) if phase=="micro" else (1-(pred.global_embedding*_text_anchor(batch["label"],pred.global_embedding.shape[-1],device)).sum(-1)).mean()
        loss=(.45*l1+.20*delta+.20*local+.15*global_) if phase=="micro" else (.40*l1+.20*delta+.20*local+.15*global_+.05*text)
        return loss,{"mfcc":float(l1.detach()),"delta":float(delta.detach()),"local_ot":float(local.detach()),"global_clip":float(global_.detach()),"text":float(text.detach())}
    for epoch in range(epochs):
        values=[];optimizer.zero_grad(set_to_none=True)
        batches_seen=0
        for step,batch in enumerate(train_loader):
            if expired(args):break
            batch=move_batch(batch,device);loss,parts=eeg_loss(batch,augment=phase!="micro")
            (loss/accum).backward();
            batches_seen+=1
            if batches_seen%accum==0:nn.utils.clip_grad_norm_(eeg.parameters(),float(cfg["training"]["grad_clip"]));optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);update_count+=1
            values.append(float(loss.detach()))
            if args.smoke_steps and step+1>=args.smoke_steps:break
        if not values:break
        if batches_seen%accum:
            remainder=batches_seen%accum
            for parameter in eeg.parameters():
                if parameter.grad is not None:parameter.grad.mul_(accum/remainder)
            nn.utils.clip_grad_norm_(eeg.parameters(),float(cfg["training"]["grad_clip"]));optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);update_count+=1
        eeg.eval();dev_values=[]
        with torch.no_grad():
            for dev_batch in loader(dev_set,cfg,train=False,token=True,eeg=True):
                dev_batch=move_batch(dev_batch,device);dev_loss,_=eeg_loss(dev_batch,augment=False);dev_values.append(float(dev_loss))
        val=float(np.mean(dev_values)) if dev_values else math.inf;history.append({"epoch":epoch+1,"train":float(np.mean(values)),"dev":val,"learning_rate":float(optimizer.param_groups[0]["lr"]),"optimizer_updates":update_count,"parts":parts})
        if val<best:best,stale=val,0;save_checkpoint(output_path(cp,cfg,f"{phase}_checkpoint"),checkpoint_schema(cfg,phase),{"eeg":eeg},history=history,audio_checkpoint_sha256=sha256_file(output_path(cp,cfg,"content_checkpoint")),parameter_count=parameter_count(eeg),primary="thinking_eeg_C_plus_canonical_P")
        else:stale+=1
        print(f"[v3 CP EEG {phase}] epoch={epoch+1}/{epochs} train={np.mean(values):.5f} dev={val:.5f} lr={optimizer.param_groups[0]['lr']:.2e}",flush=True)
        if stale>=patience or args.smoke_steps:break


def train_eeg_prosody(cp,cfg,records,device,args):
    if not args.explore:require_passed_gate(cp,cfg,"fit_gate",lineage_artifact_keys=("fit_checkpoint",))
    train_set,_=train_dev(records);audio,decoder,eeg,_,_,_=make_modules(cfg,device);load_checkpoint(output_path(cp,cfg,"fit_checkpoint"),checkpoint_schema(cfg,"fit"),{"eeg":eeg},device)
    for p in eeg.parameters():p.requires_grad_(False)
    for p in list(eeg.p_decoder.parameters())+list(eeg.duration.parameters()):p.requires_grad_(True)
    parameters=list(eeg.p_decoder.parameters())+list(eeg.duration.parameters());optimizer=torch.optim.AdamW(parameters,lr=float(cfg["training"]["eeg_prosody_lr"]));history=[]
    for epoch in range(int(cfg["training"]["eeg_prosody_epochs"])):
        values=[]
        for step,batch in enumerate(loader(train_set,cfg,train=True,token=False,eeg=True)):
            if expired(args):break
            batch=move_batch(batch,device);state=eeg(batch["eeg"].float(),batch["channel_xyz"].float(),batch["channel_mask"],batch["time_mask"]);loss,parts=_prosody_loss(state,batch);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();values.append(float(loss.detach()))
            if args.smoke_steps and step+1>=args.smoke_steps:break
        if not values:break
        history.append({"epoch":epoch+1,"loss":float(np.mean(values)),"parts":parts});save_checkpoint(output_path(cp,cfg,"eeg_prosody_checkpoint"),checkpoint_schema(cfg,"eeg_prosody"),{"eeg":eeg},history=history,exploratory_only=True,phase_metadata_available=False)
        print(f"[v3 CP EEG-P exploratory] epoch={epoch+1} loss={np.mean(values):.5f}",flush=True)
        if args.smoke_steps:break


def main():
    args=parse();cp,cfg=load_config(args.config);seed_everything(int(cfg["training"]["seed"]));records=load_prepared(output_path(cp,cfg,"prepared_cache"),expected_schema=PREPARATION_SCHEMA);device=default_device(args.device)
    if args.fresh:
        key={"oracle":"oracle_checkpoint","prosody":"prosody_checkpoint","content":"content_checkpoint","cvae":"cvae_checkpoint","micro":"micro_checkpoint","fit":"fit_checkpoint","eeg_prosody":"eeg_prosody_checkpoint"}[args.phase];path=output_path(cp,cfg,key)
        if path.is_file():
            archived=path.with_name(f"{path.stem}.before_fresh_{int(time.time())}_{os.getpid()}{path.suffix}");path.rename(archived);print(f"[v3 CP] archived prior same-schema checkpoint: {archived}",flush=True)
    if args.phase=="oracle":train_oracle(cp,cfg,records,device,args)
    elif args.phase=="prosody":train_prosody(cp,cfg,records,device,args)
    elif args.phase=="content":train_content(cp,cfg,records,device,args)
    elif args.phase=="cvae":train_cvae(cp,cfg,records,device,args)
    elif args.phase in {"micro","fit"}:train_eeg(cp,cfg,records,device,args,args.phase)
    else:train_eeg_prosody(cp,cfg,records,device,args)


if __name__=="__main__":main()
