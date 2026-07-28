#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path: sys.path.insert(0,str(APP))

from src.open_vocab_0728.data import CacheV3, DualLatentDataset, balanced_indices, collate, load_context, normalize_label
from src.open_vocab_0728.lineage import build_lineage, checkpoint_payload, validate_checkpoint
from src.open_vocab_0728.losses import channel_consistency, foreground_msssim, masked_pool, multi_positive_clip, sequence_soft_dtw, soft_iou
from src.open_vocab_0728.metrics import STSS, load_stss, ms_ssim
from src.open_vocab_0728.model import DualLatentAudioModel, DualLatentEEGToSpeech, EEGEncoder
from src.open_vocab_0728.runtime import default_device, load_config, move_batch, resolve_config_path, seed_everything, write_json


def args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description="Train independent v0728 dual-latent models")
    parser.add_argument("--config",type=Path,required=True); parser.add_argument("--phase",required=True,choices=("audio","semantic4","dual4","full11","validate")); parser.add_argument("--device",default=None); parser.add_argument("--seed",type=int,default=None); parser.add_argument("--epochs",type=int,default=None); parser.add_argument("--resume",type=Path); parser.add_argument("--smoke-steps",type=int,default=0); parser.add_argument("--loso-subject"); parser.add_argument("--strict-audio-loso",action="store_true"); return parser.parse_args()


def positive_matrix(batch: dict[str,Any], *, same_subject: bool = False) -> torch.Tensor:
    labels=[normalize_label(v) for v in batch["label"]]; subjects=batch["subject"]
    return torch.tensor([[a==b and (not same_subject or subjects[i]==subjects[j]) for j,b in enumerate(labels)] for i,a in enumerate(labels)],device=batch["mel"].device)


def loader(dataset: DualLatentDataset, *, batch_size: int, seed: int, train: bool) -> DataLoader:
    if train:
        order=balanced_indices(dataset,seed); sampler=torch.utils.data.SubsetRandomSampler(order)
        return DataLoader(dataset,batch_size=batch_size,sampler=sampler,collate_fn=collate,num_workers=0)
    return DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=collate,num_workers=0)


def paths(config_path:Path,cfg:dict[str,Any],phase:str,run_id:str|None=None) -> tuple[Path,Path,Path]:
    root=resolve_config_path(config_path,cfg["paths"]["output_root"])
    if run_id: root=root/"runs"/run_id
    name={"audio":"audio","semantic4":"eeg_semantic4","dual4":"eeg_dual4","full11":"eeg_full11"}[phase]
    return root/name/"checkpoints"/"latest.pt",root/name/"checkpoints"/"best.pt",root/name/"metrics"/"training.jsonl"


def save_checkpoint(path:Path,model:torch.nn.Module,optimizer:torch.optim.Optimizer,epoch:int,lineage:Any,extra:dict[str,Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); torch.save(checkpoint_payload(state_dict=model.state_dict(),epoch=epoch,lineage=lineage,extra={**extra,"optimizer":optimizer.state_dict()}),path)


def restore(path:Path,model:torch.nn.Module,optimizer:torch.optim.Optimizer|None,lineage:Any) -> tuple[int,float]:
    raw=torch.load(path,map_location="cpu",weights_only=False); validate_checkpoint(raw,lineage); model.load_state_dict(raw["state_dict"])
    if optimizer and raw.get("extra",{}).get("optimizer"): optimizer.load_state_dict(raw["extra"]["optimizer"])
    return int(raw["epoch"])+1,float(raw.get("extra",{}).get("best",math.inf))


def audio_loss(model:DualLatentAudioModel,batch:dict[str,Any],cfg:dict[str,Any],*,warmup:bool) -> tuple[torch.Tensor,dict[str,float]]:
    state=model(batch["hubert"],batch["hubert_mask"],batch["mel"],batch["activity"])
    content=multi_positive_clip(masked_pool(state.linguistic_latent,state.linguistic_mask),masked_pool(state.linguistic_latent,state.linguistic_mask),positive_matrix(batch),temperature=float(cfg["loss"]["contrastive_temperature"]))
    coarse=F.smooth_l1_loss(state.coarse_log_mel,batch["mel"]); activity=F.binary_cross_entropy_with_logits(state.coarse_activity_logits,batch["activity"].float())
    if warmup: return coarse+float(cfg["loss"]["audio_content_multipositive"])*content+float(cfg["loss"]["audio_activity"])*activity,{"loss":float((coarse+activity).detach()),"coarse":float(coarse.detach()),"content":float(content.detach())}
    mel=F.smooth_l1_loss(state.log_mel,batch["mel"]); structure=1-foreground_msssim(state.log_mel,batch["mel"]).mean(); full_activity=F.binary_cross_entropy_with_logits(state.activity_logits,batch["activity"].float())
    recoded,_=model.residual(state.log_mel-state.coarse_log_mel.detach(),batch["activity"]); cycle=F.smooth_l1_loss(recoded,state.realization_latent.detach())+1-F.cosine_similarity(recoded,state.realization_latent.detach(),dim=-1).mean()
    total=float(cfg["loss"]["audio_full_mel"])*mel+float(cfg["loss"]["audio_foreground_msssim"])*structure+float(cfg["loss"]["audio_activity"])*full_activity+float(cfg["loss"]["audio_residual_cycle"])*cycle
    return total,{"loss":float(total.detach()),"mel":float(mel.detach()),"structure":float(structure.detach()),"cycle":float(cycle.detach())}


@torch.no_grad()
def audio_gate(model:DualLatentAudioModel,data:DataLoader,stss:STSS,cfg:dict[str,Any],device:torch.device,train: DualLatentDataset) -> dict[str,Any]:
    maes=[]; ssim=[]; scores=[]; generations=[]; baseline_scores=[]; gains=[]; wins=[]
    grouped:dict[str,list[np.ndarray]]={}
    for source in train.indices:
        label=normalize_label(str(train.cache.raw["labels"][source])); grouped.setdefault(label,[]).append(np.asarray(train.cache.raw["mel"][source]))
    medians={label:np.median(values,axis=0) for label,values in grouped.items()}
    for batch in tqdm(data,desc="[0728 audio gate]",unit="batch",mininterval=1.0,disable=False):
        batch=move_batch(batch,device); state=model(batch["hubert"],batch["hubert_mask"],batch["mel"],batch["activity"])
        for index,label in enumerate(batch["label"]):
            pred=state.log_mel[index].cpu().numpy(); ref=batch["mel"][index].cpu().numpy(); maes.append(float(np.abs(pred-ref).mean())); ssim.append(ms_ssim(pred,ref)); scores.append(stss.score(pred,ref)); generations.append(pred)
            baseline=stss.score(medians[normalize_label(label)],ref); baseline_scores.append(baseline); gains.append(scores[-1]-baseline); wins.append(scores[-1]>baseline)
    variance=float(np.var(generations)); reference=float(np.var([value for values in grouped.values() for value in values])); ratio=variance/max(reference,1e-8)
    report={"median_log_mel_mae":float(np.median(maes)),"median_msssim":float(np.median(ssim)),"median_stss":float(np.median(scores)),"median_label_median_stss":float(np.median(baseline_scores)),"median_stss_gain_over_label_median":float(np.median(gains)),"trial_win_rate_over_label_median":float(np.mean(wins)),"generated_reference_variance_ratio":ratio,"passed":bool(np.median(maes)<=float(cfg["evaluation"]["audio_mel_mae_max_db"]) and np.median(ssim)>=float(cfg["evaluation"]["audio_msssim_minimum"]) and np.median(gains)>=float(cfg["evaluation"]["audio_stss_gain_over_label_median"]) and np.mean(wins)>=float(cfg["evaluation"]["audio_trial_win_rate"]) and float(cfg["evaluation"]["audio_variance_ratio_min"])<=ratio<=float(cfg["evaluation"]["audio_variance_ratio_max"]))}
    return report


def eeg_loss(generator:DualLatentEEGToSpeech,batch:dict[str,Any],cfg:dict[str,Any],*,phase:str) -> tuple[torch.Tensor,dict[str,float]]:
    audio=generator.audio_model
    with torch.no_grad():
        target=audio(batch["hubert"],batch["hubert_mask"],batch["mel"],batch["activity"])
    state=generator.encode(batch["eeg"],batch["channel_xyz"],batch["channel_mask"],batch["time_mask"])
    pos=positive_matrix(batch)
    semantic=multi_positive_clip(masked_pool(state.linguistic_latent,state.linguistic_mask),masked_pool(target.linguistic_latent,target.linguistic_mask),pos,temperature=float(cfg["loss"]["contrastive_temperature"]))
    seq=sequence_soft_dtw(state.linguistic_latent,target.linguistic_latent,gamma=float(cfg["loss"]["soft_dtw_gamma"]),band_fraction=float(cfg["loss"]["soft_dtw_band_fraction"]))
    coarse,coarse_activity=audio.decode_content(state.linguistic_latent)
    total=float(cfg["loss"]["semantic_contrastive"])*semantic+float(cfg["loss"]["semantic_sequence_soft_dtw"])*seq+float(cfg["loss"]["semantic_coarse_mel"])*F.smooth_l1_loss(coarse,batch["mel"])+float(cfg["loss"]["eeg_activity"])*F.binary_cross_entropy_with_logits(coarse_activity,batch["activity"].float())
    zero=generator.encode(torch.zeros_like(batch["eeg"]),batch["channel_xyz"],batch["channel_mask"],batch["time_mask"]).evidence_probability
    noise=generator.encode(torch.randn_like(batch["eeg"])*float(cfg["training"]["signal_noise_std"]),batch["channel_xyz"],batch["channel_mask"],batch["time_mask"]).evidence_probability
    evidence=F.binary_cross_entropy(state.evidence_probability,torch.ones_like(state.evidence_probability))+0.5*(F.binary_cross_entropy(zero,torch.zeros_like(zero))+F.binary_cross_entropy(noise,torch.zeros_like(noise)))
    total=total+float(cfg["loss"]["evidence"])*evidence
    if phase!="semantic4":
        realization=multi_positive_clip(masked_pool(state.realization_latent,state.realization_mask),masked_pool(target.realization_latent,target.realization_mask),torch.eye(len(batch["label"]),device=batch["eeg"].device,dtype=torch.bool),temperature=float(cfg["loss"]["contrastive_temperature"]))
        same=multi_positive_clip(masked_pool(state.realization_latent,state.realization_mask),masked_pool(target.realization_latent,target.realization_mask),positive_matrix(batch,same_subject=True),temperature=float(cfg["loss"]["contrastive_temperature"]))
        pred,act=audio.decode(state.linguistic_latent,state.realization_latent)
        structure=(1-foreground_msssim(pred,batch["mel"]).mean())+(1-soft_iou(pred,batch["mel"]).mean())
        total=total+float(cfg["loss"]["weak_paired_realization"])*realization+float(cfg["loss"]["same_subject_label_realization"])*same+float(cfg["loss"]["paired_mel"])*F.smooth_l1_loss(pred,batch["mel"])+float(cfg["loss"]["foreground_structure"])*structure+float(cfg["loss"]["eeg_activity"])*F.binary_cross_entropy_with_logits(act,batch["activity"].float())
    # Light channel consistency after one stochastic channel view.
    mask=batch["channel_mask"] & (torch.rand_like(batch["channel_mask"].float())>float(cfg["training"]["channel_dropout_max"]))
    alternate=generator.encode(batch["eeg"],batch["channel_xyz"],mask,batch["time_mask"])
    consistency=channel_consistency(state.linguistic_latent,alternate.linguistic_latent)
    total=total+float(cfg["loss"]["channel_consistency"])*consistency
    return total,{"loss":float(total.detach()),"semantic":float(semantic.detach()),"sequence":float(seq.detach()),"evidence":float(evidence.detach()),"consistency":float(consistency.detach())}


@torch.no_grad()
def evaluate_eeg(generator:DualLatentEEGToSpeech,data:DataLoader,device:torch.device) -> dict[str,float]:
    losses=[]; correct=[]; evidence=[]; predictions=[]
    for batch in tqdm(data,desc="[0728 eeg validation]",unit="batch",mininterval=1.0,disable=False):
        batch=move_batch(batch,device); state=generator.encode(batch["eeg"],batch["channel_xyz"],batch["channel_mask"],batch["time_mask"])
        target=generator.audio_model(batch["hubert"],batch["hubert_mask"],batch["mel"],batch["activity"])
        logits=F.normalize(masked_pool(state.linguistic_latent,state.linguistic_mask),dim=-1)@F.normalize(masked_pool(target.linguistic_latent,target.linguistic_mask),dim=-1).T
        predicted=logits.argmax(-1); correct.extend((batch["label_index"]==batch["label_index"][predicted]).float().cpu().tolist()); predictions.extend(batch["label_index"][predicted].cpu().tolist()); evidence.extend(state.evidence_probability.cpu().tolist())
        losses.append(float(F.smooth_l1_loss(state.linguistic_latent,target.linguistic_latent).cpu()))
    maximum=max(np.mean(np.asarray(predictions)==value) for value in set(predictions)) if predictions else 1.0
    return {"loss":float(np.mean(losses)),"content_retrieval_macro_top1":float(np.mean(correct)),"evidence_median":float(np.median(evidence)),"maximum_prediction_fraction":float(maximum)}


def run_audio(config_path:Path,cfg:dict[str,Any],device:torch.device,seed:int,epochs:int|None,resume:Path|None,smoke:int,*,loso_subject:str|None=None) -> None:
    context=load_context(config_path,cfg); lineage=build_lineage(config_path,cfg,manifest=context.manifest_path,split=context.split_path,montage=context.montage_path); root=resolve_config_path(config_path,cfg["paths"]["cache_root"])
    train=DualLatentDataset(CacheV3(root,"train"),exclude_subject=loso_subject); valid=DualLatentDataset(CacheV3(root,"validation"),exclude_subject=loso_subject); dl=loader(train,batch_size=int(cfg["training"]["audio_batch_size"]),seed=seed,train=True); vl=loader(valid,batch_size=int(cfg["training"]["audio_batch_size"]),seed=seed,train=False)
    run_id=None if loso_subject is None else f"strict_end_to_end_loso_{loso_subject.replace(':','_')}_seed_{seed}"; model=DualLatentAudioModel().to(device); latest,best_path,history=paths(config_path,cfg,"audio",run_id); warm=int(cfg["training"]["audio_content_warmup_epochs"])
    if resume:
        raw=torch.load(resume,map_location="cpu",weights_only=False); validate_checkpoint(raw,lineage); model.load_state_dict(raw["state_dict"]); start=int(raw["epoch"])+1; best=float(raw.get("extra",{}).get("best",math.inf)); warmup_complete=bool(raw.get("extra",{}).get("warmup_complete",start>=warm))
        if warmup_complete:
            for module in (model.content,model.content_decoder):
                for parameter in module.parameters(): parameter.requires_grad=False
        opt=torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],lr=float(cfg["training"]["audio_lr"]),weight_decay=float(cfg["training"]["weight_decay"]))
        if raw.get("extra",{}).get("optimizer"): opt.load_state_dict(raw["extra"]["optimizer"])
    else:
        start,best=0,math.inf; opt=torch.optim.AdamW(model.parameters(),lr=float(cfg["training"]["audio_lr"]),weight_decay=float(cfg["training"]["weight_decay"]))
    total=int(epochs or cfg["training"]["audio_epochs"]); stale=0; patience=int(cfg["training"]["audio_patience"])
    for epoch in range(start,total):
        if epoch>=warm and any(parameter.requires_grad for parameter in model.content.parameters()):
            for module in (model.content,model.content_decoder):
                for parameter in module.parameters(): parameter.requires_grad=False
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=float(cfg["training"]["audio_lr"]),weight_decay=float(cfg["training"]["weight_decay"]))
        model.train(); values=[]
        for step,batch in enumerate(tqdm(dl,desc=f"[0728 audio] {epoch+1}/{total}",unit="batch")):
            batch=move_batch(batch,device); loss,metric=audio_loss(model,batch,cfg,warmup=epoch<warm); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["training"]["grad_clip"])); opt.step(); values.append(metric)
            if smoke and step+1>=smoke: break
        model.eval(); validation=[]
        for step,batch in enumerate(tqdm(vl,desc=f"[0728 audio validation] {epoch+1}/{total}",unit="batch",mininterval=1.0,disable=False)):
            batch=move_batch(batch,device); loss,metric=audio_loss(model,batch,cfg,warmup=epoch<warm); validation.append(metric["loss"])
            if smoke and step+1>=smoke: break
        score=float(np.mean(validation)); record={"epoch":epoch+1,"train_loss":float(np.mean([v["loss"] for v in values])),"validation_loss":score}; history.parent.mkdir(parents=True,exist_ok=True); history.open("a").write(json.dumps(record)+"\n"); save_checkpoint(latest,model,opt,epoch,lineage,{"best":best,"warmup_complete":epoch>=warm})
        if score<best:
            best=score; stale=0; save_checkpoint(best_path,model,opt,epoch,lineage,{"best":best,"warmup_complete":epoch>=warm})
        else:
            stale+=1
            if epoch>=warm and stale>=patience: break
    model.load_state_dict(torch.load(best_path,map_location=device,weights_only=False)["state_dict"]); report=audio_gate(model,vl,load_stss(resolve_config_path(config_path,cfg["paths"]["metric_manifest"])),cfg,device,train); report["checkpoint"]=str(best_path); write_json(best_path.parent.parent/"metrics"/"audio_gate.json",report)
    if not report["passed"]: raise RuntimeError(f"v0728 audio gate failed: {report}")
    freeze_path=resolve_config_path(config_path,cfg["paths"]["audio_freeze_manifest"]) if loso_subject is None else best_path.parent.parent/"frozen_checkpoint.json"
    write_json(freeze_path,{"schema_version":"openvoice-0728-audio-freeze-v1","checkpoint":str(best_path),"lineage":lineage.as_dict(),"gate":report,"loso_subject":loso_subject})


def run_eeg(config_path:Path,cfg:dict[str,Any],device:torch.device,seed:int,phase:str,epochs:int|None,resume:Path|None,smoke:int,*,loso_subject:str|None=None,strict_audio_loso:bool=False) -> None:
    context=load_context(config_path,cfg); lineage=build_lineage(config_path,cfg,manifest=context.manifest_path,split=context.split_path,montage=context.montage_path); root=resolve_config_path(config_path,cfg["paths"]["cache_root"]); labels=cfg["data"]["labels_4"] if phase in {"semantic4","dual4"} else cfg["data"]["labels_11"]
    train=DualLatentDataset(CacheV3(root,"train"),labels=labels,exclude_subject=loso_subject); valid=DualLatentDataset(CacheV3(root,"validation"),labels=labels,only_subject=loso_subject) if loso_subject else DualLatentDataset(CacheV3(root,"validation"),labels=labels); dl=loader(train,batch_size=int(cfg["training"]["eeg_batch_size"]),seed=seed,train=True); vl=loader(valid,batch_size=int(cfg["training"]["eeg_batch_size"]),seed=seed,train=False)
    run_kind="strict_end_to_end" if strict_audio_loso else "shared_audio_prior"; run_id=None if loso_subject is None else f"{run_kind}_loso_{loso_subject.replace(':','_')}_seed_{seed}"
    audio_path=resolve_config_path(config_path,cfg["paths"]["audio_checkpoint"]) if not strict_audio_loso else paths(config_path,cfg,"audio",f"strict_end_to_end_loso_{loso_subject.replace(':','_')}_seed_{seed}")[1]
    audio=DualLatentAudioModel().to(device); raw=torch.load(audio_path,map_location=device,weights_only=False); validate_checkpoint(raw,lineage); audio.load_state_dict(raw["state_dict"]); audio.eval();
    for p in audio.parameters(): p.requires_grad=False
    model=DualLatentEEGToSpeech(EEGEncoder(),audio).to(device)
    if resume is None and phase in {"dual4","full11"}:
        previous="semantic4" if phase=="dual4" else "dual4"
        _,previous_best,_=paths(config_path,cfg,previous,run_id)
        if not previous_best.is_file(): raise FileNotFoundError(f"{phase} requires the frozen curriculum checkpoint: {previous_best}")
        inherited=torch.load(previous_best,map_location=device,weights_only=False); validate_checkpoint(inherited,lineage); model.load_state_dict(inherited["state_dict"],strict=True)
    opt=torch.optim.AdamW(model.eeg_encoder.parameters(),lr=float(cfg["training"]["eeg_lr"]),weight_decay=float(cfg["training"]["weight_decay"])); latest,best_path,history=paths(config_path,cfg,phase,run_id); start,best=(0,math.inf) if not resume else restore(resume,model,opt,lineage)
    default={"semantic4":"semantic4_epochs","dual4":"dual4_epochs","full11":"full11_epochs"}[phase]; total=int(epochs or cfg["training"][default])
    patience=int(cfg["training"][{"semantic4":"semantic4_patience","dual4":"dual4_patience","full11":"full11_patience"}[phase]]); stale=0
    for epoch in range(start,total):
        model.train(); values=[]
        for step,batch in enumerate(tqdm(dl,desc=f"[0728 {phase}] {epoch+1}/{total}",unit="batch")):
            batch=move_batch(batch,device); loss,metric=eeg_loss(model,batch,cfg,phase=phase); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.eeg_encoder.parameters(),float(cfg["training"]["grad_clip"])); opt.step(); values.append(metric)
            if smoke and step+1>=smoke: break
        model.eval(); validation=evaluate_eeg(model,vl,device); score=validation["loss"]; record={"epoch":epoch+1,"train_loss":float(np.mean([v["loss"] for v in values])),**validation}; history.parent.mkdir(parents=True,exist_ok=True); history.open("a").write(json.dumps(record)+"\n"); save_checkpoint(latest,model,opt,epoch,lineage,{"best":best,"phase":phase,"labels":list(labels)})
        if score<best:
            best=score; stale=0; save_checkpoint(best_path,model,opt,epoch,lineage,{"best":best,"phase":phase,"labels":list(labels)})
        else:
            stale+=1
            if stale>=patience: break
    model.load_state_dict(torch.load(best_path,map_location=device,weights_only=False)["state_dict"])
    write_json(best_path.parent.parent/"metrics"/"validation.json",evaluate_eeg(model,vl,device))


def main() -> None:
    value=args(); config_path,cfg=load_config(value.config); seed=int(value.seed or cfg["training"]["seed"]); seed_everything(seed); device=default_device(value.device)
    if value.phase=="audio": run_audio(config_path,cfg,device,seed,value.epochs,value.resume,value.smoke_steps,loso_subject=value.loso_subject if value.strict_audio_loso else None)
    elif value.phase in {"semantic4","dual4","full11"}: run_eeg(config_path,cfg,device,seed,value.phase,value.epochs,value.resume,value.smoke_steps,loso_subject=value.loso_subject,strict_audio_loso=value.strict_audio_loso)
    else:
        raise ValueError("validate requires a completed checkpoint; use gate_open_vocab_0728.py")
if __name__=="__main__": main()
