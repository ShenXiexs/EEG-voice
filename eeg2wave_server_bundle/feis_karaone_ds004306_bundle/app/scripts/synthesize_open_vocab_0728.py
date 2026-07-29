#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from scipy.io import wavfile

APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path: sys.path.insert(0,str(APP))

from src.open_vocab_0728.data import CacheV3, DualLatentDataset, collate, normalize_label
from src.open_vocab_0728.lineage import claim_locked_test_access, update_locked_test_ledger
from src.open_vocab_0728.metrics import duration_seconds, load_stss, ms_ssim, soft_dtw
from src.open_vocab_0728.model import DualLatentAudioModel, DualLatentEEGToSpeech, EEGEncoder
from src.open_vocab_0728.runtime import default_device, load_config, resolve_config_path, stable_hash, write_json
from src.open_vocab_0728.vocoder import griffin_lim_from_log_mel


def parse() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Synthesize v0728 mel/WAV and factor counterfactuals")
    p.add_argument("--config",type=Path,required=True); p.add_argument("--phase",choices=("semantic4","dual4","full11"),default="full11"); p.add_argument("--split",choices=("train","validation","locked_test","diagnostic"),default="validation"); p.add_argument("--device",default=None); p.add_argument("--access-id"); p.add_argument("--resume-existing",action="store_true"); p.add_argument("--limit",type=int,default=0); return p.parse_args()


def checkpoint_path(config:Path,cfg:dict,phase:str)->Path:
    key={"semantic4":"eeg_semantic_checkpoint","dual4":"eeg_dual4_checkpoint","full11":"eeg_full_checkpoint"}[phase]
    return resolve_config_path(config,cfg["paths"][key])


def load_model(config:Path,cfg:dict,phase:str,device:torch.device)->DualLatentEEGToSpeech:
    audio=DualLatentAudioModel().to(device); model=DualLatentEEGToSpeech(EEGEncoder(),audio).to(device)
    raw=torch.load(checkpoint_path(config,cfg,phase),map_location=device,weights_only=False); model.load_state_dict(raw["state_dict"],strict=True); model.eval(); return model


def donor(cache:CacheV3,index:int,*,same_label:bool,subject:bool=True)->int:
    label=normalize_label(str(cache.raw["labels"][index])); sub=str(cache.raw["subjects"][index]); candidates=[]
    for position in range(len(cache)):
        if position==index: continue
        if (normalize_label(str(cache.raw["labels"][position]))==label)==same_label and (not subject or str(cache.raw["subjects"][position])==sub): candidates.append(position)
    if not candidates: candidates=[p for p in range(len(cache)) if p!=index]
    if not candidates: return index
    return candidates[int(stable_hash("v0728-donor",cache.keys[index],same_label)[:8],16)%len(candidates)]


def decode(audio:DualLatentAudioModel,linguistic:torch.Tensor,realization:torch.Tensor,evidence:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    mel,act=audio.decode(linguistic,realization); gated=-80+evidence[:,None,None]*(mel+80); activity=(torch.sigmoid(act)*evidence[:,None])>=.5; return gated,activity


@torch.no_grad()
def main()->None:
    arg=parse(); config,cfg=load_config(arg.config); device=default_device(arg.device); root=resolve_config_path(config,cfg["paths"]["cache_root"])
    print(f"[0728 synth] preparing {arg.phase}/{arg.split}: device={device.type}; loading split cache", flush=True)
    ledger=None
    if arg.split=="locked_test":
        if not arg.access_id: raise PermissionError("locked-test synthesis requires --access-id")
        freeze_path=resolve_config_path(config,cfg["paths"]["locked_test_freeze"])
        if not freeze_path.exists(): raise PermissionError("locked test requires frozen validation configuration")
        ledger=claim_locked_test_access(resolve_config_path(config,cfg["paths"]["locked_test_ledger"]),freeze=json.loads(freeze_path.read_text()),access_id=arg.access_id)
    cache=CacheV3(root,arg.split,allow_locked=arg.split=="locked_test")
    train_cache=CacheV3(root,"train")
    print(f"[0728 synth] cache ready: target={len(cache)} records; loading frozen {arg.phase} checkpoint", flush=True)
    model=load_model(config,cfg,arg.phase,device)
    stss=load_stss(resolve_config_path(config,cfg["paths"]["metric_manifest"]))
    print("[0728 synth] checkpoint ready; building train-only label-median baselines", flush=True)
    label_medians:dict[str,np.ndarray]={}
    labels=sorted({normalize_label(str(value)) for value in train_cache.raw["labels"]})
    for label in tqdm(labels,desc="[0728 synth] label medians",unit="label",mininterval=1.0,disable=False):
        members=[np.asarray(train_cache.raw["mel"][i]) for i,value in enumerate(train_cache.raw["labels"]) if normalize_label(str(value))==label]
        label_medians[label]=np.median(members,axis=0)
    output=resolve_config_path(config,cfg["paths"]["output_root"])/"synthesis"/arg.phase/arg.split; output.mkdir(parents=True,exist_ok=True); records=[]
    print(f"[0728 synth] preparation complete; starting {len(cache) if not arg.limit else min(len(cache),arg.limit)} trial synthesis", flush=True)
    for index in tqdm(range(len(cache)),desc=f"[0728 synth] {arg.phase}/{arg.split}",unit="trial"):
        item=cache.item(index); base=output/item["sample_key"]
        if arg.resume_existing and (base.with_suffix(".json")).exists():
            records.append(json.loads(base.with_suffix(".json").read_text())); continue
        tensor=collate([{**item,"label_index":0}]); tensor={k:v.to(device) if torch.is_tensor(v) else v for k,v in tensor.items()}; state=model.encode(tensor["eeg"],tensor["channel_xyz"],tensor["channel_mask"],tensor["time_mask"])
        donor_index=donor(cache,index,same_label=True); donor_item=cache.item(donor_index); donor_tensor=collate([{**donor_item,"label_index":0}]); donor_tensor={k:v.to(device) if torch.is_tensor(v) else v for k,v in donor_tensor.items()}; donor_state=model.encode(donor_tensor["eeg"],donor_tensor["channel_xyz"],donor_tensor["channel_mask"],donor_tensor["time_mask"])
        wrong_index=donor(cache,index,same_label=False); wrong_item=cache.item(wrong_index); wrong_tensor=collate([{**wrong_item,"label_index":0}]); wrong_tensor={k:v.to(device) if torch.is_tensor(v) else v for k,v in wrong_tensor.items()}; wrong_state=model.encode(wrong_tensor["eeg"],wrong_tensor["channel_xyz"],wrong_tensor["channel_mask"],wrong_tensor["time_mask"])
        zero_state=model.encode(torch.zeros_like(tensor["eeg"]),tensor["channel_xyz"],tensor["channel_mask"],tensor["time_mask"])
        noise_state=model.encode(torch.randn_like(tensor["eeg"])*float(cfg["training"]["signal_noise_std"]),tensor["channel_xyz"],tensor["channel_mask"],tensor["time_mask"])
        reverse_state=model.encode(torch.flip(tensor["eeg"],dims=(-1,)),tensor["channel_xyz"],tensor["channel_mask"],tensor["time_mask"])
        permutation=torch.arange(tensor["eeg"].shape[1]-1,-1,-1,device=device)
        permuted_state=model.encode(tensor["eeg"][:,permutation],tensor["channel_xyz"],tensor["channel_mask"],tensor["time_mask"])
        audio_oracle=model.audio_model(tensor["hubert"],tensor["hubert_mask"],tensor["mel"],tensor["activity"])
        conditions={
            "correct":(state.linguistic_latent,state.realization_latent,state.evidence_probability),
            "realization_shuffle":(state.linguistic_latent,donor_state.realization_latent,state.evidence_probability),
            "content_shuffle":(wrong_state.linguistic_latent,state.realization_latent,state.evidence_probability),
            "all_factor_shuffle":(wrong_state.linguistic_latent,wrong_state.realization_latent,state.evidence_probability),
            "content_only":(state.linguistic_latent,torch.zeros_like(state.realization_latent),state.evidence_probability),
            "realization_only":(torch.zeros_like(state.linguistic_latent),state.realization_latent,state.evidence_probability),
            "zero_eeg":(zero_state.linguistic_latent,zero_state.realization_latent,zero_state.evidence_probability),
            "gaussian_noise_eeg":(noise_state.linguistic_latent,noise_state.realization_latent,noise_state.evidence_probability),
            "time_reversed_eeg":(reverse_state.linguistic_latent,reverse_state.realization_latent,reverse_state.evidence_probability),
            "channel_permuted_eeg":(permuted_state.linguistic_latent,permuted_state.realization_latent,permuted_state.evidence_probability),
            "audio_latent_oracle":(audio_oracle.linguistic_latent,audio_oracle.realization_latent,torch.ones_like(state.evidence_probability)),
        }
        result={"sample_key":item["sample_key"],"audio_key":item["audio_key"],"label":item["label"],"subject":item["subject"],"conditions":{}}
        reference=np.asarray(item["mel"])
        direct={"label_median_baseline":label_medians[normalize_label(item["label"])]}
        for name,(linguistic,realization,evidence) in conditions.items():
            if name=="content_only": mel,activity=model.audio_model.decode_content(linguistic); mel=-80+evidence[:,None,None]*(mel+80); activity=(torch.sigmoid(activity)*evidence[:,None])>=.5
            else: mel,activity=decode(model.audio_model,linguistic,realization,evidence)
            value=mel[0].detach().cpu(); active=activity[0].detach().cpu().numpy(); target=base.with_name(base.name+f"__{name}"); np.save(target.with_suffix(".mel.npy"),value.numpy()); wav=griffin_lim_from_log_mel(value,iterations=int(cfg["audio"]["griffin_lim_iterations"]),seed=int(stable_hash(item["sample_key"],name)[:8],16)); wavfile.write(target.with_suffix(".wav"),int(cfg["audio"]["sample_rate"]),(wav.cpu().numpy()*32767).astype(np.int16))
            result["conditions"][name]={"mel_path":str(target.with_suffix(".mel.npy")),"wav_path":str(target.with_suffix(".wav")),"stss":stss.score(value.numpy(),reference),"ms_ssim":ms_ssim(value.numpy(),reference),"soft_dtw_divergence":soft_dtw(value.numpy(),reference),"duration_seconds":duration_seconds(active),"evidence":float(evidence[0].detach().cpu())}
        for name,value in direct.items():
            target=base.with_name(base.name+f"__{name}"); np.save(target.with_suffix(".mel.npy"),value); wav=griffin_lim_from_log_mel(torch.from_numpy(value).to(device),iterations=int(cfg["audio"]["griffin_lim_iterations"]),seed=int(stable_hash(item["sample_key"],name)[:8],16)); wavfile.write(target.with_suffix(".wav"),int(cfg["audio"]["sample_rate"]),(wav.cpu().numpy()*32767).astype(np.int16))
            result["conditions"][name]={"mel_path":str(target.with_suffix(".mel.npy")),"wav_path":str(target.with_suffix(".wav")),"stss":stss.score(value,reference),"ms_ssim":ms_ssim(value,reference),"soft_dtw_divergence":soft_dtw(value,reference),"duration_seconds":duration_seconds((value>-55).any(0)),"evidence":1.0}
        write_json(base.with_suffix(".json"),result); records.append(result)
        if ledger: update_locked_test_ledger(resolve_config_path(config,cfg["paths"]["locked_test_ledger"]),ledger,completed_key=item["sample_key"])
        if arg.limit and index+1>=arg.limit: break
    manifest={"schema_version":"openvoice-0728-synthesis-v1","phase":arg.phase,"split":arg.split,"records":records,"metric_manifest":str(resolve_config_path(config,cfg["paths"]["metric_manifest"]))}; write_json(output/"synthesis_manifest.json",manifest)
    if ledger: update_locked_test_ledger(resolve_config_path(config,cfg["paths"]["locked_test_ledger"]),ledger,complete=True)
    print(output/"synthesis_manifest.json")
if __name__=="__main__": main()
