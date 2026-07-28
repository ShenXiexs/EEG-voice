#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from tqdm import tqdm

APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path: sys.path.insert(0,str(APP))

from src.open_vocab_0728.data import CacheV3, normalize_label
from src.open_vocab_0728.metrics import load_stss
from src.open_vocab_0728.model import DualLatentAudioModel
from src.open_vocab_0728.runtime import default_device, load_config, resolve_config_path, stable_hash, write_json


@torch.no_grad()
def main()->None:
    parser=argparse.ArgumentParser(description="Falsifiable v0728 audio-latent leakage and branch-usage audit")
    parser.add_argument("--config",type=Path,required=True); parser.add_argument("--device",default=None); args=parser.parse_args(); config,cfg=load_config(args.config); device=default_device(args.device); root=resolve_config_path(config,cfg["paths"]["cache_root"])
    train=CacheV3(root,"train"); valid=CacheV3(root,"validation"); model=DualLatentAudioModel().to(device); raw=torch.load(resolve_config_path(config,cfg["paths"]["audio_checkpoint"]),map_location=device,weights_only=False); model.load_state_dict(raw["state_dict"]); model.eval(); stss=load_stss(resolve_config_path(config,cfg["paths"]["metric_manifest"]))
    latent_l=[]; latent_r=[]; labels=[]; subjects=[]
    for cache_name,cache in (("train",train),("validation",valid)):
        for index in tqdm(range(len(cache)),desc=f"[0728 audit] encode {cache_name}",unit="trial",mininterval=1.0,disable=False):
            item=cache.item(index); hub=torch.from_numpy(item["hubert"]).to(device).unsqueeze(0); hm=torch.from_numpy(item["hubert_mask"]).to(device).unsqueeze(0); mel=torch.from_numpy(item["mel"]).to(device).unsqueeze(0); activity=torch.from_numpy(item["activity"]).to(device).unsqueeze(0)
            state=model(hub,hm,mel,activity); latent_l.append(state.linguistic_latent.mean(1).squeeze().cpu().numpy()); latent_r.append(state.realization_latent.mean(1).squeeze().cpu().numpy()); labels.append(normalize_label(item["label"])); subjects.append(item["subject"])
    def grouped_score(features:list[np.ndarray], target:list[str], groups:list[str], grouped:bool)->float:
        x=np.asarray(features); y=np.asarray(target); splitter=StratifiedGroupKFold(n_splits=min(6,len(set(groups))),shuffle=True,random_state=15) if grouped else StratifiedKFold(n_splits=5,shuffle=True,random_state=15)
        scores=[]
        iterator=splitter.split(x,y,groups) if grouped else splitter.split(x,y)
        for fit,test in tqdm(iterator,total=splitter.get_n_splits(),desc=f"[0728 audit] {('grouped ' if grouped else '')}probe",unit="fold",mininterval=1.0,disable=False):
            model_probe=LogisticRegression(max_iter=1000,class_weight="balanced").fit(x[fit],y[fit]); scores.append(balanced_accuracy_score(y[test],model_probe.predict(x[test])))
        return float(np.mean(scores))
    label_l=grouped_score(latent_l,labels,subjects,True); label_r=grouped_score(latent_r,labels,subjects,True); subject_l=grouped_score(latent_l,subjects,subjects,False); subject_r=grouped_score(latent_r,subjects,subjects,False)
    gains=[]; shuffle_gains=[]; content_changes=[]
    by_subject_label:dict[tuple[str,str],list[int]]=defaultdict(list)
    for index in tqdm(range(len(valid)),desc="[0728 audit] index donors",unit="trial",mininterval=1.0,disable=False): by_subject_label[(str(valid.raw["subjects"][index]),normalize_label(str(valid.raw["labels"][index])))].append(index)
    for index in tqdm(range(len(valid)),desc="[0728 audit] branch-usage swaps",unit="trial",mininterval=1.0,disable=False):
        item=valid.item(index); members=[v for v in by_subject_label[(item["subject"],normalize_label(item["label"]))] if v!=index]
        if not members: continue
        donor=members[int(stable_hash(item["sample_key"],"zr")[:8],16)%len(members)]; donor_item=valid.item(donor)
        def encode(value:dict):
            return model(torch.from_numpy(value["hubert"]).to(device).unsqueeze(0),torch.from_numpy(value["hubert_mask"]).to(device).unsqueeze(0),torch.from_numpy(value["mel"]).to(device).unsqueeze(0),torch.from_numpy(value["activity"]).to(device).unsqueeze(0))
        original,other=encode(item),encode(donor_item); full,_=model.decode(original.linguistic_latent,original.realization_latent); swapped,_=model.decode(original.linguistic_latent,other.realization_latent); coarse,_=model.decode_content(original.linguistic_latent); ref=item["mel"]
        gains.append(stss.score(full[0].cpu().numpy(),ref)-stss.score(coarse[0].cpu().numpy(),ref)); shuffle_gains.append(stss.score(full[0].cpu().numpy(),ref)-stss.score(swapped[0].cpu().numpy(),ref)); content_changes.append(float(torch.mean(torch.abs(original.linguistic_latent-other.linguistic_latent)).cpu()))
    report={"label_probe":{"linguistic":label_l,"realization":label_r,"difference":label_l-label_r},"subject_probe":{"linguistic":subject_l,"realization":subject_r},"branch_usage":{"median_full_minus_content_stss":float(np.median(gains)) if gains else None,"median_correct_minus_same_label_realization_shuffle_stss":float(np.median(shuffle_gains)) if shuffle_gains else None,"same_label_linguistic_distance_diagnostic":float(np.median(content_changes)) if content_changes else None},"passed":bool(label_l>label_r and (not gains or np.median(gains)>=.03) and (not shuffle_gains or np.median(shuffle_gains)>=.05))}
    output=resolve_config_path(config,cfg["paths"]["output_root"])/"audio"/"metrics"/"disentanglement_probes.json"; write_json(output,report); print(report)
    if not report["passed"]: raise RuntimeError("audio disentanglement audit failed")
if __name__=="__main__": main()
