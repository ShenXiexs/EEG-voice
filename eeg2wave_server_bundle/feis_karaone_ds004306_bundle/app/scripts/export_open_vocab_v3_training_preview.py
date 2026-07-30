#!/usr/bin/env python3
"""Export audible training reconstructions before any held-out evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:sys.path.insert(0,str(APP))

from scripts.export_open_vocab_v3_pairs import heatmap,light_cleaned_reference,manifest_paths,write_wave
from scripts.train_open_vocab_v3 import load_audio,load_eeg,micro_dataset
from src.open_vocab_v3.data import V3Dataset,_accepted_denoise_paths,channel_shuffled_eeg,collate,load_prepared,time_shuffled_eeg
from src.open_vocab_v3.runtime import capture_lineage,default_device,load_config,move_batch,output_path,require_passed_gate,write_json
from src.open_vocab_v3.vocoder import SpeechT5PowerDbHiFiGan,pcm16


def selected_fit(records,per_label:int)->Subset:
    base=V3Dataset(records,("fit",),eligible_only=True);groups={}
    for item,index in enumerate(base.indices):groups.setdefault(str(records.arrays["labels"][index]),[]).append(item)
    chosen=[]
    for label,items in sorted(groups.items()):
        chosen.extend(sorted(items,key=lambda item:str(records.arrays["sample_keys"][base.indices[item]]))[:per_label])
    return Subset(base,chosen)


@torch.no_grad()
def main()->None:
    parser=argparse.ArgumentParser(description="Export v3 micro/full-fit training WAV previews")
    parser.add_argument("--config",type=Path,required=True);parser.add_argument("--stage",choices=("micro","fit"),required=True);parser.add_argument("--device",default="cpu");parser.add_argument("--resume",action="store_true");args=parser.parse_args()
    config_path,cfg=load_config(args.config);device=default_device(args.device);started=time.monotonic();records=load_prepared(output_path(config_path,cfg,"prepared_cache"))
    if args.stage=="micro":
        gate=require_passed_gate(config_path,cfg,"micro_gate",lineage_artifact_keys=("micro_checkpoint","v2_gate"));dataset=micro_dataset(records,str(cfg["micro_gate"]["subject"]),int(cfg["micro_gate"]["per_label"]));root_key="micro_preview_root";manifest_key="micro_preview_manifest";lineage_keys=("micro_checkpoint","micro_gate")
    else:
        gate=require_passed_gate(config_path,cfg,"fit_gate",lineage_artifact_keys=("fit_checkpoint","micro_gate"));dataset=selected_fit(records,int(cfg["evaluation"]["training_preview_per_label"]));root_key="fit_preview_root";manifest_key="fit_preview_manifest";lineage_keys=("fit_checkpoint","fit_gate")
    eeg,_=load_eeg(config_path,cfg,device,stage=args.stage);audio,_=load_audio(config_path,cfg,device);vocoder=SpeechT5PowerDbHiFiGan(output_path(config_path,cfg,"vocoder_root"),device=device)
    destination=output_path(config_path,cfg,root_key);destination.mkdir(parents=True,exist_ok=True);paths=manifest_paths(output_path(config_path,cfg,"unified_manifest"));audio_root=output_path(config_path,cfg,"audio_root");denoised=_accepted_denoise_paths(config_path,cfg)
    gate_keys=list(map(str,gate.get("sample_keys",[])));ranks=list(gate.get("correct",{}).get("paired_rank_per_trial",[]));rank_by_key={key:int(rank) for key,rank in zip(gate_keys,ranks)}
    rows=[];samples=int(cfg["evaluation"]["variational_samples"])
    for batch in tqdm(DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=collate,num_workers=0),total=len(dataset),desc=f"[v3 {args.stage} preview]",unit="pair",dynamic_ncols=True):
        batch=move_batch(batch,device);key=batch["sample_key"][0];stem=destination/key;meta=stem.with_suffix(".json")
        base_names=("cleaned_reference","v0_vocoder_oracle","analytic_mfcc_oracle","cvae_posterior_oracle","cvae_prior_real_mfcc","eeg_prior_mean","zero_eeg","time_shuffled","channel_shuffled")
        names={name:stem.with_name(f"{stem.name}__{name}.wav") for name in base_names}
        names.update({f"eeg_prior_sample_{i+1}":stem.with_name(f"{stem.name}__eeg_prior_sample_{i+1}.wav") for i in range(samples)})
        figure=stem.with_name(f"{stem.name}__comparison.png")
        if args.resume and meta.is_file() and figure.is_file() and all(path.is_file() for path in names.values()):rows.append(json.loads(meta.read_text()));continue
        source=int(np.flatnonzero(records.arrays["sample_keys"].astype(str)==key)[0]);kwargs=(batch["channel_xyz"].float(),batch["channel_mask"],batch["time_mask"])
        predicted={"eeg":eeg(batch["eeg"].float(),*kwargs)[0],"zero_eeg":eeg(torch.zeros_like(batch["eeg"]).float(),*kwargs)[0],"time_shuffled":eeg(time_shuffled_eeg(batch["eeg"].float(),batch["time_mask"]),*kwargs)[0],"channel_shuffled":eeg(channel_shuffled_eeg(batch["eeg"].float(),batch["channel_mask"]),*kwargs)[0]}
        voice=batch["canonical_voice"].float();mean=batch["canonical_mfcc_mean"].float();std=batch["canonical_mfcc_std"].float()
        prior_real=audio.generate(batch["mfcc"].float(),voice,mean,std,stochastic=False);posterior=audio.reconstruct(batch["mfcc"].float(),voice,mean,std,batch["mel"].float(),stochastic=False)
        eeg_prior=audio.generate(predicted["eeg"],voice,mean,std,stochastic=False)
        mel={"analytic_mfcc_oracle":prior_real["analytic_mel"],"cvae_posterior_oracle":posterior["mel"],"cvae_prior_real_mfcc":prior_real["mel"],"eeg_prior_mean":eeg_prior["mel"]}
        for name in ("zero_eeg","time_shuffled","channel_shuffled"):mel[name]=audio.generate(predicted[name],voice,mean,std,stochastic=False)["mel"]
        for index in range(samples):mel[f"eeg_prior_sample_{index+1}"]=audio.generate(predicted["eeg"],voice,mean,std,stochastic=True)["mel"]
        generated={"v0_vocoder_oracle":pcm16(vocoder.synthesize(torch.from_numpy(records.arrays["vocoder_mel"][source:source+1]).to(device))[0]),**{name:pcm16(vocoder.synthesize(value)[0]) for name,value in mel.items()}}
        ref,rate=light_cleaned_reference(denoised.get(key,audio_root/paths[key]),cfg);write_wave(names["cleaned_reference"],ref,rate)
        for name,wave in generated.items():write_wave(names[name],wave,int(cfg["vocoder"]["sample_rate"]))
        target_mfcc=batch["mfcc"][0].cpu().numpy();target_mel=batch["mel"][0].cpu().numpy();heatmap(figure,target_mfcc,predicted["eeg"][0].cpu().numpy(),target_mel,eeg_prior["mel"][0].cpu().numpy(),f"{args.stage}: {key}")
        row={"stage":args.stage,"sample_key":key,"subject":batch["subject"][0],"label":batch["label"][0],"within_label_trial_retrieval_rank":rank_by_key.get(key),"mfcc_mae":float(np.mean(np.abs(predicted["eeg"][0].cpu().numpy()-target_mfcc))),"comparison_png":str(figure),**{f"{name}_wav":str(path) for name,path in names.items()}}
        write_json(meta,row);rows.append(row)
    elapsed=time.monotonic()-started;manifest_path=output_path(config_path,cfg,manifest_key)
    write_json(manifest_path,{"schema_version":"openvoice-v3-training-preview-v1","stage":args.stage,"n":len(rows),"complete":len(rows)==len(dataset),"elapsed_seconds":elapsed,"lineage":capture_lineage(config_path,cfg,artifact_keys=lineage_keys),"records":rows})
    with (destination/"manifest.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else ["sample_key"]);writer.writeheader();writer.writerows(rows)
    print(f"[v3 preview] stage={args.stage} n={len(rows)} manifest={manifest_path}",flush=True)


if __name__=="__main__":main()
