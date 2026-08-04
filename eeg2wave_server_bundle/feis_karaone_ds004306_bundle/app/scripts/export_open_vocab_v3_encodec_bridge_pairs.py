#!/usr/bin/env python3
"""Export the fit-only E1/E2/C2/M0/M1 listening bundle with lineage."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:sys.path.insert(0,str(APP))

from scripts.train_open_vocab_v3_encodec_bridge import (
    base_subset, fit_indices, load_cache, load_checkpoint, make_models, token_collate,
)
from src.open_vocab_v3.data import V3Dataset, channel_shuffled_eeg, collate, load_prepared, time_shuffled_eeg
from src.open_vocab_v3.encodec_bridge import PREPARATION_SCHEMA, SCHEMA, FrozenEnCodecRenderer
from src.open_vocab_v3.runtime import capture_lineage, checkpoint_schema, default_device, load_config, move_batch, output_path, read_json, sha256_file, write_json


def parse():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True);parser.add_argument("--device",default="cpu")
    parser.add_argument("--max-pairs",type=int,default=0);parser.add_argument("--resume",action="store_true");parser.add_argument("--explore",action="store_true")
    return parser.parse_args()


def wav(path,value):sf.write(path,np.asarray(value,dtype=np.float32),16000,subtype="PCM_16")


def heatmap(path,items,title,cmap="magma"):
    figure,axes=plt.subplots(len(items),1,figsize=(12,max(2.6,2.3*len(items))),sharex=True);axes=[axes] if len(items)==1 else axes
    for axis,(name,value) in zip(axes,items):axis.imshow(np.asarray(value),origin="lower",aspect="auto",cmap=cmap);axis.set_ylabel(name)
    figure.suptitle(title);figure.tight_layout();figure.savefig(path,dpi=140);plt.close(figure)


def batch_for_source(records,cache,mapping,source,device):
    base=V3Dataset(records,("fit",),eligible_only=True);position={int(index):row for row,index in enumerate(base.indices)}[int(source)]
    item=base[position];slot=mapping[int(source)]
    for name in ("encodec_codes","encodec_mask","target_latent","waveform_16k","waveform_mask","waveform_samples"):item[name]=cache[name][slot]
    return move_batch(token_collate([item]),device)


def save_conditions(folder, bridge, renderer, batch, eeg_content, controls, audio_content, p_bank, p_duration):
    samples=int(batch["waveform_samples"][0]);codes=batch["encodec_codes"];reference=batch["waveform_16k"][0,:samples].cpu().numpy();oracle=renderer.render_codes(codes,target_samples=samples)[0].cpu().numpy()
    def render(content,p,voice,duration):
        latent=bridge(content,p,voice,duration);code,_,_=renderer.quantize_st(latent);return renderer.render_codes(code,target_samples=samples)[0].cpu().numpy()
    real=batch["content_mfcc"].float();p=batch["p_base"].float();voice=batch["speaker_reference"].float();duration=batch["duration_fraction"].float();canonical_voice=batch["canonical_voice"].float()
    conditions={
        "00_reference.wav":reference,"01_frozen_encodec_oracle.wav":oracle,
        "02_real_C_real_P_independent_voice.wav":render(real,p,voice,duration),
        "03_zero_C_real_P.wav":render(torch.zeros_like(real),p,voice,duration),
        # Pair folders contain one trial, so cross-batch rolling would be an
        # identity.  A fixed temporal reversal is a non-identity C/P mismatch
        # control with no label or target-audio lookup.
        "04_shuffled_C_real_P.wav":render(real.flip(-1),p,voice,duration),
        "05_real_C_shuffled_P.wav":render(real,p.flip(1),voice,duration),
        "06_real_C_duration_only_P.wav":render(real,torch.zeros_like(p),voice,duration),
        "08_pred_audio_C_real_P.wav":render(audio_content,p,voice,duration),
    }
    for index,value in enumerate(p_bank):
        p_value=torch.from_numpy(value).to(real.device).unsqueeze(0);d_value=torch.full_like(duration,float(p_duration[index]))
        conditions[f"07_canonical_P{index}_canonical_voice.wav"]=render(real,p_value,canonical_voice,d_value)
    canonical_p=torch.from_numpy(p_bank[0]).to(real.device).unsqueeze(0);canonical_duration=torch.full_like(duration,float(p_duration[0]))
    conditions["09_eeg_C_primary_P0.wav"]=render(eeg_content,canonical_p,canonical_voice,canonical_duration)
    conditions["10_zero_eeg.wav"]=render(controls["zero"],canonical_p,canonical_voice,canonical_duration)
    conditions["11_time_shuffled_eeg.wav"]=render(controls["time"],canonical_p,canonical_voice,canonical_duration)
    conditions["12_channel_shuffled_eeg.wav"]=render(controls["channel"],canonical_p,canonical_voice,canonical_duration)
    for name,value in conditions.items():wav(folder/name,value)
    return conditions


@torch.inference_mode()
def main():
    args=parse();cp,cfg=load_config(args.config);device=default_device(args.device);records=load_prepared(output_path(cp,cfg,"prepared_cache"),expected_schema=PREPARATION_SCHEMA);cache,mapping=load_cache(cp,cfg)
    subjects=sorted(set(records.arrays["subjects"][fit_indices(records,dev=False)].astype(str).tolist()));audio,decoder,eeg,bridge=make_models(cfg,device,len(subjects))
    load_checkpoint(output_path(cp,cfg,"bridge_checkpoint"),checkpoint_schema(cfg,"bridge"),{"bridge":bridge},device);load_checkpoint(output_path(cp,cfg,"audio_c_checkpoint"),checkpoint_schema(cfg,"audio_c"),{"audio":audio,"decoder":decoder},device)
    bridge.eval();audio.eval();decoder.eval();renderer=FrozenEnCodecRenderer(output_path(cp,cfg,"encodec_root"),device=device,bandwidth=float(cfg["audio"]["encodec_bandwidth"]))
    prediction_path=output_path(cp,cfg,"micro_m1_predictions") if output_path(cp,cfg,"micro_m1_predictions").is_file() else output_path(cp,cfg,"micro_m0_predictions")
    prediction=np.load(prediction_path,allow_pickle=False)
    if str(prediction["schema"].item())!=SCHEMA:raise RuntimeError("stale non-bridge prediction cache rejected")
    source_indices=prediction["source_indices"].astype(int).tolist();eeg_values=prediction["prediction"].astype(np.float32);controls={name:prediction[name].astype(np.float32) for name in ("zero","time","channel")}
    root=output_path(cp,cfg,"preview_root");root.mkdir(parents=True,exist_ok=True);bank=records.arrays["canonical_p_bank"].astype(np.float32);bank_duration=records.arrays["canonical_p_bank_duration_fraction"].astype(np.float32);rows=[]
    expected=("00_reference.wav","01_frozen_encodec_oracle.wav","02_real_C_real_P_independent_voice.wav","03_zero_C_real_P.wav","04_shuffled_C_real_P.wav","05_real_C_shuffled_P.wav","06_real_C_duration_only_P.wav","08_pred_audio_C_real_P.wav","09_eeg_C_primary_P0.wav","10_zero_eeg.wav","11_time_shuffled_eeg.wav","12_channel_shuffled_eeg.wav","content_mfcc.png","mel_proxy_comparison.png","token_similarity.png","metadata.json")
    for row,source in enumerate(source_indices):
        batch=batch_for_source(records,cache,mapping,source,device);key=batch["sample_key"][0];folder=root/key.replace(":","_");folder.mkdir(parents=True,exist_ok=True);metadata_path=folder/"metadata.json"
        if args.resume and metadata_path.is_file() and all((folder/name).is_file() for name in expected):
            rows.append({"sample_key":key,"label":batch["label"][0],"folder":str(folder),"metadata":str(metadata_path)});continue
        audio_state=audio(batch["encodec_codes"],batch["encodec_mask"]);audio_content,_=decoder(audio_state.local,audio_state.token_mask)
        eeg_content=torch.from_numpy(eeg_values[row]).to(device).unsqueeze(0);control_content={name:torch.from_numpy(value[row]).to(device).unsqueeze(0) for name,value in controls.items()}
        conditions=save_conditions(folder,bridge,renderer,batch,eeg_content,control_content,audio_content,bank,bank_duration)
        heatmap(folder/"content_mfcc.png",[("real C",batch["content_mfcc"][0].cpu()),("Audio-C",audio_content[0].cpu()),("EEG-C",eeg_content[0].cpu()),("zero EEG",control_content["zero"][0].cpu())],"VAD-active CMVN content MFCC (c1–c39)")
        heatmap(folder/"mel_proxy_comparison.png",[("real C",batch["content_mfcc"][0].cpu()),("EEG C",eeg_content[0].cpu()),("P0",bank[0].T)],"C/P rendering conditions")
        similarity=F.normalize(audio_state.local[0],dim=-1)@F.normalize(audio_state.local[0],dim=-1).T
        heatmap(folder/"token_similarity.png",[("Audio-C self-similarity",similarity.cpu())],"Audio-C local token similarity",cmap="viridis")
        files={name:sha256_file(folder/name) for name in conditions}|{name:sha256_file(folder/name) for name in ("content_mfcc.png","mel_proxy_comparison.png","token_similarity.png")}
        metadata={"schema_version":SCHEMA,"sample_key":key,"label":batch["label"][0],"primary":"thinking_EEG_C_plus_fit_train_P0_and_canonical_voice","exploratory":bool(args.explore),"source_index":int(source),"p_bank_keys":records.arrays["canonical_p_bank_keys"].astype(str).tolist(),"speaker_reference_keys":str(records.arrays["speaker_reference_keys"][source]),"bridge_checkpoint_sha256":sha256_file(output_path(cp,cfg,"bridge_checkpoint")),"audio_c_checkpoint_sha256":sha256_file(output_path(cp,cfg,"audio_c_checkpoint")),"prediction_cache_sha256":sha256_file(prediction_path),"files":files}
        write_json(metadata_path,metadata);rows.append({"sample_key":key,"label":batch["label"][0],"folder":str(folder),"metadata":str(metadata_path)})
        print(f"[v3 bridge export] {len(rows)}/{len(source_indices)} {key}",flush=True)
        if args.max_pairs and len(rows)>=args.max_pairs:break
    complete=not args.max_pairs or len(rows)==len(source_indices);manifest={"schema_version":SCHEMA,"complete":complete,"exploratory":bool(args.explore),"n":len(rows),"prediction_cache":str(prediction_path),"lineage":capture_lineage(cp,cfg,artifact_keys=("bridge_checkpoint","audio_c_checkpoint","micro_m0_checkpoint","micro_m1_checkpoint" if output_path(cp,cfg,"micro_m1_checkpoint").is_file() else "micro_m0_checkpoint")),"pairs":rows};write_json(output_path(cp,cfg,"preview_manifest"),manifest)
    with (root/"manifest.csv").open("w",newline="",encoding="utf-8") as handle:writer=csv.DictWriter(handle,fieldnames=("sample_key","label","folder","metadata"));writer.writeheader();writer.writerows(rows)
    print(root,flush=True)


if __name__=="__main__":main()
