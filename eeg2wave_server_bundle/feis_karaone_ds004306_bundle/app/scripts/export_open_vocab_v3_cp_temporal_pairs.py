#!/usr/bin/env python3
"""Export CP-temporal training pairs, controls, and diagnostic figures."""
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
from torch.utils.data import DataLoader

APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:sys.path.insert(0,str(APP))

from scripts.train_open_vocab_v3_cp_temporal import (TokenDataset,attach_codes,
    load_checkpoint,make_modules,micro_dataset,token_collate,train_dev)
from src.open_vocab_0724.audio_features import AudioPreparationConfig
from src.open_vocab_v3.cp_temporal import PREPARATION_SCHEMA,SCHEMA
from src.open_vocab_v3.data import (V3Dataset,_accepted_denoise_paths,_read_waveform,
    channel_shuffled_eeg,collate,light_prepare_waveform,load_prepared,time_shuffled_eeg)
from src.open_vocab_v3.encodec_content import EnCodecGenerator
from src.open_vocab_v3.runtime import (capture_lineage,checkpoint_schema,
    default_device,load_config,move_batch,output_path,read_json,sha256_file,write_json)


def parse():
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--stage",choices=("micro","fit","final"),default="fit");p.add_argument("--device",default="cpu");p.add_argument("--max-pairs",type=int,default=0);p.add_argument("--resume",action="store_true");p.add_argument("--explore",action="store_true");return p.parse_args()


def save_wav(path,value):sf.write(path,np.asarray(value,dtype=np.float32),16000,subtype="PCM_16")


def heatmap(path,rows,title,cmap="magma"):
 fig,axes=plt.subplots(len(rows),1,figsize=(12,max(2.4,2.15*len(rows))),sharex=True);axes=[axes] if len(rows)==1 else axes
 for axis,(name,value) in zip(axes,rows):axis.imshow(np.asarray(value),origin="lower",aspect="auto",cmap=cmap);axis.set_ylabel(name)
 fig.suptitle(title);fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


@torch.inference_mode()
def main():
 args=parse();cp,cfg=load_config(args.config);device=default_device(args.device);records=load_prepared(output_path(cp,cfg,"prepared_cache"),expected_schema=PREPARATION_SCHEMA);checkpoint_stage="micro" if args.stage=="micro" else "fit"
 if args.stage=="final":dataset=V3Dataset(records,tuple(sorted(set(records.roles.tolist()))),eligible_only=True);pair_collate=collate
 else:
  base=micro_dataset(records,cfg) if args.stage=="micro" else train_dev(records)[0];cache,mapping=attach_codes(records,cp,cfg);dataset=TokenDataset(base,cache,mapping);pair_collate=token_collate
 if args.stage=="final" and not args.explore:
  review=output_path(cp,cfg,"training_review");payload=read_json(review) if review.is_file() else {};expected=capture_lineage(cp,cfg,artifact_keys=("fit_checkpoint","fit_gate","fit_preview_manifest"))
  if not payload.get("passed",False) or payload.get("lineage")!=expected:raise RuntimeError("CP-temporal final export refused before exact training-WAV approval")
 audio,decoder,eeg,backbone,teacher,cvae=make_modules(cfg,device);load_checkpoint(output_path(cp,cfg,"content_checkpoint"),checkpoint_schema(cfg,"content"),{"audio":audio,"decoder":decoder},device);load_checkpoint(output_path(cp,cfg,"cvae_checkpoint"),checkpoint_schema(cfg,"cvae"),{"cvae":cvae,"teacher":teacher},device);load_checkpoint(output_path(cp,cfg,f"{checkpoint_stage}_checkpoint"),checkpoint_schema(cfg,checkpoint_stage),{"eeg":eeg},device)
 eeg_p_path=output_path(cp,cfg,"eeg_prosody_checkpoint");eeg_p=None
 if eeg_p_path.is_file():
  _,_,eeg_p,_,_,_=make_modules(cfg,device);load_checkpoint(eeg_p_path,checkpoint_schema(cfg,"eeg_prosody"),{"eeg":eeg_p},device);eeg_p.eval()
 from transformers import SpeechT5HifiGan
 vocoder=SpeechT5HifiGan.from_pretrained(str(output_path(cp,cfg,"vocoder_root")),local_files_only=True).to(device).eval();codec=EnCodecGenerator(output_path(cp,cfg,"encodec_root"),device=device,bandwidth=float(cfg["audio"]["encodec_bandwidth"]));audio.eval();decoder.eval();eeg.eval();cvae.eval();backbone=cvae.backbone.eval()
 with output_path(cp,cfg,"unified_manifest").open(newline="",encoding="utf-8") as handle:paths={str(row["sample_key"]):str(row["audio_relpath"]) for row in csv.DictReader(handle) if row.get("dataset")=="karaone"}
 audio_root=output_path(cp,cfg,"audio_root");denoised=_accepted_denoise_paths(cp,cfg);preparation=AudioPreparationConfig(sample_rate=16000,max_active_seconds=float(cfg["audio"]["max_active_seconds"]),target_rms=float(cfg["audio"]["target_rms"]));root=output_path(cp,cfg,"micro_preview_root" if args.stage=="micro" else "fit_preview_root" if args.stage=="fit" else "pair_root");root.mkdir(parents=True,exist_ok=True);manifest=[];checkpoint=output_path(cp,cfg,f"{checkpoint_stage}_checkpoint")
 expected_pair_files=("00_reference.wav","01_frozen_encodec_oracle.wav","02_real_C_real_P_deterministic.wav","03_real_C_canonical_P.wav","04_pred_C_real_P.wav","05_real_C_pred_P.wav","06_pred_C_pred_P.wav","07_eeg_C_canonical_P_primary.wav","08_eeg_C_eeg_P_exploratory.wav","09_zero_eeg.wav","10_time_shuffle.wav","11_channel_shuffle.wav","content_mfcc.png","prosody_base_and_plus.png","mel_comparison.png","local_ot_similarity.png","cross_attention.png")
 def waveform(mel):return vocoder(mel.transpose(1,2))[0].detach().cpu().numpy()
 def prior(content,p,voice,plus=None):return cvae(content,p,voice,plus,target=None,stochastic=False)["mel"]
 for batch in DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=pair_collate,num_workers=0):
  batch=move_batch(batch,device);sample=batch["sample_key"][0];folder=root/sample.replace(":","_");folder.mkdir(parents=True,exist_ok=True)
  metadata_path=folder/"metadata.json"
  if args.resume and metadata_path.is_file():
   previous=read_json(metadata_path);files=previous.get("files",{});valid=previous.get("schema_version")==SCHEMA and previous.get("checkpoint_sha256")==sha256_file(checkpoint) and previous.get("config_sha256")==sha256_file(cp)
   valid=valid and all((folder/name).is_file() and files.get(name)==sha256_file(folder/name) for name in expected_pair_files)
   if valid:
    manifest.append({"sample_key":sample,"label":batch["label"][0],"folder":str(folder),"metadata":str(metadata_path)});print(f"[v3 CP export] resumed {len(manifest)}/{min(len(dataset),args.max_pairs) if args.max_pairs else len(dataset)} {sample}",flush=True)
    if args.max_pairs and len(manifest)>=args.max_pairs:break
    continue
  source=denoised.get(sample,audio_root/paths[sample]);raw,rate=_read_waveform(source);prepared,_=light_prepare_waveform(raw,rate,preparation);reference=prepared.waveform[:prepared.valid_samples]
  codes,mask=codec.encode(torch.from_numpy(reference).unsqueeze(0).to(device));codec_oracle=codec.decode(codes[:,:,:int(mask[0].sum())],target_samples_16k=len(reference))[0].cpu().numpy()
  if args.stage=="final":
   padded=torch.zeros((1,8,192),dtype=torch.long,device=device);padded_mask=torch.zeros((1,192),dtype=torch.bool,device=device);steps=min(codes.shape[-1],192);padded[:,:,:steps]=codes[:,:,:steps].long();padded_mask[:,:steps]=mask[:,:steps];batch["encodec_codes"],batch["encodec_mask"]=padded,padded_mask
  audio_state=audio(batch["encodec_codes"],batch["encodec_mask"]);real_c=batch["content_mfcc"].float();real_p=batch["p_base"].float();pred_c_real,_,audio_diag=decoder(audio_state.local,audio_state.token_mask,real_p,batch["duration_fraction"].float());pred_c_pred,_,_=decoder(audio_state.local,audio_state.token_mask,audio_state.p_base,audio_state.duration_fraction);canonical_p=batch["canonical_p_base"].float();voice=batch["canonical_voice"].float();duration=batch["canonical_duration_fraction"].float()
  eeg_state=eeg(batch["eeg"].float(),batch["channel_xyz"].float(),batch["channel_mask"],batch["time_mask"]);eeg_c,_,eeg_diag=decoder(eeg_state.local,eeg_state.token_mask,canonical_p,duration)
  conditions={
   "02_real_C_real_P_deterministic.wav":backbone(real_c,real_p,voice,batch["p_plus"].float()),
   "03_real_C_canonical_P.wav":backbone(real_c,canonical_p,voice,None),
   "04_pred_C_real_P.wav":backbone(pred_c_real,real_p,voice,None),
   "05_real_C_pred_P.wav":backbone(real_c,audio_state.p_base,voice,None),
   "06_pred_C_pred_P.wav":backbone(pred_c_pred,audio_state.p_base,voice,None),
   "07_eeg_C_canonical_P_primary.wav":prior(eeg_c,canonical_p,voice),
   "08_eeg_C_eeg_P_exploratory.wav":prior(eeg_c,(eeg_p if eeg_p is not None else eeg)(batch["eeg"].float(),batch["channel_xyz"].float(),batch["channel_mask"],batch["time_mask"]).p_base,voice),
  }
  control_signals={"09_zero_eeg.wav":torch.zeros_like(batch["eeg"]),"10_time_shuffle.wav":time_shuffled_eeg(batch["eeg"],batch["time_mask"]),"11_channel_shuffle.wav":channel_shuffled_eeg(batch["eeg"],batch["channel_mask"])}
  control_mfcc={}
  for name,signal in control_signals.items():state=eeg(signal.float(),batch["channel_xyz"].float(),batch["channel_mask"],batch["time_mask"]);value,_,_=decoder(state.local,state.token_mask,canonical_p,duration);control_mfcc[name]=value;conditions[name]=prior(value,canonical_p,voice)
  save_wav(folder/"00_reference.wav",reference);save_wav(folder/"01_frozen_encodec_oracle.wav",codec_oracle)
  for name,mel in conditions.items():save_wav(folder/name,waveform(mel))
  heatmap(folder/"content_mfcc.png",[("real audio grid",real_c[0].cpu()),("EEG canonical target",batch["eeg_content_mfcc"][0].cpu()),("audio predicted",pred_c_real[0].cpu()),("EEG predicted",eeg_c[0].cpu()),("zero",control_mfcc["09_zero_eeg.wav"][0].cpu())],"Content MFCC (c1-c39)")
  heatmap(folder/"prosody_base_and_plus.png",[("real P",real_p[0].T.cpu()),("audio P",audio_state.p_base[0].T.cpu()),("EEG P exploratory",eeg_state.p_base[0].T.cpu()),("P+ audio-only",batch["p_plus"][0].T.cpu())],"P/P+ diagnostics")
  heatmap(folder/"mel_comparison.png",[("real",batch["speech_t5_mel"][0].cpu()),("oracle",conditions["02_real_C_real_P_deterministic.wav"][0].cpu()),("EEG primary",conditions["07_eeg_C_canonical_P_primary.wav"][0].cpu()),("zero",conditions["09_zero_eeg.wav"][0].cpu())],"Native SpeechT5 Mel")
  similarity=F.normalize(eeg_state.local[0],dim=-1)@F.normalize(audio_state.local[0],dim=-1).T;heatmap(folder/"local_ot_similarity.png",[("EEG/audio",similarity.cpu())],"Local token cosine")
  heatmap(folder/"cross_attention.png",[("audio",audio_diag["attention"][0].cpu()),("EEG",eeg_diag["attention"][0].cpu())],"Duration-aware cross attention",cmap="viridis")
  metadata={"schema_version":SCHEMA,"sample_key":sample,"label":batch["label"][0],"stage":args.stage,"primary":"thinking_EEG_C_plus_fit_only_canonical_P","eeg_prosody_exploratory":True,"phase_metadata_available":False,"exploratory_gate_bypass":bool(args.explore),"source_audio":str(source),"source_audio_sha256":sha256_file(source),"config_sha256":sha256_file(cp),"checkpoint_sha256":sha256_file(checkpoint),"content_checkpoint_sha256":sha256_file(output_path(cp,cfg,"content_checkpoint")),"cvae_checkpoint_sha256":sha256_file(output_path(cp,cfg,"cvae_checkpoint")),"attention":{"audio_coverage":float(audio_diag["coverage"][0]),"audio_slope":float(audio_diag["slope"][0]),"eeg_coverage":float(eeg_diag["coverage"][0]),"eeg_slope":float(eeg_diag["slope"][0])}}
  metadata["files"]={name:sha256_file(folder/name) for name in expected_pair_files};write_json(metadata_path,metadata);manifest.append({"sample_key":sample,"label":batch["label"][0],"folder":str(folder),"metadata":str(metadata_path)});print(f"[v3 CP export] {len(manifest)}/{min(len(dataset),args.max_pairs) if args.max_pairs else len(dataset)} {sample}",flush=True)
  if args.max_pairs and len(manifest)>=args.max_pairs:break
 complete=not args.max_pairs or len(manifest)==len(dataset);name="export_manifest.json" if args.stage=="final" else "preview_manifest.json";lineage=capture_lineage(cp,cfg,artifact_keys=(f"{checkpoint_stage}_checkpoint","content_checkpoint","cvae_checkpoint"));write_json(root/name,{"schema_version":SCHEMA,"stage":args.stage,"complete":complete,"n":len(manifest),"exploratory_gate_bypass":bool(args.explore),"lineage":lineage,"pairs":manifest})
 with (root/"manifest.csv").open("w",newline="",encoding="utf-8") as handle:writer=csv.DictWriter(handle,fieldnames=("sample_key","label","folder","metadata"));writer.writeheader();writer.writerows(manifest)
 print(root,flush=True)


if __name__=="__main__":main()
