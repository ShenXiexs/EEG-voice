#!/usr/bin/env python3
"""Export only fit/M1 listening pairs for repair-v3; never reads held-out roles."""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:sys.path.insert(0,str(APP))
from scripts.train_open_vocab_v3_encodec_rvq_repair import TokenDataset,base_subset,fit_indices,load_cache,load_checkpoint,make_models,token_collate
from src.open_vocab_v3.data import V3Dataset,load_prepared,time_shuffled_eeg,channel_shuffled_eeg
from src.open_vocab_v3.encodec_rvq_repair import PREPARATION_SCHEMA,SCHEMA,FrozenEnCodecRVQ
from src.open_vocab_v3.runtime import capture_lineage,checkpoint_schema,default_device,load_config,move_batch,output_path,sha256_file,write_json
def parse():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--max-pairs',type=int,default=0);p.add_argument('--explore',action='store_true');return p.parse_args()
def wav(path,x):sf.write(path,np.asarray(x,np.float32),16000,subtype='PCM_16')
def plot(path,items,title):
 fig,axes=plt.subplots(len(items),1,figsize=(12,2.5*len(items)),sharex=True);axes=np.atleast_1d(axes)
 for ax,(name,x) in zip(axes,items):ax.imshow(np.asarray(x),origin='lower',aspect='auto',cmap='magma');ax.set_ylabel(name)
 fig.suptitle(title);fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)
def batch(records,cache,mapping,source,device):
 base=V3Dataset(records,('fit',),eligible_only=True);item=base[{int(x):i for i,x in enumerate(base.indices)}[int(source)]];row=mapping[int(source)]
 for k in ('encodec_codes','encodec_mask','audio_scales','waveform_16k','waveform_mask','waveform_samples'):item[k]=cache[k][row]
 return move_batch(token_collate([item]),device)
@torch.no_grad()
def main():
 a=parse();cp,cfg=load_config(a.config);device=default_device(a.device);records=load_prepared(output_path(cp,cfg,'prepared_cache'),expected_schema=PREPARATION_SCHEMA);cache,mapping=load_cache(cp,cfg);bridge,audio,decoder,eeg=make_models(cfg,device);load_checkpoint(output_path(cp,cfg,'rvq_bridge_checkpoint'),checkpoint_schema(cfg,'rvq_bridge'),{'bridge':bridge},device);load_checkpoint(output_path(cp,cfg,'audio_c_checkpoint'),checkpoint_schema(cfg,'audio_c'),{'audio':audio,'decoder':decoder},device);m1=output_path(cp,cfg,'micro_m1_predictions');predpath=m1 if m1.is_file() else output_path(cp,cfg,'micro_m0b_predictions');raw=np.load(predpath,allow_pickle=False)
 if str(raw['schema'].item())!=SCHEMA:raise RuntimeError('stale repair prediction rejected')
 ck='micro_m1_checkpoint' if predpath==m1 else 'micro_m0b_checkpoint';
 if ck=='micro_m0b_checkpoint':load_checkpoint(output_path(cp,cfg,ck),checkpoint_schema(cfg,'micro_m0b'),{'eeg':eeg},device)
 renderer=FrozenEnCodecRVQ(output_path(cp,cfg,'encodec_root'),device=device,bandwidth=float(cfg['audio']['encodec_bandwidth']));root=output_path(cp,cfg,'preview_root');root.mkdir(parents=True,exist_ok=True);rows=[];bank=records.arrays['canonical_p_bank'];bd=records.arrays['canonical_p_bank_duration_fraction']
 for row,source in enumerate(raw['source_indices'].astype(int).tolist()):
  b=batch(records,cache,mapping,source,device);folder=root/b['sample_key'][0].replace(':','_');folder.mkdir(parents=True,exist_ok=True);n=int(b['waveform_samples'][0]);steps=int(b['encodec_mask'][0].sum());reference=b['waveform_16k'][0,:n].cpu().numpy();oracle=renderer.decode_codes(b['encodec_codes'][...,:steps],b['audio_scales'],target_samples=n)[0].cpu().numpy();audio_state=audio(b['encodec_codes'],b['encodec_mask'],b['hubert'].float(),b['hubert_mask']);ac,_=decoder(audio_state.local,audio_state.token_mask);eegc=torch.from_numpy(raw['prediction'][row]).to(device).unsqueeze(0);controls={k:torch.from_numpy(raw[k][row]).to(device).unsqueeze(0) for k in ('zero','time','channel')}
  def render(c,p,v,d,code_mask=None):
   codes=bridge.hard_codes(bridge(c,p,v,d),code_mask=code_mask,duration_fraction=d);valid=int(code_mask[0].sum()) if code_mask is not None else int(torch.ceil(d[0]*codes.shape[-1]).item());return renderer.decode_codes(codes[...,:valid],b['audio_scales'],target_samples=n)[0].cpu().numpy()
  real=b['content_mfcc'].float();p=b['p_base'].float();v=b['speaker_reference'].float();d=b['duration_fraction'].float();p0=torch.from_numpy(bank[0]).to(device).unsqueeze(0);d0=torch.full_like(d,float(bd[0]));cv=b['canonical_voice'].float()
  files={'00_reference.wav':reference,'01_frozen_encodec_oracle.wav':oracle,'02_real_C_real_P_independent_voice.wav':render(real,p,v,d,b['encodec_mask']),'03_zero_C_real_P.wav':render(torch.zeros_like(real),p,v,d,b['encodec_mask']),'04_shuffled_C_real_P.wav':render(real.flip(-1),p,v,d,b['encodec_mask']),'05_real_C_duration_only_P.wav':render(real,torch.zeros_like(p),v,d,b['encodec_mask']),'06_pred_audio_C_real_P.wav':render(ac,p,v,d,b['encodec_mask']),'07_eeg_C_P0_canonical_voice.wav':render(eegc,p0,cv,d0),'08_zero_eeg.wav':render(controls['zero'],p0,cv,d0),'09_time_shuffle_eeg.wav':render(controls['time'],p0,cv,d0),'10_channel_shuffle_eeg.wav':render(controls['channel'],p0,cv,d0)}
  for name,x in files.items():wav(folder/name,x)
  plot(folder/'content_mfcc.png',[('real C',real[0].cpu()),('Audio-C',ac[0].cpu()),('EEG-C',eegc[0].cpu()),('zero',controls['zero'][0].cpu())],'content MFCC c1…c39')
  plot(folder/'rvq_codes.png',[('true q0…q7',b['encodec_codes'][0].cpu()),('pred q0…q7',bridge.hard_codes(bridge(real,p,v,d),code_mask=b['encodec_mask'])[0].cpu())],'sequential RVQ codes')
  meta={'schema_version':SCHEMA,'exploratory':bool(a.explore),'sample_key':b['sample_key'][0],'source_index':source,'prediction_cache_sha256':sha256_file(predpath),'rvq_bridge_sha256':sha256_file(output_path(cp,cfg,'rvq_bridge_checkpoint')),'audio_c_sha256':sha256_file(output_path(cp,cfg,'audio_c_checkpoint')),'files':{k:sha256_file(folder/k) for k in files}|{'content_mfcc.png':sha256_file(folder/'content_mfcc.png'),'rvq_codes.png':sha256_file(folder/'rvq_codes.png')}};write_json(folder/'metadata.json',meta);rows.append({'sample_key':b['sample_key'][0],'label':b['label'][0],'folder':str(folder),'metadata':str(folder/'metadata.json')});print(f'[v3 rvq export] {len(rows)}/{len(raw["source_indices"])} {b["sample_key"][0]}',flush=True)
  if a.max_pairs and len(rows)>=a.max_pairs:break
 with (root/'manifest.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=('sample_key','label','folder','metadata'));w.writeheader();w.writerows(rows)
 write_json(output_path(cp,cfg,'preview_manifest'),{'schema_version':SCHEMA,'exploratory':bool(a.explore),'n':len(rows),'pairs':rows,'lineage':capture_lineage(cp,cfg,artifact_keys=('rvq_bridge_checkpoint','audio_c_checkpoint','micro_m1_checkpoint' if m1.is_file() else 'micro_m0b_checkpoint'))})
 print(root,flush=True)
if __name__=='__main__':main()
