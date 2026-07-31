#!/usr/bin/env python3
"""Fit-only adaptation for the *generation-path* EnCodec, HiFi-GAN and ECAPA.

HuBERT and the independent ECAPA metric instance are intentionally absent from
this program: they remain immutable evaluation instruments.
"""
from __future__ import annotations
import argparse, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

APP=Path(__file__).resolve().parents[1]; sys.path.insert(0, str(APP)) if str(APP) not in sys.path else None
from src.open_vocab_v3.audio_adaptation import parameter_change, tensor_state, selected_audio_indices, multi_resolution_stft_loss, envelope_loss
from src.open_vocab_v3.data import load_prepared
from src.open_vocab_v3.encodec_content import _resample
from src.open_vocab_v3.native_mel import native_speecht5_mel
from src.open_vocab_v3.runtime import default_device, load_config, output_path, seed_everything, sha256_file, write_json
from scripts.finetune_open_vocab_v3_audio_models import AudioDomainDataset, collate as audio_collate, _fit_speaker

def args():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True);p.add_argument('--scope',choices=('fit','all'),default='fit');p.add_argument('--device',default='auto');p.add_argument('--deadline-epoch',type=float,default=0);p.add_argument('--smoke-steps',type=int,default=0);p.add_argument('--explore',action='store_true',help='record an A0 failure but continue the exploratory pipeline');return p.parse_args()
def deadline(a): return bool(a.deadline_epoch and time.time()>=a.deadline_epoch)
def buffer_state(module):return {n:x.detach().cpu().clone() for n,x in module.named_buffers() if torch.is_floating_point(x)}
def buffer_change(before,module):
 current=dict(module.named_buffers());changed=sum(int(not torch.equal(value,current[name].detach().cpu())) for name,value in before.items());return {'buffer_tensors':len(before),'changed_buffer_tensors':changed,'changed_buffer_fraction':float(changed/max(len(before),1))}
def paths(cp,cfg,scope):
 if scope=='fit': return {'encodec':output_path(cp,cfg,'encodec_adapted_root'),'encodec_manifest':output_path(cp,cfg,'encodec_manifest'),'vocoder':output_path(cp,cfg,'vocoder_adapted_root'),'vocoder_manifest':output_path(cp,cfg,'vocoder_manifest'),'speaker':output_path(cp,cfg,'speaker_adapted_checkpoint'),'speaker_manifest':output_path(cp,cfg,'speaker_adaptation_manifest'),'gate':output_path(cp,cfg,'audio_adaptation_gate')}
 root=output_path(cp,cfg,'output_root')/'audio_adaptation'/'transductive_all_encodec_clip_v1';return {'encodec':root/'encodec','encodec_manifest':root/'encodec_manifest.json','vocoder':root/'hifigan','vocoder_manifest':root/'hifigan_manifest.json','speaker':root/'ecapa/adapted_backbone.pt','speaker_manifest':root/'ecapa/adaptation_manifest.json','gate':root/'A0.json'}
def fit_encodec(cp,cfg,data,dst,device,a):
 from transformers import EncodecModel
 base=output_path(cp,cfg,'encodec_root'); model=EncodecModel.from_pretrained(str(base),local_files_only=True).to(device)
 for x in model.parameters():x.requires_grad_(True)
 before=tensor_state(model);before_encoder=tensor_state(model.encoder);before_decoder=tensor_state(model.decoder);before_quantizer=buffer_state(model.quantizer);opt=torch.optim.AdamW(model.parameters(),lr=float(cfg['audio_adaptation']['encodec_lr']),weight_decay=float(cfg['training']['weight_decay']))
 loader=DataLoader(data,batch_size=int(cfg['audio_adaptation']['encodec_batch_size']),shuffle=True,collate_fn=audio_collate,num_workers=0);first=best=None;history=[]
 for epoch in range(int(cfg['audio_adaptation']['encodec_epochs'])):
  if deadline(a):break
  values=[]
  for step,b in enumerate(loader):
   if deadline(a):break
   wave=_resample(b['waveform'].to(device),16000,24000).unsqueeze(1); mask=torch.ones(wave.shape[0],wave.shape[-1],dtype=torch.bool,device=device)
   out=model(wave,padding_mask=mask,bandwidth=float(cfg['audio']['encodec_bandwidth'])); pred=out.audio_values; target=wave[...,:pred.shape[-1]]
   loss=F.l1_loss(pred,target)+0.25*multi_resolution_stft_loss(pred,target,fft_sizes=cfg['audio_adaptation']['stft_fft_sizes'],hop_sizes=cfg['audio_adaptation']['stft_hop_sizes'])
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['training']['grad_clip']));opt.step();values.append(float(loss.detach()))
   if a.smoke_steps and step+1>=a.smoke_steps:break
  if not values:break
  val=float(np.mean(values));first=val if first is None else first;history.append({'epoch':epoch+1,'loss':val})
  if best is None or val<best:
   best=val;dst.mkdir(parents=True,exist_ok=True);model.save_pretrained(dst)
  if a.smoke_steps:break
 if first is None:raise RuntimeError('EnCodec adaptation performed zero optimizer steps')
 selected=EncodecModel.from_pretrained(str(dst),local_files_only=True);change=parameter_change(before,selected);groups={'encoder':parameter_change(before_encoder,selected.encoder),'quantizer_ema_buffers':buffer_change(before_quantizer,selected.quantizer),'decoder':parameter_change(before_decoder,selected.decoder)}
 return {'component':'Encodec_encoder_quantizer_decoder','base_root':str(base),'adapted_root':str(dst),'all_pretrained_generator_parameters_trainable':all(x.requires_grad for x in model.parameters()),'quantizer_note':'HF EnCodec quantizer has zero Parameters; its EMA codebook state is checked as trainable buffers','component_change':groups,'epochs_completed':len(history),'first_epoch_loss':first,'best_loss':best,'relative_loss_improvement':float((first-best)/max(abs(first),1e-8)),'pretrained_parameter_change':change,'history':history}
def fit_hifigan(cp,cfg,data,dst,device,a):
 from transformers import SpeechT5HifiGan
 base=output_path(cp,cfg,'vocoder_root');model=SpeechT5HifiGan.from_pretrained(str(base),local_files_only=True).to(device)
 for x in model.parameters():x.requires_grad_(True)
 before=tensor_state(model);opt=torch.optim.AdamW(model.parameters(),lr=float(cfg['audio_adaptation']['vocoder_lr']),weight_decay=float(cfg['training']['weight_decay']));loader=DataLoader(data,batch_size=int(cfg['audio_adaptation']['vocoder_batch_size']),shuffle=True,collate_fn=audio_collate,num_workers=0);first=best=None;history=[]
 for epoch in range(int(cfg['audio_adaptation']['vocoder_epochs'])):
  if deadline(a):break
  values=[]
  for step,b in enumerate(loader):
   if deadline(a):break
   wave=b['waveform'].to(device); mel=b['native_mel'].to(device);pred=model(mel.transpose(1,2));target=wave[...,:pred.shape[-1]]
   if target.shape[-1]<pred.shape[-1]:target=F.pad(target,(0,pred.shape[-1]-target.shape[-1]))
   loss=multi_resolution_stft_loss(pred,target,fft_sizes=cfg['audio_adaptation']['stft_fft_sizes'],hop_sizes=cfg['audio_adaptation']['stft_hop_sizes'])+float(cfg['audio_adaptation']['waveform_l1_weight'])*F.l1_loss(pred,target)+float(cfg['audio_adaptation']['envelope_weight'])*envelope_loss(pred,target)
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['training']['grad_clip']));opt.step();values.append(float(loss.detach()))
   if a.smoke_steps and step+1>=a.smoke_steps:break
  if not values:break
  val=float(np.mean(values));first=val if first is None else first;history.append({'epoch':epoch+1,'loss':val})
  if best is None or val<best:best=val;dst.mkdir(parents=True,exist_ok=True);model.save_pretrained(dst)
  if a.smoke_steps:break
 if first is None:raise RuntimeError('SpeechT5 HiFi-GAN adaptation performed zero optimizer steps')
 selected=SpeechT5HifiGan.from_pretrained(str(dst),local_files_only=True);change=parameter_change(before,selected)
 return {'component':'SpeechT5HiFiGAN_native_SpeechT5_Mel','base_root':str(base),'adapted_root':str(dst),'native_mel_contract':cfg['vocoder']['native_contract'],'all_pretrained_generator_parameters_trainable':True,'epochs_completed':len(history),'first_epoch_loss':first,'best_loss':best,'relative_loss_improvement':float((first-best)/max(abs(first),1e-8)),'pretrained_parameter_change':change,'history':history}
def main():
 a=args();cp,cfg=load_config(a.config);seed_everything(int(cfg['training']['seed']));device=default_device(a.device);records=load_prepared(output_path(cp,cfg,'prepared_cache'));indices=selected_audio_indices(records,a.scope);data=AudioDomainDataset(records,indices,config_path=cp,cfg=cfg);p=paths(cp,cfg,a.scope)
 if a.scope=='fit' and any(records.roles[i]!='fit' for i in indices):raise RuntimeError('fit-only audio adaptation attempted held-out WAV access')
 enc=fit_encodec(cp,cfg,data,p['encodec'],device,a);voc=fit_hifigan(cp,cfg,data,p['vocoder'],device,a);spk=_fit_speaker(cp,cfg,data,p['speaker'],device,a)
 write_json(p['encodec_manifest'],enc);write_json(p['vocoder_manifest'],voc);write_json(p['speaker_manifest'],spk)
 threshold=cfg['audio_adaptation'];checks={'fit_only':a.scope!='fit' or all(records.roles[i]=='fit' for i in indices),'encodec_all_trainable':enc['all_pretrained_generator_parameters_trainable'],'encodec_encoder_changed':enc['component_change']['encoder']['changed_parameter_fraction']>0,'encodec_quantizer_changed':enc['component_change']['quantizer_ema_buffers']['changed_buffer_fraction']>0,'encodec_decoder_changed':enc['component_change']['decoder']['changed_parameter_fraction']>0,'encodec_changed':enc['pretrained_parameter_change']['changed_parameter_fraction']>=float(threshold['min_changed_parameter_fraction']),'encodec_improved':enc['relative_loss_improvement']>=float(threshold['min_relative_loss_improvement']),'hifigan_changed':voc['pretrained_parameter_change']['changed_parameter_fraction']>=float(threshold['min_changed_parameter_fraction']),'hifigan_improved':voc['relative_loss_improvement']>=float(threshold['min_relative_loss_improvement']),'ecapa_changed':spk['pretrained_parameter_change']['changed_parameter_fraction']>=float(threshold['min_changed_parameter_fraction']),'ecapa_improved':spk['relative_loss_improvement']>=float(threshold['min_relative_loss_improvement'])}
 gate={'schema_version':'openvoice-v3-encodec-clip-adaptation-v1','scope':a.scope,'corpus_size':len(indices),'heldout_eeg_claims_allowed':a.scope=='fit','exploratory_gate_bypass':bool(a.explore),'components':{'encodec':enc,'hifigan':voc,'ecapa':spk},'checks':checks,'passed':bool(all(checks.values()))};write_json(p['gate'],gate);print(f"[v3 adaptation] scope={a.scope} n={len(indices)} passed={gate['passed']}",flush=True)
 if not gate['passed'] and not a.explore:raise SystemExit(2)
if __name__=='__main__':main()
