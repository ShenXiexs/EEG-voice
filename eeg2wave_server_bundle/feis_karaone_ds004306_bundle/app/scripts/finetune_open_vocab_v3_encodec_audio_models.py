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
from src.open_vocab_v3.audio_adaptation import parameter_change, tensor_state, selected_audio_indices, multi_resolution_stft_loss, envelope_loss, si_sdr_loss
from src.open_vocab_v3.data import load_prepared
from src.open_vocab_v3.encodec_content import _resample
from src.open_vocab_v3.native_mel import native_speecht5_mel
from src.open_vocab_v3.runtime import content_schema, default_device, load_config, output_path, read_json, seed_everything, sha256_file, write_json
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
 # HF stores codebooks as buffers. Promote them before building AdamW so the
 # manifest can prove that the actual codebooks, not only the decoder, moved.
 promoted=[]
 for index,layer in enumerate(model.quantizer.layers):
  book=layer.codebook
  if not isinstance(book.embed,torch.nn.Parameter):
   initial=book._buffers.pop('embed');book.register_parameter('embed',torch.nn.Parameter(initial.detach().clone()))
  promoted.append(f'quantizer.layers.{index}.codebook.embed')
 for x in model.parameters():x.requires_grad_(True)
 before=tensor_state(model);before_encoder=tensor_state(model.encoder);before_decoder=tensor_state(model.decoder);before_quantizer=tensor_state(model.quantizer);opt=torch.optim.AdamW(model.parameters(),lr=float(cfg['audio_adaptation']['encodec_lr']),weight_decay=float(cfg['training']['weight_decay']))
 optimizer_names={name for name,value in model.named_parameters() if any(value is item for group in opt.param_groups for item in group['params'])}
 loader=DataLoader(data,batch_size=int(cfg['audio_adaptation']['encodec_batch_size']),shuffle=True,collate_fn=audio_collate,num_workers=0);first=best=None;history=[]
 for epoch in range(int(cfg['audio_adaptation']['encodec_epochs'])):
  if deadline(a):break
  values=[]
  for step,b in enumerate(loader):
   if deadline(a):break
   wave=_resample(b['waveform'].to(device),16000,24000).unsqueeze(1)
   # A hard EnCodec index has no gradient to the encoder.  This straight-
   # through path supplies it, while the commitment term updates codebooks.
   scale=None
   if model.config.normalize:
    scale=wave.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(1.e-8);input_wave=wave/scale
   else:input_wave=wave
   embedding=model.encoder(input_wave);codes=model.quantizer.encode(embedding,bandwidth=float(cfg['audio']['encodec_bandwidth']));quantized=model.quantizer.decode(codes)
   pred=model.decoder(embedding+(quantized-embedding).detach())
   if scale is not None:pred=pred*scale
   target=wave[...,:pred.shape[-1]];commit=F.mse_loss(quantized,embedding.detach())
   loss=float(cfg['audio_adaptation'].get('waveform_l1_weight',1.0))*F.l1_loss(pred,target)+float(cfg['audio_adaptation'].get('stft_weight',.25))*multi_resolution_stft_loss(pred,target,fft_sizes=cfg['audio_adaptation']['stft_fft_sizes'],hop_sizes=cfg['audio_adaptation']['stft_hop_sizes'])+float(cfg['audio_adaptation'].get('si_sdr_weight',0.0))*si_sdr_loss(pred,target)+float(cfg['audio_adaptation'].get('envelope_weight',0.0))*envelope_loss(pred,target)+float(cfg['audio_adaptation'].get('quantizer_adapter_weight',.05))*commit
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['training']['grad_clip']));opt.step();values.append(float(loss.detach()))
   if a.smoke_steps and step+1>=a.smoke_steps:break
  if not values:break
  val=float(np.mean(values));first=val if first is None else first;history.append({'epoch':epoch+1,'loss':val})
  if best is None or val<best:
   best=val;dst.mkdir(parents=True,exist_ok=True);model.save_pretrained(dst)
  if a.smoke_steps:break
 if first is None:raise RuntimeError('EnCodec adaptation performed zero optimizer steps')
 selected=EncodecModel.from_pretrained(str(dst),local_files_only=True)
 for layer in selected.quantizer.layers:
  book=layer.codebook
  if not isinstance(book.embed,torch.nn.Parameter):
   restored=book._buffers.pop('embed');book.register_parameter('embed',torch.nn.Parameter(restored))
 change=parameter_change(before,selected);groups={'encoder':parameter_change(before_encoder,selected.encoder),'quantizer_codebooks':parameter_change(before_quantizer,selected.quantizer),'decoder':parameter_change(before_decoder,selected.decoder)}
 return {'component':'Encodec_encoder_quantizer_decoder','base_root':str(base),'adapted_root':str(dst),'all_pretrained_generator_parameters_trainable':all(x.requires_grad for x in model.parameters()),'quantizer_embed_promoted_to_parameter':True,'quantizer_adapter_parameters':promoted,'optimizer_parameter_count':sum(x.numel() for group in opt.param_groups for x in group['params']),'optimizer_contains_quantizer_adapter':all(name in optimizer_names for name in promoted),'component_change':groups,'epochs_completed':len(history),'first_epoch_loss':first,'best_loss':best,'relative_loss_improvement':float((first-best)/max(abs(first),1e-8)),'pretrained_parameter_change':change,'history':history}
def recover_encodec_for_explore(cp,cfg,dst):
 """Recover a valid best checkpoint saved before a downstream crash.

 This is deliberately explore-only: loss history was held in memory when the
 old process died, so improvement is recorded as unverified (0.0) and the A0
 improvement check remains false rather than being fabricated.
 """
 from transformers import EncodecModel
 base_root=output_path(cp,cfg,'encodec_root');base=EncodecModel.from_pretrained(str(base_root),local_files_only=True);selected=EncodecModel.from_pretrained(str(dst),local_files_only=True)
 before=tensor_state(base);before_encoder=tensor_state(base.encoder);before_decoder=tensor_state(base.decoder);before_quantizer=buffer_state(base.quantizer)
 change=parameter_change(before,selected);groups={'encoder':parameter_change(before_encoder,selected.encoder),'quantizer_ema_buffers':buffer_change(before_quantizer,selected.quantizer),'decoder':parameter_change(before_decoder,selected.decoder)}
 return {'component':'Encodec_encoder_quantizer_decoder','base_root':str(base_root),'adapted_root':str(dst),'all_pretrained_generator_parameters_trainable':all(x.requires_grad for x in selected.parameters()),'quantizer_note':'recovered after downstream crash; quantizer EMA buffers compared with frozen base','component_change':groups,'epochs_completed':None,'first_epoch_loss':None,'best_loss':None,'relative_loss_improvement':0.0,'pretrained_parameter_change':change,'history':[],'recovered_exploratory_checkpoint':True,'improvement_unverified':True}
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
 if a.explore and p['encodec_manifest'].is_file() and p['encodec'].is_dir():
  enc=read_json(p['encodec_manifest'])
  if bool(cfg.get('experiment',{}).get('require_fresh_audio_adaptation',False)) and (enc.get('recovered_exploratory_checkpoint') or not enc.get('history') or not enc.get('quantizer_embed_promoted_to_parameter',False)):
   raise RuntimeError('content-repair v3 rejects recovered/incomplete EnCodec adaptation; use a fresh artifact root')
  print('[v3 adaptation] resume completed EnCodec manifest',flush=True)
 elif a.explore and (p['encodec']/'config.json').is_file() and (p['encodec']/'model.safetensors').is_file():
  # A bare directory has no optimizer history.  Recovery was useful for the
  # old exploratory namespace after a process crash, but is categorically
  # invalid for the repaired protocol: it cannot establish that encoder,
  # decoder and codebook were jointly adapted from the frozen checkpoint.
  if bool(cfg.get('experiment',{}).get('require_fresh_audio_adaptation',False)):
   raise RuntimeError('content-repair v3 rejects an EnCodec directory without a fresh complete manifest; choose a new artifact root or rerun adaptation from the frozen model')
  enc=recover_encodec_for_explore(cp,cfg,p['encodec']);write_json(p['encodec_manifest'],enc);print('[v3 adaptation] recovered completed EnCodec checkpoint; improvement remains unverified',flush=True)
 else:enc=fit_encodec(cp,cfg,data,p['encodec'],device,a);write_json(p['encodec_manifest'],enc)
 if a.explore and p['vocoder_manifest'].is_file() and p['vocoder'].is_dir():voc=read_json(p['vocoder_manifest']);print('[v3 adaptation] resume completed HiFi-GAN manifest',flush=True)
 else:voc=fit_hifigan(cp,cfg,data,p['vocoder'],device,a);write_json(p['vocoder_manifest'],voc)
 if a.explore and p['speaker_manifest'].is_file() and p['speaker'].is_file():spk=read_json(p['speaker_manifest']);print('[v3 adaptation] resume completed ECAPA manifest',flush=True)
 else:spk=_fit_speaker(cp,cfg,data,p['speaker'],device,a);write_json(p['speaker_manifest'],spk)
 threshold=cfg['audio_adaptation'];quantizer=enc['component_change'].get('quantizer_codebooks',enc['component_change'].get('quantizer_ema_buffers',{}));checks={'fit_only':a.scope!='fit' or all(records.roles[i]=='fit' for i in indices),'encodec_all_trainable':enc['all_pretrained_generator_parameters_trainable'],'encodec_encoder_changed':enc['component_change']['encoder']['changed_parameter_fraction']>0,'encodec_quantizer_changed':quantizer.get('changed_parameter_fraction',quantizer.get('changed_buffer_fraction',0))>0,'encodec_quantizer_trainable':bool(enc.get('quantizer_embed_promoted_to_parameter',False)) and bool(enc.get('optimizer_contains_quantizer_adapter',False)),'encodec_decoder_changed':enc['component_change']['decoder']['changed_parameter_fraction']>0,'encodec_changed':enc['pretrained_parameter_change']['changed_parameter_fraction']>=float(threshold['min_changed_parameter_fraction']),'encodec_improved':enc['relative_loss_improvement']>=float(threshold['min_relative_loss_improvement']),'hifigan_changed':voc['pretrained_parameter_change']['changed_parameter_fraction']>=float(threshold['min_changed_parameter_fraction']),'hifigan_improved':voc['relative_loss_improvement']>=float(threshold['min_relative_loss_improvement']),'ecapa_changed':spk['pretrained_parameter_change']['changed_parameter_fraction']>=float(threshold['min_changed_parameter_fraction']),'ecapa_improved':spk['relative_loss_improvement']>=float(threshold['min_relative_loss_improvement'])}
 gate={'schema_version':content_schema(cfg),'scope':a.scope,'corpus_size':len(indices),'heldout_eeg_claims_allowed':a.scope=='fit','exploratory_gate_bypass':bool(a.explore),'components':{'encodec':enc,'hifigan':voc,'ecapa':spk},'checks':checks,'passed':bool(all(checks.values()))};write_json(p['gate'],gate);print(f"[v3 adaptation] scope={a.scope} n={len(indices)} passed={gate['passed']}",flush=True)
 if not gate['passed'] and not a.explore:raise SystemExit(2)
if __name__=='__main__':main()
