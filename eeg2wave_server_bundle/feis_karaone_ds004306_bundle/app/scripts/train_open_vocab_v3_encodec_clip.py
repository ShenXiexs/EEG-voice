#!/usr/bin/env python3
"""Train the v3 EnCodec-token content chain, not direct EEG→MFCC."""
from __future__ import annotations
import argparse, hashlib, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
APP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(APP)) if str(APP) not in sys.path else None
from src.open_vocab_v3.data import V3Dataset,collate,load_prepared
from src.open_vocab_v3.encodec_content import AudioContentEncoder,EEGContentEncoder,SharedMFCCDecoder,SCHEMA
from src.open_vocab_v3.metrics import (audio_content_encoder_loss, clip_token_global_losses,
    cvae_audio_loss, delta_l1, mfcc_l1, shared_mfcc_decoder_loss,
    temporal_cosine_loss)
from src.open_vocab_v3.model import NativeSpeechT5MFCCMelCVAE
from src.open_vocab_v3.runtime import checkpoint_path,checkpoint_schema,content_schema,default_device,load_config,move_batch,output_path,require_passed_gate,seed_everything,sha256_file,write_json

def parse():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--phase',choices=('audio_content','cvae','micro','fit'),required=True);p.add_argument('--device',default='auto');p.add_argument('--deadline-epoch',type=float,default=0);p.add_argument('--smoke-steps',type=int,default=0);p.add_argument('--fresh',action='store_true');p.add_argument('--explore',action='store_true',help='continue after failed prerequisite gates; checkpoints are exploratory only');return p.parse_args()
def dead(a):return bool(a.deadline_epoch and time.time()>=a.deadline_epoch)
def save(path,schema,modules,**extra):path.parent.mkdir(parents=True,exist_ok=True);torch.save({'schema_version':schema,'modules':{k:v.state_dict() for k,v in modules.items()},'extra':extra},path)
def load(path,schema,modules,device):
 raw=torch.load(path,map_location=device,weights_only=False)
 if raw.get('schema_version')!=schema:raise ValueError(f'stale/incorrect checkpoint {path}: {raw.get("schema_version")}')
 for k,v in modules.items():v.load_state_dict(raw['modules'][k],strict=True)
 return raw
def text_anchor(labels,dim,device):
 out=[]
 for label in labels:
  digest=hashlib.sha256(str(label).strip().lower().encode()).digest();base=torch.tensor(list(digest),device=device,dtype=torch.float32)/127.5-1;out.append(base.repeat(math.ceil(dim/base.numel()))[:dim])
 return torch.stack(out)
def attach_codes(records,cp,cfg):
 raw=np.load(output_path(cp,cfg,'encodec_cache'),allow_pickle=False)
 if str(raw['schema'].item())!=content_schema(cfg) or str(raw['source_prepared_sha256'].item())!=sha256_file(output_path(cp,cfg,'prepared_cache')):raise ValueError('stale EnCodec cache rejected; rebuild after current preparation')
 mapping={int(i):j for j,i in enumerate(raw['source_indices'].tolist())};return raw,mapping
class TokenDataset(Dataset):
 def __init__(self,base,cache,mapping):self.base,self.cache,self.mapping=base,cache,mapping
 def __len__(self):return len(self.base)
 def __getitem__(self,i):
  out=self.base[i]
  if 'source_index' not in out:raise RuntimeError('v3 dataset item lacks immutable source_index')
  source=int(out['source_index'])
  if source not in self.mapping:raise RuntimeError('attempted to access held-out / absent EnCodec token before approval')
  j=self.mapping[source];out['encodec_codes']=self.cache['encodec_codes'][j];out['encodec_mask']=self.cache['encodec_mask'][j];return out
def token_collate(items):
 out=collate(items);out['encodec_codes']=torch.as_tensor(np.stack([x['encodec_codes'] for x in items])).long();out['encodec_mask']=torch.as_tensor(np.stack([x['encodec_mask'] for x in items])).bool();return out
def loader(ds,cfg,train,kind='audio'):
 if kind not in {'audio','eeg'}:raise ValueError(f'unknown v3 loader kind: {kind}')
 batch_key='audio_batch_size' if kind=='audio' else 'eeg_batch_size'
 batch_size=int(cfg['training'][batch_key] if train else cfg['evaluation']['batch_size'])
 # Fit subjects and labels are imbalanced.  Sampling inverse (subject,label)
 # frequencies prevents the full-fit objective from becoming a few-subject
 # word-template classifier; metadata still never enters model.forward.
 if train and bool(cfg.get('experiment',{}).get('require_fresh_audio_adaptation',False)):
  pairs=[(str(ds[i]['subject']),str(ds[i]['label'])) for i in range(len(ds))];counts={pair:pairs.count(pair) for pair in set(pairs)};weights=torch.as_tensor([1.0/counts[pair] for pair in pairs],dtype=torch.double);sampler=WeightedRandomSampler(weights,num_samples=len(weights),replacement=True)
  return DataLoader(ds,batch_size=batch_size,sampler=sampler,collate_fn=token_collate,num_workers=0)
 return DataLoader(ds,batch_size=batch_size,shuffle=train,collate_fn=token_collate,num_workers=0)
def modules(cfg,device):
 d=int(cfg['model']['content_dimension']);kw=dict(dimension=d,heads=int(cfg['model']['heads']),layers=int(cfg['model']['content_layers']),dropout=float(cfg['model']['dropout']))
 return AudioContentEncoder(codebooks=8,vocabulary=1024,tokens=32,**kw).to(device),SharedMFCCDecoder(dimension=d,token_steps=32,frames=256).to(device),EEGContentEncoder(tokens=32,**kw).to(device)
class Reverse(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x):return x
 @staticmethod
 def backward(ctx,g):return -g

def _fivefold_centroid_probe(values,targets):
 values=np.asarray(values,dtype=np.float32);targets=np.asarray(targets,dtype=str);pred=[];truth=[]
 for fold in range(5):
  test=np.arange(len(targets))%5==fold;train=~test
  centers={name:values[train&(targets==name)].mean(0) for name in sorted(set(targets)) if np.any(train&(targets==name))}
  for row,name in zip(values[test],targets[test]):pred.append(min(centers,key=lambda key:float(np.sum((row-centers[key])**2))));truth.append(name)
 return float(np.mean(np.asarray(pred)==np.asarray(truth)))

@torch.no_grad()
def _token_probe(audio,ds,cfg,device):
 vectors=[];labels=[];subjects=[]
 for batch in DataLoader(ds,batch_size=int(cfg['evaluation']['batch_size']),shuffle=False,collate_fn=token_collate,num_workers=0):
  batch=move_batch(batch,device);vectors.append(audio(batch['encodec_codes'],batch['encodec_mask']).mean(1).cpu().numpy());labels+=batch['label'];subjects+=batch['subject']
 values=np.concatenate(vectors)
 return {'n':len(labels),'label_probe_5fold':_fivefold_centroid_probe(values,labels),'speaker_probe_5fold':_fivefold_centroid_probe(values,subjects)}

def train_audio_content(cp,cfg,records,device,a):
 cache,map_=attach_codes(records,cp,cfg);base=V3Dataset(records,('fit',),eligible_only=True);ds=TokenDataset(base,cache,map_);audio,decoder,_=modules(cfg,device);subjects=sorted(set(records.arrays['subjects'][base.indices].astype(str)));labels=sorted(set(records.arrays['labels'][base.indices].astype(str)));sid={x:i for i,x in enumerate(subjects)};lid={x:i for i,x in enumerate(labels)};dimension=int(cfg['model']['content_dimension']);teacher_dim=int(cfg['model'].get('hubert_teacher_dimension',768));teacher=nn.Sequential(nn.LayerNorm(teacher_dim),nn.Linear(teacher_dim,dimension)).to(device);label_head=nn.Linear(dimension,len(labels)).to(device);adv=nn.Linear(dimension,len(subjects)).to(device)
 weights={'hubert':float(cfg['model'].get('hubert_teacher_weight',.25)),'label':float(cfg['model'].get('content_label_weight',.10)),'variance':float(cfg['model'].get('variance_weight',.15)),'covariance':float(cfg['model'].get('covariance_weight',.05)),'diversity':float(cfg['model'].get('diversity_weight',.10)),'global_clip':float(cfg['model'].get('audio_global_clip_weight',.0)),'token_consistency':float(cfg['model'].get('audio_token_consistency_weight',.0))}
 history=[]
 # Stage 1: learn and audit a content-token space before allowing an MFCC
 # decoder to optimize reconstruction.  No label/text is a forward input.
 encoder_epochs=int(cfg['training'].get('audio_content_encoder_epochs',cfg['training']['audio_content_epochs']))
 token_opt=torch.optim.AdamW(list(audio.parameters())+list(teacher.parameters())+list(label_head.parameters())+list(adv.parameters()),lr=float(cfg['training']['audio_lr']),weight_decay=float(cfg['training']['weight_decay']))
 for epoch in range(encoder_epochs):
  values=[]
  for step,b in enumerate(loader(ds,cfg,True,'audio')):
   if dead(a):break
   b=move_batch(b,device);tokens=audio(b['encodec_codes'],b['encodec_mask']);speaker=torch.tensor([sid[x] for x in b['subject']],device=device);label=torch.tensor([lid[x] for x in b['label']],device=device)
   loss,parts=audio_content_encoder_loss(tokens,b['hubert'].float(),b['hubert_mask'],teacher,label_head(tokens.mean(1)),label,adv(Reverse.apply(tokens.mean(1))),speaker,b['label'],weights)
   token_opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(list(audio.parameters())+list(teacher.parameters())+list(label_head.parameters())+list(adv.parameters()),float(cfg['training']['grad_clip']));token_opt.step();values.append(float(loss.detach()))
   if a.smoke_steps and step+1>=a.smoke_steps:break
  if not values:break
  val=float(np.mean(values));history.append({'stage':'token_encoder','epoch':epoch+1,'loss':val,'components':parts});print(f'[v3 audio token] epoch={epoch+1}/{encoder_epochs} loss={val:.5f}',flush=True)
  if a.smoke_steps:break
 if not history or history[-1]['stage']!='token_encoder':raise RuntimeError('audio content encoder had zero steps')
 audio.eval();probe=_token_probe(audio,ds,cfg,device);probe_checks={'label':probe['label_probe_5fold']>=float(cfg['gates']['t1d']['label_probe_min']),'speaker':probe['speaker_probe_5fold']<=float(cfg['gates']['t1d']['speaker_probe_max'])};write_json(output_path(cp,cfg,'token_probe_gate'),{'schema_version':content_schema(cfg),'gate':'T1d_token_encoder_predecoder','metrics':probe,'thresholds':cfg['gates']['t1d'],'checks':probe_checks,'passed':bool(all(probe_checks.values())),'exploratory_gate_bypass':bool(a.explore)});print(f"[v3 audio token probe] label={probe['label_probe_5fold']:.4f} speaker={probe['speaker_probe_5fold']:.4f} passed={all(probe_checks.values())}",flush=True)
 if not all(probe_checks.values()) and not a.explore:raise RuntimeError('audio content token probe failed before MFCC decoder training')
 if dead(a):raise RuntimeError('audio content deadline expired before frozen-decoder stage')
 for parameter in audio.parameters():parameter.requires_grad_(False)
 audio.eval();decoder_opt=torch.optim.AdamW(decoder.parameters(),lr=float(cfg['training']['audio_lr']),weight_decay=float(cfg['training']['weight_decay']));best=math.inf
 decoder_epochs=int(cfg['training'].get('audio_content_decoder_epochs',cfg['training']['audio_content_epochs']))
 for epoch in range(decoder_epochs):
  values=[]
  for step,b in enumerate(loader(ds,cfg,True,'audio')):
   if dead(a):break
   b=move_batch(b,device)
   with torch.no_grad():tokens=audio(b['encodec_codes'],b['encodec_mask'])
   pred=decoder(tokens);loss,parts=shared_mfcc_decoder_loss(pred,b['mfcc'].float(),b['label'],weights)
   decoder_opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(decoder.parameters(),float(cfg['training']['grad_clip']));decoder_opt.step();values.append(float(loss.detach()))
   if a.smoke_steps and step+1>=a.smoke_steps:break
  if not values:break
  val=float(np.mean(values));history.append({'stage':'frozen_token_mfcc_decoder','epoch':epoch+1,'loss':val,'components':parts})
  if val<best:
   best=val;save(output_path(cp,cfg,'audio_content_checkpoint'),checkpoint_schema(cfg,'audio'),{'audio':audio,'decoder':decoder,'teacher':teacher,'label_head':label_head,'adversary':adv},history=history,subjects=subjects,labels=labels,token_encoder_frozen=True,exploratory_gate_bypass=bool(a.explore))
  print(f'[v3 MFCC decoder] epoch={epoch+1}/{decoder_epochs} loss={val:.5f}',flush=True)
  if a.smoke_steps:break
 if not np.isfinite(best):raise RuntimeError('frozen MFCC decoder had zero steps')
 return history
def train_cvae(cp,cfg,records,device,a):
 if not a.explore:require_passed_gate(cp,cfg,'t1_gate')
 base=V3Dataset(records,('fit',),eligible_only=True);model=NativeSpeechT5MFCCMelCVAE(mfcc_bins=40,mel_bins=80,dimension=int(cfg['model']['audio_dimension']),voice_dim=int(cfg['speaker']['embedding_dimension']),latent_dim=int(cfg['model']['audio_latent_dimension']),residual_limit_log10=float(cfg['model']['audio_residual_limit_log10'])).to(device);opt=torch.optim.AdamW(model.parameters(),lr=float(cfg['training']['audio_lr']),weight_decay=float(cfg['training']['weight_decay']));best=math.inf;history=[]
 for epoch in range(int(cfg['training']['audio_epochs'])):
  values=[]
  for step,b in enumerate(DataLoader(base,batch_size=int(cfg['training']['audio_batch_size']),shuffle=True,collate_fn=collate,num_workers=0)):
   if dead(a):break
   b=move_batch(b,device);target=b['speech_t5_mel'].float();target_256=F.interpolate(target,size=256,mode='linear',align_corners=False);v=model.distributions(b['mfcc'].float(),b['canonical_voice'].float(),b['canonical_mfcc_mean'].float(),b['canonical_mfcc_std'].float(),target_256);post=model.decode(v['analytic_mel'],v['content_hidden'],v['voice_hidden'],model._sample(v['posterior_mean'],v['posterior_logvar'],True));prior=model.decode(v['analytic_mel'],v['content_hidden'],v['voice_hidden'],v['prior_mean']);post,prior,analytic=(F.interpolate(x,size=target.shape[-1],mode='linear',align_corners=False) for x in (post,prior,v['analytic_mel']));loss,parts=cvae_audio_loss(post,prior,analytic,target,v['posterior_mean'],v['posterior_logvar'],v['prior_mean'],v['prior_logvar'],kl_beta=float(cfg['training']['cvae_kl_beta_max']),free_bits=float(cfg['training']['cvae_free_bits']),prior_weight=float(cfg['training']['cvae_prior_reconstruction_weight']),analytic_consistency_weight=float(cfg['training']['cvae_analytic_consistency_weight']),mask=b['speech_t5_mel_mask']);target_voice=model.generate(b['mfcc'].float(),b['speaker_reference'].float(),b['speaker_reference_mfcc_mean'].float(),b['speaker_reference_mfcc_std'].float(),stochastic=False)['mel'];target_voice=F.interpolate(target_voice,size=target.shape[-1],mode='linear',align_corners=False);speaker_loss=F.smooth_l1_loss(target_voice,target);rolled=torch.roll(b['mfcc'].float(),1,0);swapped=model.generate(rolled,b['canonical_voice'].float(),b['canonical_mfcc_mean'].float(),b['canonical_mfcc_std'].float(),stochastic=False)['mel'];content_floor=F.relu(.5*F.l1_loss(v['analytic_mel'],torch.roll(v['analytic_mel'],1,0))-F.l1_loss(prior, F.interpolate(swapped,size=target.shape[-1],mode='linear',align_corners=False)));loss=loss+float(cfg['model'].get('cvae_speaker_weight',0.0))*speaker_loss+float(cfg['model'].get('cvae_content_weight',0.0))*content_floor;parts|={'speaker':float(speaker_loss.detach()),'content_swap':float(content_floor.detach())};opt.zero_grad();loss.backward();opt.step();values.append(float(loss.detach()))
   if a.smoke_steps and step+1>=a.smoke_steps:break
  if not values:break
  val=float(np.mean(values));history.append({'epoch':epoch+1,'loss':val,'components':parts})
  if val<best:best=val;save(output_path(cp,cfg,'cvae_checkpoint'),checkpoint_schema(cfg,'cvae'),{'cvae':model},history=history,native_contract=cfg['vocoder']['native_contract'],exploratory_gate_bypass=bool(a.explore))
  print(f'[v3 CVAE] epoch={epoch+1} loss={val:.5f}',flush=True)
  if a.smoke_steps:break
 if not history:raise RuntimeError('CVAE had zero steps')
 return history
def micro_subset(records,cfg):
 src=V3Dataset(records,('fit',),eligible_only=True);sub=str(cfg['micro_gate']['subject']);per=int(cfg['micro_gate']['per_label']);by={}
 for pos,index in enumerate(src.indices):
  if str(records.arrays['subjects'][index])==sub:by.setdefault(str(records.arrays['labels'][index]),[]).append(pos)
 selected=[]
 for label in sorted(by):selected.extend(sorted(by[label],key=lambda p:str(records.arrays['sample_keys'][src.indices[p]]))[:per])
 if len(selected)!=50:raise RuntimeError(f'MM05 micro set must be exactly 50 pairs, got {len(selected)}')
 return Subset(src,selected)
def train_eeg(cp,cfg,records,device,a,phase):
 if not a.explore:
  require_passed_gate(cp,cfg,'t3_gate')
  if phase=='fit':require_passed_gate(cp,cfg,'micro_gate',lineage_artifact_keys=('micro_checkpoint',))
 cache,map_=attach_codes(records,cp,cfg);base=micro_subset(records,cfg) if phase=='micro' else V3Dataset(records,('fit',),eligible_only=True);ds=TokenDataset(base,cache,map_);audio,decoder,eeg=modules(cfg,device);load(output_path(cp,cfg,'audio_content_checkpoint'),checkpoint_schema(cfg,'audio'),{'audio':audio,'decoder':decoder},device);audio.eval();decoder.eval()
 for x in list(audio.parameters())+list(decoder.parameters()):x.requires_grad_(False)
 opt=torch.optim.AdamW(eeg.parameters(),lr=float(cfg['training']['eeg_lr']),weight_decay=float(cfg['training']['weight_decay']));epochs=int(cfg['training']['micro_epochs'] if phase=='micro' else cfg['training']['fit_epochs']);history=[];best=math.inf;stale=0;warmup=max(int(cfg['training'].get('warmup_epochs',0)),1);base_lr=float(cfg['training']['eeg_lr'])
 for epoch in range(epochs):
  for group in opt.param_groups:group['lr']=base_lr*min(1.0,float(epoch+1)/warmup)
  values=[]
  for step,b in enumerate(loader(ds,cfg,True,'eeg')):
   if dead(a):break
   b=move_batch(b,device);tok=eeg(b['eeg'].float(),b['channel_xyz'].float(),b['channel_mask'],b['time_mask']);pred=decoder(tok);l1=mfcc_l1(pred,b['mfcc'].float());delta=delta_l1(pred,b['mfcc'].float());token,global_=clip_token_global_losses(tok,audio(b['encodec_codes'],b['encodec_mask']).detach(),b['label'],eeg.clip_logit_scale)
   if phase=='micro':loss=.55*l1+.20*delta+.15*token+.10*global_
   else:loss=.50*l1+.20*delta+.15*token+.15*global_
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(eeg.parameters(),float(cfg['training']['grad_clip']));opt.step();values.append(float(loss.detach()))
   if a.smoke_steps and step+1>=a.smoke_steps:break
  if not values:break
  val=float(np.mean(values));history.append({'epoch':epoch+1,'loss':val})
  if val<best:best=val;stale=0;save(output_path(cp,cfg,f'{phase}_checkpoint'),checkpoint_schema(cfg,phase),{'eeg':eeg},history=history,audio_checkpoint_sha256=sha256_file(output_path(cp,cfg,'audio_content_checkpoint')),exploratory_gate_bypass=bool(a.explore))
  else:stale+=1
  print(f'[v3 {phase}] epoch={epoch+1} loss={val:.5f}',flush=True)
  if phase=='fit' and stale>=int(cfg['training'].get('fit_patience',epochs)):break
  if a.smoke_steps:break
 if not history:raise RuntimeError(f'{phase} had zero steps')
def main():
 a=parse();cp,cfg=load_config(a.config);seed_everything(int(cfg['training']['seed']));records=load_prepared(output_path(cp,cfg,'prepared_cache'));device=default_device(a.device)
 if a.phase=='audio_content':train_audio_content(cp,cfg,records,device,a)
 elif a.phase=='cvae':train_cvae(cp,cfg,records,device,a)
 else:train_eeg(cp,cfg,records,device,a,a.phase)
if __name__=='__main__':main()
