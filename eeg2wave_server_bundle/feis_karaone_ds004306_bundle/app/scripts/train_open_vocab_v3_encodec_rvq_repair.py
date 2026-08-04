#!/usr/bin/env python3
"""Fit-only sequential-RVQ, Audio-C, and EEG-C stages for repair-v3.

This executable intentionally exposes no validation/test phase.  The strict
runner may only reach M1 and then waits for explicit human WAV review.
"""
from __future__ import annotations
import argparse, math, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset,Subset

APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:sys.path.insert(0,str(APP))
from src.open_vocab_v3.data import V3Dataset,collate,load_prepared,time_shuffled_eeg,channel_shuffled_eeg
from src.open_vocab_v3.encodec_rvq_repair import PREPARATION_SCHEMA,SCHEMA,AudioCTeacher,ContentMFCCDecoder,DirectEEGMFCC,SequentialRVQBridge,diagonal_band_infonce,masked_l1,soft_dtw_token_clip,temporal_delta
from src.open_vocab_v3.runtime import checkpoint_schema,default_device,load_config,move_batch,output_path,seed_everything,sha256_file

def parse():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--phase',choices=('rvq_micro','rvq','audio_c','m0a','m0b','m1'),required=True);p.add_argument('--device',default='auto');p.add_argument('--deadline-epoch',type=float,default=0);p.add_argument('--smoke-steps',type=int,default=0);p.add_argument('--fresh',action='store_true');return p.parse_args()
def expired(a):return bool(a.deadline_epoch and time.time()>=a.deadline_epoch)
def save_checkpoint(path,schema,modules,**extra):
 path.parent.mkdir(parents=True,exist_ok=True);torch.save({'schema_version':schema,'modules':{k:v.state_dict() for k,v in modules.items()},'extra':extra},path)
def load_checkpoint(path,schema,modules,device):
 raw=torch.load(path,map_location=device,weights_only=False)
 if raw.get('schema_version')!=schema:raise RuntimeError(f'stale/non-repair checkpoint rejected: {path}')
 for k,v in modules.items():v.load_state_dict(raw['modules'][k],strict=True)
 return raw

class TokenDataset(Dataset):
 def __init__(self,base,cache,mapping):self.base,self.cache,self.mapping=base,cache,mapping
 def __len__(self):return len(self.base)
 def __getitem__(self,i):
  out=dict(self.base[i]);row=self.mapping.get(int(out['source_index']))
  if row is None:raise RuntimeError('non-fit source was requested from fit-only RVQ cache')
  for k in ('encodec_codes','encodec_mask','audio_scales','waveform_16k','waveform_mask','waveform_samples'):out[k]=self.cache[k][row]
  return out
def token_collate(items):
 out=collate(items)
 for k in ('encodec_codes','encodec_mask','audio_scales','waveform_16k','waveform_mask','waveform_samples'):out[k]=torch.as_tensor(np.stack([x[k] for x in items]))
 out['encodec_codes']=out['encodec_codes'].long();out['encodec_mask']=out['encodec_mask'].bool();out['waveform_mask']=out['waveform_mask'].bool();return out
def load_cache(cp,cfg):
 raw=np.load(output_path(cp,cfg,'encodec_cache'),allow_pickle=False)
 if str(raw['schema'].item())!=SCHEMA:raise RuntimeError('stale bridge-v2/other cache rejected; rebuild repair-v3 cache')
 if str(raw['prepared_cache_sha256'].item())!=sha256_file(output_path(cp,cfg,'prepared_cache')):raise RuntimeError('repair cache lineage mismatch')
 c={k:np.asarray(raw[k]) for k in raw.files};return c,{int(x):i for i,x in enumerate(c['source_indices'])}
def fit_indices(records,dev=None):
 mask=(records.roles=='fit')&records.arrays['fit_eligible'].astype(bool)
 if dev is not None:mask &= records.arrays['fit_internal_dev'].astype(bool) if dev else ~records.arrays['fit_internal_dev'].astype(bool)
 return np.flatnonzero(mask)
def base_subset(records,indices):
 base=V3Dataset(records,('fit',),eligible_only=True);pos={int(x):i for i,x in enumerate(base.indices)};return Subset(base,[pos[int(x)] for x in indices])
def micro_indices(records,cfg):
 fit=fit_indices(records,False);sub=str(cfg['micro_gate']['subject']);per=int(cfg['micro_gate']['per_label']);out=[]
 for label in sorted(set(records.arrays['labels'][fit].astype(str))):
  choices=sorted([int(x) for x in fit if str(records.arrays['subjects'][x])==sub and str(records.arrays['labels'][x])==label],key=lambda x:str(records.arrays['sample_keys'][x]));out+=choices[:per]
 if len(out)!=50:raise RuntimeError(f'M0/M1 requires 50 MM05 records, got {len(out)}')
 return np.asarray(out,np.int32)
def folds(indices,keys,labels):
 groups={}
 for i in indices.tolist():groups.setdefault(str(labels[i]),[]).append(int(i))
 if len(groups)!=10 or any(len(x)!=5 for x in groups.values()):raise RuntimeError('M1 needs 10 labels × 5 trials')
 groups={k:sorted(v,key=lambda x:str(keys[x])) for k,v in groups.items()};allv=[i for k in sorted(groups) for i in groups[k]];out=[]
 for fold in range(5):
  held=np.asarray([groups[k][fold] for k in sorted(groups)],np.int32);train=np.asarray([i for i in allv if i not in set(held.tolist())],np.int32)
  # Inner development is disjoint from the outer-held EEG trials and only
  # selects a checkpoint inside the forty-trial fold.
  inner=np.asarray([sorted([i for i in train if str(labels[i])==k],key=lambda x:str(keys[x]))[0] for k in sorted(groups)],np.int32)
  actual=np.asarray([i for i in train if i not in set(inner.tolist())],np.int32);out.append((actual,inner,held))
 return out
class Grouped(torch.utils.data.Sampler):
 def __init__(self,dataset,batch,seed):self.dataset,self.batch,self.seed=dataset,batch,seed
 def __iter__(self):
  rng=np.random.default_rng(self.seed);groups={}
  for i in range(len(self.dataset)):groups.setdefault(str(self.dataset[i]['label']),[]).append(i)
  labels=sorted(groups);n=max(1,math.ceil(len(self.dataset)/self.batch));orders={k:rng.permutation(v).tolist() for k,v in groups.items()};ptr={k:0 for k in labels}
  for step in range(n):
   chosen=[]
   for k in [labels[(step+j)%len(labels)] for j in range(max(2,self.batch//2))]:
    for _ in range(2):chosen.append(orders[k][ptr[k]%len(orders[k])]);ptr[k]+=1
   yield chosen[:self.batch]
 def __len__(self):return max(1,math.ceil(len(self.dataset)/self.batch))
def loader(ds,cfg,train=False,grouped=False):
 if train:
  size=int(cfg['training']['audio_batch_size'] if grouped else cfg['training']['eeg_batch_size']);sampler=Grouped(ds,size,int(cfg['training']['seed'])) if grouped else None
  return DataLoader(ds,batch_sampler=sampler,collate_fn=token_collate,num_workers=0) if sampler else DataLoader(ds,batch_size=size,shuffle=True,collate_fn=token_collate,num_workers=0)
 return DataLoader(ds,batch_size=int(cfg['evaluation']['batch_size']),shuffle=False,collate_fn=token_collate,num_workers=0)
def make_models(cfg,device):
 m=cfg['model'];dim=int(m['content_dimension']);heads=int(m['heads']);drop=float(m['dropout'])
 return (SequentialRVQBridge(voice_dimension=int(cfg['speaker']['embedding_dimension']),dimension=int(m['rvq_dimension'])).to(device),AudioCTeacher(dimension=dim,heads=heads,layers=int(m['audio_c_layers']),dropout=drop).to(device),ContentMFCCDecoder(dimension=dim,heads=heads,layers=int(m['decoder_layers']),dropout=drop).to(device),DirectEEGMFCC(dimension=int(m['eeg_dimension']),heads=heads,layers=int(m['eeg_layers']),dropout=drop).to(device))
def train_loop(modules,opt,train,dev,loss_fn,epochs,patience,checkpoint,schema,args,device,label):
 best,stale=math.inf,0
 for epoch in range(int(epochs)):
  if expired(args):break
  for m in modules.values():m.train()
  vals=[]
  for step,b in enumerate(train):
   b=move_batch(b,device);loss,parts=loss_fn(b,epoch)
   if not torch.isfinite(loss):raise RuntimeError(f'non-finite {label} loss')
   opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_([p for m in modules.values() for p in m.parameters() if p.requires_grad],float(1));opt.step();vals.append(float(loss.detach()))
   if args.smoke_steps and step+1>=args.smoke_steps:break
  if not vals:break
  if dev:
   for m in modules.values():m.eval();dv=[]
   with torch.no_grad():
    for b in dev:
     x,_=loss_fn(move_batch(b,device),epoch);dv.append(float(x))
   score=float(np.mean(dv)) if dv else math.inf
  else:score=float(np.mean(vals))
  if score<best:best,stale=score,0;save_checkpoint(checkpoint,schema,modules,best_dev=best,epoch=epoch+1)
  else:stale+=1
  print(f'[v3 rvq {label}] epoch={epoch+1}/{epochs} train={np.mean(vals):.5f} dev={score:.5f}',flush=True)
  if stale>=int(patience) or args.smoke_steps:break
def frozen_codebook_embeddings(cp,cfg,device):
 from transformers import EncodecModel
 model=EncodecModel.from_pretrained(str(output_path(cp,cfg,'encodec_root')),local_files_only=True).eval();count=int(model.quantizer.get_num_quantizers_for_bandwidth(float(cfg['audio']['encodec_bandwidth'])));value=torch.stack([model.quantizer.layers[q].codebook.embed.detach().float().cpu() for q in range(count)]).to(device);del model;return value
def rvq_loss(bridge,batch,cfg,epoch,codebook_embeddings):
 r=cfg['rvq'];start,end=float(r['teacher_forcing_start']),float(r['teacher_forcing_end']);ratio=min(1,epoch/max(1,int(cfg['training']['rvq_epochs'])-1));tf=start+(end-start)*ratio;tf=1. if epoch<int(r['scheduled_sampling_start_epoch']) else tf
 logits=bridge(batch['content_mfcc'].float(),batch['p_base'].float(),batch['speaker_reference'].float(),batch['duration_fraction'].float(),targets=batch['encodec_codes'],teacher_forcing=tf);target=batch['encodec_codes'];mask=batch['encodec_mask'].float();weights=torch.tensor(r['codebook_ce_weights'],device=logits.device,dtype=logits.dtype);ce=[]
 for q in range(8):
  value=F.cross_entropy(logits[:,q],target[:,q],reduction='none');ce.append((value*mask).sum()/mask.sum().clamp_min(1))
 ce_loss=(torch.stack(ce)*weights).sum()/weights.sum();prob=logits[:,:4].softmax(2);embeddings=codebook_embeddings[:4].to(prob.dtype);expected=torch.einsum('bqvt,qvd->bqtd',prob,embeddings).sum(1);wanted=torch.stack([F.embedding(target[:,q],embeddings[q]) for q in range(4)],1).sum(1);coarse_frame=1-F.cosine_similarity(expected,wanted,dim=-1);coarse=(coarse_frame*mask).sum()/mask.sum().clamp_min(1);adjacent=mask[:,1:]*mask[:,:-1];transition_frame=(temporal_delta(expected.transpose(1,2))-temporal_delta(wanted.transpose(1,2))).abs().mean(1);transition=(transition_frame*adjacent).sum()/adjacent.sum().clamp_min(1);loss=(.8*ce_loss+float(r['coarse_cosine_weight'])*coarse+float(r['transition_weight'])*transition)
 return loss,{'ce':float(ce_loss.detach()),'coarse_cosine':float(coarse.detach()),'transition':float(transition.detach()),'teacher_forcing':tf}
def content_loss(audio,decoder,batch,stage):
 state=audio(batch['encodec_codes'],batch['encodec_mask'],batch['hubert'].float(),batch['hubert_mask']);pred,_=decoder(state.local,state.token_mask);target=batch['content_mfcc'].float();mfcc=F.l1_loss(pred,target);delta=F.l1_loss(temporal_delta(pred),temporal_delta(target));temporal=1-F.cosine_similarity(temporal_delta(pred),temporal_delta(target),dim=1).mean();variance=(pred.std((0,2))-target.std((0,2))).abs().mean();token,_=diagonal_band_infonce(audio.hubert_token(state.local),F.interpolate(batch['hubert'].float().transpose(1,2),size=96,mode='linear',align_corners=False).transpose(1,2),state.token_mask,state.token_mask,labels=batch['label']);weight=batch['hubert_mask'].float().unsqueeze(-1);global_target=F.normalize((batch['hubert'].float()*weight).sum(1)/weight.sum(1).clamp_min(1),dim=-1);global_loss=1-F.cosine_similarity(state.global_embedding,global_target,dim=1).mean();loss=.45*mfcc+.2*delta+.15*temporal+.1*variance
 if stage>=1:loss=loss+.1*token
 if stage>=2:loss=loss+.1*global_loss
 return loss,{'mfcc':float(mfcc.detach()),'delta':float(delta.detach()),'temporal':float(temporal.detach()),'variance':float(variance.detach()),'diagonal_token':float(token.detach()),'global':float(global_loss.detach())}
def eeg_loss(eeg,batch,teacher=None,clip=False):
 pred,local,mask=eeg(batch['eeg'].float(),batch['channel_xyz'].float(),batch['channel_mask'],batch['time_mask']);target=batch['content_mfcc'].float();mfcc=F.l1_loss(pred,target);delta=F.l1_loss(temporal_delta(pred),temporal_delta(target));temporal=1-F.cosine_similarity(temporal_delta(pred),temporal_delta(target),dim=1).mean();loss=.7*mfcc+.2*delta+.1*temporal
 if clip:
  with torch.no_grad():state=teacher(batch['encodec_codes'],batch['encodec_mask'],batch['hubert'].float(),batch['hubert_mask'])
  token,_=soft_dtw_token_clip(local,state.local,mask,state.token_mask);glob=1-F.cosine_similarity(F.normalize(local.mean(1),dim=-1),F.normalize(state.local.mean(1),dim=-1),dim=1).mean();loss=.6*mfcc+.2*delta+.15*token+.05*glob
 return loss,pred,local,mask
def metrics(pred,target,labels):
 p=np.asarray(pred);t=np.asarray(target);d=((p[:,None]-t[None])**2).mean((2,3));names=sorted(set(labels));ld=np.stack([d[:,[i for i,x in enumerate(labels) if x==n]].mean(1) for n in names],1);ranks=[];margins=[];candidate_counts=[]
 for row,label in enumerate(labels):
  candidates=np.asarray([i for i,x in enumerate(labels) if x==label],dtype=int);order=candidates[np.argsort(d[row,candidates],kind='stable')];rank=int(np.flatnonzero(order==row)[0])+1;ranks.append(rank);candidate_counts.append(len(candidates));negative=np.delete(d[row,candidates],np.flatnonzero(candidates==row));margins.append(float(np.mean(negative)-d[row,row]) if len(negative) else 0.0)
 rng=np.random.default_rng(31);margin=np.asarray(margins,np.float32);boot=np.asarray([margin[rng.integers(0,len(margin),len(margin))].mean() for _ in range(1000)]) if len(margin) else np.asarray([0.0]);template=np.stack([t[[i for i,x in enumerate(labels) if x==label]].mean(0) for label in labels]);return {'label_top1':float(np.mean(np.asarray(names)[ld.argmin(1)]==np.asarray(labels))),'paired_r1':float(np.mean(np.asarray(ranks)<=1)),'paired_r5':float(np.mean(np.asarray(ranks)<=5)),'paired_mrr':float(np.mean(1/np.asarray(ranks,dtype=np.float32))),'paired_margin_mean':float(margin.mean()),'paired_margin_ci_low':float(np.percentile(boot,2.5)),'paired_margin_ci_high':float(np.percentile(boot,97.5)),'within_label_chance':float(np.mean(1/np.asarray(candidate_counts,dtype=np.float32))),'template_improvement':float(1-((p-t)**2).mean()/max(((template-t)**2).mean(),1e-8)),'variance_ratio':float(p.var()/max(t.var(),1e-8))}
def write_pred(path,source,labels,pred,target,controls):
 path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,schema=np.asarray(SCHEMA),source_indices=np.asarray(source,np.int32),labels=np.asarray(labels),prediction=np.asarray(pred,np.float32),target=np.asarray(target,np.float32),**{k:np.asarray(v,np.float32) for k,v in controls.items()})
def predict(eeg,ds,cfg,device,clip=False,teacher=None):
 out=[];target=[];source=[];labels=[];controls={'zero':[],'time':[],'channel':[]}
 eeg.eval();
 with torch.no_grad():
  for b in loader(ds,cfg):
   b=move_batch(b,device)
   def reconstruct(signal):return eeg(signal.float(),b['channel_xyz'].float(),b['channel_mask'],b['time_mask'])[0]
   p=reconstruct(b['eeg']);out+=list(p.cpu().numpy());target+=list(b['content_mfcc'].cpu().numpy());source+=b['source_index'].cpu().tolist();labels+=b['label']
   for name,sig in {'zero':torch.zeros_like(b['eeg']),'time':time_shuffled_eeg(b['eeg'],b['time_mask']),'channel':channel_shuffled_eeg(b['eeg'],b['channel_mask'])}.items():
    controls[name]+=list(reconstruct(sig).cpu().numpy())
 return source,labels,out,target,controls
def frozen_teacher(cp,cfg,device):
 _,audio,decoder,_=make_models(cfg,device);load_checkpoint(output_path(cp,cfg,'audio_c_checkpoint'),checkpoint_schema(cfg,'audio_c'),{'audio':audio,'decoder':decoder},device)
 for m in (audio,decoder):m.eval();[x.requires_grad_(False) for x in m.parameters()]
 return audio,decoder
def main():
 a=parse();cp,cfg=load_config(a.config);seed_everything(int(cfg['training']['seed']));device=default_device(a.device);records=load_prepared(output_path(cp,cfg,'prepared_cache'),expected_schema=PREPARATION_SCHEMA);cache,mapping=load_cache(cp,cfg);train=TokenDataset(base_subset(records,fit_indices(records,False)),cache,mapping);dev=TokenDataset(base_subset(records,fit_indices(records,True)),cache,mapping)
 if a.phase in ('rvq_micro','rvq'):
  bridge,_,_,_=make_models(cfg,device);codebook_embeddings=frozen_codebook_embeddings(cp,cfg,device);opt=torch.optim.AdamW(bridge.parameters(),lr=float(cfg['training']['rvq_lr']),weight_decay=float(cfg['training']['weight_decay']))
  if a.phase=='rvq_micro':
   selected=micro_indices(records,cfg);micro=TokenDataset(base_subset(records,selected),cache,mapping);train_loop({'bridge':bridge},opt,loader(micro,cfg,True),None,lambda b,e:rvq_loss(bridge,b,cfg,e,codebook_embeddings),cfg['training']['rvq_epochs'],cfg['training']['rvq_patience'],output_path(cp,cfg,'rvq_micro_checkpoint'),checkpoint_schema(cfg,'rvq_micro'),a,device,'E1a sequential RVQ micro')
  else:train_loop({'bridge':bridge},opt,loader(train,cfg,True),loader(dev,cfg),lambda b,e:rvq_loss(bridge,b,cfg,e,codebook_embeddings),cfg['training']['rvq_epochs'],cfg['training']['rvq_patience'],output_path(cp,cfg,'rvq_bridge_checkpoint'),checkpoint_schema(cfg,'rvq_bridge'),a,device,'E1b sequential RVQ fit-dev')
 elif a.phase=='audio_c':
  _,audio,decoder,_=make_models(cfg,device);total=int(cfg['training']['audio_c_epochs']);third=max(1,total//3);stage_epochs=(third,third,max(1,total-2*third));final_path=output_path(cp,cfg,'audio_c_checkpoint')
  for stage,count in enumerate(stage_epochs):
   if stage==2:
    for module in (audio.q0,audio.q1,audio.hubert,audio.fuse,audio.local,audio.hubert_token,decoder):
     for parameter in module.parameters():parameter.requires_grad_(False)
    audio.position.requires_grad_(False)
    audio.eval();decoder.eval();audio.global_head.train();parameters=[p for p in audio.global_head.parameters() if p.requires_grad];opt=torch.optim.AdamW(parameters,lr=float(cfg['training']['audio_c_lr']),weight_decay=float(cfg['training']['weight_decay']));stage_path=final_path.with_name('stage_2_global_best.pt');train_loop({'global':audio.global_head},opt,loader(train,cfg,True,True),loader(dev,cfg),lambda b,e:content_loss(audio,decoder,b,2),count,cfg['training']['audio_c_patience'],stage_path,checkpoint_schema(cfg,'audio_c'),a,device,'C1 Audio-C stage=2 global-only')
    if not stage_path.is_file():raise RuntimeError('Audio-C stage 2 ended without checkpoint (deadline reached)')
    load_checkpoint(stage_path,checkpoint_schema(cfg,'audio_c'),{'global':audio.global_head},device);save_checkpoint(final_path,checkpoint_schema(cfg,'audio_c'),{'audio':audio,'decoder':decoder},stage='local_frozen_global_projection')
   else:
    parameters=list(audio.parameters())+list(decoder.parameters());opt=torch.optim.AdamW(parameters,lr=float(cfg['training']['audio_c_lr']),weight_decay=float(cfg['training']['weight_decay']));stage_path=final_path.with_name(f'stage_{stage}_best.pt');train_loop({'audio':audio,'decoder':decoder},opt,loader(train,cfg,True,True),loader(dev,cfg),lambda b,e,s=stage:content_loss(audio,decoder,b,s),count,cfg['training']['audio_c_patience'],stage_path,checkpoint_schema(cfg,'audio_c'),a,device,f'C1 Audio-C stage={stage}')
    if not stage_path.is_file():raise RuntimeError(f'Audio-C stage {stage} ended without checkpoint (deadline reached)')
    load_checkpoint(stage_path,checkpoint_schema(cfg,'audio_c'),{'audio':audio,'decoder':decoder},device)
 else:
  micro=micro_indices(records,cfg)
  if a.phase in ('m0a','m0b'):
   ds=TokenDataset(base_subset(records,micro),cache,mapping);_,_,_,eeg=make_models(cfg,device);teacher=None;clip=a.phase=='m0b'
   if clip:
    teacher,shared_decoder=frozen_teacher(cp,cfg,device)
    load_checkpoint(output_path(cp,cfg,'micro_m0a_checkpoint'),checkpoint_schema(cfg,'micro_m0a'),{'eeg':eeg},device)
    # M0b may update only EEG-C.  The MFCC renderer is copied from C1 and
    # frozen so cross-modal alignment cannot repair a weak teacher by altering
    # the acoustic decoder.
    eeg.decoder.load_state_dict(shared_decoder.state_dict(),strict=True)
    for parameter in eeg.decoder.parameters(): parameter.requires_grad_(False)
   opt=torch.optim.AdamW(eeg.parameters(),lr=float(cfg['training']['eeg_micro_lr']),weight_decay=float(cfg['training']['weight_decay']));key='micro_m0b_checkpoint' if clip else 'micro_m0a_checkpoint';sch='micro_m0b' if clip else 'micro_m0a';train_loop({'eeg':eeg},opt,loader(ds,cfg,True),None,lambda b,e: (lambda z:(z[0],{'mfcc':float(F.l1_loss(z[1],b['content_mfcc'].float()).detach())}))(eeg_loss(eeg,b,teacher,clip)),cfg['training']['micro_m0_epochs'],cfg['training']['micro_m0_patience'],output_path(cp,cfg,key),checkpoint_schema(cfg,sch),a,device,'M0b EEG CLIP' if clip else 'M0a EEG direct-MFCC');load_checkpoint(output_path(cp,cfg,key),checkpoint_schema(cfg,sch),{'eeg':eeg},device);src,labs,pred,targ,controls=predict(eeg,ds,cfg,device,clip,teacher);write_pred(output_path(cp,cfg,'micro_m0b_predictions' if clip else 'micro_m0a_predictions'),src,labs,pred,targ,controls)
  else:
   teacher,shared_decoder=frozen_teacher(cp,cfg,device);allv=[[],[],[],[],{'zero':[],'time':[],'channel':[]}];states=[]
   for fold,(actual,inner,held) in enumerate(folds(micro,records.arrays['sample_keys'],records.arrays['labels'])):
    _,_,_,eeg=make_models(cfg,device);load_checkpoint(output_path(cp,cfg,'micro_m0a_checkpoint'),checkpoint_schema(cfg,'micro_m0a'),{'eeg':eeg},device);eeg.decoder.load_state_dict(shared_decoder.state_dict(),strict=True);[p.requires_grad_(False) for p in eeg.decoder.parameters()];opt=torch.optim.AdamW([p for p in eeg.parameters() if p.requires_grad],lr=float(cfg['training']['eeg_micro_lr']),weight_decay=float(cfg['training']['weight_decay']));tr=TokenDataset(base_subset(records,actual),cache,mapping);dv=TokenDataset(base_subset(records,inner),cache,mapping);path=output_path(cp,cfg,'micro_m1_checkpoint').with_name(f'fold_{fold}.pt');train_loop({'eeg':eeg},opt,loader(tr,cfg,True),loader(dv,cfg),lambda b,e:(lambda z:(z[0],{'mfcc':float(F.l1_loss(z[1],b['content_mfcc'].float()).detach())}))(eeg_loss(eeg,b,teacher,True)),cfg['training']['micro_m1_epochs'],cfg['training']['micro_m1_patience'],path,checkpoint_schema(cfg,'micro_m1'),a,device,f'M1 fold {fold}');load_checkpoint(path,checkpoint_schema(cfg,'micro_m1'),{'eeg':eeg},device);heldds=TokenDataset(base_subset(records,held),cache,mapping);src,labs,pred,targ,controls=predict(eeg,heldds,cfg,device,True,teacher);allv[0]+=src;allv[1]+=labs;allv[2]+=pred;allv[3]+=targ;[allv[4][k].extend(v) for k,v in controls.items()];states.append(eeg.state_dict())
   checkpoint=output_path(cp,cfg,'micro_m1_checkpoint');checkpoint.parent.mkdir(parents=True,exist_ok=True);torch.save({'schema_version':checkpoint_schema(cfg,'micro_m1'),'fold_states':states,'protocol':'5 outer folds; outer held never updates; per-label inner dev selected only inside outer train'},checkpoint);write_pred(output_path(cp,cfg,'micro_m1_predictions'),allv[0],allv[1],allv[2],allv[3],allv[4])
if __name__=='__main__':main()
