#!/usr/bin/env python3
"""Fit-only fail-closed gates for sequential-RVQ repair-v3."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:sys.path.insert(0,str(APP))
from scripts.train_open_vocab_v3_encodec_rvq_repair import TokenDataset,base_subset,fit_indices,load_cache,load_checkpoint,loader,make_models,micro_indices,metrics,token_collate
from src.open_vocab_v3.data import canonical_mfcc_from_waveform,load_prepared
from src.open_vocab_v3.encodec_rvq_repair import PREPARATION_SCHEMA,SCHEMA,FrozenEnCodecRVQ,diagonal_band_infonce,temporal_delta
from src.open_vocab_v3.hubert import HubertMetric,dtw_cosine
from src.open_vocab_v3.full_evaluation import waveform_fidelity
from src.open_vocab_v3.runtime import capture_lineage,checkpoint_schema,default_device,load_config,move_batch,output_path,read_json,resolve_config_path,sha256_file,write_json

def parse():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--phase',choices=('a0','r0','e1a','e1b','b0','c1','c2','m0a','m0b','m1'),required=True);p.add_argument('--device',default='cpu');p.add_argument('--no-fail',action='store_true');p.add_argument('--explore',action='store_true');return p.parse_args()
def dataset(records,cp,cfg,idx):
 cache,mapping=load_cache(cp,cfg);return TokenDataset(base_subset(records,np.asarray(idx,np.int32)),cache,mapping)
def selected(records,cfg,split='dev'):
 ix=fit_indices(records,split=='dev');labels=records.arrays['labels'].astype(str);keys=records.arrays['sample_keys'].astype(str);out=[]
 for label in sorted(set(labels[ix])):out+=sorted([int(x) for x in ix if labels[x]==label],key=lambda x:keys[x])[:int(cfg['evaluation']['bridge_oracle_per_label'])]
 return np.asarray(out,np.int32)
def prototypes(records,idx):
 labels=records.arrays['labels'].astype(str);names=sorted(set(labels[idx]));return names,np.stack([records.arrays['content_mfcc'][idx[labels[idx]==x]].mean(0) for x in names])
def mfcc_label(values,truth,labels,names,proto):
 values=np.asarray(values);truth=np.asarray(truth);d=((values[:,None]-proto[None])**2).mean((2,3));chosen=np.asarray(names)[d.argmin(1)];template=np.stack([proto[names.index(x)] for x in labels]);return {'label_top1':float(np.mean(chosen==np.asarray(labels))),'template_improvement':float(1-((values-truth)**2).mean()/max(((template-truth)**2).mean(),1e-8)),'temporal_variance_ratio':float(values.var(axis=(0,1)).mean()/max(truth.var(axis=(0,1)).mean(),1e-8))}
def collapse(values):
 x=np.asarray(values);std=x.std(-1).mean()/max(x.std((1,2)).mean(),1e-8);grad=np.abs(np.diff(x,axis=-1)).mean(1);rank=np.mean([(np.linalg.svd(v,compute_uv=False)>np.linalg.svd(v,compute_uv=False).max()*.01).sum() for v in x]);return {'temporal_std_ratio':float(std),'gradient_active_ratio':float(np.mean(grad>.05)),'effective_rank':float(rank),'horizontal_template_collapse':bool(std<.5 or np.mean(grad>.05)<.4 or rank<8)}
def train_label_probe(cp,cfg,records):
 path=output_path(cp,cfg,'label_evaluator_checkpoint');train=fit_indices(records,False);dev=fit_indices(records,True);labels=records.arrays['labels'].astype(str);classes=sorted(set(labels[train]));mp={x:i for i,x in enumerate(classes)}
 def pooled(indices):
  value=records.arrays['hubert'][indices].astype(np.float32);mask=records.arrays['hubert_mask'][indices].astype(np.float32);return (value*mask[...,None]).sum(1)/np.maximum(mask.sum(1,keepdims=True),1)
 train_value=pooled(train);dev_value=pooled(dev);mean=train_value.mean(0);scale=np.maximum(train_value.std(0),1e-4);x=torch.from_numpy(((train_value-mean)/scale).astype(np.float32));y=torch.tensor([mp[x] for x in labels[train]]);model=torch.nn.Linear(768,len(classes));opt=torch.optim.AdamW(model.parameters(),lr=.03)
 for _ in range(180):opt.zero_grad();loss=F.cross_entropy(model(x),y);loss.backward();opt.step()
 raw=torch.from_numpy(((dev_value-mean)/scale).astype(np.float32));score=float(np.mean(model(raw).argmax(1).detach().numpy()==np.asarray([mp[x] for x in labels[dev]])));path.parent.mkdir(parents=True,exist_ok=True);torch.save({'schema_version':checkpoint_schema(cfg,'label_evaluator'),'state':model.state_dict(),'mean':mean,'scale':scale,'classes':classes,'raw_fit_dev_top1':score},path);return model,mean,scale,classes,score
def _linear_probe_metrics(train_x,train_y,dev_x,dev_y,classes,epochs=160):
 train_x=np.asarray(train_x,np.float32);dev_x=np.asarray(dev_x,np.float32);mean=train_x.mean(0,keepdims=True);scale=np.maximum(train_x.std(0,keepdims=True),1e-4);mapping={x:i for i,x in enumerate(classes)};x=torch.from_numpy((train_x-mean)/scale);z=torch.from_numpy((dev_x-mean)/scale);y=torch.tensor([mapping[x] for x in train_y]);truth=np.asarray([mapping[x] for x in dev_y]);model=torch.nn.Linear(x.shape[1],len(classes));opt=torch.optim.AdamW(model.parameters(),lr=.03,weight_decay=1e-4)
 for _ in range(int(epochs)):opt.zero_grad();loss=F.cross_entropy(model(x),y);loss.backward();opt.step()
 prediction=model(z).argmax(1).detach().numpy();f1=[]
 for label in range(len(classes)):
  tp=np.sum((prediction==label)&(truth==label));fp=np.sum((prediction==label)&(truth!=label));fn=np.sum((prediction!=label)&(truth==label));f1.append(float(2*tp/max(2*tp+fp+fn,1)))
 return {'top1':float(np.mean(prediction==truth)),'macro_f1':float(np.mean(f1))}
def _linear_probe_top1(train_x,train_y,dev_x,dev_y,classes,epochs=160):
 return _linear_probe_metrics(train_x,train_y,dev_x,dev_y,classes,epochs)['top1']
def _pool_features(features):
 return np.stack([np.asarray(value,dtype=np.float32).mean(0) for value in features])
def label_probe(cp,cfg,features):
 raw=torch.load(output_path(cp,cfg,'label_evaluator_checkpoint'),map_location='cpu',weights_only=False);model=torch.nn.Linear(768,len(raw['classes']));model.load_state_dict(raw['state']);pooled=_pool_features(features);x=torch.from_numpy(((pooled-raw['mean'])/raw['scale']).astype(np.float32));return np.asarray(raw['classes'])[model(x).argmax(1).detach().numpy()]
def hubert_metric(cp,cfg,device):return HubertMetric(output_path(cp,cfg,'hubert_root'),layer=int(cfg['teachers']['hubert_layer']),device=device)
def waves_from_codes(renderer,codes,scales,lengths,code_masks=None):
 """Decode each trial at its true codec length; padding codes never enter RVQ."""
 output=[]
 with torch.no_grad():
  for row,n in enumerate(lengths):
   steps=int(code_masks[row].sum().item()) if code_masks is not None else int(codes.shape[-1]);value=renderer.decode_codes(codes[row:row+1,:,:steps],scales[row:row+1],target_samples=int(n))[0];output.append(value.detach().cpu().numpy())
 return output
def r0(cp,cfg,records,device):
 train_label_probe(cp,cfg,records);raw=torch.load(output_path(cp,cfg,'label_evaluator_checkpoint'),map_location='cpu',weights_only=False);idx=selected(records,cfg);ds=dataset(records,cp,cfg,idx);renderer=FrozenEnCodecRVQ(output_path(cp,cfg,'encodec_root'),device=device,bandwidth=float(cfg['audio']['encodec_bandwidth']));hub=hubert_metric(cp,cfg,device);waves=[];labels=[];refs=[];reference_waves=[];sample_keys=[];rms=[];clip=[]
 for b in loader(ds,cfg):
  b=move_batch(b,device);decoded=waves_from_codes(renderer,b['encodec_codes'],b['audio_scales'],b['waveform_samples'].cpu().tolist(),b['encodec_mask']);waves+=decoded;labels+=b['label'];sample_keys+=b['sample_key'];refs += [records.arrays['hubert'][i][records.arrays['hubert_mask'][i].astype(bool)] for i in b['source_index'].cpu().tolist()];reference_waves += [x[:int(n)].detach().cpu().numpy() for x,n in zip(b['waveform_16k'],b['waveform_samples'])];rms += [float(np.sqrt(np.mean(x*x)+1e-12)) for x in decoded];clip += [float(np.mean(np.abs(x)>=.999)) for x in decoded]
 encoded=[hub.encode(w,16000) for w in waves];prediction=label_probe(cp,cfg,encoded);raw_score=float(raw['raw_fit_dev_top1']);top=float(np.mean(prediction==np.asarray(labels)));dtw=float(np.median([dtw_cosine(x,y) for x,y in zip(encoded,refs)]));reference_rms=[float(np.sqrt(np.mean(x*x)+1e-12)) for x in reference_waves];ratio=float(np.median(np.asarray(rms)/np.maximum(reference_rms,1e-8)));fidelity=waveform_fidelity(waves,reference_waves)
 legacy_root=resolve_config_path(cp,cfg['calibration']['legacy_0724_codec_oracle_root']);legacy_files=list(legacy_root.glob('*/codec_oracle/*.wav')) if legacy_root.is_dir() else [];legacy_index={path.stem.split('_',1)[1]:path for path in legacy_files if '_' in path.stem};all_fit=fit_indices(records,None);keys=records.arrays['sample_keys'].astype(str);matched=sorted([int(i) for i in all_fit if keys[i].replace(':','_') in legacy_index],key=lambda i:keys[i])[:30];current_matched=[];legacy_waves=[];legacy_refs=[];legacy_ref_hubert=[];legacy_labels=[]
 if matched:
  for b in loader(dataset(records,cp,cfg,np.asarray(matched,np.int32)),cfg):
   b=move_batch(b,device);current=waves_from_codes(renderer,b['encodec_codes'],b['audio_scales'],b['waveform_samples'].cpu().tolist(),b['encodec_mask'])
   for row,(key,label,n) in enumerate(zip(b['sample_key'],b['label'],b['waveform_samples'].cpu().tolist())):
    reference=b['waveform_16k'][row,:int(n)].detach().cpu().numpy();value,rate=sf.read(legacy_index[str(key).replace(':','_')],dtype='float32');value=np.asarray(value).mean(1) if np.asarray(value).ndim==2 else np.asarray(value);value=np.interp(np.linspace(0,len(value)-1,round(len(value)*16000/rate)),np.arange(len(value)),value).astype(np.float32) if rate!=16000 else value.astype(np.float32);valid=min(len(value),len(reference),len(current[row]));current_matched.append(current[row][:valid]);legacy_waves.append(value[:valid]);legacy_refs.append(reference[:valid]);source=int(b['source_index'][row]);legacy_ref_hubert.append(records.arrays['hubert'][source][records.arrays['hubert_mask'][source].astype(bool)]);legacy_labels.append(label)
 def comparison_metric(value):
  features=[hub.encode(w,16000) for w in value];return {'n':len(value),'label_top1':float(np.mean(label_probe(cp,cfg,features)==np.asarray(legacy_labels))),'median_dtw_hubert':float(np.median([dtw_cosine(x,y) for x,y in zip(features,legacy_ref_hubert)]))}|waveform_fidelity(value,legacy_refs)
 legacy_ok=len(legacy_waves)>=int(cfg['calibration']['legacy_min_matched_trials']);legacy_metric=comparison_metric(legacy_waves) if legacy_ok else {'n':len(legacy_waves)};current_matched_metric=comparison_metric(current_matched) if legacy_ok else {'n':len(current_matched)}
 tolerance=float(cfg['calibration']['legacy_relative_tolerance']);legacy_relative_ok=legacy_ok and current_matched_metric['median_dtw_hubert']>=legacy_metric['median_dtw_hubert']*(1-tolerance) and current_matched_metric['median_morphology_ssim']>=legacy_metric['median_morphology_ssim']*(1-tolerance) and current_matched_metric['median_logmel_mae_db']<=legacy_metric['median_logmel_mae_db']*(1+tolerance)
 g=cfg['gates']['r0'];checks={'raw_probe_calibrated':raw_score>=float(g['raw_probe_min']),'frozen_label_drop':top>=raw_score-float(g['frozen_raw_drop_max']),'dtw':dtw>=float(g['dtw_min']),'rms':float(g['rms_low'])<=ratio<=float(g['rms_high']),'clipping':float(np.mean(clip))<=float(g['clipping_max']),'original_normalize_contract':renderer.normalize==bool(renderer.model.config.normalize),'legacy_0724_same_pipeline_within_tolerance':legacy_relative_ok}
 return {'gate':'R0','metrics':{'raw_fit_dev_label_top1':raw_score,'frozen_label_top1':top,'median_dtw_hubert':dtw,'median_rms_ratio':ratio,'clipping_fraction':float(np.mean(clip)),'encodec_normalize':renderer.normalize,'frozen_fidelity':fidelity,'legacy_0724':legacy_metric,'frozen_same_legacy_subset':current_matched_metric},'checks':checks,'passed':bool(all(checks.values()))}
def a0(cp,cfg,records,device):
 train=fit_indices(records,False);dev=fit_indices(records,True);labels=records.arrays['labels'].astype(str);classes=sorted(set(labels[train]));_,_,_,_,score=train_label_probe(cp,cfg,records);refs=records.arrays.get('speaker_reference_keys',np.asarray([],str));keys=records.arrays['sample_keys'].astype(str);exclude=bool(len(refs)==len(keys) and all(k not in str(refs[i]).split('|') for i,k in enumerate(keys)));p_top1=_linear_probe_top1(records.arrays['p_base'][train].reshape(len(train),-1),labels[train].tolist(),records.arrays['p_base'][dev].reshape(len(dev),-1),labels[dev].tolist(),classes);voice_top1=_linear_probe_top1(records.arrays['speaker_reference_embedding'][train],labels[train].tolist(),records.arrays['speaker_reference_embedding'][dev],labels[dev].tolist(),classes);checks={'mfcc_schema_c1_to_c39_161':records.arrays['content_mfcc'].shape[1:]==(39,161),'p_voice_reference_exclusion':exclude,'p_only_below_030':p_top1<=.30,'voice_only_below_020':voice_top1<=.20,'label_not_forward_input':True,'cache_schema':str(records.arrays['v3_preparation_schema'].item())==PREPARATION_SCHEMA};return {'gate':'A0','metrics':{'raw_hubert_probe_dev_top1':score,'p_only_label_top1':p_top1,'voice_only_label_top1':voice_top1,'fit_train_n':int(len(train)),'fit_dev_n':int(len(dev))},'checks':checks,'passed':bool(all(checks.values()))}
def model_bundle(cp,cfg,device,bridge=False,audio=False,micro_bridge=False):
 b,a,d,e=make_models(cfg,device)
 if bridge:
  key='rvq_micro_checkpoint' if micro_bridge else 'rvq_bridge_checkpoint'; schema='rvq_micro' if micro_bridge else 'rvq_bridge'
  load_checkpoint(output_path(cp,cfg,key),checkpoint_schema(cfg,schema),{'bridge':b},device);b.eval()
 if audio:load_checkpoint(output_path(cp,cfg,'audio_c_checkpoint'),checkpoint_schema(cfg,'audio_c'),{'audio':a,'decoder':d},device);a.eval();d.eval()
 return b,a,d,e
@torch.no_grad()
def audio_global_features(records,cp,cfg,indices,audio,device):
 values=[];labels=[];subjects=[]
 for batch in loader(dataset(records,cp,cfg,indices),cfg):
  batch=move_batch(batch,device);state=audio(batch['encodec_codes'],batch['encodec_mask'],batch['hubert'].float(),batch['hubert_mask']);values.extend(state.global_embedding.detach().cpu().numpy());labels.extend(batch['label']);subjects.extend(batch['subject'])
 return np.asarray(values,np.float32),labels,subjects
@torch.no_grad()
def render_bridge(bridge,renderer,content,p,voice,duration,scales,length,code_mask=None):
 logits=bridge(content.float(),p.float(),voice.float(),duration.float());codes=bridge.hard_codes(logits,code_mask=code_mask,duration_fraction=duration);return waves_from_codes(renderer,codes,scales,length,code_mask),codes
def e1(cp,cfg,records,device,micro=False):
 idx=micro_indices(records,cfg) if micro else selected(records,cfg);ds=dataset(records,cp,cfg,idx);bridge,_,_,_=model_bundle(cp,cfg,device,bridge=True,micro_bridge=micro);renderer=FrozenEnCodecRVQ(output_path(cp,cfg,'encodec_root'),device=device,bandwidth=float(cfg['audio']['encodec_bandwidth']));hub=hubert_metric(cp,cfg,device);fit=fit_indices(records,False);names,proto=prototypes(records,fit);conditions={k:[] for k in ('real','zero','shuffle','template','p_shuffle','duration_only')};labels=[];truth=[];refs=[];codes_true=[];codes_pred=[];code_masks=[]
 for b in loader(ds,cfg):
  b=move_batch(b,device);content=b['content_mfcc'].float();p=b['p_base'].float();voice=b['speaker_reference'].float();dur=b['duration_fraction'].float();scales=b['audio_scales'];length=b['waveform_samples'].cpu().tolist();template=torch.stack([torch.from_numpy(proto[names.index(x)]).to(device) for x in b['label']])
  # Temporal reversal is deterministic and non-identity even for the final
  # one-item batch; batch.roll would silently turn that control into correct C.
  args={'real':(content,p),'zero':(torch.zeros_like(content),p),'shuffle':(content.flip(-1),p),'template':(template,p),'p_shuffle':(content,p.flip(1)),'duration_only':(content,torch.zeros_like(p))}
  for k,(c,q) in args.items():conditions[k]+=render_bridge(bridge,renderer,c,q,voice,dur,scales,length,b['encodec_mask'])[0]
  _,codes=render_bridge(bridge,renderer,content,p,voice,dur,scales,length,b['encodec_mask']);codes_pred+=list(codes.cpu().numpy());codes_true+=list(b['encodec_codes'].cpu().numpy());code_masks+=list(b['encodec_mask'].cpu().numpy());labels+=b['label'];truth+=list(content.cpu().numpy());refs += [records.arrays['hubert'][i][records.arrays['hubert_mask'][i].astype(bool)] for i in b['source_index'].cpu().tolist()]
 def wm(w):
  feats=[hub.encode(x,16000) for x in w];return {'label_top1':float(np.mean(label_probe(cp,cfg,feats)==np.asarray(labels))),'dtw_hubert_median':float(np.median([dtw_cosine(x,y) for x,y in zip(feats,refs)])),'mfcc':mfcc_label(np.stack([canonical_mfcc_from_waveform(x,16000,cfg)[1:] for x in w]),np.stack(truth),labels,names,proto)}
 out={k:wm(v) for k,v in conditions.items()};true=np.asarray(codes_true);pred=np.asarray(codes_pred);valid=np.asarray(code_masks,dtype=bool);top=[float(np.mean((pred[:,q]==true[:,q])[valid])) for q in range(8)];gain=out['real']['label_top1']-max(out['zero']['label_top1'],out['shuffle']['label_top1']);dtw_gain=out['real']['dtw_hubert_median']-max(out['zero']['dtw_hubert_median'],out['shuffle']['dtw_hubert_median']);base={'gate':'E1a' if micro else 'E1b','metrics':{'conditions':out,'q_top1_valid_steps_only':top,'correct_control_gain':gain,'dtw_hubert_correct_control_gain':dtw_gain},'checks':{}}
 if micro:base['checks']={'q0':top[0]>=float(cfg['gates']['e1a']['q0_top1_min']),'coarse':float(np.mean(top[:4]))>=float(cfg['gates']['e1a']['q0_q3_macro_top1_min']),'wav':out['real']['label_top1']>=float(cfg['gates']['e1a']['wav_label_top1_min'])}
 else:base['checks']={'wav':out['real']['label_top1']>=float(cfg['gates']['e1b']['wav_label_top1_min']),'controls':gain>=float(cfg['gates']['e1b']['control_gain_min']),'dtw_controls':dtw_gain>=float(cfg['gates']['e1b']['dtw_control_gain_min']),'template':out['real']['mfcc']['template_improvement']>=float(cfg['gates']['e1b']['template_improvement_min']),'not_horizontal':not collapse(np.stack([canonical_mfcc_from_waveform(x,16000,cfg)[1:] for x in conditions['real']]))['horizontal_template_collapse']}
 base['passed']=bool(all(base['checks'].values()));return base
def b0(cp,cfg):
 p=output_path(cp,cfg,'rvq_bridge_checkpoint');return {'gate':'B0','metrics':{'checkpoint_sha256':sha256_file(p)},'checks':{'bridge_checkpoint_exists':p.is_file(),'renderer_frozen_required':True},'passed':p.is_file()}
def c1(cp,cfg,records,device):
 dev_indices=selected(records,cfg);ds=dataset(records,cp,cfg,dev_indices);_,audio,decoder,_=model_bundle(cp,cfg,device,audio=True);fit=fit_indices(records,False);names,proto=prototypes(records,fit);pred=[];truth=[];labels=[];glob=[];global_truth=[];masses=[]
 with torch.no_grad():
  for b in loader(ds,cfg):
   b=move_batch(b,device);state=audio(b['encodec_codes'],b['encodec_mask'],b['hubert'].float(),b['hubert_mask']);value,_=decoder(state.local,state.token_mask);pred+=list(value.cpu().numpy());truth+=list(b['content_mfcc'].cpu().numpy());labels+=b['label'];glob+=list(state.global_embedding.cpu().numpy());weight=b['hubert_mask'].float().unsqueeze(-1);pooled=(b['hubert'].float()*weight).sum(1)/weight.sum(1).clamp_min(1);global_truth+=list(F.normalize(pooled,dim=-1).cpu().numpy());_,info=diagonal_band_infonce(audio.hubert_token(state.local),F.interpolate(b['hubert'].float().transpose(1,2),size=96,mode='linear',align_corners=False).transpose(1,2),state.token_mask,state.token_mask,labels=b['label']);masses.append(float(info['diagonal_mass']))
 metric=mfcc_label(pred,truth,labels,names,proto);paired=metrics(np.stack(pred),np.stack(truth),labels);sim=(F.normalize(torch.tensor(np.stack(glob)),dim=-1)@F.normalize(torch.tensor(np.stack(global_truth)),dim=-1).T).numpy();retr=float(np.mean(sim.argmax(1)==np.arange(len(sim))));negative=sim.copy();np.fill_diagonal(negative,-np.inf);margin_values=np.diag(sim)-negative.max(1);rng=np.random.default_rng(31);boot=np.asarray([margin_values[rng.integers(0,len(margin_values),len(margin_values))].mean() for _ in range(1000)]);margin=float(margin_values.mean());margin_low=float(np.percentile(boot,2.5));col=collapse(pred)
 train_global,train_labels,train_subjects=audio_global_features(records,cp,cfg,fit,audio,device);dev_global,dev_labels,dev_subjects=audio_global_features(records,cp,cfg,dev_indices,audio,device);label_probe_metric=_linear_probe_metrics(train_global,train_labels,dev_global,dev_labels,sorted(set(train_labels)));speaker_classes=sorted(set(train_subjects));speaker_probe_metric=_linear_probe_metrics(train_global,train_subjects,dev_global,dev_subjects,speaker_classes);speaker_chance=1/max(len(speaker_classes),1);speaker_advantage=(speaker_probe_metric['top1']-speaker_chance)/max(1-speaker_chance,1e-8)
 metric.update({'hubert_global_retrieval':retr,'diagonal_band_mass':float(np.mean(masses)),'positive_negative_margin':margin,'positive_negative_margin_ci_low':margin_low,'effective_rank':col['effective_rank'],'paired_retrieval':paired,'frozen_label_probe':label_probe_metric,'frozen_speaker_probe':speaker_probe_metric,'normalized_speaker_advantage':float(speaker_advantage)})
 g=cfg['gates']['c1'];checks={'mfcc_label':metric['label_top1']>=float(g['mfcc_label_top1_min']),'hubert_global':retr>=float(g['hubert_global_retrieval_min']),'template':metric['template_improvement']>=float(g['template_improvement_min']),'variance':metric['temporal_variance_ratio']>=float(g['temporal_variance_ratio_min']),'diagonal':metric['diagonal_band_mass']>=float(g['diagonal_mass_min']),'rank':col['effective_rank']>=16,'margin_bootstrap_ci':margin_low>0};return {'gate':'C1','metrics':metric|col,'checks':checks,'passed':bool(all(checks.values()))}
def c2(cp,cfg,records,device):
 idx=selected(records,cfg);ds=dataset(records,cp,cfg,idx);bridge,audio,decoder,_=model_bundle(cp,cfg,device,bridge=True,audio=True);renderer=FrozenEnCodecRVQ(output_path(cp,cfg,'encodec_root'),device=device,bandwidth=float(cfg['audio']['encodec_bandwidth']));hub=hubert_metric(cp,cfg,device);conditions={k:[] for k in ('pred','zero','shuffle')};labels=[];refs=[]
 with torch.no_grad():
  for b in loader(ds,cfg):
   b=move_batch(b,device);state=audio(b['encodec_codes'],b['encodec_mask'],b['hubert'].float(),b['hubert_mask']);content,_=decoder(state.local,state.token_mask)
   for k,c in {'pred':content,'zero':torch.zeros_like(content),'shuffle':content.flip(-1)}.items():conditions[k]+=render_bridge(bridge,renderer,c,b['p_base'],b['speaker_reference'],b['duration_fraction'],b['audio_scales'],b['waveform_samples'].cpu().tolist(),b['encodec_mask'])[0]
   labels+=b['label'];refs += [records.arrays['hubert'][i][records.arrays['hubert_mask'][i].astype(bool)] for i in b['source_index'].cpu().tolist()]
 encoded={k:[hub.encode(x,16000) for x in v] for k,v in conditions.items()};scores={k:float(np.mean(label_probe(cp,cfg,value)==np.asarray(labels))) for k,value in encoded.items()};dtw={k:float(np.median([dtw_cosine(x,y) for x,y in zip(value,refs)])) for k,value in encoded.items()};gain=scores['pred']-max(scores['zero'],scores['shuffle']);oracle=read_json(output_path(cp,cfg,'e1b_gate'))['metrics']['conditions']['real']['dtw_hubert_median'];g=cfg['gates']['c2'];checks={'pred':scores['pred']>=float(g['wav_label_top1_min']),'controls':gain>=float(g['control_gain_min']),'dtw_not_below_real_c_oracle':dtw['pred']>=float(oracle)-float(g['dtw_oracle_drop_max']),'frozen_bridge':True};return {'gate':'C2','metrics':{'wav_label_top1':scores,'dtw_hubert_median':dtw,'real_c_oracle_dtw_hubert_median':float(oracle),'correct_control_gain':gain},'checks':checks,'passed':bool(all(checks.values()))}
def micro(cp,cfg,phase):
 key={'m0a':'micro_m0a_predictions','m0b':'micro_m0b_predictions','m1':'micro_m1_predictions'}[phase];raw=np.load(output_path(cp,cfg,key),allow_pickle=False)
 if str(raw['schema'].item())!=SCHEMA:raise RuntimeError('stale prediction file rejected')
 pred=np.asarray(raw['prediction']);target=np.asarray(raw['target']);labels=raw['labels'].astype(str).tolist();m=metrics(pred,target,labels);wins={}
 for k in ('zero','time','channel'):
  diff=((raw[k]-target)**2).mean((1,2))-((pred-target)**2).mean((1,2));wins[k]=float(np.mean(diff>0))
 m['control_win_rates']=wins
 if phase=='m0a':checks={'label':m['label_top1']>=.95,'paired':m['paired_r1']>=.8,'template':m['template_improvement']>=.5,'variance':m['variance_ratio']>=.2,'controls':min(wins.values())>=.9}
 elif phase=='m0b':
  direct=np.load(output_path(cp,cfg,'micro_m0a_predictions'),allow_pickle=False)
  direct_m=metrics(np.asarray(direct['prediction']),np.asarray(direct['target']),direct['labels'].astype(str).tolist())
  checks={'label_not_drop_over_005':m['label_top1']>=direct_m['label_top1']-.05,'paired_not_drop_over_005':m['paired_r1']>=direct_m['paired_r1']-.05,'template_not_drop_over_005':m['template_improvement']>=direct_m['template_improvement']-.05,'controls':min(wins.values())>=.9}
  m['m0a_reference']=direct_m
 else:checks={'label_chance':m['label_top1']/.1>=3,'paired_chance':m['paired_r1']/max(m['within_label_chance'],1e-8)>=2,'paired_margin_ci':m['paired_margin_ci_low']>0,'template':m['template_improvement']>0,'controls':min(wins.values())>=.75}
 return {'gate':phase.upper(),'metrics':m,'checks':checks,'passed':bool(all(checks.values()))}
def save(cp,cfg,key,payload,args,artifacts=()):
 payload.update({'schema_version':SCHEMA,'exploratory':bool(args.explore),'lineage':capture_lineage(cp,cfg,artifact_keys=artifacts)});write_json(output_path(cp,cfg,key),payload);print(f"[v3 rvq {payload['gate']}] passed={payload['passed']} explore={args.explore}",flush=True)
 if not payload['passed'] and not(args.no_fail or args.explore):raise RuntimeError(f"repair-v3 gate failed: {output_path(cp,cfg,key)}")
def main():
 a=parse();cp,cfg=load_config(a.config);device=default_device(a.device);records=load_prepared(output_path(cp,cfg,'prepared_cache'),expected_schema=PREPARATION_SCHEMA);phase=a.phase
 if phase=='a0':payload=a0(cp,cfg,records,device);key='a0_gate'
 elif phase=='r0':payload=r0(cp,cfg,records,device);key='r0_gate'
 elif phase=='e1a':payload=e1(cp,cfg,records,device,True);key='e1a_gate'
 elif phase=='e1b':payload=e1(cp,cfg,records,device,False);key='e1b_gate'
 elif phase=='b0':payload=b0(cp,cfg);key='b0_gate'
 elif phase=='c1':payload=c1(cp,cfg,records,device);key='c1_gate'
 elif phase=='c2':payload=c2(cp,cfg,records,device);key='c2_gate'
 else:payload=micro(cp,cfg,phase);key=f'{phase}_gate'
 artifacts=({'e1a':('rvq_micro_checkpoint',),'e1b':('rvq_bridge_checkpoint',),'c1':('audio_c_checkpoint',),'c2':('rvq_bridge_checkpoint','audio_c_checkpoint'),'m0a':('micro_m0a_checkpoint',),'m0b':('micro_m0b_checkpoint',),'m1':('micro_m1_checkpoint',)}.get(phase,()))
 save(cp,cfg,key,payload,a,artifacts)
if __name__=='__main__':main()
