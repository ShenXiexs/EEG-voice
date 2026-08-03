#!/usr/bin/env python3
"""Fail-closed gates for v3 CP-temporal-large."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from scripts.train_open_vocab_v3_cp_temporal import (
    TokenDataset, attach_codes, labels_and_subjects, load_checkpoint,
    make_modules, micro_dataset, token_collate, train_dev,
)
from src.open_vocab_v3.cp_temporal import (
    PREPARATION_SCHEMA, SCHEMA, horizontal_diagnostics, parameter_count,
)
from src.open_vocab_v3.data import (V3Dataset, channel_shuffled_eeg, collate,
                                    load_prepared, time_shuffled_eeg)
from src.open_vocab_v3.encodec_content import EnCodecGenerator
from src.open_vocab_v3.full_evaluation import (ReferenceAudio, hubert_metrics,
                                               waveform_fidelity)
from src.open_vocab_v3.hubert import HubertMetric
from src.open_vocab_v3.metrics import (bootstrap_mean_gain, paired_win_rate,
                                      pairwise_mfcc_l1, retrieval,
                                      same_label_template, variance_ratio)
from src.open_vocab_v3.native_mel import native_speecht5_mel
from src.open_vocab_v3.runtime import (capture_lineage, checkpoint_schema,
                                       default_device, load_config, move_batch,
                                       output_path, read_json,
                                       require_passed_gate, sha256_file,
                                       write_json)


def parse():
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--phase",choices=("t0","oracle","prosody","content","intervention","cvae","micro","fit","eeg_prosody","validation","locked","locked_unseen"),required=True)
    parser.add_argument("--device",default="cpu");parser.add_argument("--no-fail",action="store_true");parser.add_argument("--explore",action="store_true")
    return parser.parse_args()


def batches(dataset,cfg,device,token=False):
    for batch in DataLoader(dataset,batch_size=int(cfg["evaluation"]["batch_size"]),shuffle=False,collate_fn=token_collate if token else collate,num_workers=0):
        yield move_batch(batch,device)


def render(vocoder,mel):
    with torch.inference_mode():return [row.detach().cpu().numpy().astype(np.float32) for row in vocoder(mel.transpose(1,2))]


def vocoder_model(cp,cfg,device):
    from transformers import SpeechT5HifiGan
    # CP-temporal uses the verified frozen/native-contract backend.  No
    # exploratory adaptation can be selected implicitly.
    return SpeechT5HifiGan.from_pretrained(str(output_path(cp,cfg,"vocoder_root")),local_files_only=True).to(device).eval()


def save_gate(cp,cfg,key,payload,args,artifacts=()):
    payload.update({"schema_version":SCHEMA,"config_sha256":sha256_file(cp),"lineage":capture_lineage(cp,cfg,artifact_keys=tuple(artifacts)),"exploratory_gate_bypass":bool(args.explore)})
    payload["passed"]=bool(all(payload["checks"].values()));write_json(output_path(cp,cfg,key),payload);print(f"[v3 CP {payload['gate']}] passed={payload['passed']} explore={args.explore}",flush=True)
    if not payload["passed"] and not args.no_fail:raise SystemExit(2)


def require(args,cp,cfg,key,artifacts=()):
    if not args.explore:require_passed_gate(cp,cfg,key,lineage_artifact_keys=tuple(artifacts))


def selected_audio(records,cfg,dev=True,per_label=0):
    train,development=train_dev(records);dataset=development if dev else train
    if not per_label:return dataset
    groups={}
    for index in range(len(dataset)):groups.setdefault(str(dataset[index]["label"]),[]).append(index)
    positions=[index for _,items in sorted(groups.items()) for index in items[:per_label]]
    return torch.utils.data.Subset(dataset,positions)


def gate_t0(cp,cfg,records,device):
    dataset=selected_audio(records,cfg,dev=True,per_label=int(cfg["evaluation"]["oracle_per_label"]));reference=ReferenceAudio(cp,cfg);codec=EnCodecGenerator(output_path(cp,cfg,"encodec_root"),device=device,bandwidth=float(cfg["audio"]["encodec_bandwidth"]));vocoder=vocoder_model(cp,cfg,device);teacher=HubertMetric(output_path(cp,cfg,"hubert_root"),layer=int(cfg["teachers"]["hubert_layer"]),device=device)
    side_root=output_path(cp,cfg,"encodec_adapted_root");side_gate_path=output_path(cp,cfg,"audio_adaptation_gate");side_codec=None;side_load_error=None
    if (side_root/"config.json").is_file():
        try:side_codec=EnCodecGenerator(side_root,device=device,bandwidth=float(cfg["audio"]["encodec_bandwidth"]))
        except Exception as exc:side_load_error=f"{type(exc).__name__}: {exc}"
    references=[];codec_wavs=[];side_wavs=[];vocoder_wavs=[];labels=[]
    for batch in batches(dataset,cfg,device):
        for key,label in zip(batch["sample_key"],batch["label"]):
            wave=reference(key);references.append(wave);labels.append(label)
            value=torch.from_numpy(wave).unsqueeze(0).to(device)
            codes,mask=codec.encode(value);valid=int(mask[0].sum())
            decoded=codec.decode(codes[:,:,:valid],target_samples_16k=len(wave))
            codec_wavs.append(decoded[0].detach().cpu().numpy())
            if side_codec is not None:
                side_codes,side_mask=side_codec.encode(value);side_valid=int(side_mask[0].sum())
                side_decoded=side_codec.decode(side_codes[:,:,:side_valid],target_samples_16k=len(wave))
                side_wavs.append(side_decoded[0].detach().cpu().numpy())
        vocoder_wavs.extend(render(vocoder,batch["speech_t5_mel"].float()))
    raw_metric=hubert_metrics(references,references,labels,teacher)|waveform_fidelity(references,references);codec_metric=hubert_metrics(codec_wavs,references,labels,teacher)|waveform_fidelity(codec_wavs,references);vocoder_metric=hubert_metrics(vocoder_wavs,references,labels,teacher)|waveform_fidelity(vocoder_wavs,references);metrics={"raw_reference":raw_metric,"frozen_encodec":codec_metric,"native_mel_vocoder":vocoder_metric,"0724_oracle":"legacy 0724 WAVs remain read-only; only key-matched artifacts should be compared in the same evaluator"}
    promotion={"available":side_codec is not None,"allowed":False,"selected":False,"load_error":side_load_error}
    if side_wavs:
        side_metric=hubert_metrics(side_wavs,references,labels,teacher)|waveform_fidelity(side_wavs,references);metrics["side_adapted_encodec"]=side_metric;change_gate=read_json(side_gate_path) if side_gate_path.is_file() else {};changed=bool(change_gate.get("checks",{}).get("encodec_encoder_changed")) and bool(change_gate.get("checks",{}).get("encodec_quantizer_changed")) and bool(change_gate.get("checks",{}).get("encodec_decoder_changed"));allowed=changed and side_metric["median_dtw_hubert"]>=.98*codec_metric["median_dtw_hubert"] and side_metric["median_morphology_ssim"]>=.98*codec_metric["median_morphology_ssim"] and side_metric["median_logmel_mae_db"]<=1.02*codec_metric["median_logmel_mae_db"];promotion={"available":True,"parameters_changed":changed,"allowed":bool(allowed),"selected":False,"note":"manual config change is still required; main token cache remains frozen"}
    return {"gate":"T0","n":len(labels),"metrics":metrics,"adaptation_promotion":promotion,"thresholds":{"frozen_label":.95,"vocoder_label":.90,"adapted_relative_tolerance":.02},"checks":{"frozen_encodec":codec_metric["label_top1"]>=.95,"native_vocoder":vocoder_metric["label_top1"]>=.90},"default_tokenizer":"frozen_encodec_24khz_6kbps"}


def _oracle_outputs(cp,cfg,records,device):
    dataset=selected_audio(records,cfg,dev=True,per_label=int(cfg["evaluation"]["oracle_per_label"]));_,_,_,backbone,teacher,_=make_modules(cfg,device);load_checkpoint(output_path(cp,cfg,"oracle_checkpoint"),checkpoint_schema(cfg,"oracle"),{"backbone":backbone,"teacher":teacher},device);backbone.eval();values={name:[] for name in ("real","plus","template","canonical","zero")};targets=[];labels=[];keys=[];all_content=[];all_p=[]
    target_arrays=[];target_labels=[]
    for index in range(len(dataset)):target_arrays.append(dataset[index]["content_mfcc"]);target_labels.append(dataset[index]["label"])
    templates=same_label_template(np.stack(target_arrays),target_labels);offset=0
    with torch.inference_mode():
        for batch in batches(dataset,cfg,device):
            count=len(batch["label"]);content=batch["content_mfcc"].float();p=batch["p_base"].float();voice=batch["canonical_voice"].float();plus=batch["p_plus"].float();template=torch.as_tensor(templates[offset:offset+count],device=device);canonical=batch["canonical_p_base"].float()
            values["real"].append(backbone(content,p,voice,None));values["plus"].append(backbone(content,p,voice,plus));values["template"].append(backbone(template,p,voice,None));values["canonical"].append(backbone(content,canonical,voice,None));values["zero"].append(backbone(torch.zeros_like(content),p,voice,None));targets.append(batch["speech_t5_mel"].float());labels+=batch["label"];keys+=batch["sample_key"];all_content.append(content);all_p.append(p);offset+=count
    return {key:torch.cat(value) for key,value in values.items()},torch.cat(targets),labels,keys,torch.cat(all_p)[...,0]


def gate_oracle(cp,cfg,records,device):
    outputs,target,labels,keys,activity=_oracle_outputs(cp,cfg,records,device);vocoder=vocoder_model(cp,cfg,device);teacher=HubertMetric(output_path(cp,cfg,"hubert_root"),layer=int(cfg["teachers"]["hubert_layer"]),device=device);reference=ReferenceAudio(cp,cfg);references=[reference(key) for key in keys];metrics={}
    for name,mel in outputs.items():
        generated=render(vocoder,mel);metrics[name]={"mel_l1":float((mel-target).abs().mean()),**horizontal_diagnostics(mel.cpu().numpy(),target.cpu().numpy(),activity.cpu().numpy())};metrics[name]["wav"]=hubert_metrics(generated,references,labels,teacher)|waveform_fidelity(generated,references)
    real=metrics["real"];template=metrics["template"];plus=metrics["plus"];improvement=1-real["mel_l1"]/max(template["mel_l1"],1e-8);dtw_plus=(plus["wav"]["median_dtw_hubert"]-real["wav"]["median_dtw_hubert"])/max(abs(real["wav"]["median_dtw_hubert"]),1e-8);ssim_plus=(plus["wav"]["median_morphology_ssim"]-real["wav"]["median_morphology_ssim"])/max(abs(real["wav"]["median_morphology_ssim"]),1e-8);plus_improvement=max(dtw_plus,ssim_plus,0.0);g=cfg["gates"]["oracle"]
    checks={"label":real["wav"]["label_top1"]>=g["label_top1_min"],"dtw_gap":real["wav"]["correct_minus_wrong_gap"]>=g["dtw_gap_min"],"temporal":real["temporal_std_ratio"]>=g["temporal_std_ratio_min"],"change":real["spectral_change_ratio"]>=g["spectral_change_ratio_min"],"rank":real["effective_temporal_rank"]>=g["effective_temporal_rank_min"],"template":improvement>=g["template_improvement_min"],"real_beats_zero":real["mel_l1"]<metrics["zero"]["mel_l1"]}
    return {"gate":"T2D","n":len(labels),"metrics":metrics|{"template_improvement":improvement,"p_plus_relative_improvement":plus_improvement,"p_plus_selected":plus_improvement>=g["p_plus_relative_improvement_min"]},"thresholds":g,"checks":checks}


def _audio_states(cp,cfg,records,device):
    cache,mapping=attach_codes(records,cp,cfg);dataset=TokenDataset(selected_audio(records,cfg,dev=True),cache,mapping);audio,decoder,_,_,_,_=make_modules(cfg,device);return dataset,audio,decoder


def gate_prosody(cp,cfg,records,device):
    dataset,audio,_=_audio_states(cp,cfg,records,device);load_checkpoint(output_path(cp,cfg,"prosody_checkpoint"),checkpoint_schema(cfg,"prosody"),{"audio":audio},device);pred=[];target=[];duration=[];target_duration=[];plus=[];target_plus=[];masks=[]
    with torch.inference_mode():
        for batch in batches(dataset,cfg,device,token=True):state=audio(batch["encodec_codes"],batch["encodec_mask"]);pred.append(state.p_base.cpu().numpy());target.append(batch["p_base"].cpu().numpy());duration.append(state.duration_fraction.cpu().numpy());target_duration.append(batch["duration_fraction"].cpu().numpy());plus.append(state.p_plus.cpu().numpy());target_plus.append(batch["p_plus"].cpu().numpy());masks.append(batch["mfcc_mask"].cpu().numpy())
    p,t=np.concatenate(pred),np.concatenate(target);d,dt=np.concatenate(duration),np.concatenate(target_duration);pp,pt=np.concatenate(plus),np.concatenate(target_plus);valid=np.concatenate(masks).astype(bool);binary=((1/(1+np.exp(-p[...,0])))>=.5)&valid;truth=(t[...,0]>=.5)&valid;f1=2*np.logical_and(binary,truth).sum()/max(binary.sum()+truth.sum(),1);corr=float(np.corrcoef(p[...,1][valid],t[...,1][valid])[0,1]);duration_mae=float(np.mean(abs(d-dt))*float(cfg["audio"]["max_active_seconds"]));pred_var=[];target_var=[]
    for index in range(len(p)):
        support=valid[index];pred_var.append(np.var(p[index,support,1]));target_var.append(np.var(t[index,support,1]))
    vr=float(np.mean(pred_var)/max(np.mean(target_var),1e-8));voiced=((1/(1+np.exp(-pp[...,0])))>=.5)&valid;vtruth=(pt[...,0]>=.5)&valid;voicing_f1=2*np.logical_and(voiced,vtruth).sum()/max(voiced.sum()+vtruth.sum(),1);f0_mae=float(np.mean(abs(1/(1+np.exp(-pp[...,1]))[vtruth]-pt[...,1][vtruth]))) if vtruth.any() else math.nan;g=cfg["gates"]["prosody"];checks={"activity":f1>=g["activity_f1_min"],"envelope":corr>=g["envelope_corr_min"],"duration":duration_mae<=g["duration_mae_seconds_max"],"c0_variance":vr>=g["c0_variance_ratio_min"]}
    return {"gate":"T1P","n":len(p),"metrics":{"activity_f1":f1,"envelope_corr":corr,"duration_mae_seconds":duration_mae,"c0_variance_ratio":vr,"p_plus_audio_only":{"voicing_f1":voicing_f1,"coarse_f0_mae":f0_mae}},"thresholds":g,"checks":checks}


def retrieval_extended(pred,target,labels,keys):
    base=retrieval(pred,target,labels,keys);distance=pairwise_mfcc_l1(pred,target);r5=[];rr=[];chance=[];margins=[]
    names=np.asarray([str(x).strip().lower() for x in labels])
    for index,name in enumerate(names):
        candidates=np.flatnonzero(names==name);order=candidates[np.argsort(distance[index,candidates])];rank=int(np.flatnonzero(order==index)[0])+1;r5.append(rank<=5);rr.append(1/rank);chance.append(1/len(candidates));negative=distance[index,candidates[candidates!=index]];margins.append(float(np.median(negative)-distance[index,index]) if len(negative) else 0.0)
    margin_array=np.asarray(margins,dtype=np.float64);rng=np.random.default_rng(31);draws=np.asarray([margin_array[rng.integers(0,len(margin_array),len(margin_array))].mean() for _ in range(1000)]) if len(margin_array) else np.asarray([0.0])
    return base|{"r_at_5":float(np.mean(r5)),"mrr":float(np.mean(rr)),"r1_chance_multiple":float(base["paired_r_at_1"]/max(np.mean(chance),1e-8)),"positive_negative_margin":float(np.mean(margins)),"positive_negative_margin_ci_low":float(np.percentile(draws,2.5)),"positive_negative_margin_ci_high":float(np.percentile(draws,97.5))}


def _probe_metrics(values,targets):
    values=np.asarray(values);targets=np.asarray(targets);prediction=[];truth=[]
    for fold in range(5):
        test=np.arange(len(targets))%5==fold;train=~test;centers={name:values[train&(targets==name)].mean(0) for name in sorted(set(targets)) if np.any(train&(targets==name))}
        for row,name in zip(values[test],targets[test]):prediction.append(min(centers,key=lambda key:float(np.sum((row-centers[key])**2))));truth.append(name)
    prediction=np.asarray(prediction);truth=np.asarray(truth);accuracy=float(np.mean(prediction==truth));f1=[]
    for name in sorted(set(truth.tolist())):
        tp=np.logical_and(prediction==name,truth==name).sum();fp=np.logical_and(prediction==name,truth!=name).sum();fn=np.logical_and(prediction!=name,truth==name).sum();f1.append(2*tp/max(2*tp+fp+fn,1))
    return accuracy,float(np.mean(f1))


def _probe(values,targets):return _probe_metrics(values,targets)[0]


def gate_content(cp,cfg,records,device):
    dataset,audio,decoder=_audio_states(cp,cfg,records,device);raw=load_checkpoint(output_path(cp,cfg,"content_checkpoint"),checkpoint_schema(cfg,"content"),{"audio":audio,"decoder":decoder},device);pred=[];target=[];global_values=[];teacher_values=[];labels=[];subjects=[];keys=[];coverage=[];slope=[];rank_values=[]
    projection=torch.nn.Linear(768,int(cfg["model"]["content_dimension"])).to(device);label_head=torch.nn.Linear(int(cfg["model"]["content_dimension"]),len(raw["extra"]["labels"])).to(device);speaker_head=torch.nn.Linear(int(cfg["model"]["content_dimension"]),len(raw["extra"]["subjects"])).to(device);load_checkpoint(output_path(cp,cfg,"content_checkpoint"),checkpoint_schema(cfg,"content"),{"teacher_projection":projection,"label_head":label_head,"speaker_head":speaker_head},device)
    with torch.inference_mode():
        for batch in batches(dataset,cfg,device,token=True):state=audio(batch["encodec_codes"],batch["encodec_mask"]);mfcc,_,diag=decoder(state.local,state.token_mask,batch["p_base"].float(),batch["duration_fraction"].float());pred.append(mfcc.cpu().numpy());target.append(batch["content_mfcc"].cpu().numpy());global_values.append(state.global_embedding.cpu().numpy());teacher=projection(batch["hubert"].float());teacher_values.append(teacher.mean(1).cpu().numpy());labels+=batch["label"];subjects+=batch["subject"];keys+=batch["sample_key"];coverage.extend(diag["coverage"].cpu().tolist());slope.extend(diag["slope"].cpu().tolist());rank_values.append(state.local.cpu().numpy())
    p,t=np.concatenate(pred),np.concatenate(target);glob=np.concatenate(global_values);teach=np.concatenate(teacher_values);r=retrieval_extended(p,t,labels,keys);template_error=float(np.mean(abs(p-t)));baseline=float(np.mean(abs(same_label_template(t,labels)-t)));improvement=1-template_error/max(baseline,1e-8);vr=variance_ratio(p,t,labels);label_acc=_probe(glob,labels);hubert_acc=float(np.mean(np.asarray(labels)[(F.normalize(torch.from_numpy(glob),dim=-1)@F.normalize(torch.from_numpy(teach),dim=-1).T).argmax(1).numpy()]==np.asarray(labels)));speaker_acc,speaker_f1=_probe_metrics(glob,subjects);chance=1/max(len(set(subjects)),1);speaker_adv=(speaker_acc-chance)/max(1-chance,1e-8);local=np.concatenate(rank_values).reshape(-1,int(cfg["model"]["content_dimension"]));singular=np.linalg.svd(local-local.mean(0),compute_uv=False);weight=singular/max(singular.sum(),1e-8);effective=float(np.exp(-(weight*np.log(np.maximum(weight,1e-12))).sum()));g=cfg["gates"]["content"];checks={"label":label_acc>=g["global_label_top1_min"],"hubert":hubert_acc>=g["hubert_global_retrieval_min"],"speaker":speaker_adv<=g["normalized_speaker_advantage_max"],"template":improvement>=g["template_improvement_min"],"variance":vr>=g["temporal_variance_ratio_min"],"coverage":float(np.mean(coverage))>=g["attention_coverage_min"],"slope":float(np.mean(slope))>0,"rank":effective>=g["token_effective_rank_min"],"r1":r["paired_r_at_1"]>=g["full_r1_min"],"r5":r["r_at_5"]>=g["full_r5_min"],"mrr":r["mrr"]>=g["full_mrr_min"],"chance_multiple":r["r1_chance_multiple"]>=g["r1_chance_multiple_min"],"margin_ci":r["positive_negative_margin_ci_low"]>0}
    return {"gate":"T1C","n":len(labels),"metrics":r|{"label_probe":label_acc,"hubert_global_retrieval":hubert_acc,"speaker_probe":speaker_acc,"speaker_macro_f1":speaker_f1,"speaker_chance":chance,"normalized_speaker_advantage":speaker_adv,"template_improvement":improvement,"variance_ratio":vr,"attention_coverage":float(np.mean(coverage)),"attention_slope":float(np.mean(slope)),"token_effective_rank":effective},"thresholds":g,"checks":checks}


def _load_complete(cp,cfg,device):
    audio,decoder,_,backbone,teacher,_=make_modules(cfg,device);load_checkpoint(output_path(cp,cfg,"content_checkpoint"),checkpoint_schema(cfg,"content"),{"audio":audio,"decoder":decoder},device);load_checkpoint(output_path(cp,cfg,"oracle_checkpoint"),checkpoint_schema(cfg,"oracle"),{"backbone":backbone,"teacher":teacher},device);return audio.eval(),decoder.eval(),backbone.eval(),teacher.eval()


def gate_intervention(cp,cfg,records,device):
    cache,mapping=attach_codes(records,cp,cfg);dataset=TokenDataset(selected_audio(records,cfg,dev=True),cache,mapping);audio,decoder,backbone,_=_load_complete(cp,cfg,device);conditions={name:[] for name in ("realC_realP","predC_realP","realC_predP","predC_predP")};targets=[];activity=[];labels=[];keys=[]
    with torch.inference_mode():
        for batch in batches(dataset,cfg,device,token=True):state=audio(batch["encodec_codes"],batch["encodec_mask"]);real_p=batch["p_base"].float();pred_c_real,_,_=decoder(state.local,state.token_mask,real_p,batch["duration_fraction"].float());pred_c_pred,_,_=decoder(state.local,state.token_mask,state.p_base,state.duration_fraction);voice=batch["canonical_voice"].float();real_c=batch["content_mfcc"].float();conditions["realC_realP"].append(backbone(real_c,real_p,voice,None));conditions["predC_realP"].append(backbone(pred_c_real,real_p,voice,None));conditions["realC_predP"].append(backbone(real_c,state.p_base,voice,None));conditions["predC_predP"].append(backbone(pred_c_pred,state.p_base,voice,None));targets.append(batch["speech_t5_mel"].float());activity.append(batch["p_base"][...,0]);labels+=batch["label"];keys+=batch["sample_key"]
    target=torch.cat(targets);active=torch.cat(activity).cpu().numpy();metrics={};template=same_label_template(target.cpu().numpy(),labels)
    for name,parts in conditions.items():mel=torch.cat(parts);distance=float((mel-target).abs().mean());metrics[name]={"mel_l1":distance,"template_improvement":1-distance/max(float(np.mean(abs(template-target.cpu().numpy()))),1e-8),**horizontal_diagnostics(mel.cpu().numpy(),target.cpu().numpy(),active),**retrieval_extended(mel.cpu().numpy(),target.cpu().numpy(),labels,keys)}
    complete=metrics["predC_predP"];g=cfg["gates"]["intervention"];checks={"label":complete["label_top1"]>=g["complete_label_top1_min"],"temporal":complete["temporal_std_ratio"]>=g["complete_temporal_std_ratio_min"],"template":complete["template_improvement"]>=g["complete_template_improvement_min"],"real_upper_bound":metrics["realC_realP"]["mel_l1"]<=complete["mel_l1"]}
    return {"gate":"T1CP","n":len(labels),"metrics":metrics,"thresholds":g,"checks":checks}


def gate_cvae(cp,cfg,records,device):
    dataset=selected_audio(records,cfg,dev=True,per_label=int(cfg["evaluation"]["oracle_per_label"]));_,_,_,_,mel_teacher,cvae=make_modules(cfg,device);load_checkpoint(output_path(cp,cfg,"cvae_checkpoint"),checkpoint_schema(cfg,"cvae"),{"cvae":cvae,"teacher":mel_teacher},device);target=[];det=[];prior=[];post=[];samples=[[] for _ in range(int(cfg["evaluation"]["variational_samples"]))];activity=[]
    with torch.inference_mode():
        for batch in batches(dataset,cfg,device):kwargs=(batch["content_mfcc"].float(),batch["p_base"].float(),batch["canonical_voice"].float(),None);one=cvae(*kwargs,target=batch["speech_t5_mel"].float(),stochastic=False);two=cvae(*kwargs,target=None,stochastic=False);target.append(batch["speech_t5_mel"].cpu());det.append(two["deterministic"].cpu());prior.append(two["mel"].cpu());post.append(one["mel"].cpu());activity.append(batch["p_base"][...,0].cpu());
        # deterministic samples are evaluated with independent prior draws
        for batch in batches(dataset,cfg,device):
            for index in range(len(samples)):samples[index].append(cvae(batch["content_mfcc"].float(),batch["p_base"].float(),batch["canonical_voice"].float(),None,target=None,stochastic=True)["mel"].cpu())
    target,det,prior,post=map(lambda x:torch.cat(x),(target,det,prior,post));sample_tensors=[torch.cat(x) for x in samples];sample=np.stack([x.numpy() for x in sample_tensors]);post_improvement=1-float((post-target).abs().mean())/max(float((det-target).abs().mean()),1e-8);residual=prior-det;residual_ratio=float(residual.square().mean().sqrt()/det.square().mean().sqrt().clamp_min(1e-8));real_var=float(np.var(target.numpy(),axis=0).mean());diversity=float(np.var(sample,axis=0).mean()/max(real_var,1e-8));active=torch.cat(activity).numpy()>0.5;absolute=np.abs(residual.numpy()).mean(1);inactive_active=float(absolute[~active].mean()/max(absolute[active].mean(),1e-8)) if active.any() and (~active).any() else math.inf;g=cfg["gates"]["cvae"]
    labels=[dataset[index]["label"] for index in range(len(dataset))];keys=[dataset[index]["sample_key"] for index in range(len(dataset))];pr=retrieval_extended(prior.numpy(),target.numpy(),labels,keys);po=retrieval_extended(post.numpy(),target.numpy(),labels,keys);de=retrieval_extended(det.numpy(),target.numpy(),labels,keys)
    # Final content gates use an independent, frozen HuBERT evaluator on
    # generated waveforms. The fit-trained Mel teacher never grades itself.
    vocoder=vocoder_model(cp,cfg,device);reference=ReferenceAudio(cp,cfg);references=[reference(key) for key in keys];hubert=HubertMetric(output_path(cp,cfg,"hubert_root"),layer=int(cfg["teachers"]["hubert_layer"]),device=device)
    wav_metrics={"deterministic":hubert_metrics(render(vocoder,det.to(device)),references,labels,hubert),"prior":hubert_metrics(render(vocoder,prior.to(device)),references,labels,hubert),"posterior":hubert_metrics(render(vocoder,post.to(device)),references,labels,hubert)}
    sample_wav_metrics=[hubert_metrics(render(vocoder,value.to(device)),references,labels,hubert) for value in sample_tensors]
    dtw_drop=wav_metrics["deterministic"]["median_dtw_hubert"]-wav_metrics["prior"]["median_dtw_hubert"]
    checks={"label":wav_metrics["prior"]["label_top1"]>=g["label_top1_min"],"dtw_retention":dtw_drop<=g["dtw_drop_max"],"posterior":post_improvement>=g["posterior_improvement_min"],"prior_posterior":abs(wav_metrics["prior"]["label_top1"]-wav_metrics["posterior"]["label_top1"])<=g["prior_posterior_label_gap_max"],"sample_content_retention":all(value["label_top1"]>=g["label_top1_min"] for value in sample_wav_metrics),"residual_budget":residual_ratio<=g["residual_rms_ratio_max"],"diversity_low":diversity>=g["diversity_ratio_min"],"diversity_high":diversity<=g["diversity_ratio_max"],"inactive":inactive_active<=g["inactive_active_residual_ratio_max"]}
    return {"gate":"T2V","n":len(labels),"metrics":{"mel_retrieval":{"deterministic":de,"prior":pr,"posterior":po},"independent_hubert_wav":wav_metrics,"prior_sample_hubert_wav":sample_wav_metrics,"dtw_hubert_drop":dtw_drop,"posterior_improvement":post_improvement,"residual_rms_ratio":residual_ratio,"prior_diversity_ratio":diversity,"inactive_active_residual_ratio":inactive_active},"thresholds":g,"checks":checks}


def _eeg_predictions(cp,cfg,records,device,stage,dataset):
    # Held-out EEG evaluation needs only the frozen shared MFCC decoder. Its
    # targets are cached MFCCs; requiring fit-only EnCodec codes here would
    # both be unnecessary and make the held-out firewall impossible to use.
    _,decoder,eeg,_,_,_=make_modules(cfg,device);load_checkpoint(output_path(cp,cfg,"content_checkpoint"),checkpoint_schema(cfg,"content"),{"decoder":decoder},device);load_checkpoint(output_path(cp,cfg,f"{stage}_checkpoint"),checkpoint_schema(cfg,stage),{"eeg":eeg},device);pred=[];target=[];controls={name:[] for name in ("zero","time","channel")};labels=[];keys=[]
    with torch.inference_mode():
        for batch in batches(dataset,cfg,device,token=False):
            canonical=batch["canonical_p_base"].float();duration=batch["canonical_duration_fraction"].float()
            # Loop body kept explicit so all controls share the frozen decoder.
            state=eeg(batch["eeg"].float(),batch["channel_xyz"].float(),batch["channel_mask"],batch["time_mask"]);pred.append(decoder(state.local,state.token_mask,canonical,duration)[0].cpu().numpy());target.append(batch["eeg_content_mfcc"].cpu().numpy())
            variants={"zero":torch.zeros_like(batch["eeg"]),"time":time_shuffled_eeg(batch["eeg"],batch["time_mask"]),"channel":channel_shuffled_eeg(batch["eeg"],batch["channel_mask"])}
            for name,signal in variants.items():one=eeg(signal.float(),batch["channel_xyz"].float(),batch["channel_mask"],batch["time_mask"]);controls[name].append(decoder(one.local,one.token_mask,canonical,duration)[0].cpu().numpy())
            labels+=batch["label"];keys+=batch["sample_key"]
    return np.concatenate(pred),np.concatenate(target),{k:np.concatenate(v) for k,v in controls.items()},labels,keys


def gate_eeg(cp,cfg,records,device,stage,dataset):
    pred,target,controls,labels,keys=_eeg_predictions(cp,cfg,records,device,stage,dataset);r=retrieval_extended(pred,target,labels,keys);wins={name:paired_win_rate(pred,value,target) for name,value in controls.items()};vr=variance_ratio(pred,target,labels);baseline=float(np.mean(abs(same_label_template(target,labels)-target)));error=float(np.mean(abs(pred-target)));improvement=1-error/max(baseline,1e-8);g=cfg["gates"][stage];checks={"label":r["label_top1"]>=g["label_top1_min"],"variance":vr>=g["variance_ratio_min"],**{f"{name}_win":value>=g["paired_win_rate_min"] for name,value in wins.items()}}
    if stage=="micro":checks|={"r1":r["paired_r_at_1"]>=g["paired_r_at_1_min"],"template":improvement>=g["template_improvement_min"]}
    else:
        control_gain=bootstrap_mean_gain(pred,controls["zero"],target,samples=int(cfg["evaluation"]["bootstrap_samples"]),seed=int(cfg["training"]["seed"]));checks["paired_margin_ci"]=r["positive_negative_margin_ci_low"]>0;r["correct_vs_zero_error_bootstrap"]=control_gain
    return {"gate":"C" if stage=="micro" else "D","n":len(labels),"metrics":r|{"control_win_rates":wins,"variance_ratio":vr,"template_improvement":improvement},"thresholds":g,"checks":checks}


def heldout(cp,cfg,records,device,phase,args):
    if not args.explore:
        review=output_path(cp,cfg,"training_review");payload=read_json(review) if review.is_file() else {};expected=capture_lineage(cp,cfg,artifact_keys=("fit_checkpoint","fit_gate","fit_preview_manifest"))
        if not payload.get("passed",False) or payload.get("lineage")!=expected:raise RuntimeError("held-out access refused before exact CP-temporal training-WAV approval")
    roles={"validation":("subject_holdout_seen",),"locked":("locked_test_seen_label",),"locked_unseen":("locked_test_unseen_label",)}[phase];dataset=V3Dataset(records,roles,eligible_only=True);pred,target,controls,labels,keys=_eeg_predictions(cp,cfg,records,device,"fit",dataset)
    _,_,_,_,mel_teacher,cvae=make_modules(cfg,device);load_checkpoint(output_path(cp,cfg,"cvae_checkpoint"),checkpoint_schema(cfg,"cvae"),{"cvae":cvae,"teacher":mel_teacher},device);cvae.eval();vocoder=vocoder_model(cp,cfg,device);generated={name:[] for name in ("correct","zero","time","channel")};offset=0
    with torch.inference_mode():
        for batch in batches(dataset,cfg,device,token=False):
            count=len(batch["label"]);p=batch["canonical_p_base"].float();voice=batch["canonical_voice"].float()
            for name,value in (("correct",pred),("zero",controls["zero"]),("time",controls["time"]),("channel",controls["channel"])):
                content=torch.as_tensor(value[offset:offset+count],device=device).float();mel=cvae(content,p,voice,None,target=None,stochastic=False)["mel"];generated[name].extend(render(vocoder,mel))
            offset+=count
    reference=ReferenceAudio(cp,cfg);references=[reference(key) for key in keys];hubert=HubertMetric(output_path(cp,cfg,"hubert_root"),layer=int(cfg["teachers"]["hubert_layer"]),device=device);wav_metrics={name:hubert_metrics(value,references,labels,hubert) for name,value in generated.items()};rng=np.random.default_rng(int(cfg["training"]["seed"]));bootstrap={}
    correct=np.asarray(wav_metrics["correct"]["paired_dtw"],dtype=np.float64)
    for name in ("zero","time","channel"):
        difference=correct-np.asarray(wav_metrics[name]["paired_dtw"],dtype=np.float64);draws=np.asarray([difference[rng.integers(0,len(difference),len(difference))].mean() for _ in range(int(cfg["evaluation"]["bootstrap_samples"]))]);bootstrap[name]={"mean_gain":float(difference.mean()),"ci_low":float(np.percentile(draws,2.5)),"ci_high":float(np.percentile(draws,97.5))}
    report={"schema_version":SCHEMA,"phase":phase,"role":roles[0],"n":len(labels),"exploratory":bool(args.explore) or phase=="locked_unseen","primary":"thinking_EEG_C_plus_fit_only_canonical_P","metrics":retrieval_extended(pred,target,labels,keys)|{"variance_ratio":variance_ratio(pred,target,labels),"control_win_rates":{name:paired_win_rate(pred,value,target) for name,value in controls.items()},"independent_hubert_wav":wav_metrics,"correct_minus_control_dtw_bootstrap":bootstrap},"lineage":capture_lineage(cp,cfg,artifact_keys=("fit_checkpoint","fit_gate") if args.explore else ("fit_checkpoint","fit_gate","fit_preview_manifest","training_review"))};write_json(output_path(cp,cfg,{"validation":"validation_report","locked":"locked_report","locked_unseen":"locked_unseen_report"}[phase]),report);print(f"[v3 CP {phase}] n={len(labels)} exploratory={report['exploratory']}",flush=True)


def main():
    args=parse();cp,cfg=load_config(args.config);records=load_prepared(output_path(cp,cfg,"prepared_cache"),expected_schema=PREPARATION_SCHEMA);device=default_device(args.device)
    if args.phase=="t0":save_gate(cp,cfg,"t0_gate",gate_t0(cp,cfg,records,device),args)
    elif args.phase=="oracle":require(args,cp,cfg,"t0_gate");save_gate(cp,cfg,"oracle_gate",gate_oracle(cp,cfg,records,device),args,("oracle_checkpoint",))
    elif args.phase=="prosody":require(args,cp,cfg,"oracle_gate",("oracle_checkpoint",));save_gate(cp,cfg,"prosody_gate",gate_prosody(cp,cfg,records,device),args,("prosody_checkpoint",))
    elif args.phase=="content":require(args,cp,cfg,"prosody_gate",("prosody_checkpoint",));save_gate(cp,cfg,"content_gate",gate_content(cp,cfg,records,device),args,("content_checkpoint",))
    elif args.phase=="intervention":require(args,cp,cfg,"content_gate",("content_checkpoint",));save_gate(cp,cfg,"intervention_gate",gate_intervention(cp,cfg,records,device),args,("content_checkpoint","oracle_checkpoint"))
    elif args.phase=="cvae":require(args,cp,cfg,"intervention_gate",("content_checkpoint","oracle_checkpoint"));save_gate(cp,cfg,"cvae_gate",gate_cvae(cp,cfg,records,device),args,("cvae_checkpoint",))
    elif args.phase=="micro":require(args,cp,cfg,"cvae_gate",("cvae_checkpoint",));save_gate(cp,cfg,"micro_gate",gate_eeg(cp,cfg,records,device,"micro",micro_dataset(records,cfg)),args,("micro_checkpoint",))
    elif args.phase=="fit":require(args,cp,cfg,"micro_gate",("micro_checkpoint",));save_gate(cp,cfg,"fit_gate",gate_eeg(cp,cfg,records,device,"fit",train_dev(records)[1]),args,("fit_checkpoint",))
    elif args.phase=="eeg_prosody":
        payload={"gate":"P","metrics":{"phase_metadata_available":False,"interpretation":"thinking EEG-P exploratory only"},"thresholds":{},"checks":{"checkpoint_exists":output_path(cp,cfg,"eeg_prosody_checkpoint").is_file()}};save_gate(cp,cfg,"eeg_prosody_gate",payload,args,("eeg_prosody_checkpoint",))
    else:heldout(cp,cfg,records,device,args.phase,args)


if __name__=="__main__":main()
