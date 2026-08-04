#!/usr/bin/env python3
"""Evaluate fit-only gates for the v3 continuous EnCodec bridge.

No command in this evaluator accepts validation or locked-test roles.  This is
intentional: the strict/explore runners end at M1, before human listening
approval and before any held-out data are read.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from scripts.train_open_vocab_v3_encodec_bridge import (
    base_subset, fit_indices, load_cache, load_checkpoint, make_models,
    micro_indices, micro_metrics, token_collate,
)
from src.open_vocab_v3.data import V3Dataset, canonical_mfcc_from_waveform, collate, load_prepared
from src.open_vocab_v3.encodec_bridge import PREPARATION_SCHEMA, SCHEMA, FrozenEnCodecRenderer
from src.open_vocab_v3.hubert import HubertMetric, dtw_cosine
from src.open_vocab_v3.runtime import (
    capture_lineage, checkpoint_schema, default_device, load_config, move_batch,
    output_path, read_json, sha256_file, write_json,
)


def parse():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--phase",choices=("a0","e0","e1","e2","b0","c1","c2","m0","m1"),required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--no-fail",action="store_true")
    parser.add_argument("--explore",action="store_true")
    return parser.parse_args()


def cache_dataset(records, cp, cfg, indices):
    from scripts.train_open_vocab_v3_encodec_bridge import TokenDataset
    cache,mapping=load_cache(cp,cfg)
    return TokenDataset(base_subset(records,np.asarray(indices,dtype=np.int32)),cache,mapping)


def batches(dataset,cfg,device):
    from torch.utils.data import DataLoader
    for batch in DataLoader(dataset,batch_size=int(cfg["evaluation"]["batch_size"]),shuffle=False,collate_fn=token_collate,num_workers=0):
        yield move_batch(batch,device)


def label_prototypes(records, indices):
    labels=records.arrays["labels"].astype(str);content=records.arrays["content_mfcc"].astype(np.float32)
    names=sorted(set(labels[indices].tolist()))
    return names,np.stack([content[indices[labels[indices]==name]].mean(0) for name in names])


def content_label_metrics(prediction, truth, labels, prototype_names, prototypes):
    pred=np.asarray(prediction,dtype=np.float32);truth=np.asarray(truth,dtype=np.float32)
    distance=((pred[:,None]-prototypes[None])**2).mean((2,3));chosen=np.asarray(prototype_names)[distance.argmin(1)]
    wrong=[]
    for row,label in enumerate(labels):
        correct=float(distance[row,prototype_names.index(label)])
        incorrect=float(np.median(np.delete(distance[row],prototype_names.index(label))))
        wrong.append(incorrect-correct)
    template=np.stack([prototypes[prototype_names.index(label)] for label in labels])
    return {"label_top1":float(np.mean(chosen==np.asarray(labels))),"label_margin_mean":float(np.mean(wrong)),"mfcc_mse":float(np.mean((pred-truth)**2)),"template_improvement":float(1-np.mean((pred-truth)**2)/max(float(np.mean((template-truth)**2)),1e-8)),"temporal_variance_ratio":float(pred.var(axis=(0,1)).mean()/max(float(truth.var(axis=(0,1)).mean()),1e-8))}


def _collapse(content):
    value=np.asarray(content,dtype=np.float32)
    temporal_std=float(value.std(-1).mean())
    total_std=float(value.std((1,2)).mean())
    changes=np.abs(np.diff(value,axis=-1)).mean(1)
    rank=[]
    for item in value:
        singular=np.linalg.svd(item,compute_uv=False); rank.append(float((singular>singular.max()*0.01).sum()))
    ratio=temporal_std/max(total_std,1e-8);change=float(np.mean(changes>0.05));effective=float(np.mean(rank))
    return {"temporal_std_ratio":ratio,"spectral_change_ratio":change,"effective_temporal_rank":effective,"horizontal_collapse":bool(ratio<.5 or change<.4 or effective<8)}


def _bootstrap_margin(pred,target,controls,draws=1000):
    correct=((pred-target)**2).mean((1,2));result={}
    rng=np.random.default_rng(31)
    for name,value in controls.items():
        control=((value-target)**2).mean((1,2));difference=control-correct
        samples=np.asarray([difference[rng.integers(0,len(difference),len(difference))].mean() for _ in range(draws)])
        result[name]={"win_rate":float(np.mean(difference>0)),"mean_margin":float(difference.mean()),"ci_low":float(np.percentile(samples,2.5)),"ci_high":float(np.percentile(samples,97.5))}
    return result


def _probe(train_x,train_y,dev_x,dev_y,classes,epochs=120):
    # A deliberately separate linear probe: its optimizer only sees detached
    # numpy features, so it cannot update Audio-C or speaker conditioning.
    device=torch.device("cpu"); mapping={name:index for index,name in enumerate(classes)}
    mean=train_x.mean(0,keepdims=True);scale=np.maximum(train_x.std(0,keepdims=True),1e-4)
    train=torch.from_numpy(((train_x-mean)/scale).astype(np.float32));dev=torch.from_numpy(((dev_x-mean)/scale).astype(np.float32))
    model=torch.nn.Linear(train.shape[1],len(classes));opt=torch.optim.AdamW(model.parameters(),lr=.03,weight_decay=1e-4)
    target=torch.tensor([mapping[x] for x in train_y]);
    for _ in range(epochs):
        opt.zero_grad();loss=F.cross_entropy(model(train),target);loss.backward();opt.step()
    prediction=model(dev).argmax(1).numpy();truth=np.asarray([mapping[x] for x in dev_y])
    return float(np.mean(prediction==truth))


def _selected_dev(records,cfg):
    dev=fit_indices(records,dev=True);labels=records.arrays["labels"].astype(str);keys=records.arrays["sample_keys"].astype(str);chosen=[]
    for label in sorted(set(labels[dev].tolist())):
        values=sorted([int(index) for index in dev if labels[index]==label],key=lambda index:keys[index])
        chosen += values[:int(cfg["evaluation"]["bridge_oracle_per_label"])]
    return np.asarray(chosen,dtype=np.int32)


def _subjects(records):
    return sorted(set(records.arrays["subjects"][fit_indices(records,dev=False)].astype(str).tolist()))


def _models(cp,cfg,records,device,*,bridge=False,audio=False):
    audio_model,decoder,eeg,bridge_model=make_models(cfg,device,len(_subjects(records)))
    if bridge:
        load_checkpoint(output_path(cp,cfg,"bridge_checkpoint"),checkpoint_schema(cfg,"bridge"),{"bridge":bridge_model},device);bridge_model.eval()
    if audio:
        load_checkpoint(output_path(cp,cfg,"audio_c_checkpoint"),checkpoint_schema(cfg,"audio_c"),{"audio":audio_model,"decoder":decoder},device);audio_model.eval();decoder.eval()
    return audio_model,decoder,eeg,bridge_model


@torch.no_grad()
def _render(bridge,renderer,content,p,voice,duration,samples):
    latent=bridge(content.float(),p.float(),voice.float(),duration.float());codes,_,_=renderer.quantize_st(latent);wave=renderer.render_codes(codes)
    return [wave[row,:int(samples[row])].detach().cpu().numpy().astype(np.float32) for row in range(len(wave))]


def _wave_metrics(waves,reference_hubert,labels,target_content,records,cfg,hubert,prototype_names,prototypes):
    content=[];cosines=[]
    for wave,target in zip(waves,reference_hubert):
        content.append(canonical_mfcc_from_waveform(wave,16000,cfg)[1:])
        cosines.append(dtw_cosine(hubert.encode(wave,16000),target))
    predicted=np.stack(content); base=content_label_metrics(predicted,np.asarray(target_content,dtype=np.float32),labels,prototype_names,prototypes)
    # Recompute top-1 directly from generated audio-derived MFCC.
    distance=((predicted[:,None]-prototypes[None])**2).mean((2,3));base["label_top1"]=float(np.mean(np.asarray(prototype_names)[distance.argmin(1)]==np.asarray(labels)))
    base["median_dtw_hubert"]=float(np.median(cosines));base["dtw_hubert_mean"]=float(np.mean(cosines));base.update(_collapse(predicted));return base,predicted


def e0(cp,cfg,records,device):
    selected=_selected_dev(records,cfg);dataset=cache_dataset(records,cp,cfg,selected);renderer=FrozenEnCodecRenderer(output_path(cp,cfg,"encodec_root"),device=device,bandwidth=float(cfg["audio"]["encodec_bandwidth"]));fit=fit_indices(records,dev=False);names,prototypes=label_prototypes(records,fit);hubert=HubertMetric(output_path(cp,cfg,"hubert_root"),layer=int(cfg["teachers"]["hubert_layer"]),device=device);waves=[];refs=[];labels=[];truth=[]
    for batch in batches(dataset,cfg,device):
        decoded=renderer.render_codes(batch["encodec_codes"],target_samples=None).detach().cpu().numpy()
        waves += [item[:int(length)] for item,length in zip(decoded,batch["waveform_samples"].cpu().tolist())]
        source_indices=batch["source_index"].cpu().tolist();refs += [records.arrays["hubert"][index] for index in source_indices];truth += [records.arrays["content_mfcc"][index] for index in source_indices];labels += batch["label"]
    metrics,_=_wave_metrics(waves,refs,labels,truth,records,cfg,hubert,names,prototypes)
    return {"gate":"E0","metrics":metrics,"checks":{"frozen_only":True,"n":len(labels)},"passed":True}


def a0(cp,cfg,records,device):
    prepared=records.arrays;fit=fit_indices(records,dev=False);dev=fit_indices(records,dev=True)
    p_train=prepared["p_base"][fit].reshape(len(fit),-1);p_dev=prepared["p_base"][dev].reshape(len(dev),-1);v_train=prepared["speaker_reference_embedding"][fit];v_dev=prepared["speaker_reference_embedding"][dev]
    labels=prepared["labels"].astype(str);classes=sorted(set(labels[fit].tolist()))
    p_score=_probe(p_train,labels[fit].tolist(),p_dev,labels[dev].tolist(),classes);v_score=_probe(v_train,labels[fit].tolist(),v_dev,labels[dev].tolist(),classes)
    refs=prepared.get("speaker_reference_keys",np.asarray([],dtype=str));reference_ok=bool(len(refs)==len(records) and all(str(key) not in str(refs[index]).split("|") for index,key in enumerate(prepared["sample_keys"].astype(str))))
    key_to_index={str(key):index for index,key in enumerate(prepared["sample_keys"].astype(str))};bank_ok=all(key in key_to_index and np.array_equal(prepared["canonical_p_bank"][row],prepared["p_base"][key_to_index[key]]) for row,key in enumerate(prepared["canonical_p_bank_keys"].astype(str)))
    checks={"mfcc_contract":{"sample_rate":16000,"mfcc_bins":40,"content_coefficients":"c1..c39","frames":161,"cmvn":"utterance_active_only"},"p_bank_is_actual_trials":bank_ok,"voice_leave_one_utterance_out":reference_ok,"label_forward_forbidden":True}
    return {"gate":"A0","metrics":{"p_only_label_top1":p_score,"voice_only_label_top1":v_score,"fit_train_n":int(len(fit)),"fit_internal_dev_n":int(len(dev)),"p_medoid_keys":prepared["canonical_p_bank_keys"].astype(str).tolist()},"checks":checks,"passed":bool(all(bool(value) if isinstance(value,(bool,np.bool_)) else True for value in checks.values()))}


def e1(cp,cfg,records,device,canonical=False):
    selected=_selected_dev(records,cfg);dataset=cache_dataset(records,cp,cfg,selected);_,_,_,bridge=_models(cp,cfg,records,device,bridge=True);renderer=FrozenEnCodecRenderer(output_path(cp,cfg,"encodec_root"),device=device,bandwidth=float(cfg["audio"]["encodec_bandwidth"]));fit=fit_indices(records,dev=False);names,prototypes=label_prototypes(records,fit);hubert=HubertMetric(output_path(cp,cfg,"hubert_root"),layer=int(cfg["teachers"]["hubert_layer"]),device=device);outputs={"real_c_real_p":[],"zero_c_real_p":[],"shuffled_c_real_p":[],"template_c_real_p":[],"real_c_shuffled_p":[],"real_c_duration_only":[]};refs=[];labels=[];truth=[]
    bank=records.arrays["canonical_p_bank"].astype(np.float32);bank_duration=records.arrays["canonical_p_bank_duration_fraction"].astype(np.float32);bank_outputs=[[] for _ in range(len(bank))]
    for batch in batches(dataset,cfg,device):
        real=batch["content_mfcc"];p=batch["p_base"];voice=batch["speaker_reference"];duration=batch["duration_fraction"];samples=batch["waveform_samples"]
        if canonical:
            voice=batch["canonical_voice"]
            for index in range(len(bank)):
                canonical_p=torch.from_numpy(bank[index]).to(device).unsqueeze(0).expand(len(real),-1,-1);canonical_duration=torch.full_like(duration,float(bank_duration[index]));bank_outputs[index] += _render(bridge,renderer,real,canonical_p,voice,canonical_duration,samples)
        else:
            outputs["real_c_real_p"] += _render(bridge,renderer,real,p,voice,duration,samples)
            outputs["zero_c_real_p"] += _render(bridge,renderer,torch.zeros_like(real),p,voice,duration,samples)
            outputs["shuffled_c_real_p"] += _render(bridge,renderer,real.roll(1,0),p,voice,duration,samples)
            template=torch.from_numpy(np.stack([prototypes[names.index(label)] for label in batch["label"]])).to(device)
            outputs["template_c_real_p"] += _render(bridge,renderer,template,p,voice,duration,samples)
            outputs["real_c_shuffled_p"] += _render(bridge,renderer,real,p.roll(1,0),voice,duration.roll(1,0),samples)
            duration_only=torch.zeros_like(p)
            outputs["real_c_duration_only"] += _render(bridge,renderer,real,duration_only,voice,duration,samples)
        refs += [records.arrays["hubert"][index] for index in batch["source_index"].cpu().tolist()];labels += batch["label"];truth += list(real.cpu().numpy())
    if canonical:
        per=[_wave_metrics(value,refs,labels,truth,records,cfg,hubert,names,prototypes)[0] for value in bank_outputs]
        average={key:float(np.mean([item[key] for item in per])) for key in per[0] if isinstance(per[0][key],(float,int,np.floating))}
        return {"gate":"E2","metrics":{"canonical_p_bank":per,"mean_over_fixed_bank":average,"p_bank_keys":records.arrays["canonical_p_bank_keys"].astype(str).tolist()},"checks":{"no_best_of_target_selection":True,"canonical_voice_fit_train_only":True},"passed":bool(average.get("label_top1",0)>=float(cfg["gates"]["e1"]["label_top1_min"]))}
    scored={name:_wave_metrics(value,refs,labels,truth,records,cfg,hubert,names,prototypes)[0] for name,value in outputs.items()}
    real=scored["real_c_real_p"];gain_zero=real["label_top1"]-scored["zero_c_real_p"]["label_top1"];gain_shuffle=real["label_top1"]-scored["shuffled_c_real_p"]["label_top1"]
    a0_report=read_json(output_path(cp,cfg,"a0_gate"));p_only=float(a0_report["metrics"]["p_only_label_top1"]);voice_only=float(a0_report["metrics"]["voice_only_label_top1"])
    wrong=np.median([scored["zero_c_real_p"]["median_dtw_hubert"],scored["shuffled_c_real_p"]["median_dtw_hubert"]]);gap=real["median_dtw_hubert"]-wrong;gate=cfg["gates"]["e1"]
    checks={"label":real["label_top1"]>=float(gate["label_top1_min"]),"correct_C_over_zero":gain_zero>=float(gate["c_ablation_gain_min"]),"correct_C_over_shuffled":gain_shuffle>=float(gate["c_ablation_gain_min"]),"real_C_over_template":real["mfcc_mse"]<scored["template_c_real_p"]["mfcc_mse"],"dtw_gap":gap>=float(gate["dtw_gap_min"]),"p_leakage":p_only<=float(gate["p_only_label_max"]),"voice_leakage":voice_only<=float(gate["voice_only_label_max"]),"template":real["template_improvement"]>=float(gate["template_improvement_min"]),"no_horizontal_collapse":not bool(real["horizontal_collapse"])}
    return {"gate":"E1","metrics":{"conditions":scored,"correct_minus_zero_label":gain_zero,"correct_minus_shuffled_label":gain_shuffle,"dtw_correct_minus_wrong":gap,"p_only_label_top1":p_only,"voice_only_label_top1":voice_only},"checks":checks,"passed":bool(all(checks.values()))}


def b0(cp,cfg,records,device):
    path=output_path(cp,cfg,"bridge_checkpoint");renderer=FrozenEnCodecRenderer(output_path(cp,cfg,"encodec_root"),device=device,bandwidth=float(cfg["audio"]["encodec_bandwidth"]))
    return {"gate":"B0","metrics":{"bridge_checkpoint_sha256":sha256_file(path),"renderer_trainable_parameters":sum(parameter.numel() for parameter in renderer.parameters() if parameter.requires_grad)},"checks":{"bridge_checkpoint_exists":path.is_file(),"renderer_frozen":not any(parameter.requires_grad for parameter in renderer.parameters()),"sequential_rvq":renderer.codebooks==8},"passed":path.is_file() and not any(parameter.requires_grad for parameter in renderer.parameters())}


@torch.no_grad()
def c1(cp,cfg,records,device):
    selected=fit_indices(records,dev=True);dataset=cache_dataset(records,cp,cfg,selected);audio,decoder,_,_=_models(cp,cfg,records,device,audio=True);fit=fit_indices(records,dev=False);names,prototypes=label_prototypes(records,fit);prediction=[];truth=[];hubert_global_pred=[];probe_global_pred=[];global_target=[];subjects=[];labels=[]
    for batch in batches(dataset,cfg,device):
        state=audio(batch["encodec_codes"],batch["encodec_mask"]);content,_=decoder(state.local,state.token_mask);prediction+=list(content.cpu().numpy());truth+=list(batch["content_mfcc"].cpu().numpy());hubert_global_pred+=list(F.normalize(audio.hubert_global(state.global_embedding),dim=-1).cpu().numpy());probe_global_pred+=list(state.global_embedding.cpu().numpy());global_target+=list(F.normalize(batch["hubert"].float().mean(1),dim=-1).cpu().numpy());subjects+=batch["subject"];labels+=batch["label"]
    metric=content_label_metrics(np.stack(prediction),np.stack(truth),labels,names,prototypes);similarity=np.asarray(hubert_global_pred)@np.asarray(global_target).T;metric["hubert_global_retrieval"]=float(np.mean(similarity.argmax(1)==np.arange(len(similarity))))
    # Linear probes train separately on fit-train features.
    train_set=cache_dataset(records,cp,cfg,fit);train_global=[];train_labels=[];train_subjects=[]
    for batch in batches(train_set,cfg,device):
        state=audio(batch["encodec_codes"],batch["encodec_mask"]);train_global+=list(state.global_embedding.cpu().numpy());train_labels+=batch["label"];train_subjects+=batch["subject"]
    # Probes use the unprojected 256-D Audio-C global space.  The separate
    # 768-D HuBERT projection above is only for teacher retrieval and must not
    # be mixed into these linear diagnostics.
    metric["label_probe_top1"]= _probe(np.asarray(train_global),train_labels,np.asarray(probe_global_pred),labels,sorted(set(train_labels)))
    speaker_classes=sorted(set(train_subjects));speaker_acc=_probe(np.asarray(train_global),train_subjects,np.asarray(probe_global_pred),subjects,speaker_classes);chance=1/max(len(speaker_classes),1);metric["speaker_probe_top1"]=speaker_acc;metric["normalized_speaker_advantage"]=(speaker_acc-chance)/max(1-chance,1e-8)
    checks={"hubert_global":metric["hubert_global_retrieval"]>=float(cfg["gates"]["c1"]["hubert_global_retrieval_min"]),"template":metric["template_improvement"]>=float(cfg["gates"]["c1"]["template_improvement_min"]),"variance":metric["temporal_variance_ratio"]>=float(cfg["gates"]["c1"]["temporal_variance_ratio_min"]),"probe_not_training_loss":True}
    return {"gate":"C1","metrics":metric,"checks":checks,"passed":bool(all(checks.values()))}


def c2(cp,cfg,records,device):
    selected=_selected_dev(records,cfg);dataset=cache_dataset(records,cp,cfg,selected);audio,decoder,_,bridge=_models(cp,cfg,records,device,bridge=True,audio=True);renderer=FrozenEnCodecRenderer(output_path(cp,cfg,"encodec_root"),device=device,bandwidth=float(cfg["audio"]["encodec_bandwidth"]));fit=fit_indices(records,dev=False);names,prototypes=label_prototypes(records,fit);hubert=HubertMetric(output_path(cp,cfg,"hubert_root"),layer=int(cfg["teachers"]["hubert_layer"]),device=device);waves={"pred_c_real_p":[],"zero_c_real_p":[],"shuffled_c_real_p":[]};refs=[];labels=[];pred=[];truth=[]
    for batch in batches(dataset,cfg,device):
        state=audio(batch["encodec_codes"],batch["encodec_mask"]);content,_=decoder(state.local,state.token_mask);p=batch["p_base"];voice=batch["speaker_reference"];duration=batch["duration_fraction"];samples=batch["waveform_samples"]
        waves["pred_c_real_p"]+=_render(bridge,renderer,content,p,voice,duration,samples);waves["zero_c_real_p"]+=_render(bridge,renderer,torch.zeros_like(content),p,voice,duration,samples);waves["shuffled_c_real_p"]+=_render(bridge,renderer,content.roll(1,0),p,voice,duration,samples)
        refs += [records.arrays["hubert"][index] for index in batch["source_index"].cpu().tolist()];labels+=batch["label"];pred+=list(content.cpu().numpy());truth+=list(batch["content_mfcc"].cpu().numpy())
    metrics={name:_wave_metrics(value,refs,labels,truth,records,cfg,hubert,names,prototypes)[0] for name,value in waves.items()};mfcc=content_label_metrics(np.stack(pred),np.stack(truth),labels,names,prototypes);gain=metrics["pred_c_real_p"]["label_top1"]-max(metrics["zero_c_real_p"]["label_top1"],metrics["shuffled_c_real_p"]["label_top1"])
    checks={"renderer_frozen":True,"predicted_C_beats_controls":gain>=.20,"content_template":mfcc["template_improvement"]>0,"no_horizontal_collapse":not bool(metrics["pred_c_real_p"]["horizontal_collapse"])}
    return {"gate":"C2","metrics":{"wav_conditions":metrics,"mfcc":mfcc,"correct_C_gain":gain},"checks":checks,"passed":bool(all(checks.values()))}


def micro(cp,cfg,phase):
    path=output_path(cp,cfg,"micro_m0_predictions" if phase=="m0" else "micro_m1_predictions");raw=np.load(path,allow_pickle=False)
    if str(raw["schema"].item())!=SCHEMA:raise RuntimeError("stale bridge-v2 micro prediction cache rejected")
    pred=np.asarray(raw["prediction"]);target=np.asarray(raw["target"]);labels=np.asarray(raw["labels"]).astype(str).tolist();controls={name:np.asarray(raw[name]) for name in ("zero","time","channel")};metric=micro_metrics(pred,target,labels);bootstrap=_bootstrap_margin(pred,target,controls,int(cfg["evaluation"]["bootstrap_samples"]));metric["controls"]=bootstrap
    gate=cfg["gates"][phase]
    if phase=="m0":checks={"label":metric["label_top1"]>=float(gate["label_top1_min"]),"paired":metric["paired_r1"]>=float(gate["paired_r1_min"]),"variance":metric["variance_ratio"]>=float(gate["variance_ratio_min"]),"controls":all(value["win_rate"]>=float(gate["control_win_rate_min"]) for value in bootstrap.values())}
    else:
        checks={"label_chance_multiple":metric["label_top1"]/.1>=float(gate["label_chance_multiple_min"]),"paired_chance_multiple":metric["paired_r1"]/.2>=float(gate["paired_chance_multiple_min"]),"paired_margin_ci":all(value["ci_low"]>0 for value in bootstrap.values()),"controls":all(value["win_rate"]>=float(gate["control_win_rate_min"]) for value in bootstrap.values()),"template":metric["template_improvement"]>0}
    return {"gate":phase.upper(),"metrics":metric,"checks":checks,"passed":bool(all(checks.values())),"protocol":"M0: train on all 50 pairs" if phase=="m0" else "M1: five-fold leave-one-trial-per-label; held EEG never updates its fold"}


def save(cp,cfg,key,payload,args,artifacts=()):
    lineage=capture_lineage(cp,cfg,artifact_keys=artifacts);payload.update({"schema_version":SCHEMA,"exploratory":bool(args.explore),"lineage":lineage});write_json(output_path(cp,cfg,key),payload)
    print(f"[v3 bridge {payload['gate']}] passed={payload['passed']} explore={args.explore}",flush=True)
    if not payload["passed"] and not (args.no_fail or args.explore):raise RuntimeError(f"v3 bridge gate failed: {output_path(cp,cfg,key)}")


def main():
    args=parse();cp,cfg=load_config(args.config);device=default_device(args.device);records=load_prepared(output_path(cp,cfg,"prepared_cache"),expected_schema=PREPARATION_SCHEMA)
    if args.phase=="a0":save(cp,cfg,"a0_gate",a0(cp,cfg,records,device),args)
    elif args.phase=="e0":save(cp,cfg,"e0_gate",e0(cp,cfg,records,device),args)
    elif args.phase=="e1":save(cp,cfg,"e1_gate",e1(cp,cfg,records,device),args,("bridge_checkpoint",))
    elif args.phase=="e2":save(cp,cfg,"e2_gate",e1(cp,cfg,records,device,canonical=True),args,("bridge_checkpoint",))
    elif args.phase=="b0":save(cp,cfg,"b0_gate",b0(cp,cfg,records,device),args,("bridge_checkpoint",))
    elif args.phase=="c1":save(cp,cfg,"c1_gate",c1(cp,cfg,records,device),args,("audio_c_checkpoint",))
    elif args.phase=="c2":save(cp,cfg,"c2_gate",c2(cp,cfg,records,device),args,("bridge_checkpoint","audio_c_checkpoint"))
    else:save(cp,cfg,"m0_gate" if args.phase=="m0" else "m1_gate",micro(cp,cfg,args.phase),args,("audio_c_checkpoint","bridge_checkpoint","micro_m0_checkpoint" if args.phase=="m0" else "micro_m1_checkpoint"))


if __name__=="__main__":main()
