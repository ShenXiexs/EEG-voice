"""Complete audio/EEG gates for the EnCodec-CLIP-MFCC v3 experiment."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter
from torch.utils.data import DataLoader, Subset

from src.open_vocab_0724.audio_features import AudioPreparationConfig
from .data import (V3Dataset, _accepted_denoise_paths, _read_waveform,
                   channel_shuffled_eeg, collate, light_prepare_waveform,
                   time_shuffled_eeg)
from .encodec_content import EnCodecGenerator
from .hubert import HubertMetric, dtw_cosine
from .metrics import (bootstrap_mean_gain, paired_r_at_1_above_chance,
                      paired_win_rate, retrieval, same_label_template,
                      variance_ratio)
from .model import NativeSpeechT5MFCCMelCVAE
from .native_mel import CONTRACT, native_speecht5_mel
from .runtime import (capture_lineage, checkpoint_schema, move_batch, output_path, read_json,
                      sha256_file)
from .speaker import ECAPAEncoder, speaker_distribution


def selected(records, roles: Iterable[str], *, eligible: bool = False, per_label: int = 0):
    base = V3Dataset(records, tuple(roles), eligible_only=eligible)
    if not per_label:
        return base
    groups: dict[str, list[int]] = {}
    for position, index in enumerate(base.indices):
        groups.setdefault(str(records.arrays["labels"][index]), []).append(position)
    positions = [p for label, values in sorted(groups.items()) for p in sorted(
        values, key=lambda x: str(records.arrays["sample_keys"][base.indices[x]]))[:per_label]]
    return Subset(base, positions)


def standard_batches(dataset, cfg, device):
    for batch in DataLoader(dataset, batch_size=int(cfg["evaluation"]["batch_size"]),
                            shuffle=False, collate_fn=collate, num_workers=0):
        yield move_batch(batch, device)


class ReferenceAudio:
    def __init__(self, config_path: Path, cfg: dict[str, Any]):
        with output_path(config_path, cfg, "unified_manifest").open(newline="", encoding="utf-8") as handle:
            self.paths = {str(row["sample_key"]): str(row["audio_relpath"]) for row in csv.DictReader(handle)
                          if row.get("dataset") == "karaone"}
        self.root = output_path(config_path, cfg, "audio_root")
        self.denoised = _accepted_denoise_paths(config_path, cfg)
        self.prep = AudioPreparationConfig(sample_rate=16000,
            max_active_seconds=float(cfg["audio"]["max_active_seconds"]),
            target_rms=float(cfg["audio"]["target_rms"]))

    def __call__(self, key: str) -> np.ndarray:
        wave, rate = _read_waveform(self.denoised.get(key, self.root / self.paths[key]))
        prepared, _ = light_prepare_waveform(wave, rate, self.prep)
        return np.asarray(prepared.waveform[:max(1, prepared.valid_samples)], dtype=np.float32)


def _pool(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).mean(0)
    return result / max(float(np.linalg.norm(result)), 1e-8)


def hubert_metrics(generated: list[np.ndarray], references: list[np.ndarray], labels: list[str], teacher: HubertMetric) -> dict[str, Any]:
    left = [teacher.encode(x, 16000) for x in generated]
    right = [teacher.encode(x, 16000) for x in references]
    dtw = np.asarray([dtw_cosine(x, y) for x, y in zip(left, right)], dtype=np.float32)
    if len(left) <= 100:
        similarity=np.asarray([[dtw_cosine(x,y) for y in right] for x in left],dtype=np.float32);backend="all-to-all DTW-HuBERT cosine"
    else:
        similarity=np.stack([_pool(x) for x in left]) @ np.stack([_pool(x) for x in right]).T;backend="pooled HuBERT retrieval plus paired DTW"
    nearest = similarity.argmax(1); canonical = [str(x).strip().strip("/").lower() for x in labels]
    top1 = float(np.mean([canonical[i] == canonical[j] for i, j in enumerate(nearest)]))
    gaps = []
    for i, label in enumerate(canonical):
        wrong = similarity[i, np.asarray(canonical) != label]
        gaps.append(float(similarity[i, i] - np.median(wrong)) if len(wrong) else 0.0)
    return {"label_top1": top1, "median_dtw_hubert": float(np.median(dtw)),
            "mean_dtw_hubert": float(np.mean(dtw)), "correct_minus_wrong_gap": float(np.median(gaps)),
            "paired_dtw": dtw.tolist(), "nearest_index": nearest.tolist(),"retrieval_backend":backend}


def _logmel_db(wave: np.ndarray, frames: int = 160) -> np.ndarray:
    import librosa
    mel = librosa.feature.melspectrogram(y=np.asarray(wave), sr=16000, n_fft=512,
        win_length=400, hop_length=160, n_mels=80, fmin=0, fmax=8000, power=2.0)
    db = librosa.power_to_db(mel, ref=np.max)
    return F.interpolate(torch.from_numpy(db).float().unsqueeze(0), size=frames,
                         mode="linear", align_corners=False).squeeze(0).numpy()


def _ssim(left: np.ndarray, right: np.ndarray) -> float:
    x, y = np.asarray(left, np.float64), np.asarray(right, np.float64)
    dynamic = max(float(max(x.max(), y.max()) - min(x.min(), y.min())), 1e-6)
    ux, uy = uniform_filter(x, 7), uniform_filter(y, 7)
    vx, vy = np.maximum(uniform_filter(x*x, 7)-ux*ux,0), np.maximum(uniform_filter(y*y, 7)-uy*uy,0)
    vxy = uniform_filter(x*y, 7)-ux*uy
    c1, c2 = max((.01*dynamic)**2,1e-8), max((.03*dynamic)**2,1e-8)
    score = ((2*ux*uy+c1)*(2*vxy+c2))/((ux*ux+uy*uy+c1)*(vx+vy+c2)+1e-12)
    return float(np.clip(np.mean(score), -1, 1))


def waveform_fidelity(generated: list[np.ndarray], references: list[np.ndarray]) -> dict[str, float]:
    ssim, mae = [], []
    for one, two in zip(generated, references):
        x, y = _logmel_db(one), _logmel_db(two)
        ssim.append(_ssim(x, y)); mae.append(float(np.mean(np.abs(x-y))))
    return {"median_morphology_ssim": float(np.median(ssim)), "median_logmel_mae_db": float(np.median(mae))}


def _vocoder(config_path, cfg, device):
    from transformers import SpeechT5HifiGan
    model = SpeechT5HifiGan.from_pretrained(str(output_path(config_path, cfg, "vocoder_adapted_root")), local_files_only=True).to(device).eval()
    return model


def _cvae(config_path, cfg, device):
    from scripts.train_open_vocab_v3_encodec_clip import load
    model = NativeSpeechT5MFCCMelCVAE(mfcc_bins=40, mel_bins=80,
        dimension=int(cfg["model"]["audio_dimension"]), voice_dim=int(cfg["speaker"]["embedding_dimension"]),
        latent_dim=int(cfg["model"]["audio_latent_dimension"]),
        residual_limit_log10=float(cfg["model"]["audio_residual_limit_log10"])).to(device)
    load(output_path(config_path, cfg, "cvae_checkpoint"), checkpoint_schema(cfg, "cvae"), {"cvae": model}, device)
    return model.eval()


def render(vocoder, mel: torch.Tensor) -> list[np.ndarray]:
    return [x.detach().cpu().numpy().astype(np.float32) for x in vocoder(mel.transpose(1, 2))]


@torch.no_grad()
def mfcc_prior_wavs(config_path, cfg, records, device, mfcc: np.ndarray, batch_size: int = 8) -> list[np.ndarray]:
    """Primary synthesis: prior mean, fit-only canonical voice/statistics."""
    model=_cvae(config_path,cfg,device);voc=_vocoder(config_path,cfg,device);voice=torch.as_tensor(records.arrays["canonical_voice"],device=device).float();mean=torch.as_tensor(records.arrays["canonical_mfcc_mean"],device=device).float();std=torch.as_tensor(records.arrays["canonical_mfcc_std"],device=device).float();fit=(records.roles=="fit")&records.arrays["fit_eligible"].astype(bool);fixed_frames=int(np.median(records.arrays["speech_t5_mel_mask"][fit].sum(1)));fixed_samples=fixed_frames*int(cfg["vocoder"]["hop_length"]);result=[]
    for start in range(0,len(mfcc),batch_size):
        value=torch.as_tensor(mfcc[start:start+batch_size],device=device).float();count=len(value);generated=model.generate(value,voice.unsqueeze(0).expand(count,-1),mean.unsqueeze(0).expand(count,-1),std.unsqueeze(0).expand(count,-1),stochastic=False)["mel"];result += [wave[:fixed_samples] for wave in render(voc,F.interpolate(generated,size=int(cfg["audio"]["native_mel_frames"]),mode="linear",align_corners=False))]
    return result


@torch.no_grad()
def gate_t0(config_path, cfg, records, device):
    dataset = selected(records, ("fit",), eligible=True, per_label=int(cfg["evaluation"]["oracle_per_label"]))
    ref = ReferenceAudio(config_path, cfg); adapted = EnCodecGenerator(output_path(config_path,cfg,"encodec_adapted_root"),device=device,bandwidth=float(cfg["audio"]["encodec_bandwidth"])); frozen = EnCodecGenerator(output_path(config_path,cfg,"encodec_root"),device=device,bandwidth=float(cfg["audio"]["encodec_bandwidth"])); generated=[]; baseline=[]; references=[];labels=[];steps=[]
    for batch in standard_batches(dataset,cfg,device):
        for key,label in zip(batch["sample_key"],batch["label"]):
            truth=ref(key);tensor=torch.from_numpy(truth).unsqueeze(0);codes,mask=adapted.encode(tensor);valid=int(mask[0].sum());generated.append(adapted.decode(codes[:,:,:valid],target_samples_16k=len(truth))[0].cpu().numpy());base_codes,_=frozen.encode(tensor);baseline.append(frozen.decode(base_codes[:,:,:valid],target_samples_16k=len(truth))[0].cpu().numpy());references.append(truth);labels.append(label);steps.append(valid)
    teacher=HubertMetric(output_path(config_path,cfg,"hubert_root"),layer=int(cfg["teachers"]["hubert_layer"]),device=device);metric=hubert_metrics(generated,references,labels,teacher)|waveform_fidelity(generated,references);base=hubert_metrics(baseline,references,labels,teacher)|waveform_fidelity(baseline,references);g=cfg["gates"]["t0"]
    relative=(metric["median_logmel_mae_db"]-base["median_logmel_mae_db"])/max(base["median_logmel_mae_db"],1e-8)
    checks={"label_retrieval":metric["label_top1"]>=g["label_top1_min"],"hubert":metric["median_dtw_hubert"]>=g["hubert_dtw_min"],"ssim":metric["median_morphology_ssim"]>=g["ssim_min"],"logmel":metric["median_logmel_mae_db"]<=g["logmel_mae_db_max"],"frozen_non_degradation":relative<=g["frozen_relative_degradation_max"],"codec_steps":max(steps)<=192}
    return {"gate":"T0","n":len(labels),"metrics":{"adapted":metric,"frozen":base,"relative_logmel_degradation":relative,"codec_steps":steps},"thresholds":g,"checks":checks}


@torch.no_grad()
def gate_t0b(config_path,cfg,records,device):
    dataset=selected(records,("fit",),eligible=True,per_label=int(cfg["evaluation"]["oracle_per_label"]));voc=_vocoder(config_path,cfg,device);ref=ReferenceAudio(config_path,cfg);generated=[];references=[];labels=[]
    for b in standard_batches(dataset,cfg,device):
        waves=render(voc,b["speech_t5_mel"].float());generated += [wave[:int(mask.sum())*int(cfg["vocoder"]["hop_length"])] for wave,mask in zip(waves,b["speech_t5_mel_mask"])];references += [ref(k) for k in b["sample_key"]];labels+=b["label"]
    teacher=HubertMetric(output_path(config_path,cfg,"hubert_root"),layer=int(cfg["teachers"]["hubert_layer"]),device=device);metric=hubert_metrics(generated,references,labels,teacher);g=cfg["gates"]["t0b"];return {"gate":"T0b","n":len(labels),"metrics":metric,"thresholds":g,"native_mel_contract":CONTRACT,"checks":{"label":metric["label_top1"]>=g["label_top1_min"],"hubert":metric["median_dtw_hubert"]>=g["hubert_dtw_min"],"gap":metric["correct_minus_wrong_gap"]>=g["dtw_gap_min"]}}


def _load_audio_content(config_path,cfg,device):
    from scripts.train_open_vocab_v3_encodec_clip import load,modules
    audio,decoder,_=modules(cfg,device);load(output_path(config_path,cfg,"audio_content_checkpoint"),checkpoint_schema(cfg,"audio"),{"audio":audio,"decoder":decoder},device);return audio.eval(),decoder.eval()


@torch.no_grad()
def gate_t1(config_path,cfg,records,device,token_dataset,batcher):
    audio,decoder=_load_audio_content(config_path,cfg,device);pred=[];target=[];tokens=[];labels=[];keys=[]
    for b in batcher(token_dataset,cfg,device):
        value=audio(b["encodec_codes"],b["encodec_mask"]);pred.append(decoder(value).cpu().numpy());target.append(b["mfcc"].cpu().numpy());tokens.append(value.cpu().numpy());labels+=b["label"];keys+=b["sample_key"]
    p,t,z=np.concatenate(pred),np.concatenate(target),np.concatenate(tokens);r=retrieval(p,t,labels,keys);ratio=float(np.mean(abs(p-t))/max(float(np.mean(abs(same_label_template(t,labels)-t))),1e-8));vr=variance_ratio(p,t,labels);singular=np.linalg.svd(z.reshape(-1,z.shape[-1])-z.reshape(-1,z.shape[-1]).mean(0),compute_uv=False);weight=singular/max(singular.sum(),1e-8);rank=float(np.exp(-(weight*np.log(np.maximum(weight,1e-12))).sum()));correlation=float(np.mean((p-p.mean())*(t-t.mean()))/max(float(p.std()*t.std()),1e-8));g=cfg["gates"]["t1"];checks={"label":r["label_top1"]>=g["label_top1_min"],"paired":r["paired_r_at_1"]>=g["paired_r_at_1_min"],"template":ratio<=g["template_ratio_max"],"variance":vr>=g["variance_ratio_min"],"token_effective_rank":rank>=float(g.get("token_effective_rank_min",0.0)),"target_covariance":correlation>=float(g.get("target_covariance_min",-1.0))};return {"gate":"T1","n":len(labels),"metrics":r|{"template_error_ratio":ratio,"variance_ratio":vr,"token_effective_rank":rank,"predicted_target_correlation":correlation},"thresholds":g,"checks":checks}


def _cv_nearest_centroid(x: np.ndarray, y: list[str], folds: int = 5) -> float:
    y=np.asarray(y);predict=[];truth=[]
    for fold in range(folds):
        test=np.arange(len(y))%folds==fold;train=~test;centers={label:x[train&(y==label)].mean(0) for label in sorted(set(y)) if np.any(train&(y==label))}
        for row,label in zip(x[test],y[test]):predict.append(min(centers,key=lambda key:float(np.sum((row-centers[key])**2))));truth.append(label)
    return float(np.mean(np.asarray(predict)==np.asarray(truth)))


@torch.no_grad()
def gate_t1d(config_path,cfg,records,device,token_dataset,batcher):
    audio,_=_load_audio_content(config_path,cfg,device);x=[];labels=[];subjects=[]
    for b in batcher(token_dataset,cfg,device):x.append(audio(b["encodec_codes"],b["encodec_mask"]).mean(1).cpu().numpy());labels+=b["label"];subjects+=b["subject"]
    values=np.concatenate(x);lp,sp=_cv_nearest_centroid(values,labels),_cv_nearest_centroid(values,subjects);g=cfg["gates"]["t1d"];return {"gate":"T1d","n":len(labels),"metrics":{"label_probe_5fold":lp,"speaker_probe_5fold":sp},"thresholds":g,"checks":{"label":lp>=g["label_probe_min"],"speaker":sp<=g["speaker_probe_max"]}}


@torch.no_grad()
def gate_t2_family(config_path,cfg,records,device,phase):
    dataset=selected(records,("fit",),eligible=True,per_label=int(cfg["evaluation"]["oracle_per_label"]));model=_cvae(config_path,cfg,device);voc=_vocoder(config_path,cfg,device);ref=ReferenceAudio(config_path,cfg);teacher=HubertMetric(output_path(config_path,cfg,"hubert_root"),layer=int(cfg["teachers"]["hubert_layer"]),device=device);prior_w=[];post_w=[];analytic_error=[];post_error=[];diversity=[];references=[];labels=[];target_w=[];canonical_w=[];target_embedding=[];sample_wave_sets=[[] for _ in range(int(cfg["evaluation"]["variational_samples"]))];native_frames=int(cfg["audio"]["native_mel_frames"])
    for b in standard_batches(dataset,cfg,device):
        mfcc=b["mfcc"].float();target=F.interpolate(b["speech_t5_mel"].float(),size=256,mode="linear",align_corners=False);prior=model.generate(mfcc,b["canonical_voice"].float(),b["canonical_mfcc_mean"].float(),b["canonical_mfcc_std"].float(),stochastic=False);post=model.reconstruct(mfcc,b["canonical_voice"].float(),b["canonical_mfcc_mean"].float(),b["canonical_mfcc_std"].float(),target,stochastic=False);prior_m=F.interpolate(prior["mel"],size=native_frames,mode="linear",align_corners=False);post_m=F.interpolate(post["mel"],size=native_frames,mode="linear",align_corners=False);analytic=F.interpolate(prior["analytic_mel"],size=native_frames,mode="linear",align_corners=False);lengths=[int(mask.sum())*int(cfg["vocoder"]["hop_length"]) for mask in b["speech_t5_mel_mask"]];prior_w += [wave[:length] for wave,length in zip(render(voc,prior_m),lengths)];post_w += [wave[:length] for wave,length in zip(render(voc,post_m),lengths)];weight=b["speech_t5_mel_mask"].float().unsqueeze(1);analytic_error+=((abs(analytic-b["speech_t5_mel"].float())*weight).sum((1,2))/(weight.sum((1,2))*analytic.shape[1]).clamp_min(1)).cpu().tolist();post_error+=((abs(post_m-b["speech_t5_mel"].float())*weight).sum((1,2))/(weight.sum((1,2))*post_m.shape[1]).clamp_min(1)).cpu().tolist();references += [ref(k) for k in b["sample_key"]];labels+=b["label"]
        samples=[]
        for sample_index in range(int(cfg["evaluation"]["variational_samples"])):
            sample=F.interpolate(model.generate(mfcc,b["canonical_voice"].float(),b["canonical_mfcc_mean"].float(),b["canonical_mfcc_std"].float(),stochastic=True)["mel"],size=native_frames,mode="linear",align_corners=False);samples.append(sample);sample_wave_sets[sample_index] += [wave[:length] for wave,length in zip(render(voc,sample),lengths)]
        stack=torch.stack(samples);diversity+=torch.mean(torch.var(stack,dim=0),dim=(1,2)).cpu().tolist()
        target_prior=model.generate(mfcc,b["speaker_reference"].float(),b["speaker_reference_mfcc_mean"].float(),b["speaker_reference_mfcc_std"].float(),stochastic=False);target_w += [wave[:length] for wave,length in zip(render(voc,F.interpolate(target_prior["mel"],size=native_frames,mode="linear",align_corners=False)),lengths)];canonical_w += [wave[:length] for wave,length in zip(render(voc,prior_m),lengths)];target_embedding+=list(b["speaker_audit_reference"].cpu().numpy())
    pm=hubert_metrics(prior_w,references,labels,teacher);qm=hubert_metrics(post_w,references,labels,teacher)
    if phase=="t2":
        g=cfg["gates"]["t2"];difference=abs(pm["label_top1"]-qm["label_top1"]);return {"gate":"T2","n":len(labels),"metrics":{"prior":pm,"posterior":qm,"prior_posterior_label_gap":difference},"thresholds":g,"checks":{"label":pm["label_top1"]>=g["label_top1_min"],"dtw_gap":pm["correct_minus_wrong_gap"]>=g["dtw_gap_min"],"prior_posterior":difference<=g["prior_posterior_label_gap_max"]}}
    if phase=="t2v":
        g=cfg["gates"]["t2v"];improvement=1-float(np.mean(post_error))/max(float(np.mean(analytic_error)),1e-8);sample_metrics=[hubert_metrics(waves,references,labels,teacher) for waves in sample_wave_sets];retention=float(np.mean([x["label_top1"] for x in sample_metrics]));return {"gate":"T2v","n":len(labels),"metrics":{"analytic_mel_l1":float(np.mean(analytic_error)),"posterior_mel_l1":float(np.mean(post_error)),"posterior_improvement":improvement,"prior_sample_mel_variance":float(np.mean(diversity)),"prior_samples":sample_metrics,"mean_prior_sample_label_retention":retention},"thresholds":g,"checks":{"improvement":improvement>=g["posterior_analytic_improvement_min"],"diversity":float(np.mean(diversity))>0,"label_retention":retention>=g["label_top1_min"]}}
    auditor=ECAPAEncoder(source=str(cfg["speaker"]["model_id"]),savedir=output_path(config_path,cfg,"speaker_model_root"),device=device);target=np.stack([auditor.encode(x) for x in target_w]);canonical=np.stack([auditor.encode(x) for x in canonical_w]);refs=np.stack(target_embedding);score=np.sum(target*refs,axis=1);swapped=np.sum(canonical*refs,axis=1);fit=(records.roles=="fit")&records.arrays["fit_eligible"].astype(bool);distribution=speaker_distribution(records.arrays["speaker_audit_target_embedding"][fit],records.arrays["subjects"][fit].astype(str).tolist(),records.arrays["labels"][fit].astype(str).tolist(),seed=int(cfg["training"]["seed"]));p90=float(distribution["different_speaker_same_label"]["p90"]);gain=score-swapped;rng=np.random.default_rng(int(cfg["training"]["seed"]));samples=int(cfg["evaluation"]["bootstrap_samples"]);draw=np.asarray([gain[rng.integers(0,len(gain),len(gain))].mean() for _ in range(samples)]);boot={"mean_gain":float(gain.mean()),"ci_low":float(np.quantile(draw,.025)),"ci_high":float(np.quantile(draw,.975))};target_content=hubert_metrics(target_w,references,labels,teacher);canon_content=hubert_metrics(canonical_w,references,labels,teacher);delta=abs(target_content["label_top1"]-canon_content["label_top1"]);g=cfg["gates"]["t3"];return {"gate":"T3","n":len(labels),"metrics":{"target_similarity":float(score.mean()),"canonical_similarity":float(swapped.mean()),"swap_margin":float(gain.mean()),"swap_bootstrap":boot,"target_over_p90_fraction":float(np.mean(score>p90)),"different_speaker_same_label_p90":p90,"content_change":delta},"thresholds":g,"checks":{"p90":float(np.mean(score>p90))>=g["target_over_p90_fraction_min"],"swap_ci":boot["ci_low"]>0,"content":delta<=g["content_change_max"]}}


@torch.no_grad()
def eeg_metrics(config_path,cfg,records,device,stage,dataset):
    from scripts.train_open_vocab_v3_encodec_clip import load,modules
    _,decoder,eeg=modules(cfg,device);load(output_path(config_path,cfg,"audio_content_checkpoint"),checkpoint_schema(cfg,"audio"),{"audio":modules(cfg,device)[0],"decoder":decoder},device);load(output_path(config_path,cfg,f"{stage}_checkpoint"),checkpoint_schema(cfg,stage),{"eeg":eeg},device);pred=[];target=[];labels=[];keys=[];subjects=[];controls={"zero":[],"time":[],"channel":[]}
    for b in standard_batches(dataset,cfg,device):
        def run(x):return decoder(eeg(x,b["channel_xyz"].float(),b["channel_mask"],b["time_mask"]))
        pred.append(run(b["eeg"].float()).cpu().numpy());target.append(b["mfcc"].cpu().numpy());controls["zero"].append(run(torch.zeros_like(b["eeg"])).cpu().numpy());controls["time"].append(run(time_shuffled_eeg(b["eeg"].float(),b["time_mask"])).cpu().numpy());controls["channel"].append(run(channel_shuffled_eeg(b["eeg"].float(),b["channel_mask"])).cpu().numpy());labels+=b["label"];keys+=b["sample_key"];subjects+=b["subject"]
    p,t=np.concatenate(pred),np.concatenate(target);control={k:np.concatenate(v) for k,v in controls.items()};r=retrieval(p,t,labels,keys);wins={k:paired_win_rate(p,v,t) for k,v in control.items()};boot={k:bootstrap_mean_gain(p,v,t,samples=int(cfg["evaluation"]["bootstrap_samples"]),seed=int(cfg["training"]["seed"])+i) for i,(k,v) in enumerate(control.items())}
    # Full-fit aggregate scores can hide a single-subject template solution.
    # Subject is never a forward input; it is retained only as a diagnostic
    # stratum in the gate report.
    subject_metrics={}
    subject_values=np.asarray(subjects)
    for subject in sorted(set(subjects)):
        index=np.flatnonzero(subject_values==subject)
        one_prediction,one_target=p[index],t[index]
        one_labels=[labels[i] for i in index];one_keys=[keys[i] for i in index]
        one_retrieval=retrieval(one_prediction,one_target,one_labels,one_keys)
        subject_metrics[str(subject)]={
            "n":int(len(index)),"label_top1":one_retrieval["label_top1"],
            "paired_r_at_1":one_retrieval["paired_r_at_1"],
            "masked_mfcc_mae":float(np.mean(abs(one_prediction-one_target))),
            "variance_ratio":variance_ratio(one_prediction,one_target,one_labels),
            "control_win_rates":{name:paired_win_rate(one_prediction,value[index],one_target) for name,value in control.items()},
        }
    return p,t,control,labels,keys,r|{"masked_mfcc_mae":float(np.mean(abs(p-t))),"variance_ratio":variance_ratio(p,t,labels),"control_win_rates":wins,"correct_minus_control_bootstrap":boot,"per_subject":subject_metrics}
