#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path: sys.path.insert(0,str(APP))

from src.open_vocab_0728.runtime import allow_failed_gates, load_config, resolve_config_path, write_json


def median(records:list[dict],condition:str,key:str)->float:
    return float(np.median([record["conditions"][condition][key] for record in records]))


def win(records:list[dict],first:str,second:str)->float:
    return float(np.mean([record["conditions"][first]["stss"]>record["conditions"][second]["stss"] for record in records]))


def main()->None:
    parser=argparse.ArgumentParser(description="Fail-closed v0728 dual-latent generation gates")
    parser.add_argument("--config",type=Path,required=True); parser.add_argument("--phase",choices=("semantic4","dual4","full11"),required=True); parser.add_argument("--manifest",type=Path,required=True); parser.add_argument("--freeze-locked-test",action="store_true")
    arg=parser.parse_args(); config,cfg=load_config(arg.config); raw=json.loads(arg.manifest.read_text()); records=raw["records"]
    if not records: raise ValueError("empty synthesis manifest")
    correct=median(records,"correct","stss"); shuffled=median(records,"realization_shuffle","stss"); content=median(records,"content_only","stss"); zero=median(records,"zero_eeg","stss")
    all_factor=median(records,"all_factor_shuffle","stss"); label_median=median(records,"label_median_baseline","stss")
    negative_evidence=np.median([value for record in records for value in (record["conditions"]["zero_eeg"]["evidence"],record["conditions"]["gaussian_noise_eeg"]["evidence"])])
    report={"phase":arg.phase,"trials":len(records),"median_stss":{"correct":correct,"realization_shuffle":shuffled,"content_only":content,"all_factor_shuffle":all_factor,"label_median":label_median,"zero":zero},"gains":{"correct_minus_realization_shuffle":correct-shuffled,"shuffle_minus_content":shuffled-content,"correct_minus_content":correct-content,"correct_minus_all_factor_shuffle":correct-all_factor,"correct_minus_label_median":correct-label_median,"correct_minus_zero":correct-zero},"win_rates":{"correct_vs_realization_shuffle":win(records,"correct","realization_shuffle"),"correct_vs_content":win(records,"correct","content_only"),"correct_vs_zero":win(records,"correct","zero_eeg"),"correct_vs_label_median":win(records,"correct","label_median_baseline")},"negative_evidence_median":float(negative_evidence)}
    root=resolve_config_path(config,cfg["paths"]["output_root"])
    metric_name={"semantic4":"eeg_semantic4","dual4":"eeg_dual4","full11":"eeg_full11"}[arg.phase]
    validation_path=root/metric_name/"metrics"/"validation.json"
    validation=json.loads(validation_path.read_text()) if validation_path.exists() else {}
    report["validation"] = validation
    real_evidence=median(records,"correct","evidence")
    common=real_evidence>=float(cfg["evaluation"]["evidence_real_minimum"]) and validation.get("maximum_prediction_fraction",1.0)<=float(cfg["evaluation"]["maximum_single_class_fraction"])
    if arg.phase=="semantic4":
        passed=common and validation.get("content_retrieval_macro_top1",0.0)>=float(cfg["evaluation"]["four_label_accuracy_minimum"]) and report["gains"]["correct_minus_zero"]>=float(cfg["evaluation"]["full_vs_content_stss_gain_minimum"])
    else:
        retrieval=float(validation.get("content_retrieval_macro_top1",0.0)); chance=0.25 if arg.phase=="dual4" else 1/11
        passed=common and retrieval>=max(float(cfg["evaluation"]["four_label_accuracy_minimum"]) if arg.phase=="dual4" else 0.0,2*chance) and report["gains"]["correct_minus_realization_shuffle"]>=float(cfg["evaluation"]["realization_stss_gain_minimum"]) and report["gains"]["shuffle_minus_content"]>=float(cfg["evaluation"]["shuffled_vs_content_stss_gain_minimum"]) and report["gains"]["correct_minus_content"]>=float(cfg["evaluation"]["full_vs_content_stss_gain_minimum"]) and report["gains"]["correct_minus_all_factor_shuffle"]>=float(cfg["evaluation"]["full_vs_content_stss_gain_minimum"]) and report["gains"]["correct_minus_label_median"]>0 and report["gains"]["correct_minus_zero"]>=float(cfg["evaluation"]["full_vs_content_stss_gain_minimum"]) and report["win_rates"]["correct_vs_realization_shuffle"]>=float(cfg["evaluation"]["minimum_trial_win_rate"]) and negative_evidence<=float(cfg["evaluation"]["evidence_negative_maximum"])
    report["passed"]=bool(passed); target=root/"evaluation"/f"gate_{arg.phase}_{raw.get('split','unknown')}.json"; write_json(target,report)
    if arg.freeze_locked_test:
        if not passed: raise RuntimeError("may not freeze locked test after failed validation gate")
        from src.open_vocab_0728.lineage import build_lineage, freeze_locked_test
        from src.open_vocab_0728.data import load_context
        context=load_context(config,cfg); lineage=build_lineage(config,cfg,manifest=context.manifest_path,split=context.split_path,montage=context.montage_path)
        fingerprints={"phase":arg.phase,"synthesis_manifest":str(arg.manifest.resolve()),"synthesis_manifest_sha256":__import__("hashlib").sha256(arg.manifest.read_bytes()).hexdigest()}
        freeze_locked_test(resolve_config_path(config,cfg["paths"]["locked_test_freeze"]),lineage=lineage,fingerprints=fingerprints)
    if not passed and not allow_failed_gates(cfg): raise RuntimeError(f"v0728 gate failed: {report}")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
