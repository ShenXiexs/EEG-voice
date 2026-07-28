#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path: sys.path.insert(0,str(APP))

from src.open_vocab_0728.runtime import allow_failed_gates, load_config, resolve_config_path, write_json


def collect(root:Path,prefix:str)->dict[str,float]:
    result={}
    for path in root.glob(f"runs/{prefix}_loso_*/eeg_full11/metrics/validation.json"):
        run=path.parts[path.parts.index("runs")+1]
        result[run]=float(json.loads(path.read_text())["content_retrieval_macro_top1"])
    return result


def main()->None:
    parser=argparse.ArgumentParser(description="Aggregate physically isolated v0728 LOSO folds")
    parser.add_argument("--config",type=Path,required=True); args=parser.parse_args(); config,cfg=load_config(args.config); root=resolve_config_path(config,cfg["paths"]["output_root"]); chance=1/11; threshold=2*chance
    def summarize(values:dict[str,float])->dict:
        seeds={15:[],31:[],47:[]}
        for run,value in values.items():
            for seed in seeds:
                if run.endswith(f"seed_{seed}"): seeds[seed].append(value)
        primary=seeds[15]
        return {"fold_counts":{str(k):len(v) for k,v in seeds.items()},"median_seed15":float(np.median(primary)) if primary else None,"folds_above_chance_seed15":int(np.sum(np.asarray(primary)>chance)) if primary else 0,"seed_medians":{str(k):float(np.median(v)) if v else None for k,v in seeds.items()}}
    payload={"schema_version":"openvoice-0728-loso-aggregate-v1","chance":chance,"shared_audio_prior":summarize(collect(root,"shared_audio_prior")),"strict_end_to_end":summarize(collect(root,"strict_end_to_end"))}
    payload["passed"]=bool(payload["shared_audio_prior"]["fold_counts"]["15"]==12 and payload["strict_end_to_end"]["fold_counts"]["15"]==12 and payload["shared_audio_prior"]["median_seed15"]>=threshold and payload["strict_end_to_end"]["median_seed15"]>=threshold and payload["shared_audio_prior"]["folds_above_chance_seed15"]>=9 and payload["strict_end_to_end"]["folds_above_chance_seed15"]>=9)
    target=root/"evaluation"/"loso_aggregate.json"; write_json(target,payload); print(json.dumps(payload,indent=2))
    if not payload["passed"] and not allow_failed_gates(cfg): raise RuntimeError("LOSO aggregate gate failed")
if __name__=="__main__": main()
