#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path: sys.path.insert(0,str(APP))

from src.open_vocab_0728.data import internal_split, load_context
from src.open_vocab_0728.runtime import load_config, resolve_config_path


def main() -> None:
    parser=argparse.ArgumentParser(description="Validate v0728 namespace, data and split before writing artifacts")
    parser.add_argument("--config",type=Path,required=True); args=parser.parse_args()
    path,cfg=load_config(args.config); context=load_context(path,cfg)
    split=internal_split(context.rows,seed=int(cfg["data"]["internal_split_seed"]),development_subjects=context.development_subjects)
    required={"train","validation","locked_test"}
    if required-set(split.values()): raise ValueError("v0728 internal split is incomplete")
    print(f"[0728 preflight] output={resolve_config_path(path,cfg['paths']['output_root'])}")
    print(f"[0728 preflight] development subjects={len(context.development_subjects)} trials={len(split)}")
    print("[0728 preflight] passed; v0724/v0725 write firewall is active")
if __name__=="__main__": main()
