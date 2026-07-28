#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main()->None:
    parser=argparse.ArgumentParser(description="Render numerical v0728 mel counterfactuals")
    parser.add_argument("--manifest",type=Path,required=True); parser.add_argument("--limit",type=int,default=24); args=parser.parse_args(); raw=json.loads(args.manifest.read_text()); root=args.manifest.parent/"figures"; root.mkdir(parents=True,exist_ok=True)
    for record in raw["records"][:args.limit]:
        names=["correct","realization_shuffle","content_shuffle","content_only","zero_eeg"]; fig,axes=plt.subplots(1,len(names),figsize=(4*len(names),4))
        for axis,name in zip(axes,names):
            axis.imshow(np.load(record["conditions"][name]["mel_path"]),origin="lower",aspect="auto",vmin=-80,vmax=0,cmap="magma"); axis.set_title(f"{name}\nSTSS={record['conditions'][name]['stss']:.3f}"); axis.set_xlabel("time"); axis.set_ylabel("mel")
        fig.suptitle(f"{record['sample_key']} | {record['label']}"); fig.tight_layout(); fig.savefig(root/f"{record['sample_key']}.png",dpi=160); plt.close(fig)
if __name__=="__main__": main()
