from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score


def _active(mel: np.ndarray, activity: np.ndarray | None = None) -> np.ndarray:
    if activity is None: activity = (mel > -55.0).any(axis=0)
    active = np.flatnonzero(np.asarray(activity, dtype=bool))
    if not len(active): return np.full((mel.shape[0], 1), -80.0, dtype=np.float32)
    return np.asarray(mel[:, active[0]:active[-1] + 1], dtype=np.float32)


def normalize_time(mel: np.ndarray, frames: int = 128) -> np.ndarray:
    source = _active(mel)
    old = np.linspace(0, 1, source.shape[1]); new = np.linspace(0, 1, frames)
    return np.stack([np.interp(new, old, row) for row in source]).astype(np.float32)


def ms_ssim(first: np.ndarray, second: np.ndarray) -> float:
    a, b = normalize_time(first), normalize_time(second)
    values = []
    for scale in (1, 2, 4):
        x, y = a[:, ::scale], b[:, ::scale]
        ux, uy = x.mean(1), y.mean(1); vx, vy = x.var(1), y.var(1); cov = ((x - ux[:, None]) * (y - uy[:, None])).mean(1)
        values.append(np.mean(((2*ux*uy + 1e-4)*(2*cov + 9e-4))/((ux*ux+uy*uy+1e-4)*(vx+vy+9e-4))))
    return float(np.clip(np.mean(values), -1, 1))


def soft_iou(first: np.ndarray, second: np.ndarray, threshold: float = -55.0) -> float:
    a, b = normalize_time(first), normalize_time(second)
    x, y = 1/(1+np.exp(-(a-threshold)/4)), 1/(1+np.exp(-(b-threshold)/4))
    return float((x*y).sum() / max((x+y-x*y).sum(), 1e-8))


def soft_dtw(first: np.ndarray, second: np.ndarray, gamma: float = 0.1, band_fraction: float = 0.25) -> float:
    x, y = normalize_time(first).T, normalize_time(second).T
    def cost(a: np.ndarray, b: np.ndarray) -> float:
        d = ((a[:,None,:]-b[None,:,:])**2).mean(-1); n,m=d.shape; band=max(abs(n-m),int(max(n,m)*band_fraction)); r=np.full((n+1,m+1),np.inf); r[0,0]=0
        for i in range(1,n+1):
            for j in range(max(1,i-band),min(m,i+band)+1):
                p=np.array([r[i-1,j],r[i,j-1],r[i-1,j-1]])
                reachable=p[np.isfinite(p)]
                if not len(reachable):
                    continue
                anchor=float(reachable.min())
                softmin=anchor-gamma*np.log(np.exp(-(reachable-anchor)/gamma).sum())
                r[i,j]=d[i-1,j-1]+softmin
        return float(r[n,m])
    return max(0.0, cost(x,y)-0.5*cost(x,x)-0.5*cost(y,y))


def duration_seconds(activity: np.ndarray, hop_seconds: float = 0.01) -> float: return float(np.asarray(activity, dtype=bool).sum() * hop_seconds)


def envelope(mel: np.ndarray) -> np.ndarray: return np.mean(np.power(10.0, np.asarray(mel)/20.0), axis=0)


def envelope_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a,b=envelope(normalize_time(first)), envelope(normalize_time(second))
    if np.std(a)<1e-8 or np.std(b)<1e-8: return 0.0
    return float(np.corrcoef(a,b)[0,1])


@dataclass(frozen=True)
class STSS:
    weights: tuple[float, float, float]
    tau: float

    def components(self, first: np.ndarray, second: np.ndarray) -> dict[str, float]:
        ssim=ms_ssim(first,second); iou=soft_iou(first,second); dtw=soft_dtw(first,second)
        return {"ms_ssim": ssim, "soft_iou": iou, "soft_dtw_divergence": dtw, "soft_dtw_similarity": float(np.exp(-dtw/max(self.tau,1e-6)))}
    def score(self, first: np.ndarray, second: np.ndarray) -> float:
        c=self.components(first,second); return float(np.dot(self.weights, [c["ms_ssim"],c["soft_iou"],c["soft_dtw_similarity"]]))


def fit_stss(positive: Iterable[tuple[np.ndarray,np.ndarray]], negative: Iterable[tuple[np.ndarray,np.ndarray]]) -> tuple[STSS, dict[str,float]]:
    positive=list(positive); negative=list(negative)
    if not positive or not negative: raise ValueError("STSS fitting requires positive and negative perturbations")
    raw_positive=np.array([[ms_ssim(a,b),soft_iou(a,b),soft_dtw(a,b)] for a,b in positive]); tau=float(np.median(raw_positive[:,2][raw_positive[:,2]>0])) if np.any(raw_positive[:,2]>0) else 1.0
    x=np.column_stack((raw_positive[:,0],raw_positive[:,1],np.exp(-raw_positive[:,2]/tau)))
    raw_negative=np.array([[ms_ssim(a,b),soft_iou(a,b),soft_dtw(a,b)] for a,b in negative]); y=np.column_stack((raw_negative[:,0],raw_negative[:,1],np.exp(-raw_negative[:,2]/tau)))
    def objective(w: np.ndarray) -> float:
        margin=(x@w)[:,None]-(y@w)[None,:]
        return float(np.logaddexp(0,-margin).mean()+1e-3*np.square(w).sum())
    result=minimize(objective,np.full(3,1/3),bounds=[(0,1)]*3,constraints={"type":"eq","fun":lambda w:w.sum()-1})
    if not result.success: raise RuntimeError("constrained STSS weight fitting failed")
    score_pos=x@result.x; score_neg=y@result.x
    auc=float(roc_auc_score(np.r_[np.ones(len(score_pos)),np.zeros(len(score_neg))],np.r_[score_pos,score_neg]))
    component_auc=max(float(roc_auc_score(np.r_[np.ones(len(x)),np.zeros(len(y))],np.r_[x[:,i],y[:,i]])) for i in range(3))
    return STSS(tuple(map(float,result.x)),tau), {"auc":auc,"best_component_auc":component_auc,"pairwise_accuracy":float((score_pos[:,None]>score_neg[None,:]).mean())}


def save_stss(path: Path, stss: STSS, report: dict[str,Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps({"schema_version":"openvoice-0728-stss-v1","weights":stss.weights,"tau":stss.tau,"report":report},indent=2)+"\n")


def load_stss(path: Path) -> STSS:
    raw=json.loads(path.read_text());
    if raw.get("schema_version")!="openvoice-0728-stss-v1": raise ValueError("unsupported STSS manifest")
    return STSS(tuple(map(float,raw["weights"])),float(raw["tau"]))
