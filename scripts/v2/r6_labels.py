"""Distil heliosli's public R6 (OAI-free, gold never trained) into our training labels.

R6 predictions on TRAIN studies (scratch_r6/work{0,1,2}/ours_sub.csv) are mostly in-sample for R6, i.e. a smoothed
copy of its Opus-based training targets. Gold58 rows are out-of-sample for R6 -> fair check of the mix.
  --eval : gold58 macro AUC of label candidates.   --out X : write mixed labels (weight --w, mode --mode).
"""
import argparse, glob, numpy as np, pandas as pd
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
V2 = "artifacts/private/v2"
T = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
ap = argparse.ArgumentParser(); ap.add_argument("--base", default=f"{V2}/labels/stud_mm2.csv")
ap.add_argument("--w", type=float, default=0.5); ap.add_argument("--mode", default="temp"); ap.add_argument("--temp", type=float, default=4.0)
ap.add_argument("--eval", action="store_true"); ap.add_argument("--out"); a = ap.parse_args()
r6 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("scratch_r6/work[0-9]/ours_sub.csv"))]).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
base = pd.read_csv(a.base).set_index("StudyInstanceUID")
ids = base.index.intersection(r6.index); print("R6 rows", len(r6), "overlap", len(ids), "of", len(base))
def soft(p, mode, temp):
    p = p.clip(1e-6, 1 - 1e-6)
    if mode == "prob": return p
    if mode == "temp": return expit(logit(p) / temp)
    if mode == "rank":  # per-finding rank mapped onto the base label's marginal distribution
        out = p.copy()
        for c in T: out[c] = np.sort(base.loc[ids, c].values)[(p[c].rank(method="first").values - 1).astype(int)] if len(p) == len(ids) else p[c]
        return out
def mix(w, mode, temp):
    r = soft(r6.loc[ids, T], mode, temp); return (1 - w) * base.loc[ids, T] + w * r
if a.eval:
    tr = pd.read_csv(f"{V2}/train.csv").set_index("StudyInstanceUID"); g = tr[tr.ACL.notna()].index.intersection(ids)
    def auc(df): return np.mean([roc_auc_score(tr.loc[g, c] > 0.5, df.loc[g, c]) for c in T])
    def per(df): return {c[:6]: round(roc_auc_score(tr.loc[g, c] > 0.5, df.loc[g, c]), 3) for c in T}
    print("gold n", len(g)); print("base", round(auc(base), 4), per(base)); print("r6  ", round(auc(r6), 4), per(r6))
    for mode in ["prob", "temp", "rank"]:
        for w in [0.25, 0.4, 0.5, 0.6, 0.75]: print(mode, w, round(auc(mix(w, mode, a.temp)), 4))
if a.out:
    m = mix(a.w, a.mode, a.temp); out = base.copy(); out.loc[ids, T] = m.values
    out.reset_index().to_csv(a.out, index=False); print("wrote", a.out, "mixed rows", len(ids))
