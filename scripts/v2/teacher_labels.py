"""Multi-teacher distillation labels: base (stud_mm2) mixed with a rank-mapped public-model teacher.

Teacher = (1 - a) * rank(R6) + a * [0.65 rank(Raptor-384) + 0.35 rank(Raptor-224)]  (heliosli 0.958 recipe uses a = 0.5).
Raptor weights are OAI-trained (nartaa; disclosed) -> labels from a > 0 carry OAI lineage (final #1 only).
Gold58 rows are out-of-sample for R6 and for Raptor (its notebook: gold never trained).
"""
import argparse, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
V2 = "artifacts/private/v2"
T = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
ap = argparse.ArgumentParser(); ap.add_argument("--base", default=f"{V2}/labels/stud_mm2.csv")
ap.add_argument("--a", type=float, default=0.5); ap.add_argument("--w", type=float, default=0.5)
ap.add_argument("--eval", action="store_true"); ap.add_argument("--out"); A = ap.parse_args()
cat = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pat))]).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
r6 = cat("scratch_r6/work[0-9]/ours_sub.csv"); r384 = cat("scratch_rap/work[0-9]/raptor_raw.csv"); r224 = cat("scratch_rap/work[0-9]/raptor224_raw.csv")
base = pd.read_csv(A.base).set_index("StudyInstanceUID")
ids = base.index.intersection(r6.index).intersection(r384.index).intersection(r224.index); print("rows", len(ids), "of", len(base))
rk = lambda d: d.loc[ids, T].rank(pct=True)
def teacher(a): return (1 - a) * rk(r6) + a * (0.65 * rk(r384) + 0.35 * rk(r224))
def mapped(t):  # rank-map teacher onto the base label's per-finding marginal distribution
    out = t.copy()
    for c in T: out[c] = np.sort(base.loc[ids, c].values)[(t[c].rank(method="first").values - 1).astype(int)]
    return out
def mix(a, w): return (1 - w) * base.loc[ids, T] + w * mapped(teacher(a))
if A.eval:
    tr = pd.read_csv(f"{V2}/train.csv").set_index("StudyInstanceUID"); g = tr[tr.ACL.notna()].index.intersection(ids)
    auc = lambda d: np.mean([roc_auc_score(tr.loc[g, c] > 0.5, d.loc[g, c]) for c in T])
    print("gold", len(g), "base", round(auc(base), 4), "r6", round(auc(r6), 4), "r384", round(auc(r384), 4), "r224", round(auc(r224), 4))
    for a in [0, 0.25, 0.5, 0.75, 1.0]:
        print("a", a, "teacher", round(auc(teacher(a)), 4), " ".join(f"w{w}:{auc(mix(a, w)):.4f}" for w in [0.3, 0.4, 0.5, 0.6, 0.7]))
if A.out:
    out = base.copy(); out.loc[ids, T] = mix(A.a, A.w).values; out.reset_index().to_csv(A.out, index=False); print("wrote", A.out)
