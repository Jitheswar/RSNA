"""OOF evaluation for v2 runs: weak AUC (report labels>0.5) on all studies and gold58 AUC with bootstrap CI.

python scripts/v2/eval_oof.py runs/<a> [runs/<b> ...] [--labels labels/x.csv]
"""
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
T = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
V2 = Path("artifacts/private/v2")

def macro(y, p):
    v = [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if len(np.unique(y[:, j])) == 2]
    return float(np.mean(v)), v

def boot(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        try: out.append(macro(y[i], p[i])[0])
        except ValueError: pass
    return np.percentile(out, [2.5, 97.5])

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--labels", default="labels/pubv2.csv")
ap.add_argument("--ensemble", action="store_true", help="also score the rank-average of all runs (common studies)")
a = ap.parse_args()
tr = pd.read_csv(V2 / "train.csv").set_index("StudyInstanceUID")
gold = tr[tr.ACL.notna()][T].astype(int)
lab = pd.read_csv(V2 / a.labels).set_index("StudyInstanceUID")
for r in a.runs:
    fs = sorted(glob.glob(str(V2 / r / "fold*/oof.csv")))
    if not fs: print(r, "no oof"); continue
    oof = pd.concat([pd.read_csv(f) for f in fs]).set_index("StudyInstanceUID")
    wk = lab.loc[oof.index, T].values > 0.5
    wm, _ = macro(wk.astype(int), oof[T].values)
    g = gold.index.intersection(oof.index)
    gm, gper = macro(gold.loc[g].values, oof.loc[g, T].values) if len(g) >= 10 else (np.nan, [])
    ci = boot(gold.loc[g].values, oof.loc[g, T].values) if len(g) >= 20 else [np.nan, np.nan]
    print(f"{r}: folds={len(fs)} n={len(oof)} weak={wm:.4f} gold{len(g)}={gm:.4f} CI=[{ci[0]:.3f},{ci[1]:.3f}]")
    if len(gper) == 12: print("   per-target gold:", " ".join(f"{t[:6]}={v:.3f}" for t, v in zip(T, gper)))

if a.ensemble and len(a.runs) > 1:
    frames = []
    for r in a.runs:
        fs = sorted(glob.glob(str(V2 / r / "fold*/oof.csv")))
        if fs:
            frames.append(pd.concat([pd.read_csv(f) for f in fs]).set_index("StudyInstanceUID")[T].rank(pct=True))
    common = frames[0].index
    for f in frames[1:]:
        common = common.intersection(f.index)
    ens = sum(f.loc[common] for f in frames) / len(frames)
    g = gold.index.intersection(common)
    gm, gper = macro(gold.loc[g].values, ens.loc[g].values)
    ci = boot(gold.loc[g].values, ens.loc[g].values) if len(g) >= 20 else [np.nan, np.nan]
    print(f"ENSEMBLE of {len(frames)}: n={len(common)} gold{len(g)}={gm:.4f} CI=[{ci[0]:.3f},{ci[1]:.3f}]")
