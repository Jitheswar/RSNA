"""Compare model families and their rank-average ensembles on OOF predictions (common studies only).

Weak AUC = macro AUC against calibrated report labels > 0.5; gold = macro AUC on the reference-labelled studies
(with bootstrap CI for single families). Every subset of the given families is scored as an equal-weight rank average.

python scripts/v2/family_eval.py runs/s_mix1 runs/s_dinob_mix1 --folds 0 --labels labels/cal_v3.csv
"""

from __future__ import annotations

import argparse
import glob
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
V2 = Path("artifacts/private/v2")


def load(run, folds):
    fs = sorted(glob.glob(str(V2 / run / "fold*/oof.csv")))
    if folds is not None:
        fs = [f for f in fs if int(Path(f).parent.name.replace("fold", "")) in folds]
    return pd.concat([pd.read_csv(f) for f in fs]).set_index("StudyInstanceUID")[T]


def macro(y, p):
    return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if len(np.unique(y[:, j])) == 2]))


def boot(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        try:
            out.append(macro(y[i], p[i]))
        except ValueError:
            pass
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--folds", default="", help="comma list of folds to use (default: all available)")
    ap.add_argument("--labels", default="labels/cal_v3.csv")
    ap.add_argument("--per-target", action="store_true")
    args = ap.parse_args()
    folds = [int(f) for f in args.folds.split(",")] if args.folds else None
    P = {r: load(r, folds) for r in args.runs}
    idx = None
    for p in P.values():
        idx = p.index if idx is None else idx.intersection(p.index)
    lab = pd.read_csv(V2 / args.labels).set_index("StudyInstanceUID").loc[idx, T]
    yw = (lab.values > 0.5).astype(int)
    tr = pd.read_csv(V2 / "train.csv").set_index("StudyInstanceUID")
    gold = tr[tr.ACL.notna()][T].astype(int)
    g = gold.index.intersection(idx)
    yg = gold.loc[g].values
    R = {r: p.loc[idx].rank(pct=True) for r, p in P.items()}
    print(f"common studies {len(idx)}, gold {len(g)}")
    rows = []
    for k in range(1, len(args.runs) + 1):
        for combo in itertools.combinations(args.runs, k):
            avg = sum(R[r] for r in combo) / k
            w = macro(yw, avg.values)
            gm = macro(yg, avg.loc[g].values)
            ci = boot(yg, avg.loc[g].values) if k == 1 else (np.nan, np.nan)
            rows.append((" + ".join(Path(r).name for r in combo), w, gm, ci[0], ci[1]))
    df = pd.DataFrame(rows, columns=["models", "weak", "gold", "gold_lo", "gold_hi"])
    print(df.round(4).to_string(index=False))
    if args.per_target:
        per = pd.DataFrame({Path(r).name: [roc_auc_score(yw[:, j], R[r].values[:, j]) for j in range(12)] for r in args.runs},
                           index=T)
        per["all_avg"] = [roc_auc_score(yw[:, j], (sum(R[r] for r in args.runs) / len(args.runs)).values[:, j])
                          for j in range(12)]
        print(per.round(4).to_string())


if __name__ == "__main__":
    main()
