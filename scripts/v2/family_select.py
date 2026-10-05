"""Greedy forward selection of model families under a Kaggle inference-time budget.

Each family = one run dir with OOF predictions for every study (5 folds); at test time a family costs
n_models * per-model seconds (measured on 2xT4: ~320 s per ConvNeXt-S 320 model per 1300 studies; cost
factors scale that for bigger inputs / more windows). At each step add the family that most improves the
rank-average objective (weak AUC vs calibrated report labels, optionally mixed with gold58 AUC).

python scripts/v2/family_select.py runs/s_mix1 runs/s_mix1_pw runs/s_mix1_k24:1.29 runs/s_mix1_384:1.44 \
    runs/s_dinob_mix1 --budget 6600 --gold-w 0.5
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
V2 = Path("artifacts/private/v2")


def load(run):
    fs = sorted(glob.glob(str(V2 / run / "fold*/oof.csv")))
    return pd.concat([pd.read_csv(f) for f in fs]).set_index("StudyInstanceUID")[T], len(fs)


def macro(y, p):
    return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if len(np.unique(y[:, j])) == 2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("families", nargs="+", help="run[:cost_factor]")
    ap.add_argument("--labels", default="labels/cal_v3.csv")
    ap.add_argument("--budget", type=float, default=6600.0, help="seconds available for our leg")
    ap.add_argument("--per-model", type=float, default=320.0)
    ap.add_argument("--gold-w", type=float, default=0.5, help="objective = (1-w)*weak + w*gold58")
    args = ap.parse_args()
    fams = {}
    for spec in args.families:
        run, _, cf = spec.partition(":")
        P, n = load(run)
        fams[run] = (P, n * args.per_model * (float(cf) if cf else 1.0))
    idx = None
    for P, _ in fams.values():
        idx = P.index if idx is None else idx.intersection(P.index)
    lab = pd.read_csv(V2 / args.labels).set_index("StudyInstanceUID").loc[idx, T]
    yw = (lab.values > 0.5).astype(int)
    tr = pd.read_csv(V2 / "train.csv").set_index("StudyInstanceUID")
    gold = tr[tr.ACL.notna()][T].astype(int)
    g = gold.index.intersection(idx)
    gpos = np.array([idx.get_loc(s) for s in g])
    yg = gold.loc[g].values
    R = {r: P.loc[idx].rank(pct=True).values for r, (P, _) in fams.items()}
    cost = {r: c for r, (_, c) in fams.items()}

    def score(sel):
        avg = sum(R[r] for r in sel) / len(sel)
        w, gm = macro(yw, avg), macro(yg, avg[gpos])
        return (1 - args.gold_w) * w + args.gold_w * gm, w, gm

    print(f"studies {len(idx)}, gold {len(g)}, budget {args.budget:.0f}s")
    for r in fams:
        s, w, gm = score([r])
        print(f"  single {r:28s} cost {cost[r]:6.0f}s  weak {w:.4f}  gold {gm:.4f}  obj {s:.4f}")
    sel, spent, best = [], 0.0, -1.0
    while True:
        cands = [(score(sel + [r]), r) for r in fams if r not in sel and spent + cost[r] <= args.budget]
        if not cands:
            break
        (s, w, gm), r = max(cands)
        if s <= best:
            break
        sel.append(r)
        spent += cost[r]
        best = s
        print(f"+ {r:28s} -> obj {s:.4f} (weak {w:.4f}, gold {gm:.4f}), spent {spent:.0f}s")
    print("selected:", sel)


if __name__ == "__main__":
    main()
