"""Per-target ensemble weights from OOF predictions (rank space), with a shrink toward uniform weights.

For each target, grid-search simplex weights over model families to maximise OOF AUC against the calibrated report
labels (binarised at 0.5) on all studies; report gold58 AUC of uniform vs fitted weights (gold is not used to fit).

python scripts/v2/ens_weights.py runs/t1_cnxs320_cal1 runs/s_mix1 runs/s_mix1_384 --labels labels/cal_v3.csv
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
V2 = Path("artifacts/private/v2")


def load(run):
    fs = sorted(glob.glob(str(V2 / run / "fold*/oof.csv")))
    return pd.concat([pd.read_csv(f) for f in fs]).set_index("StudyInstanceUID")[TARGETS].rank(pct=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", default="labels/cal_v3.csv")
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--shrink", type=float, default=0.5, help="final = shrink*fitted + (1-shrink)*uniform")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    frames = [load(r) for r in args.runs]
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.intersection(f.index)
    lab = pd.read_csv(V2 / args.labels).set_index("StudyInstanceUID").loc[idx, TARGETS]
    tr = pd.read_csv(V2 / "train.csv").set_index("StudyInstanceUID")
    gold = tr[tr.ACL.notna()][TARGETS].astype(int)
    g = gold.index.intersection(idx)
    M = len(frames)
    grid = [w for w in itertools.product(np.arange(0, 1 + 1e-9, args.step), repeat=M) if abs(sum(w) - 1) < 1e-6]
    weights = {}
    rows = []
    for t in TARGETS:
        y = (lab[t] > 0.5).astype(int).values
        P = np.stack([f.loc[idx, t].values for f in frames], 1)
        best = max(grid, key=lambda w: roc_auc_score(y, P @ np.array(w)))
        w = args.shrink * np.array(best) + (1 - args.shrink) * np.ones(M) / M
        weights[t] = w.tolist()
        Pg = np.stack([f.loc[g, t].values for f in frames], 1)
        rows.append((t, roc_auc_score(y, P.mean(1)), roc_auc_score(y, P @ w),
                     roc_auc_score(gold.loc[g, t], Pg.mean(1)), roc_auc_score(gold.loc[g, t], Pg @ w),
                     " ".join(f"{v:.2f}" for v in w)))
    df = pd.DataFrame(rows, columns=["target", "weak_uniform", "weak_fitted", "gold_uniform", "gold_fitted", "weights"])
    print(df.round(4).to_string(index=False))
    print("macro:", df[["weak_uniform", "weak_fitted", "gold_uniform", "gold_fitted"]].mean().round(4).to_dict())
    if args.out:
        Path(args.out).write_text(json.dumps({"runs": args.runs, "weights": weights}, indent=1))


if __name__ == "__main__":
    main()
