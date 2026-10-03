"""Evaluate fast-lane OOF predictions on the 58 reference studies and choose a blend weight.

Pools per-fold OOF CSVs, scores them alone, then rank-blends with the exact
scored-anchor Gold58 predictions at fixed weights. Reports macro AUC and a
paired study-bootstrap interval for the blend-minus-anchor difference.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ANCHOR = Path("artifacts/private/ticket70-scored-anchor-gold58-v4/terminal-output/development-exact-scored-anchor.csv")


def auc(y, p):
    y = np.asarray(y)
    r = pd.Series(np.asarray(p)).rank().values
    n1 = y.sum()
    n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro(g, p):
    return float(np.nanmean([auc(g[:, j], p[:, j]) for j in range(g.shape[1])]))


def rank_blend(a, b, w):
    ra = pd.DataFrame(a).rank(pct=True).values
    rb = pd.DataFrame(b).rank(pct=True).values
    return (1 - w) * ra + w * rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    train = pd.read_csv("artifacts/private/fast-v1/train.csv").dropna(subset=["ACL"])
    cols = list(train.columns[2:])
    oof = pd.concat([pd.read_csv(p, dtype={"StudyInstanceUID": str}) for p in sorted(args.run.glob("oof-fold*.csv"))])
    oof = oof.set_index("StudyInstanceUID")
    ids = [s for s in train.StudyInstanceUID if s in oof.index]
    g = train.set_index("StudyInstanceUID").loc[ids, cols].values.astype(int)
    ours = oof.loc[ids, cols].values
    anchor = pd.read_csv(ANCHOR, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, cols].values
    print(f"gold studies with OOF: {len(ids)}")
    print(f"ours alone   {macro(g, ours):.4f}")
    print(f"anchor alone {macro(g, anchor):.4f}")
    rng = np.random.default_rng(20261003)
    boots = [rng.integers(0, len(ids), len(ids)) for _ in range(args.boot)]
    for w in [0.05, 0.1, 0.15, 0.2, 0.3]:
        bl = rank_blend(anchor, ours, w)
        diffs = [macro(g[i], bl[i]) - macro(g[i], anchor[i]) for i in boots]
        lo, med, hi = np.nanpercentile(diffs, [2.5, 50, 97.5])
        print(f"w={w:.2f} blend {macro(g, bl):.4f}  delta median {med:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
