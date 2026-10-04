"""Cross-target report-to-reference calibration for pre-specified finding pairs.

The panel's Fracture label is ranked better by the report's contusion statement than by its fracture statement
(gold58 LOO AUC: fracture-only 0.788, contusion-only 0.885, logistic fusion 0.893): impaction / trabecular fractures
are often written up as bone bruises. For each pair (target <- partner), fit on gold studies
    logit P(gold_target) = a + b1 * logit(cal_target) + b2 * logit(cal_partner)
(L2-regularised) and replace the target column. Fold-specific outputs fit only on gold outside that fold, using that
fold's calibrated table, so gold OOF evaluation stays leak-free.

python scripts/v2/xcal_labels.py --base labels/cal_v3 --out labels/cal_v4 --pairs Fracture:Contusion
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
V2 = Path("artifacts/private/v2")


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def fuse(table: pd.DataFrame, gold: pd.DataFrame, fit_ids, pairs, C):
    out = table.copy()
    for t, q in pairs:
        X = np.c_[lg(table.loc[fit_ids, t]), lg(table.loc[fit_ids, q])]
        clf = LogisticRegression(C=C, max_iter=1000).fit(X, gold.loc[fit_ids, t].values)
        out[t] = clf.predict_proba(np.c_[lg(table[t]), lg(table[q])])[:, 1]
        print(f"  {t} <- {q}: coef {np.round(clf.coef_[0], 3)} intercept {clf.intercept_[0]:.3f} (fit n={len(fit_ids)})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="labels/cal_v3")
    ap.add_argument("--out", default="labels/cal_v4")
    ap.add_argument("--pairs", nargs="+", default=["Fracture:Contusion"])
    ap.add_argument("--C", type=float, default=0.3)
    args = ap.parse_args()
    pairs = [tuple(p.split(":")) for p in args.pairs]
    tr = pd.read_csv(V2 / "train.csv").set_index("StudyInstanceUID")
    gold = tr[tr.ACL.notna()][T].astype(int)
    folds = pd.read_csv(V2 / "folds.csv").set_index("StudyInstanceUID").fold
    g = gold.index
    print("full table:")
    base = pd.read_csv(V2 / f"{args.base}.csv").set_index("StudyInstanceUID")
    fuse(base, gold, g, pairs, args.C).reset_index().to_csv(V2 / f"{args.out}.csv", index=False)
    oof_rows = []
    for k in range(5):
        print(f"fold {k}:")
        bk = pd.read_csv(V2 / f"{args.base}_f{k}.csv").set_index("StudyInstanceUID")
        fit = [s for s in g if folds[s] != k]
        fk = fuse(bk, gold, fit, pairs, args.C)
        fk.reset_index().to_csv(V2 / f"{args.out}_f{k}.csv", index=False)
        oof_rows.append((fk.loc[[s for s in g if folds[s] == k]], bk.loc[[s for s in g if folds[s] == k]]))
    new = pd.concat([a for a, _ in oof_rows]).loc[g]
    old = pd.concat([b for _, b in oof_rows]).loc[g]
    for t, _ in pairs:
        print(f"gold OOF AUC {t}: base {roc_auc_score(gold[t], old[t]):.3f} -> cross-calibrated {roc_auc_score(gold[t], new[t]):.3f}")
    print("gold OOF macro: base %.4f -> %.4f" % (np.mean([roc_auc_score(gold[t], old[t]) for t in T]),
                                                np.mean([roc_auc_score(gold[t], new[t]) for t in T])))


if __name__ == "__main__":
    main()
