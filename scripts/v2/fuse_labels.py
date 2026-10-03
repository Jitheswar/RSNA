"""Report x image label fusion (noisy-student refinement).

The reference panel reads images, so when a report is silent or ambiguous the image model's out-of-fold opinion
carries information. Fit, on gold58, a pooled logistic model
    logit P(gold_t) = a_t + b * logit(report_cal_t) + c * logit(image_oof_t)
(shared b, c across targets; per-target intercepts, L2-regularised), report LOO log-loss/AUC versus the report
labels alone, and write fused soft labels for every training study (image term uses OOF predictions only).

python scripts/v2/fuse_labels.py --report labels/cal_v1.csv --oof runs/<run> --out labels/fused_v1.csv
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
V2 = Path("artifacts/private/v2")


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def design(rep, img):
    """rows = (study, target) pairs; columns = [onehot target (12), logit report, logit image]."""
    n, t = rep.shape
    X = np.zeros((n * t, t + 2))
    for j in range(t):
        X[j::t, j] = 1.0
    X[:, t] = logit(rep).reshape(-1)
    X[:, t + 1] = logit(img).reshape(-1)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--oof", nargs="+", required=True, help="run dirs whose fold*/oof.csv are averaged")
    ap.add_argument("--out", default="")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--mix", type=float, default=-1.0,
                    help="if >=0: simple soft bootstrap label = mix*report + (1-mix)*image_oof (no fitting)")
    args = ap.parse_args()
    tr = pd.read_csv(V2 / "train.csv").set_index("StudyInstanceUID")
    gold = tr[tr.ACL.notna()][TARGETS].astype(int)
    rep = pd.read_csv(V2 / args.report).set_index("StudyInstanceUID")[TARGETS]
    oofs = []
    for r in args.oof:
        fs = sorted(glob.glob(str(V2 / r / "fold*/oof.csv")))
        oofs.append(pd.concat([pd.read_csv(f) for f in fs]).set_index("StudyInstanceUID")[TARGETS])
    img = sum(o for o in oofs) / len(oofs)
    uids = gold.index.intersection(img.index).intersection(rep.index)
    R, I, Y = rep.loc[uids].values, img.loc[uids].values, gold.loc[uids].values
    # leave-one-study-out
    loo = np.zeros_like(R, dtype=float)
    for i in range(len(uids)):
        m = np.ones(len(uids), bool)
        m[i] = False
        clf = LogisticRegression(C=args.C, max_iter=2000).fit(design(R[m], I[m]), Y[m].reshape(-1))
        loo[i] = clf.predict_proba(design(R[i:i + 1], I[i:i + 1]))[:, 1]
    def ll(p):
        p = np.clip(p, 1e-3, 1 - 1e-3)
        return float(-(Y * np.log(p) + (1 - Y) * np.log(1 - p)).mean())
    def auc(p):
        return float(np.mean([roc_auc_score(Y[:, j], p[:, j]) for j in range(12)]))
    print(f"gold n={len(uids)}  report: ll={ll(R):.4f} auc={auc(R):.4f} | image oof: ll={ll(I):.4f} auc={auc(I):.4f} | "
          f"fused LOO: ll={ll(loo):.4f} auc={auc(loo):.4f}")
    clf = LogisticRegression(C=args.C, max_iter=2000).fit(design(R, I), Y.reshape(-1))
    print("coef report=%.3f image=%.3f" % (clf.coef_[0][-2], clf.coef_[0][-1]))
    if args.mix >= 0:
        mixed = args.mix * R + (1 - args.mix) * I
        print(f"simple mix {args.mix}: ll={ll(mixed):.4f} auc={auc(mixed):.4f}")
        if args.out:
            all_u = img.index.intersection(rep.index)
            m = args.mix * rep.loc[all_u].values + (1 - args.mix) * img.loc[all_u].values
            out = pd.DataFrame(m, index=all_u, columns=TARGETS)
            out.index.name = "StudyInstanceUID"
            out.reset_index().to_csv(V2 / args.out, index=False)
            print("wrote", args.out, len(out))
        return
    if args.out:
        all_u = img.index.intersection(rep.index)
        fused = clf.predict_proba(design(rep.loc[all_u].values, img.loc[all_u].values))[:, 1].reshape(len(all_u), 12)
        out = pd.DataFrame(fused, index=all_u, columns=TARGETS)
        out.index.name = "StudyInstanceUID"
        out.reset_index().to_csv(V2 / args.out, index=False)
        print("wrote", args.out, len(out))


if __name__ == "__main__":
    main()
