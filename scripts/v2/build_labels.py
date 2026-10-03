"""Gold-calibrated soft labels from LLM severity grades.

Finding: the reference panel is more liberal than the written definitions (e.g. 'small effusion' is gold-positive
45% of the time, grade-2 MCL sprain 70%), so P(gold | report) must be learned, not assumed. For every source with
ordinal grades, each target gets a monotone (isotonic, Beta-smoothed) map grade-level -> P(gold positive), fit on
the 58 gold studies; 'not addressed' (g=-1) gets its own smoothed rate. Sources are averaged; leave-one-out on
gold58 reports honest log-loss / Brier / AUC.

python scripts/v2/build_labels.py --sources sol61 qwen38 gemma4 --pub-weight 0.0 --out labels/cal_v1.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
V2 = Path("artifacts/private/v2")
# ordinal position of each grade code (special codes placed between severity levels)
ORDER = {
    "ACL": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 0.5, 7: 4.5},
    "Medial Meniscus": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 2.5},
    "Lateral Meniscus": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 2.5},
}


def level(t, g):
    if g < 0:
        return None
    return ORDER.get(t, {}).get(int(g), float(g))


def fit_map(t, grades, y, a=1.0):
    """Return a function grade -> probability, fitted on (grades, y)."""
    grades = np.asarray(grades, float)
    y = np.asarray(y, float)
    prev = (y.sum() + 1) / (len(y) + 2)
    nm = grades < 0
    # not addressed: Beta-smoothed toward a low prior (half prevalence)
    p_nm = (y[nm].sum() + a * prev * 0.5) / (nm.sum() + a)
    lv = np.array([level(t, g) for g in grades[~nm]], float)
    yy = y[~nm]
    if len(lv) == 0:
        return lambda g: p_nm
    uniq = np.unique(lv)
    pos = np.array([yy[lv == u].sum() for u in uniq])
    cnt = np.array([(lv == u).sum() for u in uniq])
    post = (pos + a * prev) / (cnt + a)  # Beta smoothing toward prevalence
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(uniq, post, sample_weight=cnt + a)

    def f(g):
        if g < 0:
            return float(p_nm)
        return float(iso.predict([level(t, g)])[0])
    return f


def load_grades(name):
    p = V2 / f"labels/{name}_grades.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).set_index("StudyInstanceUID")


def calibrate(sources, gold, all_uids, exclude=None):
    """Fit maps on gold (optionally excluding one uid) and produce calibrated probs per source."""
    outs = []
    g_idx = gold.index if exclude is None else gold.index.drop(exclude)
    for name, G in sources.items():
        P = pd.DataFrame(index=all_uids, columns=TARGETS, dtype=float)
        common = g_idx.intersection(G.index)
        for t in TARGETS:
            f = fit_map(t, G.loc[common, t].values, gold.loc[common, t].values)
            vals = G.reindex(all_uids)[t].values
            P[t] = [f(v) if v == v else np.nan for v in vals]
        outs.append(P)
    return outs


def combine(cals, pub, w_pub):
    stack = np.stack([c.values for c in cals])  # [S,N,T]
    mean = np.nanmean(stack, 0)
    out = pd.DataFrame(mean, index=cals[0].index, columns=TARGETS)
    if w_pub > 0:
        out = (1 - w_pub) * out + w_pub * pub.reindex(out.index)[TARGETS]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["sol61"])
    ap.add_argument("--pub-weight", type=float, default=0.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    tr = pd.read_csv(V2 / "train.csv").set_index("StudyInstanceUID")
    gold = tr[tr.ACL.notna()][TARGETS].astype(int)
    pub = pd.read_csv("artifacts/private/public-label-audit/llm_labels_v2.csv").set_index("StudyInstanceUID")[TARGETS]
    sources = {s: load_grades(s) for s in args.sources}
    sources = {k: v for k, v in sources.items() if v is not None}
    print("sources:", {k: len(v) for k, v in sources.items()})
    # leave-one-out on gold58
    loo = pd.DataFrame(index=gold.index, columns=TARGETS, dtype=float)
    for u in gold.index:
        cals = calibrate(sources, gold, [u], exclude=u)
        loo.loc[u] = combine(cals, pub, args.pub_weight).loc[u].values
    y = gold.values.astype(float)
    p = np.clip(loo.values.astype(float), 1e-3, 1 - 1e-3)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(0)
    br = ((p - y) ** 2).mean(0)
    au = [roc_auc_score(y[:, j], p[:, j]) for j in range(12)]
    pubp = np.clip(pub.loc[gold.index].values, 1e-3, 1 - 1e-3)
    ll_pub = -(y * np.log(pubp) + (1 - y) * np.log(1 - pubp)).mean(0)
    res = pd.DataFrame({"auc_loo": au, "logloss_loo": ll, "brier_loo": br, "logloss_pubv2": ll_pub,
                        "auc_pubv2": [roc_auc_score(y[:, j], pubp[:, j]) for j in range(12)]}, index=TARGETS)
    print(res.round(3).to_string())
    print("macro:", res.mean().round(4).to_dict())
    if args.out:
        all_uids = list(tr.index)
        cals = calibrate(sources, gold, all_uids)
        lab = combine(cals, pub, args.pub_weight)
        # studies missing from every source fall back to public labels (0.5 -> 0.1)
        miss = lab.isna().any(axis=1)
        fb = pub.reindex(lab.index).where(~np.isclose(pub.reindex(lab.index), 0.5), 0.1)
        lab[miss] = fb[miss]
        lab.index.name = "StudyInstanceUID"
        lab.reset_index().to_csv(V2 / args.out, index=False)
        print(f"wrote {args.out}: {len(lab)} rows, {int(miss.sum())} from fallback; means", lab.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
