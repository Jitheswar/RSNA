"""Gold-calibrated soft labels from several report-label sources.

Finding: the reference panel is more liberal than the written definitions (e.g. 'small effusion' is gold-positive
45% of the time, grade-2 MCL sprain 70%), so P(gold | report) must be learned, not assumed. Each source is
calibrated per target on the 58 gold studies:
  ordinal     LLM severity grades -> isotonic (Beta-smoothed) map; 'not addressed' (g=-1) gets its own rate
  categorical discrete states -> per-state Beta-smoothed rate (no ordering imposed)
  prob        continuous probabilities -> isotonic map on gold58 (binned, smoothed)
Calibrated sources are averaged with weights over the sources available for each study. Leave-one-out on gold58
reports honest log-loss / Brier / AUC.

python scripts/v2/build_labels.py --sources sol61:ordinal:2 qwen38:ordinal:1 astra:categorical:1 pubv2:prob:1 \
    --out labels/cal_v1.csv --fold-specific
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
ORDER = {
    "ACL": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 0.5, 7: 4.5},
    "Medial Meniscus": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 2.5},
    "Lateral Meniscus": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 2.5},
}


def level(t, g):
    if g < 0:
        return None
    return ORDER.get(t, {}).get(int(g), float(g))


def fit_ordinal(t, grades, y, a=1.0):
    grades, y = np.asarray(grades, float), np.asarray(y, float)
    prev = (y.sum() + 1) / (len(y) + 2)
    nm = grades < 0
    p_nm = (y[nm].sum() + a * prev * 0.5) / (nm.sum() + a)
    lv = np.array([level(t, g) for g in grades[~nm]], float)
    yy = y[~nm]
    if len(lv) == 0:
        return lambda g: float(p_nm)
    uniq = np.unique(lv)
    pos = np.array([yy[lv == u].sum() for u in uniq])
    cnt = np.array([(lv == u).sum() for u in uniq])
    post = (pos + a * prev) / (cnt + a)
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(uniq, post, sample_weight=cnt + a)
    return lambda g: float(p_nm) if g < 0 else float(iso.predict([level(t, g)])[0])


def fit_categorical(t, states, y, a=1.0):
    states, y = np.asarray(states, float), np.asarray(y, float)
    prev = (y.sum() + 1) / (len(y) + 2)
    rates = {}
    for s in np.unique(states[~np.isnan(states)]):
        m = states == s
        rates[s] = (y[m].sum() + a * prev) / (m.sum() + a)
    return lambda g: float(rates.get(g, prev))


def fit_prob(t, p, y, a=2.0):
    p, y = np.asarray(p, float), np.asarray(y, float)
    prev = (y.sum() + 1) / (len(y) + 2)
    # smooth toward the raw probability, then isotonic
    target = y
    iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
    iso.fit(np.concatenate([p, [0.0, 1.0]]), np.concatenate([target, [prev * 0.3, 1 - (1 - prev) * 0.3]]))
    return lambda v: float(iso.predict([v])[0])


FITTERS = {"ordinal": fit_ordinal, "categorical": fit_categorical, "prob": fit_prob}


def load_source(name):
    if name == "pubv2":
        p = pd.read_csv("artifacts/private/public-label-audit/llm_labels_v2.csv").set_index("StudyInstanceUID")[TARGETS]
        return p
    for suffix in ("_grades.csv",):
        path = V2 / f"labels/{name}{suffix}"
        if path.exists():
            return pd.read_csv(path).set_index("StudyInstanceUID")[TARGETS]
    return None


def calibrate(sources, gold, all_uids, exclude=None):
    outs = []
    g_idx = gold.index if exclude is None else gold.index.drop(exclude)
    for name, (kind, w, G) in sources.items():
        P = pd.DataFrame(index=all_uids, columns=TARGETS, dtype=float)
        common = g_idx.intersection(G.index)
        sub = G.reindex(all_uids)
        for t in TARGETS:
            f = FITTERS[kind](t, G.loc[common, t].values, gold.loc[common, t].values)
            P[t] = [f(v) if v == v else np.nan for v in sub[t].values]
        outs.append((w, P))
    return outs


def combine(cals):
    num = sum(w * np.nan_to_num(P.values, nan=0.0) for w, P in cals)
    den = sum(w * (~np.isnan(P.values)) for w, P in cals)
    with np.errstate(invalid="ignore"):
        out = num / den
    return pd.DataFrame(out, index=cals[0][1].index, columns=TARGETS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["sol61:ordinal:1"], help="name:kind:weight")
    ap.add_argument("--out", default="")
    ap.add_argument("--fold-specific", action="store_true")
    args = ap.parse_args()
    tr = pd.read_csv(V2 / "train.csv").set_index("StudyInstanceUID")
    gold = tr[tr.ACL.notna()][TARGETS].astype(int)
    pub = pd.read_csv("artifacts/private/public-label-audit/llm_labels_v2.csv").set_index("StudyInstanceUID")[TARGETS]
    sources = {}
    for spec in args.sources:
        name, kind, w = spec.split(":")
        G = load_source(name)
        if G is None:
            print("missing source", name)
            continue
        sources[name] = (kind, float(w), G)
    print("sources:", {k: (v[0], v[1], len(v[2])) for k, v in sources.items()})
    loo = pd.DataFrame(index=gold.index, columns=TARGETS, dtype=float)
    for u in gold.index:
        loo.loc[u] = combine(calibrate(sources, gold, [u], exclude=u)).loc[u].values
    y = gold.values.astype(float)
    p = np.clip(loo.values.astype(float), 1e-3, 1 - 1e-3)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(0)
    br = ((p - y) ** 2).mean(0)
    au = [roc_auc_score(y[:, j], p[:, j]) for j in range(12)]
    res = pd.DataFrame({"auc_loo": au, "logloss_loo": ll, "brier_loo": br}, index=TARGETS)
    print(res.round(3).to_string())
    print("macro:", res.mean().round(4).to_dict())
    if args.out:
        all_uids = list(tr.index)
        fb = pub.reindex(all_uids).where(~np.isclose(pub.reindex(all_uids), 0.5), 0.1)

        def write(g_used, path):
            lab = combine(calibrate(sources, g_used, all_uids))
            miss = lab.isna().any(axis=1)
            lab[miss] = fb[miss]
            lab.index.name = "StudyInstanceUID"
            lab.reset_index().to_csv(path, index=False)
            return lab, int(miss.sum())
        lab, nmiss = write(gold, V2 / args.out)
        print(f"wrote {args.out}: {len(lab)} rows, {nmiss} fallback; means", lab.mean().round(3).to_dict())
        if args.fold_specific:
            folds = pd.read_csv(V2 / "folds.csv").set_index("StudyInstanceUID")["fold"]
            for k in sorted(folds.unique()):
                g_k = gold[folds.reindex(gold.index).values != k]
                path = V2 / args.out.replace(".csv", f"_f{k}.csv")
                write(g_k, path)
                print("wrote", path.name, "gold used", len(g_k))


if __name__ == "__main__":
    main()
