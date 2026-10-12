"""Score multimodal label sources on the 58 gold studies: alone, and mixed 0.5/0.5 with the ens5 student OOF.

Metric = macro AUC over WITHIN-fold positive/negative pairs (the OOF of each fold comes from a different model, so
pooling folds would mix scales). Raw LLM probabilities are used as-is (no fitting on gold -> no leak).

python scripts/v2/mm_eval.py llm/mm-opus-gold.jsonl llm/mm-gemma4-gold.jsonl llm/mm-qwen38-gold.jsonl
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
V2 = Path("artifacts/private/v2")
ENS5 = ["s_mix1", "s_mix1_384", "s_mix1_pw", "s_mix1_k24", "s_dinob_mix1"]


def load_jsonl(p):
    rows = {}
    for line in Path(p).read_text().splitlines():
        r = json.loads(line)
        if r.get("parsed"):
            rows[r["StudyInstanceUID"]] = [float(r["parsed"][t]["p"]) / 100.0 for t in T]
    return pd.DataFrame.from_dict(rows, orient="index", columns=T)


def oof(run):
    fs = sorted(glob.glob(str(V2 / "runs" / run / "fold*/oof.csv")))
    return pd.concat([pd.read_csv(f) for f in fs]).set_index("StudyInstanceUID")[T]


def wf_auc(score, y, fold):
    """Macro AUC over within-fold pos/neg pairs (ties count 0.5)."""
    aucs = []
    for t in T:
        num = den = 0.0
        for f in np.unique(fold):
            m = fold == f
            s, yy = score[t].values[m], y[t].values[m]
            pos, neg = s[yy == 1], s[yy == 0]
            if len(pos) and len(neg):
                d = pos[:, None] - neg[None, :]
                num += (d > 0).sum() + 0.5 * (d == 0).sum()
                den += d.size
        if den:
            aucs.append(num / den)
    return float(np.mean(aucs))


def main():
    train = pd.read_csv(V2 / "train.csv").set_index("StudyInstanceUID")
    gold = train[train.ACL.notna()][T].astype(int)
    fold = pd.read_csv(V2 / "folds.csv").set_index("StudyInstanceUID")["fold"]
    ens = sum(oof(r).rank(pct=True) for r in ENS5) / len(ENS5)  # rank-average within the pool
    ens = ens.loc[gold.index]
    ens = ens.groupby(fold.loc[gold.index]).rank(pct=True)
    ensp = (sum(oof(r) for r in ENS5) / len(ENS5)).loc[gold.index]  # probability-space mean (label recipe)
    f = fold.loc[gold.index].values
    print(f"{'source':38s} n   alone  rank-mix prob-mix (0.5 src + 0.5 ens5)")
    print(f"{'ens5 OOF':38s} {len(gold):2d}  {wf_auc(ens, gold, f):.4f}")
    for p in sys.argv[1:]:
        src = load_jsonl(V2 / p if not Path(p).exists() else p)
        idx = gold.index.intersection(src.index)
        g, ff = gold.loc[idx], fold.loc[idx].values
        s = src.loc[idx]
        mix = 0.5 * s.groupby(ff).rank(pct=True) + 0.5 * ens.loc[idx]
        pmix = 0.5 * s + 0.5 * ensp.loc[idx]
        print(f"{p:38s} {len(idx):2d}  {wf_auc(s, g, ff):.4f}  {wf_auc(mix, g, ff):.4f}  {wf_auc(pmix, g, ff):.4f}")


if __name__ == "__main__":
    main()
