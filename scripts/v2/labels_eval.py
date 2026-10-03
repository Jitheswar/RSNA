"""Turn LLM JSONL outputs into label tables and score every label source against the 58 gold studies.

python scripts/v2/labels_eval.py  [--write]   -> prints per-target gold AUC table; --write saves labels/*.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
V2 = Path("artifacts/private/v2")
SOURCES = {"sol61": V2 / "llm/sol61-high.jsonl", "qwen38": V2 / "llm/qwen38-think.jsonl", "gemma4": V2 / "llm/gemma4.jsonl"}


def load_jsonl(path):
    rows = {}
    if not Path(path).exists():
        return None
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r.get("parsed"):
            rows[r["StudyInstanceUID"]] = r["parsed"]
    return rows


def to_frames(rows):
    uids = list(rows)
    p = np.array([[float(rows[u][t].get("p", 50)) / 100.0 for t in TARGETS] for u in uids])
    g = np.array([[float(rows[u][t].get("g", -1)) for t in TARGETS] for u in uids])
    side = [rows[u].get("side", "unknown") for u in uids]
    postop = [bool(rows[u].get("postop", False)) for u in uids]
    P = pd.DataFrame(np.clip(p, 0.0, 1.0), index=uids, columns=TARGETS)
    G = pd.DataFrame(g, index=uids, columns=TARGETS)
    return P, G, pd.Series(side, index=uids), pd.Series(postop, index=uids)


def gold_auc(scores: pd.DataFrame, gold: pd.DataFrame):
    common = gold.index.intersection(scores.index)
    out = {}
    for t in TARGETS:
        y = gold.loc[common, t].values
        s = scores.loc[common, t].values
        out[t] = roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
    return out, len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    tr = pd.read_csv(V2 / "train.csv").set_index("StudyInstanceUID")
    gold = tr[tr.ACL.notna()][TARGETS].astype(int)
    table = {}
    pub = pd.read_csv("artifacts/private/public-label-audit/llm_labels_v2.csv").set_index("StudyInstanceUID")[TARGETS]
    table["pubv2"], n = gold_auc(pub, gold)
    frames = {}
    for name, path in SOURCES.items():
        rows = load_jsonl(path)
        if not rows:
            continue
        P, G, side, postop = to_frames(rows)
        frames[name] = P
        table[name], n = gold_auc(P, gold)
        table[name + "_n"] = {"n_gold": n, "n_all": len(P)}
        print(f"{name}: {len(P)} labelled, {n} gold")
        if args.write:
            out = P.copy()
            out.index.name = "StudyInstanceUID"
            out.reset_index().to_csv(V2 / f"labels/{name}.csv", index=False)
            G.reset_index().rename(columns={"index": "StudyInstanceUID"}).to_csv(V2 / f"labels/{name}_grades.csv", index=False)
            pd.DataFrame({"side": side, "postop": postop}).rename_axis("StudyInstanceUID").reset_index() \
                .to_csv(V2 / f"labels/{name}_meta.csv", index=False)
    # simple ensembles over available sources (rank-free mean of probabilities)
    if len(frames) >= 2:
        common = sorted(set.intersection(*[set(f.index) for f in frames.values()]))
        ens = sum(f.loc[common] for f in frames.values()) / len(frames)
        table["mean_llm"], _ = gold_auc(ens, gold)
    if "sol61" in frames:
        common = frames["sol61"].index.intersection(pub.index)
        mix = 0.7 * frames["sol61"].loc[common] + 0.3 * pub.loc[common]
        table["0.7sol+0.3pub"], _ = gold_auc(mix, gold)
    res = pd.DataFrame({k: v for k, v in table.items() if not k.endswith("_n")}).T
    res["macro"] = res[TARGETS].mean(1)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
