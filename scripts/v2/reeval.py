"""Re-evaluate a trained fold checkpoint on its validation fold with different eval window counts / TTA.

python scripts/v2/reeval.py --ckpt runs/<run>/fold0/best.pt --vol-dir /tmp/rsna_v2_$USER/vol384 \
    --grid "6,4,4" "12,8,8" "18,12,12"
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import TARGETS, StudyDataset, load_series_table  # noqa: E402
from model import KneeNet  # noqa: E402
from train import auc_macro, collate, predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vol-dir", required=True)
    ap.add_argument("--grid", nargs="+", default=["6,4,4", "12,8,8"])
    ap.add_argument("--labels", default="")
    args = ap.parse_args()
    V2 = "artifacts/private/v2"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    fold = int(os.path.basename(os.path.dirname(args.ckpt)).replace("fold", ""))
    meta = load_series_table(sorted(glob.glob(a["meta"])), a["series_csv"])
    tr = pd.read_csv(a["train_csv"])
    folds = pd.read_csv(a["folds_csv"]).set_index("StudyInstanceUID")
    studies = [s for s in tr.StudyInstanceUID if s in set(meta.StudyInstanceUID)]
    va = [s for s in studies if folds.loc[s, "fold"] == fold]
    lab_path = args.labels or a["labels"].replace("{fold}", str(fold))
    lab = pd.read_csv(lab_path).set_index("StudyInstanceUID").reindex(va)[TARGETS].values
    gold = tr.set_index("StudyInstanceUID").loc[va, TARGETS]
    is_gold = gold.notna().all(1).values
    dev = torch.device("cuda")
    net = KneeNet(a["backbone"], d=a["d"], n_layers=a["layers"], pretrained=False, drop_path=0.0,
                  n_aux=12 if "aux_head.weight" in ck["state"] else 0, head=a.get("head", "tf"))
    net.load_state_dict(ck["state"])
    net.to(dev).eval()
    for g in args.grid:
        ks, kc, ka = (int(v) for v in g.split(","))
        ds = StudyDataset(va, meta, args.vol_dir, lab, n_series=a["n_series"], k=a["k"], train=False,
                          k_plane={"Sagittal": ks, "Coronal": kc, "Axial": ka})
        dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=8, collate_fn=collate)
        p, idx = predict(net, dl, dev, crop=a["img"] if a["img"] != 384 else None, micro=a.get("micro", 0) or 28,
                         zoom=a.get("zoom", 1.0))
        y = lab[idx]
        wk, _ = auc_macro((y > 0.5).astype(int), p)
        gm = auc_macro(gold.values[idx][is_gold[idx]].astype(int), p[is_gold[idx]])[0] if is_gold.sum() >= 5 else np.nan
        print(f"eval windows sag/cor/ax={g}: weak {wk:.4f} gold{int(is_gold.sum())} {gm:.4f}", flush=True)


if __name__ == "__main__":
    main()
