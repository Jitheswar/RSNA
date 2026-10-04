"""Does Kaggle-style fp16 inference (net.half() + fp16 autocast) match bf16 validation? Uses packed volumes of
fold-0 validation studies, the exact kaggle_infer.study_tensors path, and compares AUC vs labels/gold.

python scripts/v2/precision_check.py --ckpt runs/<run>/fold0/best.pt --n 300
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import TARGETS, load_series_table  # noqa: E402
from kaggle_infer import study_tensors  # noqa: E402
from model import KneeNet, center_zoom  # noqa: E402
from train import auc_macro  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--vol-dir", default="/tmp/rsna_v2_ch.jitheswar/vol384")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    fold = int(os.path.basename(os.path.dirname(args.ckpt)).replace("fold", ""))
    meta = load_series_table(sorted(glob.glob(a["meta"])), a["series_csv"])
    folds = pd.read_csv(a["folds_csv"]).set_index("StudyInstanceUID")
    tr = pd.read_csv(a["train_csv"]).set_index("StudyInstanceUID")
    va = [s for s in folds.index if folds.loc[s, "fold"] == fold][: args.n]
    lab = pd.read_csv(a["labels"].replace("{fold}", str(fold))).set_index("StudyInstanceUID").loc[va, TARGETS].values
    ser = pd.read_csv(a["series_csv"])
    smeta = {r.SeriesInstanceUID: (int(r.Fat_Suppression), int(r.Fluid_Sensitive)) for r in ser.itertuples()}
    kp = None
    if a.get("k_sag") or a.get("k_cor") or a.get("k_ax"):
        kp = {"Sagittal": a.get("k_sag") or a["k"], "Coronal": a.get("k_cor") or a["k"], "Axial": a.get("k_ax") or a["k"]}
    dev = torch.device("cuda")
    n_aux = 12 if "aux_head.weight" in ck["state"] else 0

    def build(half):
        net = KneeNet(a["backbone"], d=a["d"], n_layers=a["layers"], pretrained=False, drop_path=0.0, n_aux=n_aux,
                      head=a.get("head", "tf"), img=a.get("img"))
        net.load_state_dict(ck["state"])
        net.eval().to(dev)
        return net.half() if half else net

    nets = {"bf16": build(False), "fp16": build(True), "fp32": build(False)}
    out = {k: [] for k in nets}
    by = {s: g for s, g in meta.groupby("StudyInstanceUID")}
    for s in va:
        g = by[s]
        vols = {}
        for r in g.itertuples():
            v = np.load(f"{args.vol_dir}/{r.SeriesInstanceUID}.npy")
            vols[r.SeriesInstanceUID] = (v, {"plane": r.plane, "slice_mm": r.slice_mm})
        x, types, pos, present, sm = study_tensors(vols, smeta, a["n_series"], a["k"], k_plane=kp)
        T = [torch.from_numpy(v)[None].to(dev) for v in (x, types, pos, present, sm)]
        x, types, pos, present, sm = T
        img = a.get("img", 384)
        xx = center_zoom(x, a["zoom"], img) if a.get("zoom", 1.0) != 1.0 else (
            torch.nn.functional.interpolate(x.view(-1, 3, 384, 384).float(), size=(img, img), mode="bilinear",
                                            align_corners=False).view(*x.shape[:3], 3, img, img) if img != 384 else x)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lb, _ = nets["bf16"](xx, types, pos, present, micro=48, smask=sm)
            with torch.autocast("cuda", dtype=torch.float16):
                lh, _ = nets["fp16"](xx, types, pos.half(), present, micro=48, smask=sm)
            lf, _ = nets["fp32"](xx.float(), types, pos, present, micro=48, smask=sm)
        out["bf16"].append(torch.sigmoid(lb.float())[0].cpu().numpy())
        out["fp16"].append(torch.sigmoid(lh.float())[0].cpu().numpy())
        out["fp32"].append(torch.sigmoid(lf.float())[0].cpu().numpy())
    for k, v in out.items():
        P = np.stack(v)
        nan = int(np.isnan(P).any(1).sum())
        P = np.nan_to_num(P, nan=0.5)
        wk, _ = auc_macro((lab > 0.5).astype(int), P)
        print(f"{k}: weak AUC {wk:.4f}  NaN studies {nan}  mean|p-fp32| {np.abs(P - np.nan_to_num(np.stack(out['fp32']), nan=0.5)).mean():.5f}",
              flush=True)


if __name__ == "__main__":
    main()
