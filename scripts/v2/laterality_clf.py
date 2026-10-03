"""Self-supervised laterality checker on canonical volumes (CPU).

Correctly normalised coronal/axial slices show the lateral side (fibula) on the image left. Train a small CNN to
distinguish normalised slices from horizontally flipped ones using studies whose side came from the DICOM tag or
clear geometry; evaluate on held-out tagged studies; report the verdict for 'weak-geometry' studies.
At inference: if the checker says a weak-geometry study looks flipped, the side should be inverted.

python scripts/v2/laterality_clf.py --out artifacts/private/v2/lat_clf.pt
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

V2 = "artifacts/private/v2"


def slices_for(vol, n=3):
    D = vol.shape[0]
    idx = np.linspace(0.45 * (D - 1), 0.85 * (D - 1), n).round().astype(int)  # mid-to-posterior (fibular head)
    x = torch.from_numpy(np.ascontiguousarray(vol[idx])).float() / 255.0
    return F.interpolate(x[:, None], size=(192, 192), mode="area")[:, 0]  # [n,192,192]


def build(meta, studies, n=3):
    X, ids = [], []
    for st in studies:
        g = meta[(meta.StudyInstanceUID == st) & (meta.plane == "Coronal")].sort_values("depth", ascending=False)
        if not len(g):
            continue
        vol = np.load(f"{V2}/vol384/vol/{g.iloc[0].SeriesInstanceUID}.npy", mmap_mode="r")
        X.append(slices_for(vol, n))
        ids.append(st)
    return torch.stack(X), ids


def net():
    m = torchvision.models.resnet18(num_classes=1)
    m.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{V2}/lat_clf.pt")
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(32)
    fr = []
    for m in sorted(glob.glob(f"{V2}/vol384/meta-train-*.csv")):
        try:
            fr.append(pd.read_csv(m))
        except Exception:
            pass
    meta = pd.concat(fr).drop_duplicates("SeriesInstanceUID", keep="last")
    st = meta.groupby("StudyInstanceUID").side_src.first()
    good = st[st.isin(["tag", "geometry"])].index.tolist()
    weak = st[st == "weak-geometry"].index.tolist()
    rng = np.random.default_rng(0)
    rng.shuffle(good)
    tr, va = good[:2400], good[2400:3200]
    Xtr, _ = build(meta, tr)
    Xva, _ = build(meta, va)
    Xwk, wk_ids = build(meta, weak)
    print("train", Xtr.shape, "val", Xva.shape, "weak", Xwk.shape, flush=True)
    model = net()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), 64):
            x = Xtr[perm[i:i + 64]]
            flip = torch.rand(len(x)) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(-1), x)
            # light augmentation: random shift / contrast
            x = x * torch.empty(len(x), 1, 1, 1).uniform_(0.8, 1.2)
            logit = model(x)[:, 0]
            loss = F.binary_cross_entropy_with_logits(logit, flip.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        model.eval()
        with torch.no_grad():
            p0 = torch.sigmoid(model(Xva)[:, 0])
            p1 = torch.sigmoid(model(Xva.flip(-1))[:, 0])
        acc = ((p0 < 0.5).float().mean() + (p1 >= 0.5).float().mean()).item() / 2
        print(f"epoch {ep} loss {tot / len(Xtr):.4f} val acc (normal vs flipped) {acc:.4f}", flush=True)
    torch.save(model.state_dict(), args.out)
    with torch.no_grad():
        pw = torch.sigmoid(model(Xwk)[:, 0]).numpy()
    out = pd.DataFrame({"StudyInstanceUID": wk_ids, "p_flipped": pw})
    out.to_csv(f"{V2}/lat_weak_verdict.csv", index=False)
    print("weak-geometry studies looking flipped (p>0.5):", int((pw > 0.5).sum()), "of", len(pw))


if __name__ == "__main__":
    main()
