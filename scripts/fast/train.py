"""2.5D multi-series study classifier trained on report-derived soft labels.

Per study: up to MAX_SERIES series, each a (DEPTH, H, W) uint8 volume. Slices
are encoded as 3-channel adjacent-slice stacks by a shared timm backbone, then
pooled with a slice transformer per series and attention across series that
carries a plane/sequence-type embedding. Outputs twelve logits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
TYPES = {("Sagittal", 0): 0, ("Sagittal", 1): 1, ("Coronal", 0): 2, ("Coronal", 1): 3,
         ("Axial", 0): 4, ("Axial", 1): 5}
ROOT = Path("artifacts/private/fast-v1")
_SHARDS: dict = {}


def load_volume(row) -> np.ndarray:
    """Per-series file (cluster) or memory-mapped consolidated shard (Kaggle)."""
    if isinstance(row.path, str):
        return np.load(row.path, mmap_mode="r")
    key = (row.root, str(row.shard))
    if key not in _SHARDS:
        _SHARDS[key] = np.load(Path(row.root) / f"vols-{key[1]}.npy", mmap_mode="r")
    return _SHARDS[key][int(row.row)]


def series_type(plane: str, fluid: int) -> int:
    return TYPES.get((plane, int(fluid)), 0)


def load_tables(labels_csv: Path, train_csv: Path, series_csv: Path, nfolds: int, vol_root: str | None = None):
    if vol_root is None:
        index = pd.concat([pd.read_csv(p) for p in sorted(ROOT.glob("index-*.csv"))]
                          + [pd.read_csv(p) for p in sorted((ROOT / "gold").glob("index-gold.csv"))])
    else:  # comma-separated consolidated roots (Kaggle datasets)
        parts = []
        for root in str(vol_root).split(","):
            part = pd.read_csv(Path(root) / "index.csv", dtype={"shard": str})
            part["path"] = None
            part["root"] = root
            parts.append(part)
        index = pd.concat(parts, ignore_index=True)
    meta = pd.read_csv(series_csv)
    index = index.merge(meta, left_on=["study", "series"],
                        right_on=["StudyInstanceUID", "SeriesInstanceUID"])
    index["stype"] = [series_type(p, f) for p, f in zip(index.Anatomical_Plane, index.Fluid_Sensitive)]
    gold = pd.read_csv(train_csv).dropna(subset=["ACL"])
    soft = pd.read_csv(labels_csv).set_index("StudyInstanceUID")[TARGETS]
    soft = soft.where(soft != 0.5, 0.1)
    studies = sorted(set(index.study) & set(soft.index))
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(studies))
    fold = {studies[i]: int(k % nfolds) for k, i in enumerate(perm)}
    gold_ids = set(gold.StudyInstanceUID)
    for k, s in enumerate(sorted(gold_ids & set(studies))):  # spread gold evenly
        fold[s] = k % nfolds
    groups = {s: g for s, g in index.groupby("study")}
    return studies, fold, groups, soft, gold.set_index("StudyInstanceUID")[TARGETS]


class StudyDataset(Dataset):
    def __init__(self, studies, groups, labels, train: bool, n_slices: int, max_series: int):
        self.studies, self.groups, self.labels = studies, groups, labels
        self.train, self.n_slices, self.max_series = train, n_slices, max_series

    def __len__(self):
        return len(self.studies)

    def _slices(self, vol: np.ndarray) -> np.ndarray:
        depth = vol.shape[0]
        if self.train:
            centers = np.linspace(1, depth - 2, self.n_slices) + np.random.uniform(-0.5, 0.5)
            centers = np.clip(np.round(centers), 1, depth - 2).astype(int)
        else:
            centers = np.round(np.linspace(1, depth - 2, self.n_slices)).astype(int)
        stack = np.stack([vol[centers - 1], vol[centers], vol[centers + 1]], axis=1)
        return stack  # (S, 3, H, W)

    def __getitem__(self, i):
        study = self.studies[i]
        g = self.groups[study]
        if self.train and len(g) > self.max_series:
            g = g.sample(self.max_series)
        g = g.head(self.max_series)
        vols, types = [], []
        for row in g.itertuples():
            vols.append(self._slices(load_volume(row)))
            types.append(row.stype)
        x = np.zeros((self.max_series, self.n_slices, 3) + vols[0].shape[-2:], np.uint8)
        mask = np.zeros(self.max_series, bool)
        t = np.zeros(self.max_series, np.int64)
        for k, v in enumerate(vols):
            x[k], mask[k], t[k] = v, True, types[k]
        y = self.labels.loc[study].values.astype(np.float32) if self.labels is not None else np.zeros(12, np.float32)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(t), torch.from_numpy(y)


class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x, mask):
        s = self.score(x).squeeze(-1).masked_fill(~mask, -1e4)
        return (torch.softmax(s, -1).unsqueeze(-1) * x).sum(-2)


class StudyModel(nn.Module):
    def __init__(self, backbone: str, dim: int = 384, pretrained: bool = True):
        super().__init__()
        local = Path(os.environ.get("TIMM_WEIGHTS_DIR", "/nonexistent")) / f"{backbone}.safetensors"
        if pretrained and local.exists():  # offline environments (Kaggle TPU without internet)
            self.encoder = timm.create_model(backbone, pretrained=True, num_classes=0,
                                             pretrained_cfg_overlay={"file": str(local)})
        else:
            self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat = self.encoder.num_features
        self.proj = nn.Linear(feat, dim)
        self.pos = nn.Parameter(torch.zeros(1, 64, dim))
        self.type_emb = nn.Embedding(len(TYPES), dim)
        layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, 0.1, batch_first=True, norm_first=True)
        self.slice_tf = nn.TransformerEncoder(layer, 2)
        self.slice_pool = AttnPool(dim)
        layer2 = nn.TransformerEncoderLayer(dim, 8, dim * 2, 0.1, batch_first=True, norm_first=True)
        self.series_tf = nn.TransformerEncoder(layer2, 1)
        self.series_pool = AttnPool(dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.2), nn.Linear(dim, 12))

    def forward(self, x, mask, types):
        b, n, s, c, h, w = x.shape
        flat = x[mask]  # (V, S, 3, H, W) only valid series
        v = flat.shape[0]
        with torch.autocast(flat.device.type, dtype=AMP_DTYPE if flat.is_cuda else torch.bfloat16):
            f = self.encoder(flat.reshape(v * s, c, h, w).float().div_(255).sub_(0.45).div_(0.25))
        f = self.proj(f.float()).reshape(v, s, -1) + self.pos[:, :s]
        f = self.slice_tf(f)
        f = self.slice_pool(f, torch.ones(v, s, dtype=torch.bool, device=f.device))
        series = f.new_zeros(b, n, f.shape[-1])
        series[mask] = f
        series = series + self.type_emb(types)
        series = self.series_tf(series, src_key_padding_mask=~mask)
        return self.head(self.series_pool(series, mask))


def augment(x: torch.Tensor) -> torch.Tensor:
    """Light per-study GPU augmentation: random scale/shift crop and contrast."""
    b, n, s, c, h, w = x.shape
    x = x.float()
    out = torch.empty_like(x)
    for i in range(b):
        scale = np.random.uniform(0.85, 1.15)
        tx, ty = np.random.uniform(-0.08, 0.08, 2)
        rot = np.random.uniform(-10, 10) * math.pi / 180
        theta = torch.tensor([[scale * math.cos(rot), -scale * math.sin(rot), tx],
                              [scale * math.sin(rot), scale * math.cos(rot), ty]],
                             device=x.device, dtype=torch.float32)
        xi = x[i].reshape(n * s, c, h, w)
        grid = F.affine_grid(theta.expand(n * s, 2, 3), (n * s, c, h, w), align_corners=False)
        xi = F.grid_sample(xi, grid, align_corners=False)
        gamma = np.random.uniform(0.8, 1.25)
        xi = (xi / 255).clamp(0, 1).pow(gamma) * 255 * np.random.uniform(0.9, 1.1)
        out[i] = xi.reshape(n, s, c, h, w)
    return out.clamp(0, 255)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    r = pd.Series(p).rank().values
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


AMP_DTYPE = torch.bfloat16


def predict(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for x, m, t, _ in loader:
            out.append(torch.sigmoid(model(x.to(device), m.to(device), t.to(device)).float()).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--backbone", default="convnext_nano.in12k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr-mult", type=float, default=3.0)
    ap.add_argument("--n-slices", type=int, default=12)
    ap.add_argument("--max-series", type=int, default=6)
    ap.add_argument("--labels", type=Path, default=Path("artifacts/private/public-label-audit/llm_labels_v2.csv"))
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--series-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--vol-root", default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-hours", type=float, default=0.0)
    args = ap.parse_args()
    global AMP_DTYPE
    if args.fp16:
        AMP_DTYPE = torch.float16
    scaler = torch.amp.GradScaler(enabled=args.fp16)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.fold)
    np.random.seed(args.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())) - args.workers)

    studies, fold, groups, soft, gold = load_tables(args.labels, args.train_csv, args.series_csv, args.nfolds, args.vol_root)
    labels = soft.copy()
    labels.loc[gold.index.intersection(labels.index)] = gold.loc[gold.index.intersection(labels.index)].values
    tr = [s for s in studies if fold[s] != args.fold]
    va = [s for s in studies if fold[s] == args.fold]
    if args.limit:
        tr, va = tr[: args.limit], va[: args.limit // 4]
    dl_tr = DataLoader(StudyDataset(tr, groups, labels, True, args.n_slices, args.max_series),
                       batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=True,
                       pin_memory=True, persistent_workers=True)
    dl_va = DataLoader(StudyDataset(va, groups, labels, False, args.n_slices, 8),
                       batch_size=args.bs, num_workers=args.workers, pin_memory=True)

    model = StudyModel(args.backbone).to(device)
    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    other = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr},
                             {"params": other, "lr": args.lr * args.head_lr_mult}], weight_decay=0.05)
    total = args.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * args.head_lr_mult], total_steps=total,
                                                pct_start=0.1, anneal_strategy="cos")
    log = []
    best = -1.0
    started = time.time()
    for epoch in range(args.epochs):
        if args.max_hours and log and (time.time() - started) + log[-1]["seconds"] > args.max_hours * 3600:
            print(json.dumps({"stopped_early_at_epoch": epoch}), flush=True)
            break
        model.train()
        t0, losses = time.time(), []
        for x, m, t, y in dl_tr:
            x = augment(x.to(device, non_blocking=True))
            logits = model(x, m.to(device), t.to(device))
            loss = F.binary_cross_entropy_with_logits(logits.float(), y.to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        p = predict(model, dl_va, device)
        yv = labels.loc[va].values
        weak = [auc((yv[:, j] > 0.6).astype(int), p[:, j]) for j in range(12)]
        gmask = np.array([s in gold.index for s in va])
        g_auc = [auc(yv[gmask, j].astype(int), p[gmask, j]) for j in range(12)] if gmask.sum() else []
        pred_std = float(p.std(0).mean())
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "weak_macro": float(np.nanmean(weak)), "pred_std": pred_std,
               "gold_macro": float(np.nanmean(g_auc)) if g_auc else None, "seconds": time.time() - t0}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if pred_std < 1e-3:
            raise SystemExit(f"collapsed: prediction std {pred_std:.2e} after epoch {epoch}")
        if rec["weak_macro"] > best:
            best = rec["weak_macro"]
        torch.save(model.state_dict(), args.out / f"fold{args.fold}.pt")
        pd.DataFrame(p, columns=TARGETS, index=pd.Index(va, name="StudyInstanceUID")).to_csv(
            args.out / f"oof-fold{args.fold}.csv")
    (args.out / f"log-fold{args.fold}.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                                    "log": log}, indent=1))


if __name__ == "__main__":
    main()
