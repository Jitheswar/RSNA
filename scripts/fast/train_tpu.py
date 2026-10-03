"""TPU (PyTorch/XLA, 8 cores) trainer for the fast-lane StudyModel.

Same data, labels, folds, model weights layout and checkpoint format as
``train.py``; the forward pass uses fixed shapes (every series slot is encoded
and padded slots are masked afterwards) so XLA compiles one graph. Checkpoints
load unchanged into ``train.StudyModel`` for GPU/CPU inference.
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
import torch
import torch.nn.functional as F

import train
from train import StudyDataset, StudyModel, TARGETS, auc, load_tables


def fixed_forward(model: StudyModel, x: torch.Tensor, mask: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
    """Equivalent to StudyModel.forward for valid series, with static shapes."""
    b, n, s, c, h, w = x.shape
    f = model.encoder(x.reshape(b * n * s, c, h, w).div(255).sub(0.45).div(0.25))
    f = model.proj(f).reshape(b * n, s, -1) + model.pos[:, :s]
    f = model.slice_tf(f)
    f = model.slice_pool(f, torch.ones(b * n, s, dtype=torch.bool, device=f.device))
    series = f.reshape(b, n, -1) * mask.unsqueeze(-1) + model.type_emb(types)
    series = model.series_tf(series, src_key_padding_mask=~mask)
    return model.head(model.series_pool(series, mask))


def augment(x: torch.Tensor, gen: np.random.Generator) -> torch.Tensor:
    """Same light augmentation as train.augment, batched with static shapes."""
    b, n, s, c, h, w = x.shape
    scale = torch.tensor(gen.uniform(0.85, 1.15, b), dtype=torch.float32)
    rot = torch.tensor(gen.uniform(-10, 10, b) * math.pi / 180, dtype=torch.float32)
    shift = torch.tensor(gen.uniform(-0.08, 0.08, (b, 2)), dtype=torch.float32)
    theta = torch.stack([torch.stack([scale * torch.cos(rot), -scale * torch.sin(rot), shift[:, 0]], -1),
                         torch.stack([scale * torch.sin(rot), scale * torch.cos(rot), shift[:, 1]], -1)], 1)
    gamma = torch.tensor(gen.uniform(0.8, 1.25, b), dtype=torch.float32)
    gain = torch.tensor(gen.uniform(0.9, 1.1, b), dtype=torch.float32)
    theta = theta.to(x.device).repeat_interleave(n * s, 0)
    xi = x.reshape(b * n * s, c, h, w)
    grid = F.affine_grid(theta, list(xi.shape), align_corners=False)
    xi = F.grid_sample(xi, grid, align_corners=False).reshape(b, n * s * c * h * w)
    g = gamma.to(x.device)[:, None]
    k = gain.to(x.device)[:, None]
    xi = torch.pow((xi / 255).clamp(1e-6, 1), g) * 255 * k
    return xi.clamp(0, 255).reshape(b, n, s, c, h, w)


def worker(index: int, args) -> None:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.runtime as xr

    rank, world = xr.global_ordinal(), xr.world_size()
    device = xm.xla_device()
    torch.manual_seed(args.fold)
    gen = np.random.default_rng(1000 * args.fold + rank)
    studies, fold, groups, soft, gold = load_tables(args.labels, args.train_csv, args.series_csv,
                                                    args.nfolds, args.vol_root)
    labels = soft.copy()
    common = gold.index.intersection(labels.index)
    labels.loc[common] = gold.loc[common].values
    tr = [s for s in studies if fold[s] != args.fold]
    va = [s for s in studies if fold[s] == args.fold]
    if args.limit:
        tr, va = tr[: args.limit], va[: max(args.limit // 4, world)]
    ds_tr = StudyDataset(tr, groups, labels, True, args.n_slices, args.max_series)
    ds_va = StudyDataset(va, groups, labels, False, args.n_slices, args.max_series)
    sampler = torch.utils.data.distributed.DistributedSampler(ds_tr, world, rank, shuffle=True, drop_last=True)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.bs, sampler=sampler, num_workers=args.workers,
                                        drop_last=True, persistent_workers=True)
    va_idx = list(range(rank, len(va), world))
    dl_va = torch.utils.data.DataLoader(torch.utils.data.Subset(ds_va, va_idx), batch_size=args.bs,
                                        num_workers=args.workers)

    model = StudyModel(args.backbone, pretrained=(rank == 0) or args.pretrained_all)
    if rank == 0:
        state = model.state_dict()
    xm.rendezvous("init")
    # identical initial weights on every core: broadcast rank 0 pretrained state through the host
    if rank == 0:
        torch.save(state, "/tmp/init.pt")
    xm.rendezvous("init-saved")
    model.load_state_dict(torch.load("/tmp/init.pt", map_location="cpu"))
    model = model.to(device)

    enc = list(model.encoder.parameters())
    ids = {id(p) for p in enc}
    other = [p for p in model.parameters() if id(p) not in ids]
    lr_scale = math.sqrt(world)  # global batch is world * bs
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr * lr_scale},
                             {"params": other, "lr": args.lr * args.head_lr_mult * lr_scale}], weight_decay=0.05)
    steps = args.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr * lr_scale, args.lr * args.head_lr_mult * lr_scale],
        total_steps=steps, pct_start=0.1, anneal_strategy="cos")
    log = []
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        loss_sum = torch.zeros((), device=device)
        n = 0
        for x, m, t, y in pl.MpDeviceLoader(dl_tr, device):
            with torch.autocast("xla", dtype=torch.bfloat16):
                logits = fixed_forward(model, augment(x.float(), gen), m, t)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            xm.optimizer_step(opt)
            sched.step()
            loss_sum += loss.detach()
            n += 1
        train_loss = xm.mesh_reduce("loss", (loss_sum / max(n, 1)).item(), np.mean)

        model.eval()
        preds = []
        with torch.no_grad():
            for x, m, t, _ in pl.MpDeviceLoader(dl_va, device):
                with torch.autocast("xla", dtype=torch.bfloat16):
                    preds.append(torch.sigmoid(fixed_forward(model, x.float(), m, t).float()).cpu())
        p_local = torch.cat(preds).numpy() if preds else np.zeros((0, 12), np.float32)
        gathered = xm.mesh_reduce("preds", (va_idx, p_local), lambda parts: parts)
        if rank == 0:
            order, rows = [], []
            for idx, p in gathered:
                order += idx
                rows.append(p)
            p = np.zeros((len(va), 12), np.float32)
            p[order] = np.concatenate(rows)
            yv = labels.loc[va].values
            weak = [auc((yv[:, j] > 0.6).astype(int), p[:, j]) for j in range(12)]
            gmask = np.array([s in gold.index for s in va])
            g_auc = [auc(yv[gmask, j].astype(int), p[gmask, j]) for j in range(12)] if gmask.sum() else []
            rec = {"epoch": epoch, "loss": float(train_loss), "weak_macro": float(np.nanmean(weak)),
                   "gold_macro": float(np.nanmean(g_auc)) if g_auc else None, "gold_n": int(gmask.sum()),
                   "pred_std": float(p.std(0).mean()), "seconds": time.time() - t0}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            pd.DataFrame(p, columns=TARGETS, index=pd.Index(va, name="StudyInstanceUID")).to_csv(
                args.out / f"oof-fold{args.fold}.csv")
        xm.save(model.state_dict(), str(args.out / f"fold{args.fold}.pt"))
        collapsed = xm.mesh_reduce("collapse", float(log[-1]["pred_std"]) if rank == 0 else 1.0, min)
        if collapsed < 1e-3:
            if rank == 0:
                print(json.dumps({"collapsed_after_epoch": epoch}), flush=True)
            break
    if rank == 0:
        (args.out / f"log-fold{args.fold}.json").write_text(json.dumps(
            {"args": {k: str(v) for k, v in vars(args).items()}, "log": log}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--backbone", default="convnext_nano.in12k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=2, help="per-core batch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr-mult", type=float, default=3.0)
    ap.add_argument("--n-slices", type=int, default=12)
    ap.add_argument("--max-series", type=int, default=6)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--series-csv", type=Path, required=True)
    ap.add_argument("--vol-root", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pretrained-all", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import torch_xla
    torch_xla.launch(worker, args=(args,))


if __name__ == "__main__":
    main()
