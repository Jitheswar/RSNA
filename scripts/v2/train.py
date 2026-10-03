"""Train the v2 multi-series knee model on one or more folds (A100, bf16).

python scripts/v2/train.py --vol-dir /tmp/rsna_v2/vol384 --labels artifacts/private/v2/labels/<x>.csv \
    --backbone convnext_small.fb_in22k_ft_in1k_384 --folds 0 --out artifacts/private/v2/runs/<name>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import TARGETS, StudyDataset, load_series_table  # noqa: E402
from model import KneeNet, gpu_augment  # noqa: E402


def auc_macro(y, p, mask=None):
    from sklearn.metrics import roc_auc_score
    out = []
    for j in range(y.shape[1]):
        yy, pp = y[:, j], p[:, j]
        if mask is not None:
            yy, pp = yy[mask[:, j]], pp[mask[:, j]]
        if len(np.unique(yy)) == 2:
            out.append(roc_auc_score(yy, pp))
        else:
            out.append(np.nan)
    return float(np.nanmean(out)), out


class EMA:
    def __init__(self, model, decay=0.998):
        self.decay = decay
        self.shadow = {k: v.detach().float().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.n = 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def swap(self, model):
        cur = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))
        self.shadow_backup = cur

    def restore(self, model):
        sd = model.state_dict()
        for k, v in self.shadow_backup.items():
            sd[k].copy_(v)


def worker_init(_):
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def collate(batch):
    out = {}
    for k in batch[0]:
        if k == "idx":
            out[k] = torch.tensor([b[k] for b in batch])
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


@torch.no_grad()
def predict(model, loader, dev, crop=None, micro=0):
    model.eval()
    preds, idxs = [], []
    for b in loader:
        x = b["x"].to(dev, non_blocking=True)
        if crop and crop != x.shape[-1]:
            B, S, K, C, H, W = x.shape
            x = F.interpolate(x.view(-1, C, H, W).float(), size=(crop, crop), mode="bilinear",
                              align_corners=False).view(B, S, K, C, crop, crop)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x, b["types"].to(dev), b["pos"].to(dev), b["present"].to(dev), micro=micro)
        preds.append(torch.sigmoid(logits.float()).cpu().numpy())
        idxs.append(b["idx"].numpy())
    return np.concatenate(preds), np.concatenate(idxs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol-dir", required=True)
    ap.add_argument("--meta", default="artifacts/private/v2/vol384/meta-train-*.csv")
    ap.add_argument("--series-csv", default="artifacts/private/v2/train_series.csv")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--folds-csv", default="artifacts/private/v2/folds.csv")
    ap.add_argument("--train-csv", default="artifacts/private/v2/train.csv")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--full", action="store_true", help="train on all studies (no validation)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="convnext_small.fb_in22k_ft_in1k_384")
    ap.add_argument("--img", type=int, default=384)
    ap.add_argument("--n-series", type=int, default=6)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr-enc", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gold-weight", type=float, default=1.0)
    ap.add_argument("--label-smooth", type=float, default=0.0)
    ap.add_argument("--micro", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--save-every-epoch", action="store_true")
    ap.add_argument("--aux-grades", default="", help="grades CSV (LLM ordinal grades) -> auxiliary severity head")
    ap.add_argument("--aux-w", type=float, default=0.3)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "args.json").write_text(json.dumps(vars(args), indent=1))

    meta = load_series_table(sorted(glob.glob(args.meta)), args.series_csv)
    tr = pd.read_csv(args.train_csv)
    folds = pd.read_csv(args.folds_csv).set_index("StudyInstanceUID")
    lab_tables = {}

    def labels_for(fold):
        path = args.labels.replace("{fold}", str(fold)) if "{fold}" in args.labels else args.labels
        if path not in lab_tables:
            lab_tables[path] = pd.read_csv(path).set_index("StudyInstanceUID")
        return lab_tables[path]
    lab = labels_for(0 if "{fold}" in args.labels else -1)
    studies = [s for s in tr.StudyInstanceUID if s in set(meta.StudyInstanceUID)]
    gold = tr.set_index("StudyInstanceUID").loc[studies, TARGETS]
    is_gold = gold.notna().all(1).values
    Y = lab.reindex(studies)[TARGETS].values.astype(np.float32)
    Y = np.where(np.isnan(Y), 0.1, Y)
    Yt = np.where(is_gold[:, None], gold.values.astype(np.float32), Y)  # gold overrides report labels
    if args.label_smooth:
        Yt = Yt * (1 - args.label_smooth) + 0.5 * args.label_smooth
    fold_of = folds.reindex(studies)["fold"].values
    print(f"{len(studies)} studies with volumes, {is_gold.sum()} gold", flush=True)
    AUX = None
    if args.aux_grades:
        from build_labels import ORDER
        G = pd.read_csv(args.aux_grades).set_index("StudyInstanceUID").reindex(studies)[TARGETS]
        AUX = np.full(G.shape, -1.0, np.float32)  # -1 = masked
        for j, t_ in enumerate(TARGETS):
            lv = np.array([ORDER.get(t_, {}).get(int(g), float(g)) if g == g and g >= 0 else np.nan for g in G[t_].values])
            mx = np.nanmax(lv) if np.isfinite(lv).any() else 1.0
            AUX[:, j] = np.where(np.isfinite(lv), lv / max(mx, 1e-6), -1.0)
        print("aux grades: coverage", float((AUX >= 0).mean()), flush=True)

    run_folds = [-1] if args.full else [int(f) for f in args.folds.split(",")]
    for fold in run_folds:
        lab = labels_for(fold if fold >= 0 else 0)
        Y = lab.reindex(studies)[TARGETS].values.astype(np.float32)
        Y = np.where(np.isnan(Y), 0.1, Y)
        Yt = np.where(is_gold[:, None], gold.values.astype(np.float32), Y)
        if args.label_smooth:
            Yt = Yt * (1 - args.label_smooth) + 0.5 * args.label_smooth
        torch.manual_seed(args.seed + fold)
        np.random.seed(args.seed + fold)
        out = out_root / ("full" if fold < 0 else f"fold{fold}")
        out.mkdir(exist_ok=True)
        tr_idx = np.arange(len(studies)) if fold < 0 else np.where(fold_of != fold)[0]
        va_idx = np.array([], int) if fold < 0 else np.where(fold_of == fold)[0]
        ds_tr = StudyDataset([studies[i] for i in tr_idx], meta, args.vol_dir, Yt[tr_idx], n_series=args.n_series,
                             k=args.k, train=True, seed=args.seed + fold, aux=AUX[tr_idx] if AUX is not None else None)
        ds_va = StudyDataset([studies[i] for i in va_idx], meta, args.vol_dir, Yt[va_idx], n_series=args.n_series,
                             k=args.k, train=False)
        dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=args.workers, collate_fn=collate,
                           pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4,
                           worker_init_fn=worker_init)
        dl_va = DataLoader(ds_va, batch_size=2, shuffle=False, num_workers=args.workers, collate_fn=collate,
                           pin_memory=True, worker_init_fn=worker_init) if len(va_idx) else None
        w_gold = torch.tensor(np.where(is_gold[tr_idx], args.gold_weight, 1.0), dtype=torch.float32)

        model = KneeNet(args.backbone, d=args.d, n_layers=args.layers, drop_path=args.drop_path,
                        grad_ckpt=args.grad_ckpt, n_aux=12 if AUX is not None else 0).to(dev)
        model.enc.to(memory_format=torch.channels_last)
        enc_params = list(model.enc.parameters())
        enc_ids = {id(p) for p in enc_params}
        head_params = [p for p in model.parameters() if id(p) not in enc_ids]
        opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr_enc},
                                 {"params": head_params, "lr": args.lr_head}], weight_decay=args.wd, betas=(0.9, 0.999))
        steps_per_epoch = len(dl_tr) // args.accum
        total = args.max_steps or steps_per_epoch * args.epochs
        warm = max(1, int(args.warmup * total))
        base = [g["lr"] for g in opt.param_groups]

        def set_lr(step):
            f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * max(f, 0.01 if step >= warm else f)

        ema = EMA(model, args.ema)
        gen = torch.Generator(device=dev)
        gen.manual_seed(args.seed + fold)
        step, best, log = 0, -1.0, []
        t_start = time.time()
        for epoch in range(args.epochs):
            ds_tr.epoch = epoch
            model.train()
            t0, run_loss, nb = time.time(), 0.0, 0
            opt.zero_grad(set_to_none=True)
            for it, b in enumerate(dl_tr):
                x = b["x"].to(dev, non_blocking=True)
                present = b["present"].to(dev)
                x = gpu_augment(x, present, gen=gen, crop=args.img if args.img != x.shape[-1] else None)
                y = b["y"].to(dev)
                wg = w_gold[b["idx"]].to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, aux_logits = model(x, b["types"].to(dev), b["pos"].to(dev), present, micro=args.micro)
                loss = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none").mean(1)
                loss = (loss * wg).sum() / wg.sum()
                if aux_logits is not None and "aux" in b:
                    a = b["aux"].to(dev)
                    m = (a >= 0).float()
                    la = F.binary_cross_entropy_with_logits(aux_logits.float(), a.clamp(min=0), reduction="none")
                    loss = loss + args.aux_w * (la * m).sum() / m.sum().clamp(min=1)
                (loss / args.accum).backward()
                run_loss_t = loss.detach() if nb == 0 else run_loss_t + loss.detach()
                nb += 1
                if (it + 1) % args.accum == 0:
                    set_lr(step)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    ema.update(model)
                    step += 1
                    if step % 50 == 0:
                        run_loss = float(run_loss_t)
                        print(f"fold {fold} ep {epoch} step {step}/{total} loss {run_loss / nb:.4f} "
                              f"{(time.time() - t0) / nb:.3f}s/it", flush=True)
                    if args.max_steps and step >= args.max_steps:
                        break
            run_loss = float(run_loss_t) if nb else 0.0
            rec = {"fold": fold, "epoch": epoch, "loss": run_loss / max(nb, 1), "train_s": time.time() - t0}
            if dl_va is not None:
                ema.swap(model)
                p, idx = predict(model, dl_va, dev, crop=args.img if args.img != 384 else None, micro=args.micro)
                ema.restore(model)
                yv = Yt[va_idx][idx]
                wk_y = (yv > 0.5).astype(int)
                wk, per = auc_macro(wk_y, p)
                gmask = is_gold[va_idx][idx]
                rec.update({"weak_auc": wk, "weak_per": [round(v, 4) for v in per], "pred_std": float(p.std(0).mean())})
                if gmask.sum() >= 5:
                    rec["gold_auc_fold"] = auc_macro(yv[gmask].astype(int), p[gmask])[0]
                if wk > best:
                    best = wk
                    ema.swap(model)
                    torch.save({"state": model.state_dict(), "args": vars(args), "epoch": epoch}, out / "best.pt")
                    ema.restore(model)
                    pd.DataFrame(p, columns=TARGETS).assign(StudyInstanceUID=[studies[va_idx[i]] for i in idx]) \
                        .to_csv(out / "oof.csv", index=False)
            rec["elapsed"] = time.time() - t_start
            print(json.dumps(rec), flush=True)
            log.append(rec)
            (out / "log.json").write_text(json.dumps(log, indent=1))
            if args.save_every_epoch or epoch == args.epochs - 1:
                ema.swap(model)
                torch.save({"state": model.state_dict(), "args": vars(args), "epoch": epoch}, out / "last.pt")
                ema.restore(model)
            if args.max_steps and step >= args.max_steps:
                break
        del model, opt, ema, dl_tr, dl_va
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
