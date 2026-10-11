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
from model import KneeNet, center_zoom, gpu_augment  # noqa: E402


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
def predict(model, loader, dev, crop=None, micro=0, zoom=1.0):
    model.eval()
    preds, idxs = [], []
    for b in loader:
        x = b["x"].to(dev, non_blocking=True)
        if zoom != 1.0:
            x = center_zoom(x, zoom, crop or x.shape[-1])
        elif crop and crop != x.shape[-1]:
            B, S, K, C, H, W = x.shape
            x = F.interpolate(x.view(-1, C, H, W).float(), size=(crop, crop), mode="bilinear",
                              align_corners=False).view(B, S, K, C, crop, crop)
        with torch.autocast("cuda", dtype=AMP_DT):
            logits, _ = model(x, b["types"].to(dev), b["pos"].to(dev), b["present"].to(dev), micro=micro,
                              smask=b["smask"].to(dev))
        preds.append(torch.sigmoid(logits.float()).cpu().numpy())
        idxs.append(b["idx"].numpy())
    return np.concatenate(preds), np.concatenate(idxs)


AMP_DT = None  # set in main (bf16 default; fp16 on T4)


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
    ap.add_argument("--layer-decay", type=float, default=1.0, help="<1: layer-wise lr decay for ViT encoders")
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
    ap.add_argument("--no-lat-norm", action="store_true", help="ablation: undo laterality normalisation of left knees")
    ap.add_argument("--zoom", type=float, default=1.0, help="central zoom factor (1.6 ~ 100 mm field)")
    ap.add_argument("--head", default="tf", choices=["tf", "mil", "pool"])
    ap.add_argument("--pos-weight", action="store_true", help="BCE pos_weight = clip((1-prev)/prev, 1, 10)")
    ap.add_argument("--aug", default="full", choices=["full", "light", "none"])
    ap.add_argument("--random-pos", action="store_true", help="bag sampling: random window centres in training")
    ap.add_argument("--mirror", type=float, default=0.0, help="training: prob. of mirroring a whole study (labels kept)")
    ap.add_argument("--init-ckpt", default="", help="initialise from a trained checkpoint (may contain {fold})")
    ap.add_argument("--aucm-w", type=float, default=0.0, help=">0: add AUC-margin loss (Yuan et al. ICCV 2021) on y>0.5")
    ap.add_argument("--aucm-margin", type=float, default=1.0)
    ap.add_argument("--ktr-sag", type=int, default=0, help="training per-plane window counts (default = eval counts)")
    ap.add_argument("--ktr-cor", type=int, default=0)
    ap.add_argument("--ktr-ax", type=int, default=0)
    ap.add_argument("--k-sag", type=int, default=0, help="per-plane slice counts (0 = use --k for all planes)")
    ap.add_argument("--k-cor", type=int, default=0)
    ap.add_argument("--k-ax", type=int, default=0)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16"], help="fp16 (+GradScaler) for GPUs without bf16 (T4)")
    ap.add_argument("--comp", default="", help="raw competition root: prepare studies from DICOMs on the fly (no --vol-dir)")
    args = ap.parse_args()
    global AMP_DT
    AMP_DT = torch.float16 if args.amp == "fp16" else torch.bfloat16

    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    if gpu_mem < 40:  # e.g. A30 24 GB: keep the effective batch, trade speed for memory
        if args.bs > 1:
            args.accum *= args.bs
            args.bs = 1
        if args.img > 256:  # 224-px backbones fit at bs=1 without checkpointing (ResNet50-D 224: 3.9 GB with it)
            args.grad_ckpt = True
            args.micro = min(args.micro or 16, 16)
        print(f"small GPU ({gpu_mem:.0f} GB): bs=1 accum={args.accum} grad_ckpt={args.grad_ckpt} micro={args.micro}", flush=True)
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

    k_plane = None
    if args.k_sag or args.k_cor or args.k_ax:
        k_plane = {"Sagittal": args.k_sag or args.k, "Coronal": args.k_cor or args.k, "Axial": args.k_ax or args.k}
    k_plane_tr = k_plane
    if args.ktr_sag or args.ktr_cor or args.ktr_ax:
        k_plane_tr = {"Sagittal": args.ktr_sag or args.k, "Coronal": args.ktr_cor or args.k, "Axial": args.ktr_ax or args.k}
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
        unnorm = {}
        if args.no_lat_norm:
            sides = meta.groupby("StudyInstanceUID").side.first()
            flipped = set(pd.read_csv("artifacts/private/v2/lat_flip_v1.csv").StudyInstanceUID)
            # effective normalisation flip = (side == L) XOR (checker re-flip)
            unnorm = {s: ((v == "L") != (s in flipped)) for s, v in sides.items()}
        prep_fn = None
        if args.comp:
            from rawprep import RawPrep
            prep_fn = RawPrep(args.comp)
        ds_tr = StudyDataset([studies[i] for i in tr_idx], meta, args.vol_dir, Yt[tr_idx], n_series=args.n_series,
                             k=args.k, train=True, seed=args.seed + fold, aux=AUX[tr_idx] if AUX is not None else None,
                             unnorm_lat=unnorm, k_plane=k_plane_tr, random_pos=args.random_pos, mirror_p=args.mirror, prep_fn=prep_fn)
        ds_va = StudyDataset([studies[i] for i in va_idx], meta, args.vol_dir, Yt[va_idx], n_series=args.n_series,
                             k=args.k, train=False, unnorm_lat=unnorm, k_plane=k_plane, prep_fn=prep_fn)
        dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=args.workers, collate_fn=collate,
                           pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4,
                           worker_init_fn=worker_init)
        dl_va = DataLoader(ds_va, batch_size=2, shuffle=False, num_workers=args.workers, collate_fn=collate,
                           pin_memory=True, worker_init_fn=worker_init) if len(va_idx) else None
        w_gold = torch.tensor(np.where(is_gold[tr_idx], args.gold_weight, 1.0), dtype=torch.float32)

        model = KneeNet(args.backbone, d=args.d, n_layers=args.layers, drop_path=args.drop_path,
                        grad_ckpt=args.grad_ckpt, n_aux=12 if AUX is not None else 0, head=args.head,
                        img=args.img).to(dev)
        if args.init_ckpt:
            ck0 = torch.load(args.init_ckpt.replace("{fold}", str(fold)), map_location="cpu", weights_only=False)
            model.load_state_dict(ck0["state"], strict=True)
            print("initialised from", args.init_ckpt.replace("{fold}", str(fold)), flush=True)
        scaler = torch.cuda.amp.GradScaler(enabled=AMP_DT == torch.float16)
        aucm = None
        if args.aucm_w > 0:  # min over (model, a, b), max over alpha >= 0
            p_pos = torch.tensor(np.clip((Yt[tr_idx] > 0.5).mean(0), 0.02, 0.98), dtype=torch.float32, device=dev)
            a_ = torch.zeros(12, device=dev, requires_grad=True)
            b_ = torch.zeros(12, device=dev, requires_grad=True)
            al_ = torch.zeros(12, device=dev, requires_grad=True)
            opt_al = torch.optim.SGD([al_], lr=args.lr_head * 10, maximize=True)
            aucm = (p_pos, a_, b_, al_, opt_al)
        pw = None
        if args.pos_weight:
            prev = np.clip(Yt[tr_idx].mean(0), 0.03, 0.7)
            pw = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)
            print("pos_weight", np.round(pw.cpu().numpy(), 2).tolist(), flush=True)
        model.enc.to(memory_format=torch.channels_last)
        enc_params = list(model.enc.parameters())
        enc_ids = {id(p) for p in enc_params}
        head_params = [p for p in model.parameters() if id(p) not in enc_ids]
        if args.layer_decay < 1.0:  # ViT fine-tuning: per-block lr scale (timm grouping), head at lr_head
            from timm.optim._param_groups import param_groups_layer_decay
            groups = param_groups_layer_decay(model.enc, weight_decay=args.wd, layer_decay=args.layer_decay)
            for g in groups:
                g["lr"] = args.lr_enc * g.pop("lr_scale", 1.0)
            groups.append({"params": head_params + ([aucm[1], aucm[2]] if aucm else []), "lr": args.lr_head})
            opt = torch.optim.AdamW(groups, weight_decay=args.wd, betas=(0.9, 0.999))
        else:
            opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr_enc},
                                     {"params": head_params + ([aucm[1], aucm[2]] if aucm else []),
                                      "lr": args.lr_head}], weight_decay=args.wd, betas=(0.9, 0.999))
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
                if args.aug == "full":
                    x = gpu_augment(x, present, gen=gen, crop=args.img if args.img != x.shape[-1] else None, zoom=args.zoom)
                elif args.aug == "light":
                    x = gpu_augment(x, present, gen=gen, crop=args.img if args.img != x.shape[-1] else None, zoom=args.zoom,
                                    rot=4.0, scale=(0.95, 1.05), shift=0.03)
                else:
                    x = center_zoom(x, args.zoom, args.img)
                y = b["y"].to(dev)
                wg = w_gold[b["idx"]].to(dev)
                with torch.autocast("cuda", dtype=AMP_DT):
                    logits, aux_logits = model(x, b["types"].to(dev), b["pos"].to(dev), present, micro=args.micro,
                                               smask=b["smask"].to(dev))
                loss = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none", pos_weight=pw).mean(1)
                loss = (loss * wg).sum() / wg.sum()
                if aucm is not None:
                    p_pos, a_, b_, al_, _ = aucm
                    sc = torch.sigmoid(logits.float())
                    yb = (y > 0.5).float()
                    m_ = args.aucm_margin
                    la_ = ((1 - p_pos) * (sc - a_) ** 2 * yb + p_pos * (sc - b_) ** 2 * (1 - yb)
                           + 2 * al_ * (p_pos * (1 - p_pos) * m_ + p_pos * sc * (1 - yb) - (1 - p_pos) * sc * yb)
                           - p_pos * (1 - p_pos) * al_ ** 2)
                    loss = loss + args.aucm_w * la_.mean()
                if aux_logits is not None and "aux" in b:
                    a = b["aux"].to(dev)
                    m = (a >= 0).float()
                    la = F.binary_cross_entropy_with_logits(aux_logits.float(), a.clamp(min=0), reduction="none")
                    loss = loss + args.aux_w * (la * m).sum() / m.sum().clamp(min=1)
                scaler.scale(loss / args.accum).backward()
                run_loss_t = loss.detach() if nb == 0 else run_loss_t + loss.detach()
                nb += 1
                if (it + 1) % args.accum == 0:
                    set_lr(step)
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    if aucm is not None:
                        aucm[4].step()
                        aucm[4].zero_grad(set_to_none=True)
                        with torch.no_grad():
                            aucm[3].clamp_(min=0.0)
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
                p, idx = predict(model, dl_va, dev, crop=args.img if args.img != 384 else None, micro=args.micro,
                                 zoom=args.zoom)
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
