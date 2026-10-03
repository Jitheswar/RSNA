"""Kaggle inference for the v2 knee models (offline).

Prepares every test series with prep.py (identical code path to training packing), then runs each model on
every study (deterministic slice sampling) across all visible GPUs and writes per-model probabilities.

python kaggle_infer.py --comp /kaggle/input/rsna-knee-abnormality-detection --models <dir>/*.pt --out preds.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import queue
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prep  # noqa: E402

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


_LAT = None


def _lat_model():
    """Anatomical laterality checker (resnet18, CPU) shipped as lat_clf.pt next to this file; None if absent."""
    global _LAT
    if _LAT is None:
        path = os.path.join(HERE, "lat_clf.bin")
        if not os.path.exists(path):
            _LAT = False
        else:
            import torch
            import torch.nn as nn
            import torchvision
            torch.set_num_threads(1)
            m = torchvision.models.resnet18(num_classes=1)
            m.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            m.load_state_dict(torch.load(path, map_location="cpu"))
            _LAT = m.eval()
    return _LAT or None


def check_flip(out, side_src):
    """True if the study should be re-flipped: checker p>0.5 for weak geometry, p>0.99 for tag/geometry."""
    import torch
    import torch.nn.functional as F
    model = _lat_model()
    cor = [(v.shape[0], v) for v, info in out.values() if info["plane"] == "Coronal"]
    if model is None or not cor:
        return False
    vol = max(cor, key=lambda t: t[0])[1]
    D = vol.shape[0]
    idx = np.linspace(0.45 * (D - 1), 0.85 * (D - 1), 3).round().astype(int)
    x = torch.from_numpy(np.ascontiguousarray(vol[idx])).float() / 255.0
    x = F.interpolate(x[:, None], size=(192, 192), mode="area")[:, 0][None]
    with torch.no_grad():
        p = torch.sigmoid(model(x)[0, 0]).item()
    return p > (0.5 if side_src in ("weak-geometry", "unresolved") else 0.99)


def prepare_study(args):
    """Read all DICOMs of one test study and return {series_uid: (uint8 vol, info)} plus timing."""
    import pydicom
    study_dir, study, series_rows = args
    t0 = time.time()
    out = {}
    try:
        series_ds = {}
        for se in series_rows:
            files = sorted(glob.glob(os.path.join(study_dir, se, "*.dcm")))
            dss = []
            for f in files:
                try:
                    dss.append(pydicom.dcmread(f))
                except Exception:
                    continue
            series_ds[se] = dss
        side, side_src = prep.study_laterality([d for dss in series_ds.values() for d in dss[:3]])
        for se, dss in series_ds.items():
            if not dss:
                continue
            try:
                vol, info = prep.prepare_series(dss, series_rows[se], side)
                info["side_src"] = side_src
                out[se] = (vol, info)
            except Exception as exc:
                print(f"prep fail {study}/{se}: {type(exc).__name__}: {exc}", flush=True)
        if out:
            try:
                if check_flip(out, side_src):
                    for se, (vol, info) in list(out.items()):
                        vol = vol[::-1] if info["plane"] == "Sagittal" else vol[:, :, ::-1]
                        info["lat_checker_flip"] = True
                        out[se] = (np.ascontiguousarray(vol), info)
                    print(f"laterality checker flipped {study}", flush=True)
            except Exception as exc:
                print(f"laterality check failed {study}: {exc}", flush=True)
    except Exception as exc:
        print(f"study fail {study}: {type(exc).__name__}: {exc}", flush=True)
    return study, out, time.time() - t0


def study_tensors(vols, series_meta, n_series, k, shift=0.0):
    """Build model inputs from prepared volumes using the training-time (deterministic) selection."""
    from data import choose_series, slice_indices, series_type, PRIORITY
    rank = {t: i for i, t in enumerate(PRIORITY)}
    rows = []
    for se, (vol, info) in vols.items():
        fs, fl = series_meta.get(se, (0, 0))
        plane = info["plane"]
        rows.append({"SeriesInstanceUID": se, "plane": plane, "depth": vol.shape[0], "slice_mm": info["slice_mm"],
                     "Fat_Suppression": fs, "Fluid_Sensitive": fl, "stype": series_type(plane, fs, fl),
                     "prio": rank.get((plane, fs, fl), 99)})
    S, K = n_series, k
    x = np.zeros((S, K, 3, prep.SIZE, prep.SIZE), np.uint8)
    types = np.zeros(S, np.int64)
    pos = np.zeros((S, K, 2), np.float32)
    present = np.zeros(S, np.float32)
    if rows:
        g = pd.DataFrame(rows)
        for j, row in enumerate(choose_series(g, S, None)):
            vol = vols[row.SeriesInstanceUID][0]
            D = vol.shape[0]
            idx = slice_indices(D, K, None)
            if shift:
                step = 0.92 * (D - 1) / max(K - 1, 1)
                idx = np.clip(np.rint(idx + shift * step), 0, D - 1).astype(int)
            trip = np.stack([np.clip(idx - 1, 0, D - 1), idx, np.clip(idx + 1, 0, D - 1)], 1)
            x[j] = vol[trip]
            types[j] = int(row.stype)
            present[j] = 1.0
            pos[j, :, 0] = idx / max(D - 1, 1)
            pos[j, :, 1] = (idx - (D - 1) / 2.0) * float(row.slice_mm or 3.0) / 50.0
    return x, types, pos, present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="v2_preds.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--tta", type=int, default=1, help="1 = none; 2 = add a half-step slice-offset pass")
    args = ap.parse_args()
    import torch
    from model import KneeNet

    comp = Path(args.comp)
    ids = pd.read_csv(comp / f"{args.split}.csv", dtype=str).StudyInstanceUID.tolist()
    if args.limit:
        ids = ids[: args.limit]
    ser = pd.read_csv(comp / f"{args.split}_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    plane_of = dict(zip(ser.SeriesInstanceUID, ser.Anatomical_Plane))
    smeta = {r.SeriesInstanceUID: (int(r.Fat_Suppression), int(r.Fluid_Sensitive)) for r in ser.itertuples()}
    by_study = {s: g.SeriesInstanceUID.tolist() for s, g in ser.groupby("StudyInstanceUID")}

    n_gpu = max(1, torch.cuda.device_count())
    devs = [torch.device(f"cuda:{i}") for i in range(n_gpu)] if torch.cuda.is_available() else [torch.device("cpu")]
    paths = []
    for m in args.models:
        paths += sorted(glob.glob(m))
    models = {d: [] for d in devs}
    cfgs = []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        a = ck["args"]
        cfgs.append(a)
        for d in devs:
            n_aux = int(ck["state"]["aux_head.weight"].shape[0]) if "aux_head.weight" in ck["state"] else 0
            net = KneeNet(a["backbone"], d=a["d"], n_layers=a["layers"], pretrained=False, drop_path=0.0, n_aux=n_aux)
            net.load_state_dict(ck["state"], strict=True)
            net.eval().to(d)
            if args.fp16 and d.type == "cuda":
                net.half()
            models[d].append(net)
    print(f"{len(paths)} models on {len(devs)} devices; {len(ids)} studies", flush=True)
    S = max(a["n_series"] for a in cfgs)
    Ks = sorted({a["k"] for a in cfgs})

    preds = {}
    q: queue.Queue = queue.Queue(maxsize=16)

    def consumer(dev):
        while True:
            item = q.get()
            if item is None:
                return
            study, vols = item
            per_model = []
            with torch.no_grad():
                inputs = {}
                shifts = [0.0, 0.5][: max(1, args.tta)]
                for kk in Ks:
                    for sh in shifts:
                        x, types, pos, present = study_tensors(vols, smeta, S, kk, shift=sh)
                        inputs[(kk, sh)] = (torch.from_numpy(x)[None].to(dev), torch.from_numpy(types)[None].to(dev),
                                            torch.from_numpy(pos)[None].to(dev), torch.from_numpy(present)[None].to(dev))
                for net, a in zip(models[dev], cfgs):
                    tta_probs = []
                    for sh in shifts:
                        x, types, pos, present = inputs[(a["k"], sh)]
                        ns = a["n_series"]
                        if ns < S:
                            x, types, pos, present = x[:, :ns], types[:, :ns], pos[:, :ns], present[:, :ns]
                        if present.sum() == 0:
                            break
                        img = a.get("img", 384)
                        xx = x
                        if img != x.shape[-1]:
                            B_, S_, K_, C_, H_, W_ = x.shape
                            xx = torch.nn.functional.interpolate(x.view(-1, C_, H_, W_).float(), size=(img, img),
                                                                 mode="bilinear", align_corners=False
                                                                 ).view(B_, S_, K_, C_, img, img)
                        if args.fp16 and dev.type == "cuda":
                            with torch.autocast("cuda", dtype=torch.float16):
                                logits, _ = net(xx, types, pos.half(), present, micro=48)
                        else:
                            logits, _ = net(xx, types, pos, present, micro=48)
                        tta_probs.append(torch.sigmoid(logits.float())[0].cpu().numpy())
                    per_model.append(np.mean(tta_probs, 0) if tta_probs else np.full(12, np.nan, np.float32))
            preds[study] = np.stack(per_model)

    threads = [threading.Thread(target=consumer, args=(d,), daemon=True) for d in devs]
    for t in threads:
        t.start()
    t0 = time.time()
    jobs = [(str(comp / f"{args.split}_series" / s), s, {se: plane_of.get(se) for se in by_study.get(s, [])}) for s in ids]
    with ProcessPoolExecutor(args.workers) as pool:
        for n, (study, vols, dt) in enumerate(pool.map(prepare_study, jobs, chunksize=1)):
            q.put((study, vols))
            if (n + 1) % 50 == 0:
                print(f"prepared {n + 1}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    for _ in threads:
        q.put(None)
    for t in threads:
        t.join()
    rows = []
    for s in ids:
        p = preds.get(s)
        rows.append(np.full((len(paths), 12), np.nan, np.float32) if p is None else p)
    arr = np.stack(rows)  # [N, M, 12]
    np.save(Path(args.out).with_suffix(".npy"), arr)
    mean = np.nanmean(arr, 1)
    mean = np.where(np.isnan(mean), 0.5, mean)
    df = pd.DataFrame(mean, columns=TARGETS)
    df.insert(0, "StudyInstanceUID", ids)
    df.to_csv(args.out, index=False)
    print(f"done {len(ids)} studies in {time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
