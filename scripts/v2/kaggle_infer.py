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
    except Exception as exc:
        print(f"study fail {study}: {type(exc).__name__}: {exc}", flush=True)
    return study, out, time.time() - t0


def study_tensors(vols, series_meta, n_series, k):
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
            net = KneeNet(a["backbone"], d=a["d"], n_layers=a["layers"], pretrained=False, drop_path=0.0)
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
                for kk in Ks:
                    x, types, pos, present = study_tensors(vols, smeta, S, kk)
                    inputs[kk] = (torch.from_numpy(x)[None].to(dev), torch.from_numpy(types)[None].to(dev),
                                  torch.from_numpy(pos)[None].to(dev), torch.from_numpy(present)[None].to(dev))
                for net, a in zip(models[dev], cfgs):
                    x, types, pos, present = inputs[a["k"]]
                    if a["n_series"] < S:
                        x, types, pos, present = x[:, :a["n_series"]], types[:, :a["n_series"]], pos[:, :a["n_series"]], present[:, :a["n_series"]]
                    if present.sum() == 0:
                        per_model.append(np.full(12, np.nan, np.float32))
                        continue
                    img = a.get("img", 384)
                    xx = x
                    if img != x.shape[-1]:
                        B_, S_, K_, C_, H_, W_ = x.shape
                        xx = torch.nn.functional.interpolate(x.view(-1, C_, H_, W_).float(), size=(img, img), mode="bilinear",
                                                             align_corners=False).view(B_, S_, K_, C_, img, img)
                    if args.fp16 and dev.type == "cuda":
                        with torch.autocast("cuda", dtype=torch.float16):
                            logits, _ = net(xx, types, pos.half(), present, micro=48)
                    else:
                        logits, _ = net(xx, types, pos, present, micro=48)
                    per_model.append(torch.sigmoid(logits.float())[0].cpu().numpy())
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
