"""MRI-only inference for fast-lane StudyModel folds.

Reads raw test DICOMs with the vendored repo decoder/preprocessing (exact parity
with the training cache), resamples each series to 32 slices, averages fold
probabilities and writes a 13-column CSV. Series that fail to decode are
skipped; a study with no usable series receives the training prevalence.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import train  # noqa: E402
from train import TARGETS, StudyModel, series_type  # noqa: E402

train.AMP_DTYPE = torch.float16

DEPTH = 32


def _prep(args):
    study, series, folder, vendor = args
    sys.path.insert(0, vendor)
    from rsna_prep.dicom import decode_dataset
    from rsna_prep.domain import SeriesRecord
    from rsna_prep.preprocessing import PreprocessConfig, prepare_series

    try:
        files = sorted(Path(folder).glob("*.dcm"))
        slices = []
        for f in files:
            try:
                slices.append(decode_dataset(pydicom.dcmread(str(f)), filename=f.name))
            except Exception:
                continue
        prep = prepare_series(SeriesRecord(study, series, tuple(slices)), PreprocessConfig(image_size=(224, 224)))
        images = np.asarray(prep.images, dtype=np.float32)[:, 0]
        n = images.shape[0]
        pos = np.linspace(0, n - 1, DEPTH)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, n - 1)
        w = (pos - lo)[:, None, None]
        vol = images[lo] * (1 - w) + images[hi] * w
        return study, series, np.clip(vol * 255.0 + 0.5, 0, 255).astype(np.uint8)
    except Exception as exc:  # unusable series: skip, keep the study row
        print(f"skip {series}: {type(exc).__name__}: {exc}", flush=True)
        return study, series, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True, help="dir with fold*.pt")
    ap.add_argument("--vendor", type=Path, required=True, help="dir containing rsna_prep package")
    ap.add_argument("--backbone", default="convnext_nano.in12k_ft_in1k")
    ap.add_argument("--n-slices", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    test = pd.read_csv(args.comp / "test.csv", dtype=str)
    meta = pd.read_csv(args.comp / "test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    jobs = [(r.StudyInstanceUID, r.SeriesInstanceUID,
             str(args.comp / "test_series" / r.StudyInstanceUID / r.SeriesInstanceUID), str(args.vendor))
            for r in meta.itertuples()]
    with ProcessPoolExecutor(args.workers) as pool:
        vols = {(s, se): v for s, se, v in pool.map(_prep, jobs, chunksize=1) if v is not None}
    print(f"prepared {len(vols)}/{len(jobs)} series in {time.time() - t0:.0f}s", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for ckpt in sorted(args.models.glob("fold*.pt")):
        m = StudyModel(args.backbone, pretrained=False)
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        models.append(m.to(device).eval())
    stype = {(r.StudyInstanceUID, r.SeriesInstanceUID): series_type(r.Anatomical_Plane, r.Fluid_Sensitive)
             for r in meta.itertuples()}
    centers = np.round(np.linspace(1, DEPTH - 2, args.n_slices)).astype(int)
    prior = np.full(12, 0.3, np.float32)
    rows = []
    for study in test.StudyInstanceUID:
        keys = [k for k in vols if k[0] == study][:8]
        if not keys:
            rows.append(prior)
            continue
        x = np.stack([np.stack([vols[k][centers - 1], vols[k][centers], vols[k][centers + 1]], 1) for k in keys])
        xt = torch.from_numpy(x)[None].to(device)
        mask = torch.ones(1, len(keys), dtype=torch.bool, device=device)
        types = torch.tensor([[stype[k] for k in keys]], device=device)
        with torch.no_grad():
            p = np.mean([torch.sigmoid(m(xt, mask, types).float()).cpu().numpy()[0] for m in models], 0)
        rows.append(p)
    out = pd.DataFrame(np.asarray(rows), columns=TARGETS)
    out.insert(0, "StudyInstanceUID", test.StudyInstanceUID.values)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} {out.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
