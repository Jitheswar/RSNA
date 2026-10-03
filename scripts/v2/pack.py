"""Pack every study of a split from the competition zip with scripts/v2/prep.py.

Writes <out>/vol/<series>.npy (uint8 [D,384,384]) and <out>/meta-<shard>.csv.
Run as a slurm array: --shard i --nshards n.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pydicom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep  # noqa: E402

ZIP = "rsna-knee-abnormality-detection.zip"
_ZF = None
FIELDS = ["StudyInstanceUID", "SeriesInstanceUID", "side", "side_src", "plane", "plane_hint", "n_src", "n_kept",
          "depth", "order", "transpose", "flip_rows", "flip_cols", "lat_flip", "crop_mm", "mm_px", "slice_mm",
          "Manufacturer", "Model", "Field", "TE", "TR", "TI", "Thick", "SeqName", "Desc", "error"]


def _tag(ds, name):
    v = getattr(ds, name, "")
    return str(v).replace(",", ";").replace("\n", " ")[:60]


def pack_study(job):
    global _ZF
    if _ZF is None:
        _ZF = zipfile.ZipFile(ZIP)
    study, series_files, plane_hints, out, size = job
    rows = []
    try:
        series_ds = {}
        for se, files in series_files.items():
            dss = []
            for f in files:
                try:
                    dss.append(pydicom.dcmread(io.BytesIO(_ZF.read(f))))
                except Exception:
                    continue
            series_ds[se] = dss
        side, side_src = prep.study_laterality([d for dss in series_ds.values() for d in dss[:3]])
        for se, dss in series_ds.items():
            target = Path(out) / "vol" / f"{se}.npy"
            base = {"StudyInstanceUID": study, "SeriesInstanceUID": se, "side": side, "side_src": side_src}
            try:
                if not dss:
                    raise ValueError("no readable files")
                vol, info = prep.prepare_series(dss, plane_hints.get(se), side, size=size)
                tmp = target.with_suffix(".tmp.npy")
                np.save(tmp, vol)
                os.replace(tmp, target)
                d0 = dss[0]
                base.update(info)
                base.update({"Manufacturer": _tag(d0, "Manufacturer"), "Model": _tag(d0, "ManufacturerModelName"),
                             "Field": _tag(d0, "MagneticFieldStrength"), "TE": _tag(d0, "EchoTime"),
                             "TR": _tag(d0, "RepetitionTime"), "TI": _tag(d0, "InversionTime"),
                             "Thick": _tag(d0, "SliceThickness"), "SeqName": _tag(d0, "SequenceName"),
                             "Desc": _tag(d0, "SeriesDescription"), "error": ""})
            except Exception as exc:  # keep going; record the failure
                base["error"] = f"{type(exc).__name__}: {exc}"[:200]
            rows.append(base)
    except Exception as exc:
        rows.append({"StudyInstanceUID": study, "error": f"study {type(exc).__name__}: {exc}"[:200]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--studies", default="")
    ap.add_argument("--skip-meta", nargs="*", default=[])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    (args.out / "vol").mkdir(parents=True, exist_ok=True)
    import pandas as pd
    ser = pd.read_csv(f"artifacts/private/v2/{args.split}_series.csv", dtype=str)
    hints = dict(zip(ser.SeriesInstanceUID, ser.Anatomical_Plane))
    by_study: dict[str, dict[str, list[str]]] = {}
    for name in zipfile.ZipFile(ZIP).namelist():
        if not (name.startswith(args.split + "_series/") and name.endswith(".dcm")):
            continue
        _, st, se, _f = name.split("/")
        by_study.setdefault(st, {}).setdefault(se, []).append(name)
    studies = sorted(by_study)
    if args.studies:
        keep = set(args.studies.split(","))
        studies = [s for s in studies if s in keep]
    if args.skip_meta:
        done = set()
        for m in args.skip_meta:
            try:
                dm = pd.read_csv(m)
            except Exception:
                continue
            ok = dm[dm["error"].isna()].groupby("StudyInstanceUID").SeriesInstanceUID.nunique()
            for st, n in ok.items():
                if st in by_study and n == len(by_study[st]):
                    done.add(st)
        print(f"skipping {len(done)} completed studies", flush=True)
        studies = [s for s in studies if s not in done]
    studies = studies[args.shard::args.nshards]
    if args.limit:
        studies = studies[: args.limit]
    jobs = [(st, by_study[st], {se: hints.get(se) for se in by_study[st]}, str(args.out), args.size) for st in studies]
    meta = args.out / f"meta-{args.split}-{args.tag or args.shard}.csv"
    t0 = time.time()
    n = 0
    with Pool(args.workers) as pool, open(meta, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for rows in pool.imap_unordered(pack_study, jobs, chunksize=1):
            for r in rows:
                w.writerow(r)
            n += 1
            if n % 50 == 0:
                fh.flush()
                print(f"{n}/{len(jobs)} studies {time.time() - t0:.0f}s", flush=True)
    print(f"done {n} studies in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
