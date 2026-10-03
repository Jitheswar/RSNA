"""Pack every training series from the raw competition zip at a chosen resolution.

Uses the repo decoder and ``prepare_series`` (the same code path as inference,
so parity holds at any size), resamples to a fixed depth and stores uint8
volumes ``<out>/vol/<series>.npy`` plus ``<out>/index-<shard>.csv``.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pydicom

# torch-free vendored copy of rsna_knee.{dicom,domain,geometry,preprocessing} (identical sources)
sys.path.insert(0, "artifacts/private/fast-v1/model-pkg")
from rsna_prep.dicom import decode_dataset  # noqa: E402
from rsna_prep.domain import SeriesRecord  # noqa: E402
from rsna_prep.preprocessing import PreprocessConfig, prepare_series  # noqa: E402

ZIP = "rsna-knee-abnormality-detection.zip"
_ZF = None


def resample_depth(images: np.ndarray, depth: int) -> np.ndarray:
    n = images.shape[0]
    pos = np.linspace(0, n - 1, depth)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (pos - lo)[:, None, None]
    return images[lo] * (1 - w) + images[hi] * w


def pack(job):
    global _ZF
    study, series, files, out, size, depth = job
    target = Path(out) / "vol" / f"{series}.npy"
    if _ZF is None:
        _ZF = zipfile.ZipFile(ZIP)
    try:
        if not target.exists():
            slices = []
            for f in files:
                try:
                    slices.append(decode_dataset(pydicom.dcmread(io.BytesIO(_ZF.read(f))), filename=f))
                except Exception:
                    continue
            images = np.asarray(prepare_series(SeriesRecord(study, series, tuple(slices)),
                                               PreprocessConfig(image_size=(size, size))).images, np.float32)[:, 0]
            vol = np.clip(resample_depth(images, depth) * 255.0 + 0.5, 0, 255).astype(np.uint8)
            tmp = target.with_suffix(".tmp.npy")
            np.save(tmp, vol)
            tmp.rename(target)
            n = images.shape[0]
        else:
            n = -1
        return f"{study},{series},{n},{target}"
    except Exception as exc:
        print(f"skip {series}: {type(exc).__name__}: {exc}", flush=True)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    (args.out / "vol").mkdir(parents=True, exist_ok=True)
    by_series: dict[tuple[str, str], list[str]] = {}
    for name in zipfile.ZipFile(ZIP).namelist():
        parts = name.split("/")
        if len(parts) == 4 and parts[0] == "train_series" and name.endswith(".dcm"):
            by_series.setdefault((parts[1], parts[2]), []).append(name)
    keys = sorted(by_series)[args.shard :: args.nshards]
    jobs = [(st, se, by_series[(st, se)], str(args.out), args.size, args.depth) for st, se in keys]
    with Pool(args.workers) as pool:
        rows = [r for r in pool.imap_unordered(pack, jobs, chunksize=4) if r]
    (args.out / f"index-{args.shard}.csv").write_text("study,series,n_slices,path\n" + "\n".join(rows) + "\n")
    print(f"shard {args.shard}: {len(rows)}/{len(jobs)} series", flush=True)


if __name__ == "__main__":
    main()
