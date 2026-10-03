"""Pack Ticket 35 series caches into fixed-depth uint8 volumes for GPU training.

Each series becomes ``vol/<series_uid>.npy`` with shape (DEPTH, 224, 224) uint8,
resampled along the geometry-ordered slice axis. An index CSV records study,
series, original slice count and path. Shard with --shard/--nshards.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

CACHE = Path("artifacts/private/ticket35-expanded-image-preparation/series-cache/entries")
DEPTH = 32


def resample_depth(images: np.ndarray, depth: int) -> np.ndarray:
    n = images.shape[0]
    pos = np.linspace(0, n - 1, depth)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (pos - lo)[:, None, None]
    return images[lo] * (1 - w) + images[hi] * w


def pack(entry: Path, out: Path) -> tuple[str, str, int, str] | None:
    meta_files = list(entry.glob("*.json"))
    if not meta_files:
        return None
    meta = json.loads(meta_files[0].read_text())
    if meta.get("state") != "complete":
        return None
    prepared = meta["prepared"]
    series = prepared["series_instance_uid"]
    target = out / f"{series}.npy"
    images = np.load(entry / meta["array"]["filename"], mmap_mode="r")[:, 0]
    if not target.exists():
        vol = resample_depth(np.asarray(images, dtype=np.float32), DEPTH)
        tmp = target.with_suffix(".tmp.npy")
        np.save(tmp, np.clip(vol * 255.0 + 0.5, 0, 255).astype(np.uint8))
        tmp.rename(target)
    return prepared["study_instance_uid"], series, int(images.shape[0]), str(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/private/fast-v1"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    vol = args.out / "vol"
    vol.mkdir(parents=True, exist_ok=True)
    entries = sorted(p for p in CACHE.glob("*/*") if p.is_dir())[args.shard :: args.nshards]
    with Pool(args.workers) as pool:
        rows = pool.starmap(pack, [(e, vol) for e in entries], chunksize=8)
    lines = ["study,series,n_slices,path"] + [",".join(map(str, r)) for r in rows if r]
    (args.out / f"index-{args.shard}.csv").write_text("\n".join(lines) + "\n")
    print(f"shard {args.shard}: {len(lines) - 1} of {len(entries)} entries packed")


if __name__ == "__main__":
    main()
