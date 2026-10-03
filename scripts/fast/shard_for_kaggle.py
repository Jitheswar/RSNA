"""Consolidate packed per-series volumes into a few large uint8 arrays for upload.

Writes ``<out>/vols-<k>.npy`` of shape (N_k, 32, 224, 224) and ``<out>/index.csv``
with shard/row columns so a Kaggle notebook can memory-map them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("artifacts/private/fast-v1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "kaggle-upload")
    ap.add_argument("--nshards", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index = pd.concat([pd.read_csv(p) for p in sorted(ROOT.glob("index-*.csv"))]).sort_values(["study", "series"])
    index = index.reset_index(drop=True)
    index["shard"] = index.index % args.nshards
    index["row"] = index.groupby("shard").cumcount()
    for k, part in index.groupby("shard"):
        arr = np.lib.format.open_memmap(args.out / f"vols-{k}.npy", mode="w+", dtype=np.uint8,
                                        shape=(len(part), 32, 224, 224))
        for row, path in zip(part.row, part.path):
            arr[row] = np.load(path)
        arr.flush()
        print(f"shard {k}: {len(part)} series")
    index.drop(columns=["path"]).to_csv(args.out / "index.csv", index=False)


if __name__ == "__main__":
    main()
