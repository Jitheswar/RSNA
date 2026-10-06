"""Invert the laterality normalisation of selected studies in a packed volume directory (idempotent).

Studies whose DICOM side was a weak geometric guess and which the anatomical checker flags as flipped are
re-normalised: sagittal slice order reversed, coronal/axial flipped horizontally. A marker file in the volume
directory records every fixed series so a second run never flips back.

python scripts/v2/fix_laterality.py --vol-dir <dir> --flip-csv artifacts/private/v2/lat_flip_v1.csv
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

V2 = "artifacts/private/v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol-dir", required=True)
    ap.add_argument("--flip-csv", required=True)
    args = ap.parse_args()
    vol_dir = Path(args.vol_dir)
    marker = vol_dir / ".latfix_done.txt"
    done = set(marker.read_text().split()) if marker.exists() else set()
    flips = set(pd.read_csv(args.flip_csv).StudyInstanceUID)
    fr = []
    for m in sorted(glob.glob(f"{V2}/vol384/meta-train-*.csv")):
        try:
            fr.append(pd.read_csv(m))
        except Exception:
            pass
    meta = pd.concat(fr).drop_duplicates("SeriesInstanceUID", keep="last")
    plane = dict(zip(meta.SeriesInstanceUID, meta.plane))
    want = set(meta[meta.StudyInstanceUID.isin(flips)].SeriesInstanceUID)
    # flipping is an involution: series flipped under an older list but absent from this one are flipped back, so the
    # directory always matches --flip-csv exactly; the marker is updated after every file (crash-safe)
    todo = sorted(want - done) + sorted(done - want)
    n = n_back = 0
    for se in todo:
        p = vol_dir / f"{se}.npy"
        if not p.exists() or se not in plane:
            continue
        v = np.load(p)
        v = v[::-1] if plane[se] == "Sagittal" else v[:, :, ::-1]
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, np.ascontiguousarray(v))
        os.replace(tmp, p)
        if se in done:
            done.discard(se)
            n_back += 1
        else:
            done.add(se)
            n += 1
        tmpm = marker.with_suffix(".tmp")
        tmpm.write_text("".join(x + "\n" for x in sorted(done)))
        os.replace(tmpm, marker)
    print(f"{vol_dir}: flipped {n} series, restored {n_back}; {len(flips)} studies listed, {len(done)} series flipped now")


if __name__ == "__main__":
    main()
