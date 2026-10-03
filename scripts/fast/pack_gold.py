"""Pack the 58 reference-labelled studies from raw DICOM with exact cache parity."""
import io, json, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd, pydicom
sys.path.insert(0, "src")
from rsna_knee.dicom import decode_dataset
from rsna_knee.domain import SeriesRecord
from rsna_knee.preprocessing import PreprocessConfig, prepare_series
sys.path.insert(0, "scripts/fast")
from pack_volumes import DEPTH, resample_depth

out = Path("artifacts/private/fast-v1/gold")
(out / "vol").mkdir(parents=True, exist_ok=True)
zf = zipfile.ZipFile("rsna-knee-abnormality-detection.zip")
train = pd.read_csv("artifacts/private/fast-v1/train.csv")
gold = set(train.dropna(subset=["ACL"]).StudyInstanceUID)
by_series = {}
for n in zf.namelist():
    parts = n.split("/")
    if len(parts) == 4 and parts[0] == "train_series" and parts[1] in gold and n.endswith(".dcm"):
        by_series.setdefault((parts[1], parts[2]), []).append(n)
cfg = PreprocessConfig(image_size=(224, 224))
rows, vols = [], []
for (st, se), files in sorted(by_series.items()):
    try:
        slices = tuple(decode_dataset(pydicom.dcmread(io.BytesIO(zf.read(f))), filename=f) for f in files)
        images = np.asarray(prepare_series(SeriesRecord(st, se, slices), cfg).images, np.float32)[:, 0]
    except Exception as exc:
        print("skip", se, type(exc).__name__, exc, flush=True)
        continue
    vol = np.clip(resample_depth(images, DEPTH) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    path = out / "vol" / f"{se}.npy"
    np.save(path, vol)
    rows.append((st, se, images.shape[0], str(path)))
    vols.append(vol)
pd.DataFrame(rows, columns=["study", "series", "n_slices", "path"]).to_csv(out / "index-gold.csv", index=False)
k = pd.DataFrame(rows, columns=["study", "series", "n_slices", "path"]).drop(columns="path")
k["shard"] = "gold"; k["row"] = range(len(k))
k.to_csv(out / "index.csv", index=False)
np.save(out / "vols-gold.npy", np.stack(vols))
print(f"packed {len(rows)} series from {len({r[0] for r in rows})} gold studies")
