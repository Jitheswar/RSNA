"""Check that repo DICOM preprocessing reproduces cached training pixels."""
import io, json, sys, zipfile
from pathlib import Path
import numpy as np, pydicom
sys.path.insert(0, "src")
from rsna_knee.dicom import decode_dataset
from rsna_knee.domain import SeriesRecord
from rsna_knee.preprocessing import PreprocessConfig, prepare_series

cache = Path("artifacts/private/ticket35-expanded-image-preparation/series-cache/entries")
entries = sorted(p for p in cache.glob("*/*") if p.is_dir())[::4000][:5]
zf = zipfile.ZipFile("rsna-knee-abnormality-detection.zip")
names = zf.namelist()
cfg = PreprocessConfig(image_size=(224, 224))
for e in entries:
    meta = json.loads((e / "metadata.json").read_text())["prepared"]
    st, se = meta["study_instance_uid"], meta["series_instance_uid"]
    prefix = f"train_series/{st}/{se}/"
    files = [n for n in names if n.startswith(prefix)]
    slices = tuple(decode_dataset(pydicom.dcmread(io.BytesIO(zf.read(n))), filename=n) for n in files)
    prep = prepare_series(SeriesRecord(st, se, slices), cfg)
    ref = np.load(e / "images.npy")
    got = np.asarray(prep.images if hasattr(prep, "images") else prep.pixels, dtype=np.float32)
    print(se[-12:], ref.shape, got.shape, float(np.abs(ref.reshape(got.shape) - got).max()) if ref.size == got.size else "shape-mismatch")
