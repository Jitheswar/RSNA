"""Assemble the Kaggle dataset dir for v2 inference: code + fp16 checkpoints.

python scripts/v2/package_models.py --runs artifacts/private/v2/runs/<run>/fold*/best.pt ... --out artifacts/private/v2/kaggle/rsna-v2-models
"""
import argparse, glob, json, shutil
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for f in ["prep.py", "data.py", "model.py", "kaggle_infer.py"]:
    shutil.copy(Path("scripts/v2") / f, out / f)
if Path("artifacts/private/v2/lat_clf.pt").exists():
    shutil.copy("artifacts/private/v2/lat_clf.pt", out / "lat_clf.bin")
paths = []
for c in args.ckpts:
    paths += sorted(glob.glob(c))
manifest = []
for p in paths:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    state = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["state"].items()}
    run, fold = Path(p).parent.parent.name, Path(p).parent.name
    name = f"{run}__{fold}__{Path(p).stem}.pt"
    keep_args = {k: ck["args"][k] for k in ["backbone", "d", "layers", "n_series", "k", "img"]}
    a = ck["args"]
    if a.get("head", "tf") != "tf":
        keep_args["head"] = a["head"]
    if a.get("zoom", 1.0) != 1.0:
        keep_args["zoom"] = a["zoom"]
    if a.get("k_sag") or a.get("k_cor") or a.get("k_ax"):
        keep_args["k_plane"] = {"Sagittal": a.get("k_sag") or a["k"], "Coronal": a.get("k_cor") or a["k"],
                                "Axial": a.get("k_ax") or a["k"]}
    torch.save({"state": state, "args": keep_args, "epoch": ck.get("epoch")}, out / name)
    manifest.append({"file": name, **keep_args})
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
meta = {"title": "rsna-v2-models", "id": "jitheswar/rsna-v2-models", "licenses": [{"name": "apache-2.0"}]}
(out / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
print(len(manifest), "models ->", out)
