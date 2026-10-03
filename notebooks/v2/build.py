"""Build a v2 submission notebook.

usage: build.py <slug> <weight> [--only]
  default: attributed public baseline (yamadan96/rsna-knee-d4-public0946, unchanged) + our v2 leg rank-blended at <weight>
  --only : our v2 models alone (no baseline cells)
"""
import json, sys
from pathlib import Path
here = Path(__file__).parent
slug, weight = sys.argv[1], float(sys.argv[2])
only = "--only" in sys.argv
leg = (here / "leg.py").read_text().replace("__WEIGHT__", str(weight)).replace("__ONLY__", str(only))
base_dir = here.parent / "blend"
if only:
    nb = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        "# RSNA knee v2 (ours alone)\nCanonical laterality-normalised multi-series 2.5D models trained on gold-calibrated report labels.\n"]})
else:
    nb = json.loads((base_dir / "public0946.ipynb").read_text())
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": [
        f"Baseline = public notebook `yamadan96/rsna-knee-d4-public0946` (attributed, unchanged). "
        f"Appended: our v2 models, rank-blended at a pre-fixed weight {weight}; any v2 failure keeps the baseline output.\n"]})
nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                    "source": leg.splitlines(keepends=True)})
nb.setdefault("metadata", {})["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = here / slug
out.mkdir(exist_ok=True)
(out / "notebook.ipynb").write_text(json.dumps(nb, indent=1))
meta = json.loads((base_dir / "public0946-metadata.json").read_text())
meta.update({"id": f"jitheswar/{slug}", "title": slug, "code_file": "notebook.ipynb", "is_private": True,
             "machine_shape": "NvidiaTeslaT4", "enable_gpu": True, "enable_internet": False})
meta.pop("id_no", None)
if only:
    meta["dataset_sources"], meta["kernel_sources"], meta["model_sources"] = [], [], []
meta["dataset_sources"] = meta["dataset_sources"] + ["jitheswar/rsna-v2-models"]
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
print(out, len(nb["cells"]), "cells")
