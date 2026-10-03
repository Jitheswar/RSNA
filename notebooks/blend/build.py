"""Build the blended submission notebook: public 0.946 notebook + fast-lane leg.

usage: build.py <weight> <slug> [baseline-stem]
Source: yamadan96/rsna-knee-d4-public0946 (public, attributed); only one cell is appended.
"""
import json, sys
from pathlib import Path
here = Path(__file__).parent
weight, slug = sys.argv[1], sys.argv[2]
stem = sys.argv[3] if len(sys.argv) > 3 else "public0946"
nb = json.loads((here / f"{stem}.ipynb").read_text())
leg = (here / "leg.py").read_text().replace("__WEIGHT__", str(float(weight)))
nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                    "source": leg.splitlines(keepends=True)})
nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": [
    f"Fork of public baseline `{stem}` (unchanged, attributed) with one appended cell that rank-blends "
    f"our 2.5D model at weight {weight}; any failure keeps the public output.\n"]})
nb.setdefault("metadata", {})["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = here / slug
out.mkdir(exist_ok=True)
(out / "notebook.ipynb").write_text(json.dumps(nb, indent=1))
meta = json.loads((here / f"{stem}-metadata.json").read_text())
meta.update({"id": f"jitheswar/{slug}", "title": slug, "code_file": "notebook.ipynb", "is_private": True,
             "machine_shape": "NvidiaTeslaT4"})
meta.pop("id_no", None)
meta["dataset_sources"] = [d for d in meta["dataset_sources"] if d] + ["jitheswar/rsna-fast-model"]
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
print(out, len(nb["cells"]), "cells")
