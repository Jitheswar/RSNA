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
sim = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--sim=")), 0)
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
if sim:
    sim_src = f"""# REHEARSAL ONLY: simulate a hidden test of {sim} studies from train data (timing), never submitted
import os, time, pandas as pd
_SIM_T0 = time.time()
_c = [p for p in ['/kaggle/input/competitions/rsna-knee-abnormality-detection', '/kaggle/input/rsna-knee-abnormality-detection'] if os.path.exists(p + '/train.csv')][0]
_sim = '/tmp/simroot'; os.makedirs(_sim, exist_ok=True)
_tr = pd.read_csv(_c + '/train.csv', dtype=str); _ser = pd.read_csv(_c + '/train_series.csv', dtype=str)
_ids = _tr.StudyInstanceUID.sample({sim}, random_state=0)
pd.DataFrame({{'StudyInstanceUID': _ids}}).to_csv(_sim + '/test.csv', index=False)
_ser[_ser.StudyInstanceUID.isin(set(_ids))].to_csv(_sim + '/test_series.csv', index=False)
_ss = pd.read_csv(_c + '/sample_submission.csv', nrows=1)
_sub = pd.DataFrame({{'StudyInstanceUID': _ids}})
for _col in _ss.columns[1:]: _sub[_col] = 0.5
_sub.to_csv(_sim + '/sample_submission.csv', index=False)
for _n, _t in [('test_series', 'train_series'), ('train_series', 'train_series'), ('train.csv', 'train.csv'), ('train_series.csv', 'train_series.csv')]:
    if not os.path.exists(_sim + '/' + _n): os.symlink(_c + '/' + _t, _sim + '/' + _n)
os.environ['RSNA_COMP_ROOT'] = _sim
print('SIM root', _sim, len(_ids), 'studies', flush=True)
"""
    nb["cells"].insert(0, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                           "source": sim_src.splitlines(keepends=True)})
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                        "source": ["print(f'TOTAL REHEARSAL WALL {time.time()-_SIM_T0:.0f}s', flush=True)\n"]})
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
