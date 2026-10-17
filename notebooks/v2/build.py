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
base = next((a.split("=")[1] for a in sys.argv if a.startswith("--base=")), "public0946")  # public0946 | r7
sim = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--sim=")), 0)
import os
DS = os.environ.get("DS", "rsna-v2-models")  # separate model datasets per notebook: a hidden rerun mounts the latest version
_tagf = here.parent.parent / f"artifacts/private/v2/kaggle/{DS}/release_tag.txt"
tag = _tagf.read_text().strip() if _tagf.exists() else "untagged"
leg = (here / "leg.py").read_text().replace("__WEIGHT__", str(weight)).replace("__ONLY__", str(only)).replace("__TAG__", tag)
print("model package tag", tag)
base_dir = here.parent / "blend"
if only:
    nb = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        "# RSNA knee v2 (ours alone)\nCanonical laterality-normalised multi-series 2.5D models trained on gold-calibrated report labels.\n"]})
elif base in ("r7", "r6"):
    # OAI-free attributed baseline: heliosli's R7 ConvNeXt stage alone (Raptor/OAI stage removed), capped at 6.5 h so
    # our leg has time; R7 writes /kaggle/working/submission.csv, which the leg rank-blends into
    nb = json.loads((here / f"rsna-{base}-only" / "notebook.ipynb").read_text())
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            src = "".join(c["source"]).replace("DEADLINE_HOURS = 8.3", "DEADLINE_HOURS = 6.5" if base == "r7" else "DEADLINE_HOURS = 6.0")
            c["source"] = src.splitlines(keepends=True)
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": [
        f"Baseline = heliosli's public {base.upper()} ConvNeXt stage (attributed; Raptor/OAI stages removed). "
        f"Appended: our v2 models, rank-blended at a pre-fixed weight {weight}; any v2 failure keeps the R7 output.\n"]})
else:
    nb = json.loads((base_dir / "public0946.ipynb").read_text())
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": [
        f"Baseline = public notebook `yamadan96/rsna-knee-d4-public0946` (attributed, unchanged). "
        f"Appended: our v2 models, rank-blended at a pre-fixed weight {weight}; any v2 failure keeps the baseline output.\n"]})
nb["cells"].insert(0, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                       "source": ["import time as _rsna_time\n", "RSNA_T0 = _rsna_time.time()  # session start, for the v2 leg deadline\n"]})
nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                    "source": leg.splitlines(keepends=True)})
if "--s6" in sys.argv:
    # third OAI-free arm: nartaa's S6 all-data ConvNeXt (Heliosli S6 architecture; Raptor/OAI component removed), run LAST on
    # whatever time is left; final = (1 - W_S6) * rank(R7+ours blend) + W_S6 * rank(S6) where S6 scored, else unchanged
    import base64
    b64 = base64.b64encode((here / "s6_stage.py").read_bytes()).decode()
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
        "## Third arm: S6 all-data ConvNeXt (nartaa; Heliosli's S6 architecture), OAI component removed\n",
        "Weights `nartaa/rsna-knee-s6fall-weights` (CC BY-NC 4.0; README: no OAI data; ImageNet-22k init) + Heliosli's public "
        "anatomy/code dataset. nartaa's OAI-trained CoAtNet fallback is removed and not attached. Credit: nartaa, Heliosli.\n"]})
    s6_cell = f"""import base64, os, subprocess, sys, time
import numpy as np, pandas as pd
open('/tmp/s6_stage.py', 'wb').write(base64.b64decode('{b64}'))
W_S6 = {float(os.environ.get("W_S6", "0.3333"))}
_left = _session_start() + 9 * 3600 - time.time() - 35 * 60
print(f'S6 stage: {{_left / 60:.0f}} min left before the margin', flush=True)
_ok = False
if _left > 75 * 60:
    try:
        _env = dict(os.environ, S6_GUARD_S=str(int(_left)))
        _rc = subprocess.run([sys.executable, '-u', '/tmp/s6_stage.py'], env=_env, timeout=_left + 300).returncode
        _ok = _rc == 0 and os.path.isfile('/kaggle/working/s6_sub.csv')
        print('S6 rc', _rc, 'ok', _ok, flush=True)
    except Exception as e:
        print('S6 FAILED', repr(e)[:500], flush=True)
try:
    if _ok:
        cur = pd.read_csv('/kaggle/working/submission.csv', dtype={{'StudyInstanceUID': str}})
        s6 = pd.read_csv('/kaggle/working/s6_sub.csv', dtype={{'StudyInstanceUID': str}})
        s6 = cur[['StudyInstanceUID']].merge(s6, on='StudyInstanceUID', how='left')
        out = cur.copy(); n_ok = 0
        for t in TARGETS:
            o = s6[t].values.astype(float); ok = ~np.isnan(o); n_ok = int(ok.sum())
            if ok.sum() < 2: continue
            c = cur[t].rank(pct=True).values; r = pd.Series(o).rank(pct=True).values
            v = c.copy(); v[ok] = (1 - W_S6) * c[ok] + W_S6 * r[ok]
            out[t] = v
        out.to_csv('/kaggle/working/submission.csv', index=False)
        print(f'S6 blended at W_S6={{W_S6}} on {{n_ok}} / {{len(cur)}} studies', flush=True)
    else:
        print('S6 not used: submission = R7 + v2 blend', flush=True)
except Exception as e:
    print('S6 blend FAILED, keeping R7 + v2', repr(e)[:500], flush=True)
"""
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                        "source": s6_cell.splitlines(keepends=True)})
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
os.makedirs(_sim + '/test_series', exist_ok=True)
for _st in _ids:  # only the sampled studies, so directory walks match a real 1300-study hidden test
    if not os.path.exists(_sim + '/test_series/' + _st): os.symlink(_c + '/train_series/' + _st, _sim + '/test_series/' + _st)
for _n, _t in [('train_series', 'train_series'), ('train.csv', 'train.csv'), ('train_series.csv', 'train_series.csv')]:
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
meta = json.loads(((here / f"rsna-{base}-only" / "kernel-metadata.json") if base in ("r7", "r6") else (base_dir / "public0946-metadata.json")).read_text())
meta.update({"id": f"jitheswar/{slug}", "title": slug, "code_file": "notebook.ipynb", "is_private": True,
             "machine_shape": "NvidiaTeslaT4", "enable_gpu": True, "enable_internet": False})
meta.pop("id_no", None)
if only:
    meta["dataset_sources"], meta["kernel_sources"], meta["model_sources"] = [], [], []
meta["dataset_sources"] = meta["dataset_sources"] + [f"jitheswar/{DS}"] + (["nartaa/rsna-knee-s6fall-weights"] if "--s6" in sys.argv else []) \
    + (["heliosli/rsna-orthofoundation-fold0-epoch7-20261004"] if "--s6" in sys.argv and only else [])  # S6 needs its anatomy V-Net + code
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
print(out, len(nb["cells"]), "cells")
