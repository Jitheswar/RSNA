# ===================== Fast-lane leg: our 2.5D model, rank-blended into the public output =====================
# Appended after the public notebook has written /kaggle/working/submission.csv.
# Any failure, schema problem or time-budget risk keeps the public submission unchanged.
import glob as _fl_glob
import os as _fl_os
import shutil as _fl_shutil
import subprocess as _fl_sp
import sys as _fl_sys
import time as _fl_time

import pandas as _fl_pd

_FL_WEIGHT = float(_fl_os.environ.get("FL_WEIGHT", "__WEIGHT__"))
_FL_TIME_LIMIT = 8.3 * 3600  # notebook wall budget (hidden limit is 9 h)
_FL_TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
               "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_FL_WORK = _fl_os.environ.get("FL_WORK", "/kaggle/working")
_FL_INPUT = _fl_os.environ.get("FL_INPUT", "/kaggle/input")
_FL_SUB = f"{_FL_WORK}/submission.csv"


def _fl_find(name, marker):
    for pat in [f"{_FL_INPUT}/{name}", f"{_FL_INPUT}/*/{name}", f"{_FL_INPUT}/*/*/{name}"]:
        for c in sorted(_fl_glob.glob(pat)):
            if _fl_os.path.exists(_fl_os.path.join(c, marker)):
                return c
    return None


def _fl_elapsed():
    up = float(open("/proc/uptime").read().split()[0])
    start = int(open("/proc/self/stat").read().split(")")[-1].split()[19]) / _fl_os.sysconf("SC_CLK_TCK")
    return up - start


def _fl_run():
    pub = _fl_pd.read_csv(_FL_SUB, dtype={"StudyInstanceUID": str})
    assert pub.columns.tolist() == ["StudyInstanceUID", *_FL_TARGETS], pub.columns.tolist()
    _fl_shutil.copy2(_FL_SUB, f"{_FL_WORK}/_public_submission.csv")
    pkg = _fl_find("rsna-fast-model", "infer.py")
    comp = _fl_find("rsna-knee-abnormality-detection", "test.csv")
    if pkg is None or comp is None:
        print("[fast leg] inputs missing -> public only", flush=True)
        return
    n_test = len(_fl_pd.read_csv(f"{comp}/test.csv"))
    # leave room: our leg is budgeted at ~4 s per study plus a fixed 10 min
    if _fl_elapsed() + 600 + 4 * n_test > _FL_TIME_LIMIT:
        print(f"[fast leg] time budget too tight (elapsed {_fl_elapsed():.0f}s, n={n_test}) -> public only", flush=True)
        return
    out = f"{_FL_WORK}/_fast_pred.csv"
    r = _fl_sp.run([_fl_sys.executable, f"{pkg}/infer.py", "--comp", comp, "--models", f"{pkg}/models",
                    "--vendor", pkg, "--out", out, "--workers", str(max(1, (_fl_os.cpu_count() or 4)))],
                   timeout=max(60, _FL_TIME_LIMIT - _fl_elapsed() - 300))
    if r.returncode != 0 or not _fl_os.path.exists(out):
        print(f"[fast leg] inference failed rc={r.returncode} -> public only", flush=True)
        return
    ours = _fl_pd.read_csv(out, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    pub_i = pub.set_index("StudyInstanceUID")
    assert set(ours.index) == set(pub_i.index) and ours[_FL_TARGETS].notna().all().all()
    ours = ours.loc[pub_i.index, _FL_TARGETS]
    blend = pub_i[_FL_TARGETS].copy()
    for c in _FL_TARGETS:
        rp = pub_i[c].rank(pct=True)
        ro = ours[c].rank(pct=True)
        blend[c] = ((1 - _FL_WEIGHT) * rp + _FL_WEIGHT * ro).clip(1e-4, 1 - 1e-4)
    blend.reset_index().to_csv(_FL_SUB, index=False)
    print(f"[fast leg] blended weight={_FL_WEIGHT} n={len(blend)}", flush=True)


_fl_t0 = _fl_time.time()
try:
    _fl_run()
except Exception as _fl_exc:  # keep the public result on any failure
    print(f"[fast leg] SOFT FAIL {type(_fl_exc).__name__}: {_fl_exc} -> public only", flush=True)
    if _fl_os.path.exists(f"{_FL_WORK}/_public_submission.csv"):
        _fl_shutil.copy2(f"{_FL_WORK}/_public_submission.csv", _FL_SUB)
print(f"[fast leg] done in {_fl_time.time() - _fl_t0:.0f}s", flush=True)
