# === v2 leg (our model): canonical laterality-normalised multi-series 2.5D models, rank-blended into the baseline ===
import os, sys, glob, json, subprocess, time, shutil
import numpy as np, pandas as pd
V2_WEIGHT = __WEIGHT__          # pre-fixed blend weight for our models (rank space)
V2_ONLY = __ONLY__              # True -> submit our models alone (no baseline)
_t0 = time.time()
_pkg = [p for pat in ('/kaggle/input/*/kaggle_infer.py', '/kaggle/input/*/*/kaggle_infer.py', '/kaggle/input/datasets/*/*/kaggle_infer.py', '/kaggle/input/datasets/*/*/*/kaggle_infer.py')
        for p in glob.glob(pat) if 'rsna-v2' in p]
_comp = [p for p in [os.environ.get('RSNA_COMP_ROOT', ''), '/kaggle/input/competitions/rsna-knee-abnormality-detection', '/kaggle/input/rsna-knee-abnormality-detection'] if os.path.exists(os.path.join(p, 'test.csv'))][0]
_base_path = '/kaggle/working/submission.csv'
TARGETS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
V2_TAG = '__TAG__'              # release tag of the model package this notebook was built for
if not _pkg:
    raise RuntimeError('v2 model package not mounted')
_tag_file = os.path.join(os.path.dirname(_pkg[0]), 'release_tag.txt')
_mounted = open(_tag_file).read().strip() if os.path.exists(_tag_file) else 'missing'
if _mounted != V2_TAG:
    raise RuntimeError(f'stale v2 model package mounted: tag {_mounted!r} != expected {V2_TAG!r}')
print('v2 package tag OK', V2_TAG, flush=True)
try:
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
except Exception:
    pass
try:
    pkg = os.path.dirname(_pkg[0])
    models = sorted(glob.glob(os.path.join(pkg, "*.pt")) + glob.glob(os.path.join(pkg, "*", "*.pt")))
    print('v2 package', pkg, len(models), 'models', flush=True)
    cmd = [sys.executable, os.path.join(pkg, 'kaggle_infer.py'), '--comp', _comp, '--models', *models,
           '--out', '/kaggle/working/v2_preds.csv', '--workers', '4', '--fp16', '--pure-fp16']
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    ours = pd.read_csv('/kaggle/working/v2_preds.csv', dtype={'StudyInstanceUID': str})
    test = pd.read_csv(os.path.join(_comp, 'test.csv'), dtype={'StudyInstanceUID': str})
    ours = test[['StudyInstanceUID']].merge(ours, on='StudyInstanceUID', how='left')
    if V2_ONLY or not os.path.exists(_base_path):
        out = ours.copy()
        out[TARGETS] = out[TARGETS].fillna(0.5)
    else:
        base = pd.read_csv(_base_path, dtype={'StudyInstanceUID': str})
        base = test[['StudyInstanceUID']].merge(base, on='StudyInstanceUID', how='left')
        out = base.copy()
        for t in TARGETS:
            b = base[t].rank(pct=True).values
            o = ours[t].rank(pct=True).values
            ok = ~np.isnan(o)
            v = b.copy()
            v[ok] = (1 - V2_WEIGHT) * b[ok] + V2_WEIGHT * o[ok]
            out[t] = v
    assert list(out.columns) == ['StudyInstanceUID'] + TARGETS and len(out) == len(test)
    assert np.isfinite(out[TARGETS].values).all()
    out.to_csv(_base_path, index=False)
    print(f'v2 leg done in {time.time()-_t0:.0f}s, weight {V2_WEIGHT}, only={V2_ONLY}', flush=True)
except Exception as exc:
    import traceback; traceback.print_exc()
    print('v2 leg FAILED; keeping baseline submission', exc, flush=True)
