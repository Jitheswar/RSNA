"""Prepare exact scored-baseline Gold58 inference; never launch or reserve it."""

from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from rsna_knee.scored_anchor_recipe import (
    extract_scored_recipe, notebook_code, read_notebook_code, relocate_scored_recipe,
)
from rsna_knee.targets import TARGET_COLUMNS

ANCHOR = PROJECT / "artifacts/private/public-notebook-adoption/scored-baseline-v1"
DEST = PROJECT / "artifacts/private/ticket70-scored-anchor-gold58-v1"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise SystemExit("Immutable exact-anchor preparation already exists")
    submission_request = json.loads((ANCHOR / "competition-submission-request.json").read_text())
    terminal = json.loads((ANCHOR / "terminal-scoring-evidence.json").read_text())
    capture = json.loads((ANCHOR / "remote-version-2.json").read_text())
    kernel = "jitheswar/rsna-public-ensemble-gpu-preflight-attributed"
    if not (terminal["request_checksum"] == digest(ANCHOR / "competition-submission-request.json")
            and terminal["job_id"] == "56705536" and terminal["script_version_id"] == 354125287
            and terminal["kernel_id"] == submission_request["kernel_id"] == kernel
            and terminal["kernel_version"] == submission_request["kernel_version"] == 2
            and terminal["submission"]["ref"] == 56705536
            and terminal["submission"]["status"] == "COMPLETE"
            and terminal["submission"]["publicScore"] == "0.936"
            and terminal["submission"]["url"] == f"/code/{kernel}?scriptVersionId=354125287"
            and capture["metadata"]["ref"] == kernel
            and capture["metadata"]["currentVersionNumber"] == 2):
        raise ValueError("Exact scored submission/source bridge differs")
    local_code = read_notebook_code(ANCHOR / "baseline.ipynb")
    remote_code = read_notebook_code(ANCHOR / "remote-source/rsna-public-ensemble-gpu-preflight-attributed.ipynb")
    captured_code = notebook_code(json.loads(capture["blob"]["source"]))
    if local_code != remote_code or remote_code != captured_code:
        raise ValueError("Local recipe differs from saved remote version2")
    manifest = json.loads((ANCHOR / "artifact-manifest.json").read_text())
    if digest(ANCHOR / "baseline.ipynb") != manifest["baseline.ipynb"]:
        raise ValueError("Original notebook bytes differ from retained manifest")
    original = extract_scored_recipe(local_code)
    relocated = relocate_scored_recipe(original)
    with zipfile.ZipFile(PROJECT / "rsna-knee-abnormality-detection.zip") as archive:
        train_bytes = archive.read("train.csv")
        series_bytes = archive.read("train_series.csv")
    rows = list(csv.DictReader(io.StringIO(train_bytes.decode())))
    selected = [row for row in rows if all(row[target] in ("0", "1") for target in TARGET_COLUMNS)]
    ids = [row["StudyInstanceUID"] for row in selected]
    if len(ids) != 58 or len(set(ids)) != 58:
        raise ValueError("Exactly58 complete binary reference rows required")
    uid_sha = hashlib.sha256("".join(uid + "\n" for uid in ids).encode()).hexdigest()
    allocation = json.loads((ANCHOR / "output/gpu_allocation_evidence.json").read_text())
    if allocation["exit_code"] != 0 or allocation["scope"]["test_study_count"] != 3:
        raise ValueError("Exact three-study visible execution evidence differs")
    # Conservative whole-visible scaling includes its startup cost twice.
    forecast = 2 * allocation["elapsed_wall_seconds"] * 58 / 3 + 600
    if forecast >= 7140:
        raise ValueError("Exact-anchor whole-cohort forecast exceeds proposed watchdog")
    constants = {
        "EXPECTED_TRAIN_SHA": hashlib.sha256(train_bytes).hexdigest(),
        "EXPECTED_SERIES_SHA": hashlib.sha256(series_bytes).hexdigest(),
        "EXPECTED_UID_SHA": uid_sha, "TARGETS": list(TARGET_COLUMNS),
        "ORIGINAL_CELLS": list(original), "SOURCE_CELLS": list(relocated),
    }
    code = "\n".join(f"{key} = {value!r}" for key, value in constants.items()) + "\n"
    code += RUNTIME_CODE
    compile(code, "ticket70-gold58", "exec")
    metadata = json.loads((ANCHOR / "kernel-metadata.json").read_text())
    metadata.update(id="jitheswar/rsna-exact-scored-anchor-gold58-v1",
                    title="RSNA exact scored anchor Gold58 development",
                    code_file="development.ipynb", is_private=True,
                    enable_gpu=True, enable_tpu=False, enable_internet=False)
    DEST.mkdir()
    def write(name, value):
        (DEST / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    write("development.ipynb", {"nbformat": 4, "nbformat_minor": 5, "metadata": {},
        "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": code.splitlines(keepends=True)}]})
    write("kernel-metadata.json", metadata)
    for name in ("NOTICE.md", "LICENSE-APACHE-2.0.txt"):
        shutil.copyfile(ANCHOR / name, DEST / name)
    sources = DEST / "source-snapshot"
    sources.mkdir()
    for index, source in enumerate(original):
        (sources / f"original-cell-{index}.py").write_text(source)
        (sources / f"relocated-cell-{index}.py").write_text(relocated[index])
    shutil.copyfile(Path(__file__), sources / Path(__file__).name)
    shutil.copyfile(PROJECT / "src/rsna_knee/scored_anchor_recipe.py", sources / "scored_anchor_recipe.py")
    shutil.copyfile(PROJECT / "scripts/validate_ticket70_scored_anchor_development.py",
                    sources / "validate_ticket70_scored_anchor_development.py")
    write("request.json", {
        "schema_version": "rsna-ticket70-exact-scored-anchor-development-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "kernel_id": metadata["id"],
        "scored_submission_ref": 56705536, "scored_script_version_id": 354125287,
        "scored_kernel_version": 2, "scored_public_score": "0.936",
        "original_notebook_sha256": digest(ANCHOR / "baseline.ipynb"),
        "remote_notebook_sha256": digest(ANCHOR / "remote-source/rsna-public-ensemble-gpu-preflight-attributed.ipynb"),
        "remote_capture_sha256": digest(ANCHOR / "remote-version-2.json"),
        "scoring_request_sha256": digest(ANCHOR / "competition-submission-request.json"),
        "scoring_terminal_sha256": digest(ANCHOR / "terminal-scoring-evidence.json"),
        "source_cells": [{"index": i, "original_sha256": hashlib.sha256(a.encode()).hexdigest(),
                          "relocated_sha256": hashlib.sha256(b.encode()).hexdigest()}
                         for i, (a, b) in enumerate(zip(original, relocated))],
        "only_recipe_changes": "Four competition-root string literals; all other source bytes unchanged",
        "notebook_sha256": digest(DEST / "development.ipynb"),
        "metadata_sha256": digest(DEST / "kernel-metadata.json"),
        "train_csv_sha256": constants["EXPECTED_TRAIN_SHA"],
        "train_series_csv_sha256": constants["EXPECTED_SERIES_SHA"],
        "development_uid_sha256": uid_sha, "study_count": 58,
        "visible_execution_evidence_sha256": digest(ANCHOR / "output/gpu_allocation_evidence.json"),
        "forecast_wall_seconds": forecast, "forecast_rule": "2*exact-visible-wall*58/3+600; not measured Gold58 cost",
        "assigned_gpu_count": 2, "required_accelerator": "NvidiaTeslaT4",
        "session_timeout_seconds": 7200, "execution_watchdog_seconds": 7140,
        "proposed_reserved_gpu_seconds": 14400, "maximum_launch_attempts": 1,
        "proposed_campaign_cap_seconds": 151200, "authorization_sha256": digest(AUTH),
        "resource_decision_status": "Prepared only; cap amendment, reservation and launch not performed",
        "launch_gate": "Existing TPU request terminal; fresh global accelerator vacancy, free typed quota and paid scaling off; exact resource decision/capacity/calendar/storage and reservation first",
        "external_spend_usd": 0, "model_fitting": False,
        "competition_submission": False, "final_selection": False,
        "limitations": "Adaptively reused Gold58/public parent selection; not clean OOF. Original dynamic cache planning retained and recorded. Preparation does not establish MRI predictions or runtime.",
        "frozen_files": {str(p.relative_to(DEST)): digest(p) for p in sources.iterdir()},
    })
    (DEST / "request-checksum.txt").write_text(digest(DEST / "request.json") + "\n")
    print(json.dumps({"root": str(DEST), "request_sha256": digest(DEST / "request.json"),
                      "forecast_wall_seconds": forecast, "launched": False}))


RUNTIME_CODE = r'''
from pathlib import Path
import hashlib, json, linecache, os, resource, shutil, sys, threading, time
import numpy as np
import pandas as pd
import torch
started = time.monotonic()
work = Path('/kaggle/working')
def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()
def expire():
    (work/'submission.csv').unlink(missing_ok=True)
    print('Exact scored-anchor development watchdog expired', flush=True)
    os._exit(124)
watchdog = threading.Timer(7140, expire)
watchdog.daemon = True
watchdog.start()
try:
    if torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError('Exact dual-checkpoint scored recipe requires two T4 GPUs')
    roots = [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
             Path('/kaggle/input/rsna-knee-abnormality-detection')]
    comp = next(root for root in roots if (root/'train.csv').is_file())
    if sha(comp/'train.csv')!=EXPECTED_TRAIN_SHA or sha(comp/'train_series.csv')!=EXPECTED_SERIES_SHA:
        raise RuntimeError('Original metadata bytes changed')
    training = pd.read_csv(comp/'train.csv', dtype={'StudyInstanceUID':str})
    ids = training.loc[training[TARGETS].notna().all(axis=1), 'StudyInstanceUID'].tolist()
    if len(ids)!=58 or len(set(ids))!=58 or hashlib.sha256(''.join(uid+'\n' for uid in ids).encode()).hexdigest()!=EXPECTED_UID_SHA:
        raise RuntimeError('Frozen Gold58 membership/order differs')
    view = Path('/tmp/rsna_exact_scored_gold58')
    view.mkdir()
    (view/'test_series').mkdir()
    for uid in ids:
        source = comp/'train_series'/uid
        if not source.is_dir(): raise RuntimeError('Original reference MRI missing')
        (view/'test_series'/uid).symlink_to(source, target_is_directory=True)
    pd.DataFrame({'StudyInstanceUID':ids}).to_csv(view/'test.csv', index=False)
    series = pd.read_csv(comp/'train_series.csv', dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
    series = series.loc[series.StudyInstanceUID.isin(ids)]
    if set(series.StudyInstanceUID)!=set(ids): raise RuntimeError('Gold58 series coverage differs')
    series.to_csv(view/'test_series.csv', index=False)
    sample = pd.DataFrame(np.full((58,12),.5), columns=TARGETS)
    sample.insert(0,'StudyInstanceUID',ids)
    sample.to_csv(view/'sample_submission.csv',index=False)
    # Keep the original planner's full training count, with no target values.
    pd.DataFrame({'StudyInstanceUID':training.StudyInstanceUID}).to_csv(view/'train.csv',index=False)
    del training, series, sample
    loaded = {}
    original_load = torch.load
    def capture_load(path,*args,**kwargs):
        if isinstance(path,(str,os.PathLike)):
            p = Path(path)
            loaded[str(p)] = {'sha256':sha(p),'bytes':p.stat().st_size}
        return original_load(path,*args,**kwargs)
    torch.load = capture_load
    coat_receipt = {}
    def profile(frame,event,arg):
        if event=='return' and frame.f_code.co_name=='main' and frame.f_code.co_filename=='<ticket70-source-3>':
            local = frame.f_locals
            success = local.get('legacy_success')
            coat_receipt.update(completed=bool(local.get('legacy_enabled') and isinstance(success,np.ndarray)
                and success.shape==(58,) and success.all()),
                legacy_success_count=int(success.sum()) if isinstance(success,np.ndarray) else 0)
            if 'primary_predictions' in local and 'legacy_predictions' in local:
                np.savez_compressed(work/'development-raptor-raw.npz',
                    primary=local['primary_predictions'],legacy=local['legacy_predictions'],study_uids=np.array(ids))
    class Monitor:
        def __init__(self,stream): self.stream=stream;self.tail='';self.fallbacks=0
        def __getattr__(self,name): return getattr(self.stream,name)
        def write(self,text):
            self.tail += text
            while '\n' in self.tail:
                line,self.tail=self.tail.split('\n',1)
                if ('[DINOsaur V4.2]' in line and ('FALLBACK' in line or ' fallback:' in line)) or 'calibration skipped' in line:
                    self.fallbacks += 1
            return self.stream.write(text)
        def flush(self): return self.stream.flush()
    stdout = sys.stdout
    monitor = Monitor(stdout)
    sys.stdout = monitor
    sys.setprofile(profile)
    stages = []
    try:
        for index,source in enumerate(SOURCE_CELLS):
            filename=f'<ticket70-source-{index}>'
            linecache.cache[filename]=(len(source),None,source.splitlines(keepends=True),filename)
            exec(compile(source,filename,'exec'),globals())
            frame=pd.read_csv(work/'submission.csv',dtype={'StudyInstanceUID':str})
            values=frame[TARGETS].to_numpy(float)
            if list(frame.columns)!=['StudyInstanceUID']+TARGETS or frame.StudyInstanceUID.tolist()!=ids or not np.isfinite(values).all() or (values<0).any() or (values>1).any():
                raise RuntimeError('Scored recipe stage output is invalid')
            path=work/f'development-stage-{index}.csv'
            shutil.copyfile(work/'submission.csv',path)
            stages.append({'index':index,'sha256':sha(path),'recipe_sha256':hashlib.sha256(source.encode()).hexdigest(),
                'cache_config':{key:globals().get(key) for key in ('IMG','CACHE_IMG','CACHE_SLICES','GROUP','N_GROUP')}})
    finally:
        sys.setprofile(None)
        sys.stdout = stdout
        torch.load = original_load
    if not globals().get('_RSNA_FULL_ENSEMBLE_COMPLETE') or not globals().get('V18_CALIBRATOR_APPLIED') or not coat_receipt.get('completed') or monitor.fallbacks:
        raise RuntimeError('Exact scored recipe did not complete both Raptor arms/calibration without fallback')
    final=work/'development-exact-scored-anchor.csv'
    (work/'submission.csv').replace(final)
    receipt={'schema_version':'rsna-ticket70-exact-scored-anchor-output-v1','status':'COMPLETE',
        'study_count':58,'development_uid_sha256':EXPECTED_UID_SHA,'stages':stages,
        'final_sha256':sha(final),'original_cells_sha256':[hashlib.sha256(s.encode()).hexdigest() for s in ORIGINAL_CELLS],
        'loaded_weights':loaded,'coat_receipt':coat_receipt,'fallback_events':monitor.fallbacks,
        'elapsed_seconds':time.monotonic()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'assigned_gpu_count':2,'model_fitting':False,'competition_submission':False,
        'limitation':'Adaptively reused Gold58; not clean OOF or a leaderboard score.'}
    (work/'exact-scored-anchor-receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'COMPLETE','study_count':58,'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)
except BaseException:
    (work/'submission.csv').unlink(missing_ok=True)
    raise
finally:
    watchdog.cancel()
'''


if __name__ == "__main__":
    main()
