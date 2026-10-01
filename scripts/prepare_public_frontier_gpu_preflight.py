"""Freeze the attributed public recipe and strict gates for a visible GPU trial."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "artifacts/private/external-notebook-audit/frontier-2026-09-30/d4-public0946"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", type=int, default=1)
    args = parser.parse_args()
    if args.revision < 1:
        raise SystemExit("Request revision must be positive")
    DEST = ROOT / f"artifacts/private/public-frontier-adoption/gpu-inference-preflight-v{args.revision}"
    if DEST.exists():
        raise SystemExit("Immutable request directory already exists")
    source_path = SOURCE_ROOT / "rsna-knee-d4-public0946.ipynb"
    source = json.loads(source_path.read_text())
    validator_path = ROOT / "scripts/public_frontier_validation.py"
    validator = validator_path.read_text()
    cells = []
    for index, cell in enumerate(source["cells"]):
        if cell["cell_type"] == "code":
            text = "".join(cell["source"])
            compile(text, f"upstream-cell-{index}", "exec")
            cells.append((index, text))
    cpu_root = ROOT / "artifacts/private/public-frontier-adoption/cpu-asset-preflight-v1"
    cpu = json.loads((cpu_root / "validation.json").read_text())
    assert cpu["terminal_status"] == "COMPLETE" and cpu["gpu_count"] == 0
    assert cpu["verified_manifests"] == 4 and cpu["manifest_files"] == 107
    code = '''from pathlib import Path
import os, json, hashlib, threading, time, resource, subprocess
from datetime import datetime, timezone
import pandas as pd
import numpy as np
started=time.monotonic()
def expire():
    Path('/kaggle/working/submission.csv').unlink(missing_ok=True)
    print('Visible preflight exceeded 900 seconds', flush=True)
    os._exit(124)
watchdog=threading.Timer(900,expire)
watchdog.daemon=True
watchdog.start()
comp=Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
if not (comp/'test.csv').is_file():comp=Path('/kaggle/input/rsna-knee-abnormality-detection')
test=pd.read_csv(comp/'test.csv',dtype={'StudyInstanceUID':str})
sample=pd.read_csv(comp/'sample_submission.csv',dtype={'StudyInstanceUID':str})
if not 1 <= len(sample) <= 3 or len(test)!=len(sample):
    raise RuntimeError('This exact request permits visible studies only; hidden scoring is prohibited')
expected_ids=sample.StudyInstanceUID.tolist()
expected_columns=sample.columns.tolist()
import torch
if torch.cuda.device_count()!=2 or not all('T4' in torch.cuda.get_device_name(i) for i in range(2)):
    raise RuntimeError('The pinned recipe requires exactly two T4 GPUs')
for i in range(2):torch.cuda.reset_peak_memory_stats(i)
_frontier_stop_sampling=threading.Event()
_frontier_memory={'interval_seconds':2,'device_peak_used_bytes':{},'cgroup_peak_bytes':None,'samples':0,'errors':[]}
def _frontier_sample_memory():
    while not _frontier_stop_sampling.is_set():
        try:
            result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,check=True)
            for line in result.stdout.splitlines():
                index,used=line.split(',')
                key=str(int(index.strip()))
                value=int(used.strip())*1024*1024
                _frontier_memory['device_peak_used_bytes'][key]=max(value,_frontier_memory['device_peak_used_bytes'].get(key,0))
            _frontier_memory['samples']+=1
            path=Path('/sys/fs/cgroup/memory.current')
            if path.is_file():
                value=int(path.read_text().strip())
                _frontier_memory['cgroup_peak_bytes']=max(value,_frontier_memory['cgroup_peak_bytes'] or 0)
        except Exception as error:
            detail=type(error).__name__+': '+str(error)
            if detail not in _frontier_memory['errors']:_frontier_memory['errors'].append(detail)
        _frontier_stop_sampling.wait(2)
_frontier_sampling_thread=threading.Thread(target=_frontier_sample_memory,daemon=True)
_frontier_sampling_thread.start()
'''
    code += "exec(compile(" + repr(validator) + ", 'public_frontier_validation.py', 'exec'),globals())\n"
    code += "UPSTREAM_CELLS=" + repr(cells) + "\n"
    code += '''try:
    for index,source in UPSTREAM_CELLS:
        print('UPSTREAM_CELL_START',index,flush=True)
        # CoAt launchers export helpers with inspect.getsource. Preserve the
        # exact original cell text for those child-worker source exports.
        import linecache
        filename=f'<rsna-upstream-cell-{index}>'
        linecache.cache[filename]=(len(source),None,source.splitlines(keepends=True),filename)
        exec(compile(source,filename,'exec'),globals())
        print('UPSTREAM_CELL_COMPLETE',index,flush=True)
    work=Path('/kaggle/working')
    audit=json.loads((work/'btkd_v559_complete.json').read_text())
    blend=json.loads((work/'_coat_raptor_blend_receipt.json').read_text())
    paths={'resgated_top3':work/'_coat_arm_receipt.json','global96_top3':work/'_coat_g96_receipt.json','repairv1_top3':work/'_coat_repairv1_receipt.json','d4_swa3':work/'d4_input/d4.receipt.json'}
    receipts={name:json.loads(path.read_text()) for name,path in paths.items()}
    strict=validate_public_frontier(audit,blend,receipts,len(expected_ids))
    phases=audit['phases']
    required=REQUIRED_PHASES | {name for name in phases if name.startswith('dino_')}
    for name in required:
        proof=phases[name];path=Path(proof['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=proof['sha256']:
            raise RuntimeError('Prediction bytes changed: '+name)
        with np.load(path,allow_pickle=False) as z:
            if z['study_uids'].astype(str).tolist()!=expected_ids or z['labels'].astype(str).tolist()!=expected_columns[1:]:
                raise RuntimeError('Prediction identity/order changed: '+name)
            values=z['values']
            if values.shape!=(len(expected_ids),12) or not np.isfinite(values).all():
                raise RuntimeError('Prediction values invalid: '+name)
    for log in work.rglob('*.log'):
        if '[dense-fallback]' in log.read_text(errors='replace'):
            raise RuntimeError('Reader input preparation used a fallback: '+str(log))
    output=pd.read_csv(work/'submission.csv',dtype={'StudyInstanceUID':str})
    values=output.iloc[:,1:].to_numpy()
    if output.columns.tolist()!=expected_columns or output.StudyInstanceUID.tolist()!=expected_ids:
        raise RuntimeError('Submission schema/order mismatch')
    if not np.isfinite(values).all() or (values<0).any() or (values>1).any():
        raise RuntimeError('Submission scores invalid')
    _frontier_stop_sampling.set();_frontier_sampling_thread.join(timeout=4)
    record={'schema_version':'rsna-public-frontier-visible-gpu-validation-v1','status':'COMPLETE','observed_at_utc':datetime.now(timezone.utc).isoformat(),'visible_only':True,'elapsed_seconds':time.monotonic()-started,'strict_execution':strict,'parent_max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'assigned_gpu_count':2,'parent_process_gpus':[{'name':torch.cuda.get_device_name(i),'peak_allocated_bytes':torch.cuda.max_memory_allocated(i),'peak_reserved_bytes':torch.cuda.max_memory_reserved(i)} for i in range(2)],'whole_vm_memory_sampling':_frontier_memory,'memory_limitations':'Polling may miss peaks between samples; PyTorch metrics cover only the parent process. Cgroup coverage depends on the exposed container hierarchy.','submission_sha256':hashlib.sha256((work/'submission.csv').read_bytes()).hexdigest(),'model_fitting':False,'competition_submission':False}
    (work/'strict_visible_validation.json').write_text(json.dumps(record,indent=2)+'\\n')
    print(json.dumps(record,indent=2),flush=True)
except BaseException:
    Path('/kaggle/working/submission.csv').unlink(missing_ok=True)
    raise
finally:
    _frontier_stop_sampling.set();_frontier_sampling_thread.join(timeout=4)
    Path('/kaggle/working/whole_vm_memory_samples.json').write_text(json.dumps(_frontier_memory,indent=2))
    watchdog.cancel()
'''
    compile(code, "gpu-preflight", "exec")
    metadata = json.loads((SOURCE_ROOT / "kernel-metadata.json").read_text())
    for field in ("id_no", "machine_shape", "keywords"):
        metadata.pop(field, None)
    handle = "jitheswar/rsna-public-frontier-gpu-preflight-attributed"
    metadata.update(id=handle,title=handle.split("/")[1],code_file="gpu-preflight.ipynb",
                    is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False)
    DEST.mkdir(parents=True)
    def write(name, obj):
        (DEST / name).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    credit = ("Private visible inference preflight of yamadan96/rsna-knee-d4-public0946 v2, "
              "scriptVersionId354057270. Retains the source recipe and authors' attribution; "
              "adds execution limits and strict rejection of partial/fallback output. No training, "
              "hidden scoring or final selection. Full upstream markdown credits follow.\n")
    markdown = "\n\n".join("".join(c["source"]) for c in source["cells"] if c["cell_type"]=="markdown")
    write("gpu-preflight.ipynb",{"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"cells":[{"cell_type":"markdown","metadata":{},"source":[credit+markdown]},{"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code.splitlines(keepends=True)}]})
    write("kernel-metadata.json",metadata)
    shutil.copyfile(ROOT / "artifacts/private/public-notebook-adoption/attributed-ensemble-draft/LICENSE-APACHE-2.0.txt",DEST / "LICENSE-APACHE-2.0.txt")
    (DEST / "NOTICE.md").write_text(credit+"\nSource: https://www.kaggle.com/code/yamadan96/rsna-knee-d4-public0946?scriptVersionId=354057270\n\n"+markdown)
    write("request.json",{"schema_version":"rsna-public-frontier-visible-gpu-request-v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"kernel_id":handle,"upstream_source_sha256":sha(source_path),"validator_sha256":sha(validator_path),"notebook_sha256":sha(DEST / "gpu-preflight.ipynb"),"metadata_sha256":sha(DEST / "kernel-metadata.json"),"cpu_validation_sha256":sha(cpu_root / "validation.json"),"source_code_cells":len(cells),"assigned_gpu_count":2,"walltime_reservation_seconds":1800,"reserved_gpu_seconds":3600,"execution_watchdog_seconds":900,"maximum_launch_attempts":3,"visible_study_limit":3,"external_spend_usd":0,"model_fitting":False,"competition_submission":False,"final_selection":False,"launch_command":["kaggle","kernels","push","-p",str(DEST),"--accelerator","NvidiaTeslaT4","--timeout","1800"],"execution_gate":"Do not launch while submission56705536 is active. Reconcile its terminal GPU accounting, verify available campaign capacity, free quota, paid scaling disabled, and exact Apache license evidence first.","authorization_basis":"Existing broad owner zero-spend campaign authorization; separate attributed public-baseline visible inference only, preserving Ticket39 frozen science and exact-job/final-selection boundaries."})
    (DEST / "request-checksum.txt").write_text(sha(DEST / "request.json")+"\n")
    print(DEST)


if __name__ == "__main__":
    main()
