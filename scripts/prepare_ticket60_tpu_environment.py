"""Freeze a bounded fail-on-CPU TPU environment admission without training."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "artifacts/private/ticket60-tpu-environment-v1"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket60-tpu-environment-20261002.json"
HANDLE = "jitheswar/rsna-tpu-environment-admission-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main():
    if ROOT.exists() or DECISION.exists():
        raise FileExistsError("Immutable TPU environment request already exists")
    quota = PROJECT / "artifacts/private/campaign-quota-observation-20261002/quota.json"
    snapshot = json.loads(quota.read_text())
    allowance = snapshot["quota"]["tpuQuota"]
    if allowance["timeUsed"] != "0s" or allowance["timeReserved"] != "0s" or allowance["totalTimeAllowed"] != "72000s":
        raise ValueError("Exact unused TPU quota observation required")
    ROOT.mkdir(mode=0o700)
    code = '''import os, sys, json, time, traceback, threading, platform, importlib.metadata
from pathlib import Path
from datetime import datetime, timezone
started = time.monotonic()
work = Path('/kaggle/working')
receipt = {'schema_version':'rsna-ticket60-tpu-environment-v1',
    'status':'STARTED', 'python':platform.python_version(),
    'requested_machine_shape':'TpuV5E8', 'gpu_requested':False,
    'model_inference':False, 'fitting':False, 'mri_inputs':False,
    'competition_submission':False, 'external_spend_usd':0}
def save():
    receipt['observed_at_utc'] = datetime.now(timezone.utc).isoformat()
    receipt['elapsed_seconds'] = time.monotonic()-started
    (work/'tpu_environment_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\\n')
def expire():
    receipt.update(status='WATCHDOG_EXPIRED')
    save()
    os._exit(124)
watchdog = threading.Timer(540,expire)
watchdog.daemon=True
watchdog.start()
save()
try:
    import torch
    receipt['torch_version'] = torch.__version__
    receipt['cuda_device_count'] = torch.cuda.device_count()
    receipt['cuda_available'] = torch.cuda.is_available()
    if receipt['cuda_device_count'] or receipt['cuda_available']:
        raise RuntimeError('CPU/TPU-only environment request unexpectedly has CUDA')
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    receipt['torch_xla_version']=torch_xla.__version__
    receipt['actual_device_type']=xr.device_type()
    receipt['packages']={}
    for name in ('torch','torch_xla','libtpu','timm','numpy'):
        try: receipt['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: receipt['packages'][name]=None
    if receipt['actual_device_type'] != 'TPU':
        raise RuntimeError('Requested TPU was not provisioned: '+str(receipt['actual_device_type']))
    device=xm.xla_device()
    receipt['xla_device']=str(device)
    receipt['global_runtime_devices']=xr.global_runtime_device_count()
    receipt['addressable_runtime_devices']=xr.addressable_runtime_device_count()
    if receipt['global_runtime_devices'] != 8 or receipt['addressable_runtime_devices'] < 1:
        raise RuntimeError('Requested eight-device TPU host contract differs')
    try: receipt['device_attributes']=torch_xla._XLAC._xla_get_device_attributes(str(device))
    except Exception as exc: receipt['device_attributes_unavailable']=type(exc).__name__
    torch.set_float32_matmul_precision('highest')
    compute_started=time.monotonic()
    values=torch.ones((16,16),dtype=torch.float32,device=device)
    result=values@values.T
    xm.mark_step()
    actual=result.cpu()
    receipt['calculation']={'shape':list(actual.shape),'dtype':str(actual.dtype),
        'minimum':float(actual.min()),'maximum':float(actual.max()),'sum':float(actual.sum()),
        'synchronized_elapsed_seconds':time.monotonic()-compute_started}
    if actual.shape != (16,16) or not torch.isfinite(actual).all() or not torch.equal(actual,torch.full((16,16),16.0)):
        raise RuntimeError('Actual synchronized TPU calculation differs from exact FP32 result')
    receipt['status']='ACTUAL_TPU_VERIFIED'
    save()
    print(json.dumps(receipt,indent=2),flush=True)
except BaseException as exc:
    receipt.update(status='ENVIRONMENT_ADMISSION_FAILED',error_type=type(exc).__name__,
        error=str(exc),traceback=traceback.format_exc())
    save()
    print(json.dumps(receipt,indent=2),flush=True)
    raise
finally:
    watchdog.cancel()
'''
    compile(code, "ticket60-tpu-environment", "exec")
    notebook = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": [
        "Private bounded TPU allocation admission. No MRI, reports, labels, models, training or competition submission. Fail on CPU or missing XLA; preserve actual runtime receipt.\n"]},
        {"cell_type": "code", "metadata": {}, "source": code.splitlines(keepends=True), "execution_count": None, "outputs": []}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}
    write(ROOT / "environment.ipynb", notebook)
    metadata = {"id": HANDLE, "title": "RSNA TPU Environment Admission v1", "code_file": "environment.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": False, "enable_tpu": True, "machine_shape": "TpuV5E8", "enable_internet": False,
                "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    write(ROOT / "kernel-metadata.json", metadata)
    resources = {"maximum_session_seconds": 600, "code_watchdog_seconds": 540,
                 "maximum_launch_attempts": 1, "requested_machine_shape": "TpuV5E8", "gpu_count": 0,
                 "conservative_tpu_quota_reservation_seconds": 600, "external_spend_usd": 0, "automatic_retry": False}
    spec = ROOT / "prospective-spec.md"
    shutil.copyfile(PROJECT / ".scratch/rsna-leaderboard/issues/60-establish-actual-tpu-training-capacity.md", spec)
    shutil.copyfile(__file__, ROOT / "preparation-source.py")
    inputs = [AUTH, quota, spec, ROOT / "preparation-source.py",
              PROJECT / "docs/research/competitive-next-route-2026-10-02.md"]
    request = {"schema_version": "rsna-ticket60-tpu-environment-request-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "kernel_handle": HANDLE, "resources": resources, "notebook_sha256": digest(ROOT / "environment.ipynb"),
               "metadata_sha256": digest(ROOT / "kernel-metadata.json"),
               "frozen_files": [{"path": str(path), "sha256": digest(path)} for path in inputs],
               "model_inference": False, "fitting": False, "competition_submission": False, "gpu_ledger_changed": False,
               "purpose": "Verify an actually provisioned PyTorch/XLA TPU and exact synchronized calculation before any training/resource proposal."}
    write(ROOT / "request.json", request)
    write(DECISION, {"schema_version": "rsna-campaign-autonomous-decision-v1", "decision": "approved",
                    "decision_made_by": "campaign agent under standing owner authorization",
                    "request_sha256": digest(ROOT / "request.json"), "resources": resources,
                    "authorization_sha256": digest(AUTH), "new_exact_owner_numeric_approval_claimed": False,
                    "basis": request["purpose"]})
    print(json.dumps({"request_sha256": digest(ROOT / "request.json"), "kernel_handle": HANDLE, "launched": False}))


if __name__ == "__main__":
    main()
