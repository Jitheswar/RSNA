"""Prepare the pinned public0033 bundle check without accelerator inference."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


PROJECT = Path(__file__).resolve().parents[1]
AUDIT = PROJECT / "artifacts/private/external-notebook-audit/frontier-2026-09-30/accessibility-update"
DEST = PROJECT / "artifacts/private/public-meniscus-adoption/cpu-asset-preflight-v1"
DATASET = "renta0426/rsna-knee-public0033-meniscus-bag-v1"
DATASET_VERSION = 2
DATASET_VERSION_ID = 19356288
MANIFEST_SHA = "b610229f046fe0f27ec8dcf661c5532d4991427428378975d0443da671607945"
RUNTIME_SHA = "9541f82a993d7dc2942ca2a5112e110471285f8d80701846079a3dc708761683"
AUTHORIZATION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise SystemExit("Immutable preparation directory already exists")
    if sha(AUDIT / "bundle_manifest.json") != MANIFEST_SHA:
        raise ValueError("Reviewed bundle manifest changed")
    if sha(AUDIT / "public0033_runtime.py") != RUNTIME_SHA:
        raise ValueError("Reviewed inference runtime changed")
    authorization = json.loads(AUTHORIZATION.read_text())
    if authorization.get("external_spend_usd") != 0:
        raise ValueError("Zero-spend campaign authorization missing")
    manifest = json.loads((AUDIT / "bundle_manifest.json").read_text())
    code = '''from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib, importlib.util, json, os, resource, threading, time
started = time.monotonic()
def expire():
    print('CPU specialist verification exceeded 600 seconds', flush=True)
    os._exit(124)
watchdog = threading.Timer(600, expire)
watchdog.daemon = True
watchdog.start()
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'
import torch
torch.set_num_threads(4)
if torch.cuda.device_count() or torch.cuda.is_available():
    raise RuntimeError('This request is CPU-only')
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
'''
    code += f"EXPECTED_MANIFEST_SHA = {MANIFEST_SHA!r}\n"
    code += f"EXPECTED_RUNTIME_SHA = {RUNTIME_SHA!r}\n"
    code += '''roots = [Path('/kaggle/input/datasets/renta0426/rsna-knee-public0033-meniscus-bag-v1'),
         Path('/kaggle/input/rsna-knee-public0033-meniscus-bag-v1')]
candidates = sorted({p.resolve() for p in roots if (p / 'bundle_manifest.json').is_file()})
if len(candidates) != 1:
    raise RuntimeError('Exactly one public0033 version-2 mount is required')
root = candidates[0]
if sha(root / 'bundle_manifest.json') != EXPECTED_MANIFEST_SHA:
    raise RuntimeError('Mounted manifest differs from the reviewed version')
if sha(root / 'public0033_runtime.py') != EXPECTED_RUNTIME_SHA:
    raise RuntimeError('Mounted runtime differs from the reviewed version')
spec = importlib.util.spec_from_file_location('_checked_public0033', root / 'public0033_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
manifest = runtime.verify_bundle(root)
verified = {}
for name, expected in manifest['files'].items():
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError('Bundle file escapes its attachment')
    if name == 'dataset-metadata.json' and not path.is_file():
        continue
    actual = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    if actual != expected:
        raise RuntimeError('Bundle file differs: ' + name)
    verified[name] = actual
if len(verified) != 8 + int((root / 'dataset-metadata.json').is_file()):
    raise RuntimeError('Incomplete bundle verification')
model = runtime._load_fullfit_model(root, torch.device('cpu'))
state = model.state_dict()
if len(state) != 233 or any(value.dtype != torch.float32 for value in state.values()):
    raise RuntimeError('Strict loaded state tensor count/FP32 identity changed')
if any(parameter.requires_grad is False for parameter in model.parameters()):
    raise RuntimeError('Unexpected loader structure')
state_description = {name: {'shape': list(value.shape), 'dtype': str(value.dtype)}
                     for name, value in state.items()}
state_identity = hashlib.sha256(json.dumps(state_description, sort_keys=True,
    separators=(',', ':')).encode()).hexdigest()
contract = json.loads((root / 'run_contract.json').read_text())
versions = {name: str(getattr(importlib.import_module(name), '__version__', 'available'))
            for name in ('torch', 'transformers', 'safetensors', 'numpy')}
record = {'schema_version': 'rsna-public-meniscus-cpu-preflight-v1',
          'status': 'COMPLETE', 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
          'elapsed_seconds': time.monotonic() - started,
          'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'assigned_gpu_count': torch.cuda.device_count(), 'root': str(root),
          'dataset_version_number': 2, 'expected_dataset_version_source_id': 19356288,
          'bundle_manifest_sha256': EXPECTED_MANIFEST_SHA, 'verified_files': verified,
          'strict_state_load': True, 'state_tensor_count': len(state),
          'state_structure_sha256': state_identity, 'state_structure': state_description,
          'versions': versions, 'training_cohort': contract['cohort'],
          'fold_safe_oof': contract['fold_safe_oof'],
          'inference_gpu_count_required': manifest['required_gpu_count'],
          'inference_gpu_family_required': manifest['required_gpu_family'],
          'model_fitting': False, 'model_inference': False,
          'competition_submission': False, 'final_selection': False,
          'external_spend_usd': 0,
          'limitation': 'CPU bundle/load verification; no evidence of GPU parity, predictive gain or hidden runtime.'}
output = Path('/kaggle/working/specialist_cpu_preflight.json')
output.write_text(json.dumps(record, indent=2, sort_keys=True) + '\\n')
watchdog.cancel()
print(json.dumps({key: value for key, value in record.items() if key != 'state_structure'},
                 indent=2, sort_keys=True), flush=True)
'''
    compile(code, "meniscus-cpu-preflight", "exec")
    metadata = json.loads((PROJECT / "artifacts/private/public-frontier-adoption/cpu-asset-preflight-v1/kernel-metadata.json").read_text())
    handle = "jitheswar/rsna-public-meniscus-cpu-preflight"
    metadata.update(id=handle, title=handle.split("/")[1], code_file="cpu-preflight.ipynb",
                    dataset_sources=[f"{DATASET}/{DATASET_VERSION}"], competition_sources=[],
                    kernel_sources=[], model_sources=[], is_private=True, enable_gpu=False,
                    enable_tpu=False, enable_internet=False)
    DEST.mkdir(parents=True)

    def write(name, value):
        (DEST / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    credit = ("Private CPU validation of renta0426's public0033 Meniscus Bag v2, Apache-2.0. "
              "Model and runtime are upstream work. Checks immutable file hashes, provenance and strict CPU "
              "state loading. No fitting, predictions, competition submission or final selection.\n")
    write("cpu-preflight.ipynb", {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": [credit]},
                  {"cell_type": "code", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": code.splitlines(keepends=True)}]})
    write("kernel-metadata.json", metadata)
    shutil.copyfile(PROJECT / "artifacts/private/public-notebook-adoption/attributed-ensemble-draft/LICENSE-APACHE-2.0.txt",
                    DEST / "LICENSE-APACHE-2.0.txt")
    (DEST / "NOTICE.md").write_text(credit + "\nhttps://www.kaggle.com/datasets/" + DATASET + "\n")
    write("request.json", {"schema_version": "rsna-public-meniscus-cpu-request-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "kernel_id": handle,
        "dataset": DATASET, "dataset_version_number": DATASET_VERSION,
        "dataset_version_source_id": DATASET_VERSION_ID,
        "bundle_manifest_sha256": MANIFEST_SHA, "runtime_sha256": RUNTIME_SHA,
        "checkpoint": manifest["checkpoint"],
        "notebook_sha256": sha(DEST / "cpu-preflight.ipynb"),
        "metadata_sha256": sha(DEST / "kernel-metadata.json"),
        "assigned_gpu_count": 0, "cpu_threads": 4, "execution_watchdog_seconds": 600,
        "session_timeout_seconds": 900, "maximum_launch_attempts": 1,
        "maximum_local_output_bytes": 10000000, "external_spend_usd": 0,
        "authorization_sha256": sha(AUTHORIZATION),
        "authorization_basis": "Agent campaign decision under recorded standing owner authorization; CPU-only asset admission.",
        "model_fitting": False, "model_inference": False,
        "competition_submission": False, "final_selection": False})
    (DEST / "request-checksum.txt").write_text(sha(DEST / "request.json") + "\n")
    print(DEST)


if __name__ == "__main__":
    main()
