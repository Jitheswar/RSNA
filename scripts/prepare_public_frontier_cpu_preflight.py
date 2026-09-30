"""Prepare a private mount/hash check, without importing model inference code."""

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "artifacts/private/external-notebook-audit/frontier-2026-09-30"
SOURCE = UPSTREAM / "d4-public0946/rsna-knee-d4-public0946.ipynb"
DEST = ROOT / "artifacts/private/public-frontier-adoption/cpu-asset-preflight-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise SystemExit("Immutable request directory already exists")
    notebook = json.loads(SOURCE.read_text())
    excerpts = []
    for index, names in [(3, {"ASSET_ROOTS", "_asset_walk", "_asset_find_asset"}),
                         (5, {"rsna_sha", "rsna_json", "rsna_asset_preflight"})]:
        text = "".join(notebook["cells"][index]["source"])
        seen = set()
        for node in ast.parse(text).body:
            name = node.name if isinstance(node, ast.FunctionDef) else None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                name = getattr(node.targets[0], "id", None)
            if name in names and name not in seen:
                excerpts.append(ast.get_source_segment(text, node))
                seen.add(name)
        if seen != names:
            raise ValueError(f"Missing source definitions: {names - seen}")
    manifest_hashes = {p.name: sha(p) for p in (UPSTREAM / "publisher-manifests").glob("*.json")}
    manifest_hashes["coat_resgated_ep10_top3_manifest.json"] = "98511a8fdeb9da0e6e70c78d013ff636e1476f31c80b5dc134d294b18c3f284e"
    code = '''from pathlib import Path
import os, hashlib, json, time, threading, importlib
from datetime import datetime, timezone
import pandas as pd
started=time.monotonic()
def expire():
    print('CPU asset preflight exceeded 600 seconds', flush=True)
    os._exit(124)
watchdog=threading.Timer(600, expire)
watchdog.daemon=True
watchdog.start()
import torch
if torch.cuda.is_available() or torch.cuda.device_count():
    raise RuntimeError('This request is CPU-only')
'''
    code += "\n\n".join(excerpts) + "\n"
    code += "MANIFEST_HASHES=" + repr(manifest_hashes) + "\n"
    code += '''rsna_asset_preflight()
verified={}
for name,digest in MANIFEST_HASHES.items():
    path=_asset_find_asset(name,digest)
    manifest=json.loads(path.read_text())
    files={}
    for relative,expected in manifest['files'].items():
        candidate=(path.parent/relative).resolve()
        if not candidate.is_relative_to(path.parent.resolve()):
            raise RuntimeError('Manifest path escapes its attachment')
        actual=rsna_sha(candidate)
        if actual != expected:
            raise RuntimeError('Manifest file changed: '+str(candidate))
        files[relative]={'bytes':candidate.stat().st_size,'sha256':actual}
    verified[name]={'path':str(path),'sha256':digest,'files':files}
comp=Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
if not (comp/'test.csv').is_file():
    comp=Path('/kaggle/input/rsna-knee-abnormality-detection')
sample=pd.read_csv(comp/'sample_submission.csv',dtype={'StudyInstanceUID':str})
targets=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
if list(sample.columns)!=['StudyInstanceUID']+targets:
    raise RuntimeError('Competition output schema changed')
e13_candidates=[Path('/kaggle/input/notebooks/sofiaanjenje/rsna-knee-e13-train/rsna_rad_e11/v52_e11_heads.pt'),Path('/kaggle/input/rsna-knee-e13-train/rsna_rad_e11/v52_e11_heads.pt')]
e13=next((p for p in e13_candidates if p.is_file()),None)
if e13 is None or rsna_sha(e13)!='ad9f19af73bfdf4e49263c0e45060dc3cb239e1195039b26dc8c0a3a6bcd1a8a':
    raise RuntimeError('E13 notebook-output mount missing or changed')
dino=_asset_find_asset('pytorch_model.bin')
pilkwang=Path('/kaggle/input/datasets/pilkwang/rsna-knee-weights/manifest.json')
if not pilkwang.is_file():
    pilkwang=Path('/kaggle/input/rsna-knee-weights/manifest.json')
if not pilkwang.is_file():raise RuntimeError('DINO ensemble manifest missing')
versions={name:str(getattr(importlib.import_module(name),'__version__','available')) for name in ('torch','torchvision','timm','numpy','pandas','pydicom','cv2')}
record={'schema_version':'rsna-public-frontier-cpu-asset-preflight-v1','observed_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,'gpu_count':torch.cuda.device_count(),'competition_root':str(comp),'sample_rows':len(sample),'verified_manifests':verified,'e13_path':str(e13),'dino_model':{'path':str(dino),'bytes':dino.stat().st_size,'sha256':rsna_sha(dino)},'pilkwang_manifest_sha256':rsna_sha(pilkwang),'versions':versions,'model_inference_executed':False,'note':'Package imports verify the CPU container only; model-loader identities and GPU dependencies remain separate gates.'}
rsna_json('/kaggle/working/preflight.json',record)
watchdog.cancel()
print(json.dumps(record,indent=2,sort_keys=True))
'''
    compile(code, "cpu-preflight", "exec")
    metadata = json.loads((SOURCE.parent / "kernel-metadata.json").read_text())
    for field in ("id_no", "machine_shape", "keywords"):
        metadata.pop(field, None)
    handle = "jitheswar/rsna-public-frontier-cpu-asset-preflight"
    metadata.update(id=handle, title=handle.split("/")[1], code_file="cpu-preflight.ipynb",
                    is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=False)
    DEST.mkdir(parents=True)
    def write(name, obj):
        (DEST / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    write("cpu-preflight.ipynb", {"nbformat":4,"nbformat_minor":5,
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},
        "cells":[{"cell_type":"markdown","metadata":{},"source":["Private CPU mount/hash preflight. Definitions excerpted from yamadan96/rsna-knee-d4-public0946; original pipeline credits ryokucha, mattiaangeli, antoinegg1, prvsiyan, pilkwang, marwanmath, dreaddevelopment and sofiaanjenje. Source license not independently confirmed. No public redistribution or final selection.\n"]},
                 {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code.splitlines(keepends=True)}]})
    write("kernel-metadata.json", metadata)
    write("request.json", {"schema_version":"rsna-public-frontier-cpu-preflight-request-v1",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"kernel_id":handle,
        "upstream_url":"https://www.kaggle.com/code/yamadan96/rsna-knee-d4-public0946",
        "upstream_source_sha256":sha(SOURCE),"upstream_metadata_sha256":sha(SOURCE.parent / "kernel-metadata.json"),
        "notebook_sha256":sha(DEST / "cpu-preflight.ipynb"),"metadata_sha256":sha(DEST / "kernel-metadata.json"),
        "execution_cap_seconds":600,"assigned_gpu_count":0,"external_spend_usd":0,
        "model_fitting":False,"model_inference":False,"competition_submission":False,
        "source_license_status":"not independently confirmed; private investigation only",
        "authorization_basis":"Existing broad owner zero-spend campaign authorization; private CPU input-validation stage for a separate public baseline route.",
        "maximum_launch_attempts":3})
    (DEST / "request-checksum.txt").write_text(sha(DEST / "request.json") + "\n")
    print(DEST)


if __name__ == "__main__":
    main()
