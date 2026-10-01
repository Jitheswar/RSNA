"""Freeze one paired Gold58 inference experiment; do not launch it."""

import ast
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import pandas as pd

from public_meniscus_overlay import TARGETS
from prepare_public_meniscus_cpu_preflight import (
    AUTHORIZATION, DATASET, DATASET_VERSION, DATASET_VERSION_ID, MANIFEST_SHA, RUNTIME_SHA,
)


PROJECT = Path(__file__).resolve().parents[1]
ANCHOR = PROJECT / "artifacts/private/public-frontier-adoption/scored-baseline-v1"
CPU = PROJECT / "artifacts/private/public-meniscus-adoption/cpu-asset-preflight-v1"
MRI = PROJECT / "artifacts/private/public-meniscus-adoption/cpu-mri-preflight-v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", type=int, default=1)
    args = parser.parse_args()
    if args.revision < 1:
        raise SystemExit("Preparation revision must be positive")
    DEST = PROJECT / f"artifacts/private/public-meniscus-adoption/paired-development-v{args.revision}"
    if DEST.exists():
        raise SystemExit("Immutable development experiment already exists")
    cpu = json.loads((CPU / "validation.json").read_text())
    if (cpu["status"] != "COMPLETE" or cpu["gpu_count"] != 0
            or cpu["strict_fp32_state_tensors"] != 233
            or sha(CPU / "output/specialist_cpu_preflight.json") != cpu["output_sha256"]):
        raise ValueError("Independent CPU admission is incomplete or changed")
    mri = json.loads((MRI / "validation.json").read_text())
    if (mri["status"] != "COMPLETE" or mri["gpu_count"] != 0
            or mri["study_count"] != 58 or mri["decode_failures"] != 0
            or mri["fallback_events"] != 0 or not mri["preserved_after_unlink"]
            or sha(MRI / "output/cpu_mri_preflight.json") != mri["output_sha256"]):
        raise ValueError("Actual CPU MRI admission is incomplete or changed")
    anchor_validation = json.loads((ANCHOR / "validation.json").read_text())
    if anchor_validation["status"] != "COMPLETE":
        raise ValueError("Frozen anchor visible verification is incomplete")
    anchor_notebook = json.loads((ANCHOR / "baseline.ipynb").read_text())
    anchor_text = "".join(anchor_notebook["cells"][-1]["source"])
    assignments = [node for node in ast.parse(anchor_text).body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "UPSTREAM_CELLS"
                           for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("Exact anchor source-cell definition is ambiguous")
    cells = ast.literal_eval(assignments[0].value)
    if len(cells) != 23 or sum(index == 18 for index, source in cells) != 1:
        raise ValueError("Frozen 23-cell D4 recipe changed")
    for index, source in cells:
        compile(source, f"anchor-cell-{index}", "exec")
    archive = PROJECT / "rsna-knee-abnormality-detection.zip"
    with zipfile.ZipFile(archive) as zipped:
        train_bytes = zipped.read("train.csv")
    import io
    train = pd.read_csv(io.BytesIO(train_bytes), dtype={"StudyInstanceUID": str})
    gold_ids = train.loc[train[TARGETS].notna().all(axis=1), "StudyInstanceUID"].tolist()
    if len(gold_ids) != 58 or len(set(gold_ids)) != 58:
        raise ValueError("Reference study membership differs from Gold58")
    uid_sha = hashlib.sha256("".join(uid + "\n" for uid in gold_ids).encode()).hexdigest()
    overlay_path = PROJECT / "scripts/public_meniscus_overlay.py"
    validator_path = PROJECT / "scripts/public_frontier_validation.py"
    code = '''from pathlib import Path
from datetime import datetime, timezone
import gc, hashlib, importlib.util, json, linecache, os, resource, threading, time
import numpy as np
import pandas as pd
started = time.monotonic()
work = Path('/kaggle/working')
def expire():
    (work / 'submission.csv').unlink(missing_ok=True)
    print('Paired development inference exceeded 2400 seconds', flush=True)
    os._exit(124)
watchdog = threading.Timer(2400, expire)
watchdog.daemon = True
watchdog.start()
comp = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
if not (comp / 'train.csv').is_file():
    comp = Path('/kaggle/input/rsna-knee-abnormality-detection')
'''
    code += f"EXPECTED_TRAIN_SHA = {hashlib.sha256(train_bytes).hexdigest()!r}\n"
    code += f"EXPECTED_UID_SHA = {uid_sha!r}\n"
    code += f"DEVELOPMENT_TARGETS = {TARGETS!r}\n"
    code += f"EXPECTED_MANIFEST_SHA = {MANIFEST_SHA!r}\nEXPECTED_RUNTIME_SHA = {RUNTIME_SHA!r}\n"
    code += f"EXPECTED_CACHE_SHA = {mri['cache_sha256']!r}\nEXPECTED_MASK_SHA = {mri['mask_sha256']!r}\n"
    code += '''def development_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
if development_sha(comp / 'train.csv') != EXPECTED_TRAIN_SHA:
    raise RuntimeError('Competition training metadata differs from the frozen source')
training = pd.read_csv(comp / 'train.csv', dtype={'StudyInstanceUID': str})
expected_ids = training.loc[training[DEVELOPMENT_TARGETS].notna().all(axis=1), 'StudyInstanceUID'].tolist()
if len(expected_ids) != 58 or len(set(expected_ids)) != 58:
    raise RuntimeError('This request admits exactly the frozen Gold58 development cohort')
if hashlib.sha256(''.join(uid + '\\n' for uid in expected_ids).encode()).hexdigest() != EXPECTED_UID_SHA:
    raise RuntimeError('Reference cohort identity/order changed')
# This explicit view uses training images solely for development prediction.
# Prediction code retains the original test interface and never sees labels.
view = Path('/tmp/rsna_gold58_development')
view.mkdir()
(view / 'test_series').mkdir()
for uid in expected_ids:
    original = comp / 'train_series' / uid
    if not original.is_dir():
        raise RuntimeError('Reference MRI source missing: ' + uid)
    (view / 'test_series' / uid).symlink_to(original, target_is_directory=True)
pd.DataFrame({'StudyInstanceUID': expected_ids}).to_csv(view / 'test.csv', index=False)
series = pd.read_csv(comp / 'train_series.csv', dtype={'StudyInstanceUID': str})
series = series.loc[series.StudyInstanceUID.isin(expected_ids)]
if set(series.StudyInstanceUID) != set(expected_ids):
    raise RuntimeError('Reference series metadata coverage differs')
series.to_csv(view / 'test_series.csv', index=False)
sample = pd.DataFrame(np.full((58, 12), 0.5), columns=DEVELOPMENT_TARGETS)
sample.insert(0, 'StudyInstanceUID', expected_ids)
sample.to_csv(view / 'sample_submission.csv', index=False)
# Only the number of training studies is consumed by the cache planner.
pd.DataFrame({'StudyInstanceUID': training.StudyInstanceUID}).to_csv(view / 'train.csv', index=False)
del training, series
os.environ['RSNA_COMP_ROOT'] = str(view)
os.environ['RSNA_CACHE_RAM_GIB'] = '0.1'  # Exercise the same disk-cache branch needed for hidden data.
os.environ['RSNA_PARALLEL_COAT_READERS'] = '0'  # Preserve the anchor's >48-study execution policy.
expected_columns = ['StudyInstanceUID'] + DEVELOPMENT_TARGETS
import torch
if torch.cuda.device_count() != 2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
    raise RuntimeError('This exact experiment requires two free T4 GPUs')
for i in range(2):
    torch.cuda.reset_peak_memory_stats(i)
'''
    code += "exec(compile(" + repr(overlay_path.read_text()) + ", 'public_meniscus_overlay.py', 'exec'), globals())\n"
    code += "exec(compile(" + repr(validator_path.read_text()) + ", 'public_frontier_validation.py', 'exec'), globals())\n"
    code += "UPSTREAM_CELLS = " + repr(cells) + "\n"
    code += '''def run_specialist(capture):
    roots = [Path('/kaggle/input/datasets/renta0426/rsna-knee-public0033-meniscus-bag-v1'),
             Path('/kaggle/input/rsna-knee-public0033-meniscus-bag-v1')]
    roots = sorted({root.resolve() for root in roots if (root / 'bundle_manifest.json').is_file()})
    if len(roots) != 1:
        raise RuntimeError('Exactly one admitted specialist bundle is required')
    root = roots[0]
    if development_sha(root / 'bundle_manifest.json') != EXPECTED_MANIFEST_SHA:
        raise RuntimeError('Specialist manifest differs from CPU admission')
    if development_sha(root / 'public0033_runtime.py') != EXPECTED_RUNTIME_SHA:
        raise RuntimeError('Specialist runtime differs from CPU admission')
    spec = importlib.util.spec_from_file_location('_verified_meniscus_runtime', root / 'public0033_runtime.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    cache, mask, studies = capture.take(expected_ids)
    if not isinstance(cache, np.memmap):
        raise RuntimeError('Development request must exercise a disk-backed specialist cache')
    def array_sha(array):
        digest = hashlib.sha256()
        for row in array:
            digest.update(memoryview(np.ascontiguousarray(row)))
        return digest.hexdigest()
    if array_sha(cache) != EXPECTED_CACHE_SHA or array_sha(mask) != EXPECTED_MASK_SHA:
        raise RuntimeError('Native MRI pixels or masks differ from actual CPU admission')
    result = run_preserving_random_state(runtime.run_cached_inference, cache, mask, studies, {
        'bag_raw_csv': str(work / 'public0033_bag_raw.csv'),
        'receipt_json': str(work / 'public0033_cached_inference_receipt.json'),
        'work_dir': str(work)})
    if result.get('status') != 'passed' or result.get('fallback') != 0 or result.get('study_count') != 58:
        raise RuntimeError('Specialist did not complete for the exact cohort; fallback is forbidden')
    capture.clear()
    del cache, mask
    gc.collect()
    torch.cuda.empty_cache()
try:
    for index, source in UPSTREAM_CELLS:
        print('ANCHOR_CELL_START', index, flush=True)
        filename = f'<rsna-upstream-cell-{index}>'
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        if index == 18:
            original_build_cache = build_cache
            capture = MeniscusCacheCapture(original_build_cache)
            build_cache = capture.build_cache
            try:
                exec(compile(source, filename, 'exec'), globals())
                run_specialist(capture)
            finally:
                build_cache = original_build_cache
                capture.clear()
        else:
            exec(compile(source, filename, 'exec'), globals())
        print('ANCHOR_CELL_COMPLETE', index, flush=True)
    audit = json.loads((work / 'btkd_v559_complete.json').read_text())
    blend = json.loads((work / '_coat_raptor_blend_receipt.json').read_text())
    receipt_paths = {'resgated_top3': work / '_coat_arm_receipt.json',
                     'global96_top3': work / '_coat_g96_receipt.json',
                     'repairv1_top3': work / '_coat_repairv1_receipt.json',
                     'd4_swa3': work / 'd4_input/d4.receipt.json'}
    receipts = {name: json.loads(path.read_text()) for name, path in receipt_paths.items()}
    strict = validate_public_frontier(audit, blend, receipts, 58)
    phases = audit['phases']
    required = REQUIRED_PHASES | {name for name in phases if name.startswith('dino_')}
    for name in required:
        proof = phases[name]
        if development_sha(proof['path']) != proof['sha256']:
            raise RuntimeError('Anchor prediction bytes changed: ' + name)
        with np.load(proof['path'], allow_pickle=False) as saved:
            if saved['study_uids'].astype(str).tolist() != expected_ids or saved['labels'].astype(str).tolist() != DEVELOPMENT_TARGETS:
                raise RuntimeError('Anchor prediction identity/order changed: ' + name)
            if saved['values'].shape != (58, 12) or not np.isfinite(saved['values']).all():
                raise RuntimeError('Invalid anchor predictions: ' + name)
    for log in work.rglob('*.log'):
        if '[dense-fallback]' in log.read_text(errors='replace'):
            raise RuntimeError('An anchor reader used an input fallback')
    anchor = pd.read_csv(work / 'submission.csv', dtype={'StudyInstanceUID': str}, float_precision='round_trip')
    specialist = pd.read_csv(work / 'public0033_bag_raw.csv', dtype={'StudyInstanceUID': str}, float_precision='round_trip')
    if anchor.StudyInstanceUID.tolist() != expected_ids:
        raise RuntimeError('Anchor study order differs')
    overlay = apply_meniscus_overlay(anchor, _blend_transformer, _blend_coatnet, specialist)
    (work / 'submission.csv').replace(work / 'development-anchor.csv')
    overlay.to_csv(work / 'development-overlay.csv', index=False)
    _blend_transformer.to_csv(work / 'development-transformer.csv', index=False)
    _blend_coatnet.to_csv(work / 'development-hybrid.csv', index=False)
    specialist_receipt = json.loads((work / 'public0033_cached_inference_receipt.json').read_text())
    if specialist_receipt['status'] != 'passed' or specialist_receipt['fallback'] != 0:
        raise RuntimeError('Specialist completion receipt differs')
    record = {'schema_version': 'rsna-public-meniscus-paired-development-v1',
              'status': 'COMPLETE', 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
              'elapsed_seconds': time.monotonic() - started, 'study_count': 58,
              'study_uid_sha256': EXPECTED_UID_SHA, 'anchor_code_cells_unchanged': 23,
              'strict_anchor_execution': strict, 'specialist_fallback': 0,
              'eleven_other_targets_identical': True, 'overlay_weight': 0.10,
              'input_cache_disk_branch': True,
              'input_cache_sha256': EXPECTED_CACHE_SHA, 'input_mask_sha256': EXPECTED_MASK_SHA,
              'actual_cpu_mri_parity': True,
              'anchor_random_states_preserved': True,
              'output_sha256': {name: development_sha(work / name) for name in
                  ('development-anchor.csv', 'development-overlay.csv', 'development-transformer.csv',
                   'development-hybrid.csv', 'public0033_bag_raw.csv', 'public0033_cached_inference_receipt.json')},
              'parent_max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'parent_gpu_memory': [{'device': i, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(i),
                                     'peak_reserved_bytes': torch.cuda.max_memory_reserved(i)} for i in range(2)],
              'competition_submission': False, 'model_fitting': False, 'final_selection': False,
              'limitation': 'Reused Gold58 development set; public parents were selected on these labels. This is not clean OOF or an untouched generalization estimate.'}
    (work / 'paired_development.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\\n')
    print(json.dumps(record, indent=2), flush=True)
finally:
    (work / 'submission.csv').unlink(missing_ok=True)
    watchdog.cancel()
'''
    compile(code, "paired-development", "exec")
    metadata = json.loads((ANCHOR / "kernel-metadata.json").read_text())
    handle = "jitheswar/rsna-d4-meniscus-paired-development"
    metadata.update(id=handle, title=handle.split("/")[1], code_file="development.ipynb")
    metadata["dataset_sources"].append(f"{DATASET}/{DATASET_VERSION}")
    DEST.mkdir(parents=True)

    def write(name, value):
        (DEST / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    credits = "".join(anchor_notebook["cells"][0]["source"])
    note = ("Private paired development inference of frozen D4 and one 10% Medial Meniscus "
            "specialist vote. Public0033 weights/runtime belong to renta0426 (Apache-2.0). "
            "Uses the 58 reference training studies for development prediction. "
            "No fitting, competition submission or final selection. Reused development data; not clean OOF.\n")
    write("development.ipynb", {"nbformat": 4, "nbformat_minor": 5,
        "metadata": anchor_notebook["metadata"],
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": [note + credits]},
                  {"cell_type": "code", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": code.splitlines(keepends=True)}]})
    write("kernel-metadata.json", metadata)
    shutil.copyfile(ANCHOR / "LICENSE-APACHE-2.0.txt", DEST / "LICENSE-APACHE-2.0.txt")
    (DEST / "NOTICE.md").write_text(note + "\n" + (ANCHOR / "NOTICE.md").read_text())
    write("request.json", {"schema_version": "rsna-public-meniscus-development-request-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "kernel_id": handle,
        "anchor_notebook_sha256": sha(ANCHOR / "baseline.ipynb"),
        "anchor_remote_kernel_sha256": sha(ANCHOR / "remote-kernel.json"),
        "cpu_admission_sha256": sha(CPU / "validation.json"), "overlay_module_sha256": sha(overlay_path),
        "cpu_mri_admission_sha256": sha(MRI / "validation.json"),
        "native_cache_sha256": mri["cache_sha256"], "native_mask_sha256": mri["mask_sha256"],
        "anchor_validator_sha256": sha(validator_path), "notebook_sha256": sha(DEST / "development.ipynb"),
        "metadata_sha256": sha(DEST / "kernel-metadata.json"),
        "anchor_cells": {str(i): hashlib.sha256(t.encode()).hexdigest() for i, t in cells},
        "competition_train_csv_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "development_study_count": 58, "development_study_uid_sha256": uid_sha,
        "specialist_dataset_version_number": DATASET_VERSION,
        "specialist_dataset_version_source_id": DATASET_VERSION_ID,
        "specialist_manifest_sha256": MANIFEST_SHA, "specialist_runtime_sha256": RUNTIME_SHA,
        "medial_meniscus_specialist_weight": 0.10, "weights_swept": False,
        "assigned_gpu_count": 2, "session_timeout_seconds": 2700,
        "execution_watchdog_seconds": 2400, "reserved_gpu_seconds": 5400,
        "maximum_launch_attempts": 1, "external_spend_usd": 0,
        "model_fitting": False, "competition_submission": False, "final_selection": False,
        "analysis": {"paired_study_bootstrap_resamples": 2000, "seed": 20261001,
                     "report": ["twelve-target macro AUC", "Medial Meniscus AUC", "paired intervals",
                                "target support", "three fold directions", "all unchanged columns", "actual throughput"],
                     "development_not_clean_oof": True},
        "authorization_sha256": sha(AUTHORIZATION),
        "launch_gate": "Do not launch until exact scoring 56739912 is terminal and reconciled, "
            "one GPU request across backends is clear, free quota/paid scaling are refreshed, "
            "and the concrete campaign resource amendment/reservation is recorded under standing authorization.",
        "proposed_campaign_cap_seconds": 136800,
        "resource_decision_status": "Prepared proposal; no cap amendment, reservation or launch performed."})
    (DEST / "request-checksum.txt").write_text(sha(DEST / "request.json") + "\n")
    print(DEST)


if __name__ == "__main__":
    main()
