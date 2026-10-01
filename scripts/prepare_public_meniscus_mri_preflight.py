"""Prepare real-MRI input checks for the frozen paired experiment on CPU."""

import ast
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from prepare_public_meniscus_cpu_preflight import AUTHORIZATION, RUNTIME_SHA


PROJECT = Path(__file__).resolve().parents[1]
DEVELOPMENT = PROJECT / "artifacts/private/public-meniscus-adoption/paired-development-v3"
NATIVE = PROJECT / "artifacts/private/public-meniscus-adoption/native-cache-contract/manifest.json"
NATIVE_SHA = "496949a3a3e789bc1f4ccff595205c911e471c5b5ef669366a2dd0a58e125844"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_cells(code):
    nodes = [node for node in ast.parse(code).body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "UPSTREAM_CELLS"
                     for target in node.targets)]
    if len(nodes) != 1:
        raise ValueError("Anchor source-cell assignment is ambiguous")
    return dict(ast.literal_eval(nodes[0].value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", type=int, default=1)
    args = parser.parse_args()
    if args.revision < 1:
        raise SystemExit("CPU preparation revision must be positive")
    DEST = PROJECT / f"artifacts/private/public-meniscus-adoption/cpu-mri-preflight-v{args.revision}"
    if DEST.exists():
        raise SystemExit("Immutable CPU MRI request already exists")
    request = json.loads((DEVELOPMENT / "request.json").read_text())
    if sha(DEVELOPMENT / "development.ipynb") != request["notebook_sha256"]:
        raise ValueError("Frozen paired notebook changed")
    notebook = json.loads((DEVELOPMENT / "development.ipynb").read_text())
    original = "".join(notebook["cells"][-1]["source"])
    cells = source_cells(original)
    if {str(i): hashlib.sha256(source.encode()).hexdigest() for i, source in cells.items()} != request["anchor_cells"]:
        raise ValueError("Anchor cell identities differ")
    if sha(NATIVE) != NATIVE_SHA:
        raise ValueError("Native cache manifest differs from the verified anchor")
    groups = {member["pixel_group"] for member in json.loads(NATIVE.read_text())["members"]}
    if len(groups) != 1:
        raise ValueError("This exact specialist requires one native pixel group")
    pixel_group = json.loads(next(iter(groups)))
    prefix, separator, _ = original.partition("import torch\n")
    if not separator or prefix.count("threading.Timer(2400, expire)") != 1:
        raise ValueError("Paired input-preparation prefix differs")
    prefix = prefix.replace("threading.Timer(2400, expire)", "threading.Timer(600, expire)")
    prefix = prefix.replace("Paired development inference exceeded 2400 seconds", "CPU MRI preprocessing exceeded 600 seconds")
    code = prefix + '''import re, tempfile, traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
if torch.cuda.device_count() or torch.cuda.is_available():
    raise RuntimeError('Real MRI preflight is CPU-only')
torch.set_num_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['SLOT_SCHEME'] = 'recovered'
'''
    start = cells[5].index("T0 = time.time()\n")
    fragments = [(5, cells[5][start:]), (4, cells[4])]
    log_node = next(node for node in ast.parse(cells[7]).body if isinstance(node, ast.FunctionDef) and node.name == "log")
    fragments.append((7, ast.get_source_segment(cells[7], log_node)))
    fragments.extend((index, cells[index]) for index in (8, 9, 10, 11, 12))
    adopt_node = next(node for node in ast.parse(cells[16]).body if isinstance(node, ast.FunctionDef) and node.name == "adopt_config_globals")
    fragments.append((16, ast.get_source_segment(cells[16], adopt_node)))
    for index, fragment in fragments:
        code += "exec(compile(" + repr(fragment) + f", 'cpu-anchor-fragment-{index}', 'exec'), globals())\n"
    overlay = PROJECT / "scripts/public_meniscus_overlay.py"
    if sha(overlay) != request["overlay_module_sha256"]:
        raise ValueError("Frozen cache adapter changed")
    code += "exec(compile(" + repr(overlay.read_text()) + ", 'public_meniscus_overlay.py', 'exec'), globals())\n"
    runtime = PROJECT / "artifacts/private/external-notebook-audit/frontier-2026-09-30/accessibility-update/public0033_runtime.py"
    if sha(runtime) != RUNTIME_SHA:
        raise ValueError("Reviewed specialist runtime changed")
    contract_nodes = [node for node in ast.parse(runtime.read_text()).body
                      if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                      and node.name in {"Public0033InferenceError", "_validate_cache_inputs"}]
    if len(contract_nodes) != 2:
        raise ValueError("Specialist cache contract definitions changed")
    for node in contract_nodes:
        code += ast.get_source_segment(runtime.read_text(), node) + "\n\n"
    code += f"PIXEL_GROUP = {pixel_group!r}\n"
    code += '''N_SLOTS, SLICES_PER_SLOT, IMAGE_SIZE = 6, 12, 336
ROOT = view
_RSNA_TEST_IDS = expected_ids
HDR_THREADS = PIX_THREADS = ORDER_THREADS = 8
TIME_BUDGET = 600
adopt_config_globals(PIXEL_GROUP)
capture = MeniscusCacheCapture(build_cache)
try:
    metadata = pd.read_csv(view / 'test_series.csv', dtype={'StudyInstanceUID': str})
    planes = dict(zip(metadata.SeriesInstanceUID, metadata.Anatomical_Plane))
    headers = annotate(walk('test_series'))
    if set(headers.StudyInstanceUID) != set(expected_ids):
        raise RuntimeError('Real MRI header coverage differs')
    if 'err' in headers and headers['err'].notna().any():
        raise RuntimeError('Reference MRI header errors occurred')
    slots = pick_slots(headers, planes)
    studies, pixels, mask = capture.build_cache(slots, planes, lat_of(headers, 'CPU Gold58 '), 'CPU Gold58 native')
    retained, retained_mask, retained_studies = capture.take(expected_ids)
    if retained is not pixels or retained_mask is not mask or not isinstance(pixels, np.memmap):
        raise RuntimeError('Completed native disk cache was not retained exactly')
    _validate_cache_inputs(pixels, mask, studies)
    def array_sha(array):
        digest = hashlib.sha256()
        for row in array:
            digest.update(memoryview(np.ascontiguousarray(row)))
        return digest.hexdigest()
    pixel_sha = array_sha(pixels)
    mask_sha = array_sha(mask)
    summaries = [{'uid': uid, 'present_slots': int(mask[i].sum()),
                  'minimum': int(pixels[i].min()), 'maximum': int(pixels[i].max()),
                  'nonzero_pixels': int(np.count_nonzero(pixels[i]))}
                 for i, uid in enumerate(studies)]
    if any(row['nonzero_pixels'] == 0 or row['maximum'] == row['minimum'] for row in summaries):
        raise RuntimeError('A reference study has no informative pixels')
    cache_file = Path(pixels.filename)
    rsna_release_pixels(pixels)
    if cache_file.exists() or array_sha(retained) != pixel_sha or array_sha(retained_mask) != mask_sha:
        raise RuntimeError('Retained MRI pixels/masks changed after anchor unlink')
    record = {'schema_version': 'rsna-public-meniscus-cpu-mri-preflight-v1',
              'status': 'COMPLETE', 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
              'elapsed_seconds': time.monotonic() - started,
              'assigned_gpu_count': 0, 'study_count': len(studies),
              'study_uid_sha256': EXPECTED_UID_SHA, 'training_csv_sha256': EXPECTED_TRAIN_SHA,
              'pixel_group': PIXEL_GROUP, 'cache_shape': list(pixels.shape),
              'cache_dtype': str(pixels.dtype), 'cache_bytes': int(pixels.nbytes),
              'cache_sha256': pixel_sha, 'mask_sha256': mask_sha,
              'header_series_count': len(headers), 'study_summaries': summaries,
              'events': _RSNA_AUDIT['events'], 'decode_failed_series': DECODE_FAILED,
              'strict_specialist_cache_contract': True, 'preserved_after_unlink': True,
              'parent_max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'model_inference': False, 'model_fitting': False, 'competition_submission': False,
              'external_spend_usd': 0}
    (work / 'cpu_mri_preflight.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\\n')
    print(json.dumps({key: value for key, value in record.items() if key not in ('events', 'study_summaries')}, indent=2), flush=True)
finally:
    capture.clear()
    for filename in list(_RSNA_CACHE_FILES):
        Path(filename).unlink(missing_ok=True)
    watchdog.cancel()
'''
    compile(code, "cpu-mri-preflight", "exec")
    metadata = json.loads((DEVELOPMENT / "kernel-metadata.json").read_text())
    handle = "jitheswar/rsna-meniscus-gold58-cpu-preflight"
    metadata.update(id=handle, title=handle.split("/")[1], code_file="cpu-mri-preflight.ipynb",
                    dataset_sources=[], kernel_sources=[], model_sources=[],
                    enable_gpu=False, enable_tpu=False, enable_internet=False, is_private=True)
    DEST.mkdir(parents=True)

    def write(name, value):
        (DEST / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    note = ("Private CPU MRI preflight for the paired Gold58 experiment. Uses exact preprocessing "
            "definitions from the attributed D4 anchor and the public0033 cache contract. "
            "No model inference, training, scoring or final selection. Source credits follow.\n")
    write("cpu-mri-preflight.ipynb", {"nbformat": 4, "nbformat_minor": 5, "metadata": notebook["metadata"],
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": [note + "".join(notebook["cells"][0]["source"])]},
                  {"cell_type": "code", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": code.splitlines(keepends=True)}]})
    write("kernel-metadata.json", metadata)
    write("request.json", {"schema_version": "rsna-public-meniscus-cpu-mri-request-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "kernel_id": handle,
        "parent_development_request_sha256": sha(DEVELOPMENT / "request.json"),
        "native_manifest_sha256": NATIVE_SHA, "specialist_runtime_sha256": RUNTIME_SHA,
        "source_fragments": [{"cell": index, "sha256": hashlib.sha256(fragment.encode()).hexdigest()} for index, fragment in fragments],
        "notebook_sha256": sha(DEST / "cpu-mri-preflight.ipynb"), "metadata_sha256": sha(DEST / "kernel-metadata.json"),
        "assigned_gpu_count": 0, "cpu_worker_threads": 8, "torch_threads": 1,
        "execution_watchdog_seconds": 600, "session_timeout_seconds": 900,
        "maximum_launch_attempts": 1, "expected_cache_bytes": 58 * 6 * 12 * 336 * 336,
        "maximum_saved_output_bytes": 10000000, "external_spend_usd": 0,
        "authorization_sha256": sha(AUTHORIZATION), "model_inference": False, "model_fitting": False,
        "competition_submission": False, "final_selection": False})
    (DEST / "request-checksum.txt").write_text(sha(DEST / "request.json") + "\n")
    print(DEST)


if __name__ == "__main__":
    main()
