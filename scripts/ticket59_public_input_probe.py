"""Probe unchanged public DINO preprocessing on a fixed broader MRI cohort.

This diagnostic executes fragments of the attributed public D4 source. It
does not run model inference or establish a cause for the hidden rerun error.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile


PROJECT = Path(os.environ.get("RSNA_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
ROOT = PROJECT / "artifacts/private/public-frontier-adoption/broader-cpu-input-probe-v1"
CELLS = PROJECT / "artifacts/private/public-frontier-adoption/scoring-failure-diagnosis-v1/upstream-cells"
PREFIX = PROJECT / "artifacts/private/ticket57-resnet-prefix-cohort"
NATIVE = PROJECT / "artifacts/private/public-meniscus-adoption/native-cache-contract/manifest.json"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket59-broader-cpu-input-probe-20261001.json"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False, default=str)
        handle.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fragment_def(source, name):
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(nodes) == 1, "Exact frozen preprocessing function is ambiguous")
    return ast.get_source_segment(source, nodes[0])


def prepare(storage_bytes):
    require(not ROOT.exists() and not DECISION.exists(), "Immutable broader-input probe already exists")
    require(storage_bytes > 0 and storage_bytes + 36 * 1024 ** 3 < 2 * 1024 ** 4,
        "Measured project storage must fit the 2 TiB ceiling with temporary allowance")
    prefix_plan = read(PREFIX / "plan.json")
    manifest = read(PREFIX / "features/feature-manifest.json")
    candidates = [row["study_instance_uid"] for row in manifest["features"]
        if row["source_kind"] == "ticket35-prepared-series-cache-read-only"]
    require(len(candidates) == 4170 and len(set(candidates)) == 4170, "Admitted non-reference MRI membership differs")
    selected = sorted(candidates, key=lambda uid: hashlib.sha256(("ticket59-input-probe-v1:" + uid).encode()).digest())[:256]
    integration_path = Path(prefix_plan["integration_path"])
    integration = read(integration_path)
    by_uid = {row["StudyInstanceUID"]: row for row in integration["studies"]}
    studies = [by_uid[uid] for uid in selected]
    paths = [entry for study in studies for series in study["series"] for entry in series["entries"]]
    require(len(paths) == len(set(paths)) and all(path.startswith("train_series/") and path.endswith(".dcm")
        and ".." not in Path(path).parts for path in paths), "Exact safe archive MRI entries are required")
    native = read(NATIVE)
    require(digest(NATIVE) == "496949a3a3e789bc1f4ccff595205c911e471c5b5ef669366a2dd0a58e125844",
        "Admitted public native-cache contract changed")
    groups = {row["pixel_group"] for row in native["members"]}
    require(len(groups) == 1, "Exactly one admitted DINO pixel group is required")
    ROOT.mkdir(mode=0o700)
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(CELLS, snapshot / "upstream-cells")
    shutil.copyfile(__file__, snapshot / Path(__file__).name)
    shutil.copyfile(PROJECT / "scripts/public_frontier_validation.py", snapshot / "public_frontier_validation.py")
    inputs = [AUTH, PREFIX / "plan.json", PREFIX / "features/feature-manifest.json", integration_path, NATIVE]
    inputs.extend(path for path in snapshot.rglob("*") if path.is_file())
    resources = {"partition": "cluster1-cpu", "nodes": 1, "tasks": 1, "cpus_per_task": 16,
        "memory_gib": 32, "walltime_seconds": 3600, "code_watchdog_seconds": 3540,
        "maximum_launch_attempts": 1, "maximum_concurrent_jobs": 1, "gpu_count": 0,
        "exclusive": False, "requeue": False, "external_spend_usd": 0,
        "maximum_raw_mri_bytes": 32 * 1024 ** 3, "maximum_pixel_cache_bytes": 3 * 1024 ** 3}
    plan = {"schema_version": "rsna-ticket59-broader-input-probe-plan-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_root": str(ROOT),
        "source_snapshot": str(snapshot), "decision_path": str(DECISION), "resources": resources,
        "measured_project_bytes_before_request": storage_bytes,
        "selection": "First 256 SHA256(ticket59-input-probe-v1:UID) among 4,170 admitted report-source MRI studies; reference studies excluded; no label values read.",
        "acquisition_policy": "All original train_series archive DICOM entries and original train_series.csv metadata for selected studies, including series excluded from the learning integration/cache.",
        "selected_study_uids": selected, "studies": studies, "study_count": len(selected),
        "archive_path": prefix_plan["source_archive_path"], "archive_bytes": prefix_plan["source_archive_bytes"],
        "archive_mtime_ns": prefix_plan["source_archive_mtime_ns"], "pixel_group": json.loads(next(iter(groups))),
        "worker_threads": 8, "torch_threads": 1, "expected_cache_bytes": 256 * 6 * 12 * 336 * 336,
        "runtime_versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pydicom", "torch")},
        "frozen_files": [{"path": str(path), "sha256": digest(path)} for path in inputs],
        "hidden_error_attribution": False, "model_inference": False, "fitting": False,
        "competition_submission": False, "candidate_promotion": False, "network_during_execution": False}
    write(ROOT / "plan.json", plan)
    checksum = digest(ROOT / "plan.json")
    write(DECISION, {"schema_version": "rsna-campaign-autonomous-decision-v1", "decision": "approved",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": checksum,
        "resources": resources, "authorization_sha256": digest(AUTH),
        "decision_made_by": "campaign agent under standing owner authorization",
        "basis": "Seek an accessible fatal preprocessing input using exact unchanged public-source fragments on a label-blind broader real-MRI cohort, after Gold58 GPU success failed to reproduce the hidden symptom. No model changes or inference; preserve trace and input identities.",
        "new_exact_owner_numeric_approval_claimed": False, "new_gpu_request": False,
        "gpu_ledger_cap_changed": False, "candidate_promotion": False})
    with (ROOT / "run.sbatch").open("x") as handle:
        handle.write(f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket59-input-probe
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output={ROOT}/slurm-%j.out
#SBATCH --error={ROOT}/slurm-%j.err
#SBATCH --chdir={PROJECT}
set -euo pipefail
umask 077
export RSNA_PROJECT_ROOT={PROJECT}
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
{PROJECT}/.venv/bin/python {snapshot}/{Path(__file__).name} run {ROOT}/plan.json {checksum}
''')
    subprocess.run(["bash", "-n", str(ROOT / "run.sbatch")], check=True)
    print(json.dumps({"plan_sha256": checksum, "launcher": str(ROOT / "run.sbatch"), "study_count": len(selected), "launched": False}))


def run(plan_path, checksum):
    started = time.monotonic()
    require(digest(plan_path) == checksum, "Exact broader-input plan changed")
    plan = read(plan_path)
    root, snapshot = Path(plan["run_root"]), Path(plan["source_snapshot"])
    decision = read(plan["decision_path"])
    require(decision["decision"] == "approved" and decision["plan_sha256"] == checksum
        and decision["resources"] == plan["resources"], "Exact broader-input resource decision differs")
    allocation = subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True)
    fields = dict(token.split("=", 1) for token in allocation.split() if "=" in token)
    expected = {"JobId": os.environ["SLURM_JOB_ID"], "JobState": "RUNNING", "Partition": "cluster1-cpu",
        "NumNodes": "1", "NumCPUs": "16", "NumTasks": "1", "MinMemoryNode": "32G",
        "TimeLimit": "01:00:00", "OverSubscribe": "OK", "Requeue": "0", "Restarts": "0"}
    require(all(fields.get(key) == value for key, value in expected.items())
        and "gpu" not in fields.get("TRES", "").lower(), "Actual bounded CPU probe allocation differs")
    with (root / "running-allocation.txt").open("x") as handle:
        handle.write(allocation)
    watchdog = threading.Timer(plan["resources"]["code_watchdog_seconds"], lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    for row in plan["frozen_files"]:
        require(digest(row["path"]) == row["sha256"], "Frozen preprocessing or MRI membership changed")
    require(all(importlib.metadata.version(name) == version for name, version in plan["runtime_versions"].items()),
        "Frozen CPU probe runtime differs")
    archive_path = Path(plan["archive_path"])
    stat = archive_path.stat()
    require(stat.st_size == plan["archive_bytes"] and stat.st_mtime_ns == plan["archive_mtime_ns"],
        "Original MRI archive stat identity differs")
    import numpy as np
    import pandas as pd
    import pydicom
    import torch
    import torch.nn.functional as F
    from concurrent.futures import ThreadPoolExecutor
    from public_frontier_validation import FAILED_EVENTS
    import re
    import tempfile
    from typing import Sequence

    require(not torch.cuda.is_available() and torch.version.cuda is None, "CPU-only probe runtime is required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    phase, environment, source_rows = "extract-original-mri", {}, []
    view = root / "input-view"
    view.mkdir(mode=0o700)
    (view / "test_series").mkdir(mode=0o700)
    ids = plan["selected_study_uids"]
    try:
        with zipfile.ZipFile(archive_path) as archive:
            selected_ids = set(ids)
            infos = [info for info in archive.infolist() if info.filename.startswith("train_series/")
                and info.filename.endswith(".dcm") and len(Path(info.filename).parts) == 4
                and Path(info.filename).parts[1] in selected_ids]
            source_series = {Path(info.filename).parts[2] for info in infos}
            require({Path(info.filename).parts[1] for info in infos} == selected_ids,
                "Every selected study must have original DICOM entries")
            raw_bytes = sum(info.file_size for info in infos)
            require(raw_bytes <= plan["resources"]["maximum_raw_mri_bytes"], "Actual raw MRI exceeds frozen disk allowance")
            require(shutil.disk_usage(root).free > raw_bytes + plan["expected_cache_bytes"] + 1024 ** 3,
                "Actual source/cache scratch space is insufficient")
            metadata = pd.read_csv(archive.open("train_series.csv"), dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
            metadata = metadata.loc[metadata.StudyInstanceUID.isin(ids)].copy()
            require(set(metadata.StudyInstanceUID) == selected_ids and not metadata.SeriesInstanceUID.duplicated().any(),
                "Original CSV MRI metadata study coverage differs")
            metadata.to_csv(view / "test_series.csv", index=False)
            pd.DataFrame({"StudyInstanceUID": ids}).to_csv(view / "test.csv", index=False)
            def extract(info):
                parts = Path(info.filename).parts
                require(len(parts) == 4 and parts[0] == "train_series" and parts[1] in selected_ids
                    and parts[2] in source_series and ".." not in parts, "Unsafe source MRI entry")
                target = view / "test_series" / Path(*parts[1:])
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 ** 2)
                require(target.stat().st_size == info.file_size, "Extracted MRI byte count differs")
                return {"entry": info.filename, "archive_crc32": info.CRC, "bytes": info.file_size,
                    "actual_sha256": digest(target)}
            with ThreadPoolExecutor(max_workers=8) as executor:
                for index, row in enumerate(executor.map(extract, infos)):
                    source_rows.append(row)
                    if (index + 1) % 4096 == 0:
                        print(json.dumps({"extracted_dicom_files": index + 1, "raw_bytes_planned": raw_bytes}), flush=True)
        write(root / "source-mri-entries.json", source_rows)
        print(json.dumps({"actual_studies": len(ids), "actual_dicom_files": len(source_rows), "actual_raw_bytes": raw_bytes}), flush=True)
        phase = "unchanged-public-preprocessing"
        os.environ["RSNA_PIXEL_SCRATCH"] = str(root / "pixel-scratch")
        os.environ["SLOT_SCHEME"] = "recovered"
        environment = {"__builtins__": __builtins__, "np": np, "pd": pd, "pydicom": pydicom,
            "torch": torch, "F": F, "os": os, "re": re, "time": time, "tempfile": tempfile,
            "traceback": traceback, "Path": Path, "ThreadPoolExecutor": ThreadPoolExecutor,
            "Sequence": Sequence, "WeightsError": RuntimeError}
        cells = {index: (snapshot / f"upstream-cells/cell-{index}.py").read_text() for index in (4, 5, 7, 8, 9, 10, 11, 12, 16)}
        fragments = [(5, cells[5][cells[5].index("T0 = time.time()\n"):]), (4, cells[4]),
            (7, fragment_def(cells[7], "log")), *[(index, cells[index]) for index in (8, 9, 10, 11, 12)],
            (16, fragment_def(cells[16], "adopt_config_globals"))]
        for index, fragment in fragments:
            exec(compile(fragment, f"unchanged-public-anchor-cell-{index}", "exec"), environment)
        environment.update(ROOT=view, _RSNA_TEST_IDS=ids, HDR_THREADS=8, PIX_THREADS=8, ORDER_THREADS=8,
            TIME_BUDGET=plan["resources"]["code_watchdog_seconds"] - (time.monotonic() - started))
        environment["adopt_config_globals"](plan["pixel_group"])
        headers = environment["annotate"](environment["walk"]("test_series"))
        headers.to_csv(root / "headers.csv", index=False)
        require(set(headers.StudyInstanceUID) == set(ids) and len(headers) == len(source_series),
            "Actual public header coverage differs")
        require("err" not in headers or not headers["err"].notna().any(), "Actual public MRI header errors occurred")
        planes = dict(zip(metadata.SeriesInstanceUID, metadata.Anatomical_Plane))
        slots = environment["pick_slots"](headers, planes)
        studies, pixels, mask = environment["build_cache"](slots, planes, environment["lat_of"](headers, "CPU broader "), "CPU broader native")
        require(studies == ids and pixels.dtype == np.uint8 and pixels.shape == (256, 6, 12, 336, 336)
            and pixels.nbytes == plan["expected_cache_bytes"]
            and pixels.nbytes <= plan["resources"]["maximum_pixel_cache_bytes"], "Actual broader native cache shape/identity differs")
        summaries = [{"study_instance_uid": uid, "present_slots": int(mask[index].sum()),
            "nonzero_pixels": int(np.count_nonzero(pixels[index])), "pixel_min": int(pixels[index].min()),
            "pixel_max": int(pixels[index].max())} for index, uid in enumerate(studies)]
        events = environment["_RSNA_AUDIT"]["events"]
        bad = [event for event in events if event.get("kind") in FAILED_EVENTS]
        result = {"schema_version": "rsna-ticket59-broader-input-probe-result-v1", "status": "COMPLETE",
            "observed_at_utc": datetime.now(timezone.utc).isoformat(), "job_id": os.environ["SLURM_JOB_ID"],
            "plan_sha256": checksum, "study_count": len(studies), "dicom_files": len(source_rows),
            "actual_raw_bytes": raw_bytes, "source_entries_sha256": digest(root / "source-mri-entries.json"),
            "header_series_count": len(headers), "metadata_series_count": len(metadata),
            "acquisition_policy": plan["acquisition_policy"], "native_cache_bytes": int(pixels.nbytes),
            "native_cache_disk_branch": isinstance(pixels, np.memmap), "events": events,
            "failed_fallback_events": bad, "decode_failed_series": environment["DECODE_FAILED"],
            "study_summaries": summaries, "unchanged_source_fragments": [{"cell": index,
                "fragment_sha256": hashlib.sha256(fragment.encode()).hexdigest()} for index, fragment in fragments],
            "cpu_input_guard_passed": not bad and not environment["DECODE_FAILED"],
            "hidden_error_reproduced": False, "hidden_error_attribution": False,
            "runtime_scope": "CPU native DINO preprocessing only; excludes model/GPU/CoAt execution and hidden inputs.",
            "labels_read": False, "model_inference": False, "fitting": False, "candidate_promotion": False,
            "competition_submission": False, "gpu_count": 0, "external_spend_usd": 0,
            "elapsed_seconds": time.monotonic() - started,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        write(root / "probe-result.json", result)
        require(result["cpu_input_guard_passed"], "Accessible broader MRI input has failed/fallback preprocessing events; hidden error attribution remains unverified")
        print(json.dumps({"status": "COMPLETE", "study_count": len(studies), "failed_fallback_events": 0,
            "hidden_error_reproduced": False}), flush=True)
    except Exception as exc:
        write(root / "probe-exception.json", {"schema_version": "rsna-ticket59-probe-exception-v1",
            "observed_at_utc": datetime.now(timezone.utc).isoformat(), "job_id": os.environ["SLURM_JOB_ID"],
            "plan_sha256": checksum, "phase": phase, "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "completed_source_files": len(source_rows),
            "events": environment.get("_RSNA_AUDIT", {}).get("events", []), "hidden_error_attribution": False})
        raise
    finally:
        for path in environment.get("_RSNA_CACHE_FILES", []):
            Path(path).unlink(missing_ok=True)
        watchdog.cancel()


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare(int(sys.argv[2]))
    else:
        run(Path(sys.argv[2]), sys.argv[3])
