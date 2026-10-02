"""Exercise the exact D4 residual CoAt CPU preparation seam on original MRI."""

from concurrent.futures import ProcessPoolExecutor
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def init_worker(runtime_source, cv2_site):
    sys.path[:0] = [cv2_site, runtime_source]
    import torch
    import cv2
    require(torch.version.cuda is None and not torch.cuda.is_available(), "CPU-only CoAt probe required")
    require(cv2.__version__ == "4.12.0", "Exact admitted OpenCV required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)


def prepare_one(payload):
    import input_resgated_runtime as runtime
    index, uid, records, series_root = payload
    started = time.monotonic()
    logs = io.StringIO()
    try:
        with contextlib.redirect_stdout(logs):
            actual_index, bag = runtime._prepare_bag_task((index, uid, records, Path(series_root)))
        require(actual_index == index and bag.study_uid == uid, "Actual call-site study identity changed")
        arrays = (bag.images, bag.slots, bag.metadata)
        return {"index": index, "study_uid": uid, "status": "COMPLETE", "windows": len(bag.images),
                "image_shape": list(bag.images.shape), "payload_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays],
                "payload_bytes": sum(a.nbytes for a in arrays), "preparation_errors": list(bag.errors),
                "sparse_fallback": bag.sparse_fallback, "dense_sampler_fallback": "[dense-fallback]" in logs.getvalue(),
                "stdout": logs.getvalue(), "elapsed_seconds": time.monotonic() - started,
                "peak_worker_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    except Exception as exc:
        return {"index": index, "study_uid": uid, "status": "ERROR", "error_type": type(exc).__name__,
                "error": str(exc), "traceback": traceback.format_exc(), "stdout": logs.getvalue(),
                "elapsed_seconds": time.monotonic() - started, "hidden_error_attribution": False}


def run(plan_path, checksum):
    started = time.monotonic()
    require(digest(plan_path) == checksum, "Immutable CoAt probe plan changed")
    plan = read(plan_path)
    root = Path(plan["run_root"])
    decision = read(plan["decision_path"])
    require(decision["decision"] == "approved" and decision["plan_sha256"] == checksum
            and decision["resources"] == plan["resources"], "Exact CoAt resource decision differs")
    allocation = subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True)
    fields = dict(token.split("=", 1) for token in allocation.split() if "=" in token)
    require(all(fields.get(k) == v for k, v in {"JobId": os.environ["SLURM_JOB_ID"], "JobState": "RUNNING",
            "Partition": "cluster1-cpu", "NumNodes": "1", "NumCPUs": "16", "MinMemoryNode": "64G",
            "TimeLimit": "01:00:00", "Requeue": "0", "Restarts": "0", "OverSubscribe": "OK"}.items())
            and "gpu" not in fields.get("TRES", "").lower(), "Actual CoAt CPU allocation differs")
    with (root / "running-allocation.txt").open("x") as handle:
        handle.write(allocation)
    watchdog = threading.Timer(3540, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    for row in plan["frozen_files"]:
        require(digest(row["path"]) == row["sha256"], "Exact upstream preprocessing or original membership changed")
    require(all(importlib.metadata.version(k) == v for k, v in plan["runtime_versions"].items()), "CoAt CPU runtime changed")
    try:
        payloads = [(i, row["study_uid"], row["records"], row["series_root"]) for i, row in enumerate(plan["studies"])]
        results = []
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"),
                initializer=init_worker, initargs=(plan["runtime_source"], plan["cv2_site"])) as executor:
            with (root / "study-results.jsonl").open("x") as journal:
                for row in executor.map(prepare_one, payloads, chunksize=1):
                    results.append(row)
                    journal.write(json.dumps(row, sort_keys=True) + "\n")
                    journal.flush()
                    print(json.dumps({"studies_processed": len(results), "status": row["status"],
                                      "windows": row.get("windows")}), flush=True)
        errors = [row for row in results if row["status"] != "COMPLETE"]
        dense_fallbacks = [row["study_uid"] for row in results if row.get("dense_sampler_fallback")]
        result = {"schema_version": "rsna-ticket59-coat-input-result-v1", "status": "COMPLETE" if not errors else "INPUT_FAILURE",
                  "observed_at_utc": datetime.now(timezone.utc).isoformat(), "job_id": os.environ["SLURM_JOB_ID"],
                  "plan_sha256": checksum, "study_count": len(results), "input_failure_count": len(errors),
                  "dense_sampler_fallback_studies": dense_fallbacks, "study_results_sha256": digest(root / "study-results.jsonl"),
                  "elapsed_seconds": time.monotonic() - started, "gpu_count": 0, "external_spend_usd": 0,
                  "labels_read": False, "model_inference": False, "hidden_error_attribution": False,
                  "runtime_scope": "Exact patched D4 residual CoAt _prepare_bag_task; no checkpoint load, CUDA, other readers or hidden input."}
        write(root / "probe-result.json", result)
        require(not errors, "Accessible original MRI causes CoAt preparation failure; inspect retained trace before attribution")
        print(json.dumps(result), flush=True)
    finally:
        watchdog.cancel()


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2])
