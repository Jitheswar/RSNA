"""Check exact CoAt preparation execution records without rerunning images."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate(root):
    plan, accepted, result = [read(root / name) for name in ("plan.json", "accepted-launch.json", "probe-result.json")]
    checksum = digest(root / "plan.json")
    job = str(accepted["job_id"])
    terminal_path = root / f"controller-{job}/terminal-controller.txt"
    terminal = dict(token.split("=", 1) for token in terminal_path.read_text().split() if "=" in token)
    if (accepted["plan_sha256"] != checksum or accepted["returncode"] != 0
            or str(int(accepted["stdout"].strip().split(";")[0])) != job
            or result["plan_sha256"] != checksum or result["job_id"] != job
            or result["study_count"] != plan["study_count"] or result["status"] != "COMPLETE"
            or result["input_failure_count"] or result["model_inference"] or result["labels_read"]
            or result["hidden_error_attribution"] or result["gpu_count"] or result["external_spend_usd"]):
        raise ValueError("Successful exact CPU preparation request required")
    resources = plan["resources"]
    limit = resources["walltime_seconds"]
    expected_time = f"{limit // 3600:02}:{limit % 3600 // 60:02}:{limit % 60:02}"
    expected = {"JobId": job, "JobState": "COMPLETED", "ExitCode": "0:0", "Partition": "cluster1-cpu",
                "NumNodes": "1", "NumCPUs": str(resources["cpus_per_task"]), "MinMemoryNode": f"{resources['memory_gib']}G",
                "TimeLimit": expected_time, "Requeue": "0", "Restarts": "0", "OverSubscribe": "OK"}
    if not all(terminal.get(k) == v for k, v in expected.items()) or "gpu" in terminal.get("TRES", "").lower():
        raise ValueError("Exact successful bounded allocation required")
    decision = read(plan["decision_path"])
    if decision["decision"] != "approved" or decision["plan_sha256"] != checksum or decision["resources"] != resources:
        raise ValueError("Exact resource decision differs")
    for row in plan["frozen_files"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Frozen original metadata/runtime changed")
    if digest(Path(plan["runtime_source"]) / "input_resgated_runtime.py") != plan["patched_runtime_sha256"]:
        raise ValueError("Generated actual notebook CoAt runtime differs")
    journal_path = root / "study-results.jsonl"
    if digest(journal_path) != result["study_results_sha256"]:
        raise ValueError("Actual preparation journal changed")
    rows = [json.loads(line) for line in journal_path.read_text().splitlines()]
    if ([r["study_uid"] for r in rows] != [r["study_uid"] for r in plan["studies"]]
            or [r["index"] for r in rows] != list(range(plan["study_count"]))
            or len({r["study_uid"] for r in rows}) != plan["study_count"]):
        raise ValueError("Actual original-study membership/order differs")
    for row in rows:
        n = row["windows"]
        if (row["status"] != "COMPLETE" or not 1 <= n <= 94 or row["sparse_fallback"]
                or row["image_shape"] != [n, 3, 384, 384] or row["payload_bytes"] != n * (3 * 384 * 384 + 8 + 2 * 4)
                or len(row["payload_sha256"]) != 3
                or any(not re.fullmatch(r"[0-9a-f]{64}", s) for s in row["payload_sha256"])
                or row["dense_sampler_fallback"] != ("[dense-fallback]" in row["stdout"])):
            raise ValueError("Malformed actual native bag execution record")
    dense = [r["study_uid"] for r in rows if r["dense_sampler_fallback"]]
    if dense != result["dense_sampler_fallback_studies"]:
        raise ValueError("Actual dense-fallback accounting differs")
    clock = terminal["RunTime"]
    days, _, clock = clock.rpartition("-")
    h, m, s = map(int, clock.split(":"))
    return {"schema_version": "rsna-ticket59-coat-input-independent-summary-v1", "status": "COMPLETE",
            "observed_at_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": checksum, "job_id": job,
            "result_sha256": digest(root / "probe-result.json"), "terminal_sha256": digest(terminal_path),
            "study_count": len(rows), "window_range": [min(r["windows"] for r in rows), max(r["windows"] for r in rows)],
            "original_slice_range": [min(r["original_slice_count"] for r in plan["studies"]), max(r["original_slice_count"] for r in plan["studies"])],
            "allocated_cpu_node_seconds": int(days or 0) * 86400 + h * 3600 + m * 60 + s,
            "input_failure_count": 0, "dense_sampler_fallback_count": len(dense),
            "preparation_warning_studies": sum(bool(r["preparation_errors"]) for r in rows),
            "peak_worker_rss_bytes": max(r["peak_worker_rss_bytes"] for r in rows),
            "producer_payload_hash_records_checked": True, "images_independently_replayed": False,
            "hidden_failure_reproduced": False, "gpu_count": 0, "external_spend_usd": 0,
            "limitation": "CPU residual CoAt preparation only; other readers, checkpoint/GPU/model paths and hidden inputs remain untested."}


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    destination = root / "independent-summary-validation.json"
    if destination.exists():
        raise FileExistsError("Immutable validation already exists")
    result = validate(root)
    with destination.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result))
