"""Verify full-source probe coverage against original archive metadata.

Payload verification runs in a bounded CPU allocation. It checks every
retained DICOM against its producer SHA and independently computes its CRC.
No reports, labels, model inference or changes to the tested recipe occur.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile
import zlib


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(text):
    return dict(token.split("=", 1) for token in text.split() if "=" in token)


def seconds(value):
    days, _, clock = value.rpartition("-")
    h, m, s = map(int, clock.split(":"))
    return int(days or 0) * 86400 + h * 3600 + m * 60 + s


def fragment(source, name):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(nodes) == 1, "Frozen source function is ambiguous")
    return ast.get_source_segment(source, nodes[0])


def verify_payload(item):
    path, row = item
    require(path.is_file() and not path.is_symlink(), "Missing or symlinked source DICOM")
    stat = path.stat()
    require(stat.st_size == row["bytes"], "Retained DICOM size changed")
    sha, crc = hashlib.sha256(), 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 ** 2):
            sha.update(chunk)
            crc = zlib.crc32(chunk, crc)
    after = path.stat()
    require((after.st_size, after.st_mtime_ns, after.st_ino) == (stat.st_size, stat.st_mtime_ns, stat.st_ino),
            "DICOM changed during independent verification")
    require(sha.hexdigest() == row["actual_sha256"] and crc == row["archive_crc32"],
            "Retained DICOM SHA/independent archive CRC differs")
    return row["bytes"]


def validate(root, payloads=False):
    started = time.monotonic()
    campaign = read(root / "plan.json")
    accepted = read(root / "accepted-partitions.json")
    require(accepted["campaign_plan_sha256"] == digest(root / "plan.json"), "Campaign identity changed")
    require(len(campaign["partitions"]) == len(accepted["partitions"]) == 2, "Expected two exact partitions")
    require(digest(root / "original-train-series.csv") == campaign["original_series_metadata_sha256"],
            "Original cohort metadata changed")
    metadata_bytes = (root / "original-train-series.csv").read_bytes()
    metadata = list(csv.DictReader(io.StringIO(metadata_bytes.decode("utf-8-sig"))))
    uids = sorted({r["StudyInstanceUID"] for r in metadata})
    require(len(uids) == campaign["study_count"] == 4407, "Full corpus membership differs")
    require(digest(root / "source-snapshot/run_full_partition.py") == campaign["worker_sha256"], "Worker changed")
    results, visited = [], set()
    for index in range(2):
        entry = campaign["partitions"][index]
        launch = accepted["partitions"][index]
        directory = root / f"partition-{index}"
        plan, result = read(directory / "plan.json"), read(directory / "probe-result.json")
        checksum = digest(directory / "plan.json")
        exact = read(directory / "accepted-launch.json")
        require(entry["partition"] == launch["partition"] == index
                and Path(entry["root"]) == directory and entry["plan_sha256"] == launch["plan_sha256"] == checksum
                and exact["job_id"] == launch["job_id"] and exact["plan_sha256"] == checksum
                and exact["returncode"] == 0 and int(exact["stdout"].strip()) == launch["job_id"],
                "Accepted partition identity differs")
        decision = read(plan["decision_path"])
        require(decision["decision"] == "approved" and decision["plan_sha256"] == checksum
                and decision["resources"] == plan["resources"], "Partition decision changed")
        for row in plan["frozen_files"]:
            require(digest(row["path"]) == row["sha256"], "Frozen source/input changed")
        members = plan["selected_study_uids"]
        member_set = set(members)
        require(members == uids[index::2] and not visited.intersection(members), "Partition membership differs")
        visited.update(members)
        job = str(launch["job_id"])
        terminal_path = directory / f"controller-{job}/terminal-controller.txt"
        terminal = fields(terminal_path.read_text())
        required = {"JobId": job, "JobState": "COMPLETED", "ExitCode": "0:0", "Partition": "cluster1-cpu",
                    "NumNodes": "1", "NumCPUs": "16", "NumTasks": "1", "MinMemoryNode": "64G",
                    "TimeLimit": "03:00:00", "Requeue": "0", "Restarts": "0", "OverSubscribe": "OK"}
        require(all(terminal.get(k) == v for k, v in required.items()) and "gpu" not in terminal.get("TRES", "").lower(),
                "Exact successful bounded terminal allocation is required")
        observation = read(directory / f"controller-{job}/terminal-observation.json")
        require(str(observation["job_id"]) == job and observation["state"] == "COMPLETED"
                and observation["returncode"] == 0, "Terminal observer identity differs")
        require(result["status"] == "COMPLETE" and result["job_id"] == job and result["plan_sha256"] == checksum
                and result["study_count"] == len(members) and result["cpu_input_guard_passed"]
                and not result["decode_failed_series"] and not result["failed_fallback_events"]
                and result["gpu_count"] == result["external_spend_usd"] == 0
                and not any(result[k] for k in ("labels_read", "model_inference", "fitting", "candidate_promotion",
                                                "competition_submission", "hidden_error_reproduced", "hidden_error_attribution")),
                "Exact unchanged successful CPU input result is required")
        summaries = result["study_summaries"]
        require([r["study_instance_uid"] for r in summaries] == members
                and all(0 < r["present_slots"] <= 6 and r["nonzero_pixels"] > 0
                        and 0 <= r["pixel_min"] <= r["pixel_max"] <= 255 for r in summaries), "Study input summary differs")
        snapshot = Path(plan["source_snapshot"]) / "upstream-cells"
        cells = {i: (snapshot / f"cell-{i}.py").read_text() for i in (4, 5, 7, 8, 9, 10, 11, 12, 16)}
        fragments = [(5, cells[5][cells[5].index("T0 = time.time()\n"):]), (4, cells[4]),
                     (7, fragment(cells[7], "log")), *[(i, cells[i]) for i in (8, 9, 10, 11, 12)],
                     (16, fragment(cells[16], "adopt_config_globals"))]
        require(result["unchanged_source_fragments"] == [{"cell": i, "fragment_sha256": hashlib.sha256(s.encode()).hexdigest()}
                                                         for i, s in fragments], "Executed preprocessing fragments differ")
        source_path = directory / "source-mri-entries.json"
        require(digest(source_path) == result["source_entries_sha256"], "Producer source records changed")
        rows = read(source_path)
        by_entry = {row["entry"]: row for row in rows}
        require(len(by_entry) == len(rows) == result["dicom_files"], "Duplicate source entries")
        stat = Path(plan["archive_path"]).stat()
        require(stat.st_size == plan["archive_bytes"] and stat.st_mtime_ns == plan["archive_mtime_ns"], "Source archive changed")
        with zipfile.ZipFile(plan["archive_path"]) as archive:
            require(archive.read("train_series.csv") == metadata_bytes, "Original archive metadata differs")
            infos = [info for info in archive.infolist() if info.filename.startswith("train_series/")
                     and info.filename.endswith(".dcm") and len(Path(info.filename).parts) == 4
                     and Path(info.filename).parts[1] in member_set]
        require(len(infos) == len(by_entry) and {info.filename for info in infos} == set(by_entry), "Incomplete original DICOM coverage")
        for info in infos:
            row = by_entry[info.filename]
            require(row["bytes"] == info.file_size and row["archive_crc32"] == info.CRC
                    and re.fullmatch(r"[0-9a-f]{64}", row["actual_sha256"]), "Source central-directory identity differs")
        raw_bytes = sum(row["bytes"] for row in rows)
        require(raw_bytes == result["actual_raw_bytes"] <= plan["resources"]["maximum_raw_mri_bytes"], "Raw byte total differs")
        headers = list(csv.DictReader((directory / "headers.csv").open()))
        series_ids = {Path(info.filename).parts[2] for info in infos}
        expected_metadata = [row for row in metadata if row["StudyInstanceUID"] in member_set]
        require(len(headers) == len(series_ids) == result["header_series_count"] == result["metadata_series_count"]
                and {row["SeriesInstanceUID"] for row in headers} == series_ids
                and {row["SeriesInstanceUID"] for row in expected_metadata} == series_ids
                and {row["StudyInstanceUID"] for row in headers} == set(members)
                and all(not row.get("err") for row in headers), "Original header series coverage differs")
        require(result["native_cache_bytes"] == len(members) * 6 * 12 * 336 * 336 == plan["expected_cache_bytes"]
                and result["native_cache_disk_branch"], "Native disk-cache size/branch differs")
        events = result["events"]
        require([event["kind"] for event in events] == ["scratch_allocation", "cache_complete"]
                and all(event["bytes"] == result["native_cache_bytes"] for event in events)
                and events[1]["studies"] == len(members) and events[1]["filled"] == events[1]["slot_jobs"]
                == sum(row["present_slots"] for row in summaries), "Actual cache completion events differ")
        verified_bytes = 0
        if payloads:
            items = ((directory / "input-view/test_series" / Path(*Path(row["entry"]).parts[1:]), row) for row in rows)
            with ThreadPoolExecutor(max_workers=8) as executor:
                for count, size in enumerate(executor.map(verify_payload, items), 1):
                    verified_bytes += size
                    if count % 16384 == 0:
                        print(json.dumps({"partition": index, "payloads_verified": count}), flush=True)
            require(verified_bytes == raw_bytes, "Independent payload byte total differs")
        results.append({"partition": index, "job_id": job, "study_count": len(members), "dicom_files": len(rows),
                        "raw_bytes": raw_bytes, "native_cache_bytes": result["native_cache_bytes"],
                        "allocated_cpu_node_seconds": seconds(terminal["RunTime"]),
                        "frozen_files_verified": len(plan["frozen_files"]), "present_slot_range": [min(r["present_slots"] for r in summaries), max(r["present_slots"] for r in summaries)],
                        "peak_process_rss_bytes": result["peak_process_rss_bytes"],
                        "result_sha256": digest(directory / "probe-result.json"), "terminal_sha256": digest(terminal_path),
                        "independently_verified_payload_bytes": verified_bytes})
    require(visited == set(uids), "Partition union omits source studies")
    return {"schema_version": "rsna-ticket59-full-input-independent-validation-v1", "status": "COMPLETE",
            "observed_at_utc": datetime.now(timezone.utc).isoformat(), "campaign_plan_sha256": digest(root / "plan.json"),
            "partitions": results, "study_count": len(visited), "disjoint_complete_membership": True,
            "dicom_files": sum(r["dicom_files"] for r in results), "raw_bytes": sum(r["raw_bytes"] for r in results),
            "allocated_cpu_node_seconds": sum(r["allocated_cpu_node_seconds"] for r in results),
            "source_payloads_independently_rehashed": payloads, "archive_crc_independently_computed": payloads,
            "verification_elapsed_seconds": time.monotonic() - started, "hidden_error_reproduced": False,
            "disposition": "hidden_failure_unreproduced", "gpu_count": 0, "external_spend_usd": 0,
            "limitation": "CPU native DINO preprocessing only; no other image configurations, CoAt/model/GPU execution or hidden input tested."}


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    payloads = len(sys.argv) > 2 and sys.argv[2] == "--payloads"
    if payloads:
        job = os.environ["SLURM_JOB_ID"]
        allocation = fields(subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True))
        require(all(allocation.get(k) == v for k, v in {"JobId": job, "JobState": "RUNNING", "Partition": "cluster1-cpu",
                "NumNodes": "1", "NumCPUs": "16", "MinMemoryNode": "32G", "TimeLimit": "01:00:00", "Requeue": "0"}.items())
                and "gpu" not in allocation.get("TRES", "").lower(), "Independent hashing needs bounded CPU allocation")
    destination = root / ("independent-payload-validation.json" if payloads else "independent-summary-validation.json")
    require(not destination.exists(), "Immutable validation already exists")
    result = validate(root, payloads)
    with destination.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "study_count", "dicom_files", "raw_bytes", "allocated_cpu_node_seconds", "source_payloads_independently_rehashed")}))
