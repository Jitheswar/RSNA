"""Independently verify every prefix entry and its frozen encoder reconstruction."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def run(request_path, expected_sha):
    started = time.monotonic()
    if digest(request_path) != expected_sha:
        raise ValueError("Frozen prefix verification request changed")
    request = read(request_path)
    output = Path(request["output_root"])
    allocation = subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True)
    (output / "running-allocation.txt").write_text(allocation)
    fields = dict(token.split("=", 1) for token in allocation.split() if "=" in token)
    expected = {"JobState": "RUNNING", "Partition": "cluster1-cpu", "NumNodes": "1", "NumCPUs": "16",
                "NumTasks": "1", "MinMemoryNode": "64G", "TimeLimit": "00:30:00",
                "OverSubscribe": "OK", "Requeue": "0", "Restarts": "0"}
    if any(fields.get(key) != value for key, value in expected.items()) or "gpu" in fields.get("TRES", "").lower():
        raise ValueError("Exact bounded CPU verification allocation is required")
    decision = read(request["decision_path"])
    if decision["decision"] != "approved" or decision["request_sha256"] != expected_sha or decision["resources"] != request["resources"]:
        raise ValueError("Exact prefix verification resource decision differs")
    watchdog = threading.Timer(request["resources"]["code_watchdog_seconds"], lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    for row in request["frozen_files"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Frozen verification source changed")
    for name, version in request["runtime_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Verification runtime changed")
    root = Path(request["extraction_root"])
    plan = read(root / "plan.json")
    plan_sha = digest(root / "plan.json")
    if plan_sha != request["extraction_plan_sha256"]:
        raise ValueError("Exact extraction plan changed")
    for row in plan["frozen_files"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Frozen extraction dependency changed")
    terminal_evidence, results, rows = [], [], []
    for partition in range(2):
        accepted = read(root / f"accepted-launch-{partition}.json")
        job = accepted["job_id"]
        terminal_path = root / f"controller-{job}/terminal-controller.txt"
        observation = read(terminal_path.with_name("terminal-observation.json"))
        controller = terminal_path.read_text()
        fields = dict(token.split("=", 1) for token in controller.split() if "=" in token)
        expected = {"JobId": str(job), "JobState": "COMPLETED", "ExitCode": "0:0", "Partition": "cluster1-cpu",
                    "NumNodes": "1", "NumCPUs": "16", "MinMemoryNode": "64G", "TimeLimit": "03:00:00",
                    "Requeue": "0", "Restarts": "0", "OverSubscribe": "OK"}
        if (accepted["plan_sha256"] != plan_sha or accepted["partition"] != partition
                or observation["controller"] != controller or observation["plan_sha256"] != plan_sha
                or observation["job_id"] != job or any(fields.get(key) != value for key, value in expected.items())
                or "gpu" in fields.get("TRES", "").lower()):
            raise ValueError("Successful matching extraction terminal evidence is required")
        result_path = root / f"partition-{partition}-job-{job}/partition-result.json"
        result = read(result_path)
        expected_uids = set(plan["partitions"][partition]["study_uids"])
        if (result["status"] != "COMPLETE" or result["plan_sha256"] != plan_sha or result["job_id"] != str(job)
                or result["partition"] != partition or len(result["features"]) != len(expected_uids)
                or {r["study_instance_uid"] for r in result["features"]} != expected_uids
                or any(r["job_id"] != str(job) or r["partition"] != partition for r in result["features"])
                or result["valid_slices"] != plan["partitions"][partition]["valid_slices"]
                or not result["encoder_parameters_and_buffers_unchanged"] or result["training"] or result["labels_read"]):
            raise ValueError("Completed extraction partition contract differs")
        seconds = (datetime.fromisoformat(fields["EndTime"]) - datetime.fromisoformat(fields["StartTime"])).total_seconds()
        terminal_evidence.append({"partition": partition, "job_id": job, "terminal_controller_sha256": digest(terminal_path),
                                  "allocated_cpu_node_seconds": seconds})
        results.append({"partition": partition, "job_id": job, "result_sha256": digest(result_path)})
        rows.extend(result["features"])
    rows.sort(key=lambda row: row["study_instance_uid"])
    expected_uids = set(plan["valid_slice_counts"])
    if len(rows) != 4228 or {r["study_instance_uid"] for r in rows} != expected_uids:
        raise ValueError("Complete expected activation cohort is required")
    feature_root = Path(plan["feature_root"])
    if {path.name for path in feature_root.glob("*.npz")} != {row["feature_file"] for row in rows}:
        raise ValueError("Activation directory has missing or extraneous entries")

    import numpy as np
    import torch
    from rsna_knee.model import ModelConfig, build_study_model
    from rsna_knee.prefix_cache import array_digest, identity_digest, load_prefix_entry

    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available() or identity_digest(plan["feature_identity"]) != plan["feature_identity_sha256"]:
        raise ValueError("CPU or activation representation identity differs")
    model = build_study_model(ModelConfig.from_dict(plan["model"]))
    model.load_pretrained_state(torch.load(plan["checkpoint_path"], map_location="cpu", weights_only=True))
    model.requires_grad_(False).eval()
    reference = {r["study_instance_uid"]: r for r in read(plan["reference_feature_manifest_path"])["features"]}
    total_slices, total_bytes, max_error = 0, 0, 0.0
    for index, row in enumerate(rows):
        uid = row["study_instance_uid"]
        prior = reference[uid]
        if row["source_identity_sha256"] != prior["source_identity_sha256"] or row["source_kind"] != prior["source_kind"]:
            raise ValueError("Admitted prior source identity differs")
        path = feature_root / row["feature_file"]
        entry = load_prefix_entry(path, expected_sha256=row["feature_file_sha256"], expected_study_uid=uid,
            expected_identity_sha256=plan["feature_identity_sha256"], expected_source_identity_sha256=row["source_identity_sha256"],
            expected_plan_sha256=plan_sha)
        if (entry.metadata["job_id"] != row["job_id"] or entry.metadata["partition"] != row["partition"]
                or len(entry.prefix) != row["valid_slices"] or len(entry.prefix) != plan["valid_slice_counts"][uid]
                or array_digest(entry.prefix) != row["prefix_array_sha256"] or path.stat().st_size != row["file_bytes"]):
            raise ValueError("Actual entry arrays, slice counts or accepted producer differ")
        prior_path = Path(plan["reference_feature_root"]) / prior["feature_file"]
        if digest(prior_path) != prior["feature_file_sha256"]:
            raise ValueError("Admitted prior feature bytes changed")
        with np.load(prior_path, allow_pickle=False) as previous:
            for key in ("slice_mask", "series_mask", "series_descriptors"):
                if not np.array_equal(getattr(entry, key), previous[key]):
                    raise ValueError("Prior packed mask or descriptor parity failed")
            expected_values = previous["features"][entry.slice_mask & entry.series_mask[:, None]].copy()
        prefix = torch.from_numpy(entry.prefix)
        with torch.inference_mode():
            values = torch.cat([model.encoder.forward_from_layer4(prefix[offset:offset + 16])
                                for offset in range(0, len(prefix), 16)]).numpy()
        error = float(np.max(np.abs(values - expected_values)))
        if not np.isfinite(error) or error > plan["native_feature_parity_tolerance"]:
            raise ValueError("Independent complete-cohort frozen encoder reconstruction differs")
        max_error = max(max_error, error)
        total_slices += len(prefix)
        total_bytes += path.stat().st_size
        if (index + 1) % 128 == 0:
            print(json.dumps({"verified_studies": index + 1, "verified_slices": total_slices,
                              "maximum_native_feature_error": max_error}), flush=True)
    if total_slices != 187272:
        raise ValueError("Full valid-slice count differs")
    manifest_path = feature_root / "feature-manifest.json"
    manifest = {"schema_version": "rsna-ticket57-prefix-feature-manifest-v1", "status": "COMPLETE",
                "extraction_plan_sha256": plan_sha, "verification_request_sha256": expected_sha,
                "feature_identity": plan["feature_identity"], "feature_identity_sha256": plan["feature_identity_sha256"],
                "study_count": len(rows), "valid_slices": total_slices, "file_bytes": total_bytes,
                "all_source_masks_descriptors_and_slice_order_verified": True,
                "all_native_encoder_features_verified": True, "maximum_native_feature_error": max_error,
                "held_out_images_encoded_without_fitting": True, "encoder_adapted": False, "features": rows}
    write(manifest_path, manifest)
    write(output / "verification-result.json", {"schema_version": "rsna-ticket57-prefix-verification-result-v1",
        "status": "COMPLETE", "observed_at_utc": datetime.now(timezone.utc).isoformat(), "job_id": os.environ["SLURM_JOB_ID"],
        "request_sha256": expected_sha, "extraction_plan_sha256": plan_sha,
        "feature_manifest_sha256": digest(manifest_path), "feature_manifest_path": str(manifest_path),
        "feature_files_verified": len(rows), "valid_slices_verified": total_slices, "file_bytes": total_bytes,
        "all_native_encoder_features_verified": True, "maximum_native_feature_error": max_error,
        "extraction_results": results, "terminal_evidence": terminal_evidence,
        "allocated_extraction_cpu_node_seconds": sum(row["allocated_cpu_node_seconds"] for row in terminal_evidence),
        "elapsed_seconds": time.monotonic() - started,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "training": False, "candidate_promotion": False, "gpu_count": 0, "external_spend_usd": 0})
    watchdog.cancel()
    print(json.dumps({"status": "COMPLETE", "verified_studies": len(rows), "verified_slices": total_slices}), flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2])
