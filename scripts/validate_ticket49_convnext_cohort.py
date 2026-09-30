"""Independently seal complete feature outputs and their unfitted pooled bank."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path: Path, value: object) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(request_path: Path, expected_sha: str) -> None:
    started = time.monotonic()
    if digest(request_path) != expected_sha:
        raise ValueError("exact verification request changed")
    request = json.loads(request_path.read_text())
    output = Path(request["output_root"])
    allocation = subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True)
    (output / "running-allocation.txt").write_text(allocation)
    fields = dict(token.split("=", 1) for token in allocation.split() if "=" in token)
    if any(fields.get(key) != value for key, value in {
        "JobState": "RUNNING", "Partition": "cluster1-cpu", "NumNodes": "1",
        "NumCPUs": "16", "NumTasks": "1", "MinMemoryNode": "64G",
        "TimeLimit": "00:15:00", "OverSubscribe": "OK",
    }.items()) or "gpu" in fields.get("TRES", "").lower():
        raise ValueError("verification allocation differs")
    approval = json.loads(Path(request["approval_path"]).read_text())
    if approval.get("decision") != "approved" or approval.get("request_sha256") != expected_sha:
        raise ValueError("exact verification authorization missing")
    for row in request["frozen_files"]:
        if digest(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"frozen verification dependency changed: {row['path']}")
    import numpy as np
    import torch
    from rsna_knee.frozen_head_experiment import pool_study

    if importlib.metadata.version("numpy") != "2.2.6" or torch.__version__ != "2.7.1+cpu":
        raise ValueError("verification runtime differs")
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    plan_path = Path(request["extraction_plan_path"])
    plan_sha = digest(plan_path)
    plan = json.loads(plan_path.read_text())
    for row in plan["frozen_files"]:
        if digest(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"original extraction dependency changed: {row['path']}")
    all_rows = []
    for partition in range(2):
        launch = json.loads(Path(request["launch_paths"][partition]).read_text())
        job_id = launch["job_id"]
        terminal_path = Path(plan["run_root"]) / f"controller-{job_id}/terminal-controller.txt"
        for _ in range(4):
            if terminal_path.exists():
                break
            time.sleep(5)
        terminal = terminal_path.read_text() if terminal_path.exists() else subprocess.check_output(
            ["scontrol", "show", "job", job_id, "-o"], text=True)
        fields = dict(token.split("=", 1) for token in terminal.split() if "=" in token)
        if (fields.get("JobId") != job_id or fields.get("JobState") != "COMPLETED"
                or fields.get("ExitCode") != "0:0" or "gpu" in fields.get("TRES", "").lower()):
            raise ValueError("extraction job did not complete successfully on CPUs")
        write(output / f"terminal-{job_id}.json", {"job_id": job_id, "controller": terminal,
                                                   "launch_file_sha256": digest(Path(request["launch_paths"][partition]))})
        result_path = Path(plan["run_root"]) / f"partition-{partition}-job-{job_id}/partition-result.json"
        result = json.loads(result_path.read_text())
        expected = set(plan["partitions"][partition]["study_uids"])
        rows = result["features"]
        if (result["status"] != "complete" or result["job_id"] != job_id
                or result["plan_sha256"] != plan_sha or result["partition"] != partition
                or len(rows) != len(expected) or {row["study_instance_uid"] for row in rows} != expected):
            raise ValueError("partition output identity or completeness differs")
        all_rows.extend(rows)
    rows = sorted(all_rows, key=lambda row: row["study_instance_uid"])
    if len(rows) != 4228 or len({row["study_instance_uid"] for row in rows}) != 4228:
        raise ValueError("complete feature membership differs")
    files = sorted(Path(plan["feature_root"]).glob("*.npz"))
    if len(files) != len(rows) or {path.name for path in files} != {row["feature_file"] for row in rows}:
        raise ValueError("feature directory contains missing or extraneous outputs")
    prior = {row["study_instance_uid"]: row for row in
             json.loads(Path(plan["reference_feature_manifest_path"]).read_text())["features"]}
    pooled = []
    total_valid = 0
    for index, row in enumerate(rows):
        uid = row["study_instance_uid"]
        path = Path(plan["feature_root"]) / row["feature_file"]
        if (path.is_symlink() or digest(path) != row["feature_file_sha256"]
                or row["source_identity_sha256"] != prior[uid]["source_identity_sha256"]):
            raise ValueError("feature file/source checksum differs")
        reference_path = Path(plan["reference_feature_root"]) / prior[uid]["feature_file"]
        if digest(reference_path) != prior[uid]["feature_file_sha256"]:
            raise ValueError("prior representation checksum differs")
        with np.load(path, allow_pickle=False) as feature, np.load(reference_path, allow_pickle=False) as reference:
            metadata = json.loads(str(feature["metadata"]))
            if (metadata["plan_sha256"] != plan_sha or metadata["study_instance_uid"] != uid
                    or metadata["feature_identity"] != plan["feature_identity"]
                    or metadata["source_identity_sha256"] != row["source_identity_sha256"]
                    or metadata["job_id"] != row["job_id"]):
                raise ValueError("feature metadata differs")
            values = feature["features"]
            masks = feature["slice_mask"].astype(bool)
            series = feature["series_mask"].astype(bool)
            descriptors = feature["series_descriptors"]
            valid = masks & series[:, None]
            if (values.shape != (16, 8, 768) or values.dtype != np.float32
                    or not np.isfinite(values).all() or np.any(values[~valid] != 0)
                    or int(valid.sum()) != plan["valid_slice_counts"][uid]
                    or int(valid.sum()) != row["valid_slices"]):
                raise ValueError("feature dimensions, numerics or masks differ")
            for key in ["slice_mask", "series_mask", "series_descriptors"]:
                if not np.array_equal(feature[key], reference[key]):
                    raise ValueError("prior mask/descriptor parity failed")
            total_valid += int(valid.sum())
            pooled.append(pool_study(values, masks, series, descriptors).numpy())
        if (index + 1) % 256 == 0:
            print(json.dumps({"validated_studies": index + 1}), flush=True)
    if total_valid != 187272:
        raise ValueError("complete selected-slice count differs")
    manifest_path = Path(plan["feature_root"]) / "feature-manifest.json"
    write(manifest_path, {
        "schema_version": "rsna-ticket49-frozen-convnext-feature-manifest-v1",
        "extraction_plan_sha256": plan_sha, "feature_identity": plan["feature_identity"],
        "feature_study_count": len(rows), "valid_slice_count": total_valid,
        "held_out_images_used_for_prediction_only": True,
        "encoder_adapted_on_competition_inputs": False,
        "complete_mask_descriptor_source_parity": True, "features": rows,
    })
    pooled_path = output / "pooled-cohort.npz"
    with pooled_path.open("xb") as handle:
        np.savez(handle, pooled=np.stack(pooled).astype(np.float32),
                 study_uids=np.asarray([row["study_instance_uid"] for row in rows]),
                 metadata=np.asarray(json.dumps({"feature_manifest_sha256": digest(manifest_path),
                    "extraction_plan_sha256": plan_sha, "pooling_id": "plane-stratified-series-mean-v1",
                    "pooling_source_sha256": request["pooling_source_sha256"]}, sort_keys=True)))
    write(output / "verification-result.json", {
        "schema_version": "rsna-ticket49-cohort-verification-v1", "status": "complete",
        "request_sha256": expected_sha, "job_id": os.environ["SLURM_JOB_ID"],
        "extraction_plan_sha256": plan_sha, "feature_manifest_path": str(manifest_path),
        "feature_manifest_sha256": digest(manifest_path), "pooled_cohort_path": str(pooled_path),
        "pooled_cohort_sha256": digest(pooled_path), "study_count": len(rows),
        "valid_slices": total_valid, "pooled_width": 3100,
        "elapsed_seconds": time.monotonic() - started,
        "source_identity_masks_descriptors_parity": True,
        "gpu_count": 0, "external_spend_usd": 0, "training": False,
        "candidate_promotion": False,
    })


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
