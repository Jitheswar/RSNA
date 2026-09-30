"""Bounded, offline, unsupervised frozen-encoder throughput pilot."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run(plan_path: Path, expected_plan_sha256: str) -> None:
    start = time.monotonic()
    if digest(plan_path) != expected_plan_sha256:
        raise ValueError("immutable pilot plan checksum differs")
    plan = json.loads(plan_path.read_text())
    root = Path(plan["run_root"])
    attempt = root / f"job-{os.environ['SLURM_JOB_ID']}"
    attempt.mkdir(mode=0o700)
    allocation = subprocess.check_output(
        ["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True
    )
    (attempt / "running-allocation.txt").write_text(allocation)
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z_]+)=([^\s]+)", allocation))
    if any(fields.get(key) != value for key, value in {
        "JobState": "RUNNING", "Partition": "cluster1-cpu", "NumNodes": "1",
        "NumCPUs": "16", "NumTasks": "1", "TimeLimit": "00:10:00",
        "MinMemoryNode": "64G",
    }.items()) or "gres/gpu" in fields.get("AllocTRES", ""):
        raise ValueError("live allocation differs from admitted CPU pilot")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
        raise ValueError("pilot must not expose GPUs")
    approval_path = Path(plan["approval_path"])
    approval = json.loads(approval_path.read_text())
    if (approval.get("plan_sha256") != expected_plan_sha256
            or approval.get("decision") != "approved"
            or approval.get("scope") != "ticket48-convnext-cpu-pilot"):
        raise ValueError("pilot lacks exact campaign authorization record")
    for row in plan["frozen_files"]:
        path = Path(row["path"])
        if path.is_symlink() or digest(path) != row["sha256"]:
            raise ValueError(f"frozen file differs: {path}")
    for package, version in plan["runtime_versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"runtime version differs: {package}")
    if digest(Path(__file__)) != plan["pilot_source_sha256"]:
        raise ValueError("executing pilot source differs")
    if os.environ.get("PYTHONPATH") != str(Path(plan["source_snapshot"]) / "src"):
        raise ValueError("pilot must import its frozen source snapshot")

    def timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError("pilot reached admitted 590-second code limit")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(max(1, int(590 - (time.monotonic() - start))))
    import numpy as np
    import torch
    import timm
    from safetensors.torch import load_file
    from rsna_knee.domain import SeriesRecord, StudyRecord
    from rsna_knee.preprocessing import pack_study
    from rsna_knee.real_inference import load_integration_manifest
    from rsna_knee.series_cache import SeriesCache

    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise ValueError("pilot requires the admitted CPU-only PyTorch build")
    integration = load_integration_manifest(Path(plan["integration_path"]))
    if integration.manifest_checksum != plan["integration_manifest_checksum"]:
        raise ValueError("integration semantic checksum differs")
    if (integration.preprocess_config.image_size != (224, 224)
            or integration.packing_config.max_slices != 8
            or integration.packing_config.max_series != 16):
        raise ValueError("pilot image representation differs")
    index = json.loads(Path(plan["series_index_path"]).read_text())["entries"]
    cache = SeriesCache(Path(plan["series_cache_root"]), read_only=True)
    studies = {item.study_instance_uid: item for item in integration.studies}
    reference = {
        item["study_instance_uid"]: item for item in
        json.loads(Path(plan["reference_feature_manifest_path"]).read_text())["features"]
    }
    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=1000)
    state = load_file(plan["checkpoint_path"], device="cpu")
    model.load_state_dict(state, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 28589128:
        raise ValueError("checkpoint architecture parameter count differs")
    del state
    model.reset_classifier(0, global_pool="avg")
    model.requires_grad_(False).eval().to(memory_format=torch.channels_last)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    results = []
    for uid in plan["pilot_study_uids"]:
        study_start = time.monotonic()
        manifest = studies[uid]
        records, prepared, source_rows = [], [], []
        for item in manifest.series:
            row = index[item.series_instance_uid]
            if row["study_instance_uid"] != uid:
                raise ValueError("series-index membership differs")
            metadata_path = cache.entry_path(row["cache_key"]) / "metadata.json"
            if digest(metadata_path) != row["metadata_sha256"]:
                raise ValueError("prepared cache metadata differs from frozen series index")
            prepared.append(cache.load_validated_entry(
                row["cache_key"], study_instance_uid=uid,
                series_instance_uid=item.series_instance_uid,
                expected_source_entry_names=item.entries,
                config=integration.preprocess_config,
            ))
            records.append(SeriesRecord(
                study_instance_uid=uid, series_instance_uid=item.series_instance_uid,
                slices=(), anatomical_plane=item.anatomical_plane,
                fluid_sensitive=item.fluid_sensitive, fat_suppression=item.fat_suppression,
                diagnostic_status=item.diagnostic_status, source_entry_names=item.entries,
            ))
            source_rows.append(row)
        packed = pack_study(StudyRecord(uid, tuple(records)), prepared,
                            integration.preprocess_config,
                            packing_config=integration.packing_config)
        valid = packed.slice_mask & packed.series_mask[:, None]
        valid_count = int(valid.sum())
        if valid_count != plan["pilot_valid_slice_counts"][uid]:
            raise ValueError("pilot slice count differs from the frozen selection")
        prior_row = reference[uid]
        prior_path = Path(plan["reference_feature_root"]) / prior_row["feature_file"]
        if digest(prior_path) != prior_row["feature_file_sha256"]:
            raise ValueError("prior feature file differs")
        with np.load(prior_path, allow_pickle=False) as prior:
            for key, value in {"slice_mask": packed.slice_mask,
                               "series_mask": packed.series_mask,
                               "series_descriptors": packed.series_descriptors}.items():
                if not np.array_equal(prior[key], value):
                    raise ValueError(f"existing representation differs: {key}")
        images = torch.from_numpy(packed.images[valid])
        features = np.zeros((*valid.shape, 768), dtype=np.float32)
        encoded_rows = []
        encoding_start = time.monotonic()
        with torch.inference_mode():
            for offset in range(0, valid_count, 16):
                batch = (images[offset:offset + 16].expand(-1, 3, -1, -1) - mean) / std
                encoded = model(batch.contiguous(memory_format=torch.channels_last))
                if encoded.shape != (min(16, valid_count - offset), 768):
                    raise ValueError("encoder feature shape differs")
                encoded_rows.append(encoded.cpu().numpy())
        features[valid] = np.concatenate(encoded_rows)
        if not np.isfinite(features).all() or not np.any(features[valid]):
            raise ValueError("features are nonfinite or entirely zero")
        if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
            raise ValueError("encoder is not frozen")
        encoding_seconds = time.monotonic() - encoding_start
        output = attempt / (hashlib.sha256(uid.encode()).hexdigest() + ".npz")
        np.savez(output, features=features, slice_mask=packed.slice_mask,
                 series_mask=packed.series_mask, series_descriptors=packed.series_descriptors)
        row = {"study_instance_uid": uid, "valid_slices": valid_count,
               "feature_path": str(output), "feature_file_sha256": digest(output),
               "encoding_seconds": encoding_seconds,
               "total_seconds": time.monotonic() - study_start,
               "sampled_source_indices": [list(item) for item in packed.sampled_source_indices],
               "source_cache_rows": source_rows,
               "prior_feature_file_sha256": prior_row["feature_file_sha256"]}
        results.append(row)
        with (attempt / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps({"completed_studies": len(results), "valid_slices": valid_count,
                          "elapsed_seconds": time.monotonic() - start}), flush=True)
    elapsed = time.monotonic() - start
    total_slices = sum(row["valid_slices"] for row in results)
    encoding = sum(row["encoding_seconds"] for row in results)
    overhead = sum(row["total_seconds"] - row["encoding_seconds"] for row in results)
    projection = 2 * (encoding / total_slices * plan["cohort_valid_slices"]
                      + overhead / len(results) * plan["cohort_study_count"])
    eligible = projection <= plan["maximum_full_extraction_node_hours"] * 3600
    write_json(attempt / "pilot-result.json", {
        "schema_version": "rsna-ticket48-convnext-cpu-pilot-result-v1",
        "plan_sha256": expected_plan_sha256, "job_id": os.environ["SLURM_JOB_ID"],
        "status": "complete", "study_count": len(results), "valid_slices": total_slices,
        "feature_dimension": 768, "parameter_count_before_classifier_removal": parameter_count,
        "elapsed_seconds": elapsed, "encoding_seconds": encoding,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "projection_with_twofold_allowance_seconds": projection,
        "projection_with_twofold_allowance_node_hours": projection / 3600,
        "throughput_gate": "pass" if eligible else "stop",
        "projection_excludes_gold_archive_decode_and_queue_delay": True,
        "projection_is_estimate_not_allocation_charge": True,
        "cohort_extraction_authorized_by_pilot": False,
        "scientific_model_training": False, "gpu_count": 0,
        "external_spend_usd": 0, "candidate_promotion": False, "features": results,
    })
    signal.alarm(0)


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2])
