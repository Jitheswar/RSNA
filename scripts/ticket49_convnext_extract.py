"""Offline extraction of one admitted partition of the frozen image cohort."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
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


def run(plan_path: Path, expected_sha: str, partition: int) -> None:
    started = time.monotonic()
    if digest(plan_path) != expected_sha:
        raise ValueError("exact extraction plan checksum differs")
    plan = json.loads(plan_path.read_text())
    root = Path(plan["run_root"])
    job_id = os.environ["SLURM_JOB_ID"]
    attempt = root / f"partition-{partition}-job-{job_id}"
    attempt.mkdir(mode=0o700)
    allocation = subprocess.check_output(["scontrol", "show", "job", job_id, "-o"], text=True)
    (attempt / "running-allocation.txt").write_text(allocation)
    fields = dict(token.split("=", 1) for token in allocation.split() if "=" in token)
    if any(fields.get(key) != value for key, value in {
        "JobId": job_id, "JobState": "RUNNING", "Partition": "cluster1-cpu",
        "NumNodes": "1", "NumCPUs": "16", "NumTasks": "1", "CPUs/Task": "16",
        "TimeLimit": "03:00:00", "MinMemoryNode": "64G", "OverSubscribe": "OK",
    }.items()) or any("gpu" in fields.get(key, "").lower()
                     for key in ["TRES", "AllocTRES", "Gres"]):
        raise ValueError("live allocation differs from admitted CPU extraction")
    approval = json.loads(Path(plan["approval_path"]).read_text())
    if (approval.get("decision") != "approved" or approval.get("plan_sha256") != expected_sha
            or approval.get("scope") != "ticket49-convnext-cohort-extraction"):
        raise ValueError("exact extraction authorization missing")
    def timeout(_signal: int, _frame: object) -> None:
        raise TimeoutError("extraction reached the admitted code watchdog")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(10740)
    for row in plan["frozen_files"]:
        path = Path(row["path"])
        if path.is_symlink() or digest(path) != row["sha256"]:
            raise ValueError(f"frozen extraction input differs: {path}")
    for package, version in plan["runtime_versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"runtime differs: {package}")
    if os.environ.get("PYTHONPATH") != str(Path(plan["source_snapshot"]) / "src"):
        raise ValueError("imports must use the admitted source snapshot")
    source_archive = Path(plan["source_archive_path"])
    stat = source_archive.stat()
    if stat.st_size != plan["source_archive_bytes"] or stat.st_mtime_ns != plan["source_archive_mtime_ns"]:
        raise ValueError("existing source archive stat identity changed")

    import numpy as np
    import torch
    import timm
    from safetensors.torch import load_file
    from rsna_knee.domain import SeriesRecord, StudyRecord
    from rsna_knee.preprocessing import pack_study, prepare_study
    from rsna_knee.real_inference import iter_real_studies, load_integration_manifest
    from rsna_knee.series_cache import SeriesCache
    from rsna_knee.ticket36 import _study_source_digest, _json_sha256

    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise ValueError("requires CPU-only admitted runtime")
    integration = load_integration_manifest(plan["integration_path"])
    if integration.manifest_checksum != plan["integration_manifest_checksum"]:
        raise ValueError("integration semantic checksum differs")
    index = json.loads(Path(plan["series_index_path"]).read_text())["entries"]
    cache = SeriesCache(plan["series_cache_root"], read_only=True)
    studies = {item.study_instance_uid: item for item in integration.studies}
    reference = {row["study_instance_uid"]: row for row in
                 json.loads(Path(plan["reference_feature_manifest_path"]).read_text())["features"]}
    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=1000)
    state = load_file(plan["checkpoint_path"], device="cpu")
    model.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 28589128:
        raise ValueError("full checkpoint parameter count differs")
    del state
    model.reset_classifier(0, global_pool="avg")
    model.requires_grad_(False).eval().to(memory_format=torch.channels_last)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    partition_uids = plan["partitions"][partition]["study_uids"]
    gold_uids = [uid for uid in partition_uids if reference[uid]["source_kind"] == "frozen-rsna-archive"]
    gold_iterator = iter(iter_real_studies(integration, source_archive,
                        require_complete_series=True, study_uids=gold_uids)) if gold_uids else iter(())
    # Match the archive reader's study order, while processing reports first.
    gold_order = [item.study_instance_uid for item in integration.studies if item.study_instance_uid in set(gold_uids)]
    report_uids = [uid for uid in partition_uids if uid not in set(gold_uids)]
    ordered = report_uids + gold_order
    if len(set(ordered)) != len(partition_uids) or set(ordered) != set(partition_uids):
        raise ValueError("partition membership differs")
    result_rows = []
    for uid in ordered:
        step_start = time.monotonic()
        prior_row = reference[uid]
        manifest = studies[uid]
        if uid in gold_uids:
            loaded = next(gold_iterator)
            study = loaded.study
            if study.study_instance_uid != uid:
                raise ValueError("gold archive reader order differs")
            prepared_study = prepare_study(study, integration.preprocess_config)
            if prepared_study.failures or not prepared_study.series:
                raise ValueError("gold study image preparation failed")
            prepared = prepared_study.series
            source_identity = _json_sha256({
                "archive_fingerprint": integration.archive_fingerprint,
                "study_instance_uid": uid,
                "series": [{"series_instance_uid": item.series_instance_uid,
                            "entry_names": list(item.source_entry_names),
                            "source_slice_uids": [part.sop_instance_uid for part in item.slices]}
                           for item in sorted(study.series, key=lambda item: item.series_instance_uid)],
            })
        else:
            records, prepared = [], []
            for item in manifest.series:
                row = index[item.series_instance_uid]
                if row["study_instance_uid"] != uid:
                    raise ValueError("series cache belongs to another study")
                if digest(cache.entry_path(row["cache_key"]) / "metadata.json") != row["metadata_sha256"]:
                    raise ValueError("prepared series cache differs from immutable index")
                prepared.append(cache.load_validated_entry(
                    row["cache_key"], study_instance_uid=uid,
                    series_instance_uid=item.series_instance_uid,
                    expected_source_entry_names=item.entries, config=integration.preprocess_config))
                records.append(SeriesRecord(
                    study_instance_uid=uid, series_instance_uid=item.series_instance_uid, slices=(),
                    anatomical_plane=item.anatomical_plane, fluid_sensitive=item.fluid_sensitive,
                    fat_suppression=item.fat_suppression, diagnostic_status=item.diagnostic_status,
                    source_entry_names=item.entries))
            study = StudyRecord(uid, tuple(records))
            source_identity = _study_source_digest(manifest, index,
                                                  archive_fingerprint=integration.archive_fingerprint)
        if source_identity != prior_row["source_identity_sha256"]:
            raise ValueError("image source identity differs from admitted prior features")
        packed = pack_study(study, prepared, integration.preprocess_config,
                            packing_config=integration.packing_config)
        valid = packed.slice_mask & packed.series_mask[:, None]
        valid_count = int(valid.sum())
        if valid_count != plan["valid_slice_counts"][uid]:
            raise ValueError("valid slice count differs")
        prior_path = Path(plan["reference_feature_root"]) / prior_row["feature_file"]
        if digest(prior_path) != prior_row["feature_file_sha256"]:
            raise ValueError("reference representation evidence differs")
        with np.load(prior_path, allow_pickle=False) as prior:
            for key, value in {"slice_mask": packed.slice_mask, "series_mask": packed.series_mask,
                               "series_descriptors": packed.series_descriptors}.items():
                if not np.array_equal(prior[key], value):
                    raise ValueError(f"image representation parity failed: {key}")
        features = np.zeros((*valid.shape, 768), dtype=np.float32)
        images = torch.from_numpy(packed.images[valid])
        encoded_rows = []
        with torch.inference_mode():
            for offset in range(0, valid_count, 16):
                batch = (images[offset:offset + 16].expand(-1, 3, -1, -1) - mean) / std
                encoded = model(batch.contiguous(memory_format=torch.channels_last))
                if encoded.shape != (min(16, valid_count - offset), 768):
                    raise ValueError("encoded feature shape differs")
                encoded_rows.append(encoded.cpu().numpy())
        features[valid] = np.concatenate(encoded_rows)
        if not np.isfinite(features).all() or not np.any(features[valid]):
            raise ValueError("invalid encoder features")
        metadata = {"study_instance_uid": uid, "feature_identity": plan["feature_identity"],
                    "plan_sha256": expected_sha, "source_identity_sha256": source_identity,
                    "source_kind": prior_row["source_kind"], "job_id": job_id,
                    "sampled_source_indices": [list(item) for item in packed.sampled_source_indices]}
        output = Path(plan["feature_root"]) / (hashlib.sha256(uid.encode()).hexdigest() + ".npz")
        temporary = output.with_suffix(f".npz.partial-{job_id}")
        with temporary.open("xb") as handle:
            np.savez(handle, features=features, slice_mask=packed.slice_mask,
                     series_mask=packed.series_mask, series_descriptors=packed.series_descriptors,
                     metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output, follow_symlinks=False)
        temporary.unlink()
        row = {"study_instance_uid": uid, "feature_file": output.name,
               "feature_file_sha256": digest(output), "valid_slices": valid_count,
               "source_identity_sha256": source_identity, "source_kind": prior_row["source_kind"],
               "partition": partition, "job_id": job_id,
               "elapsed_seconds": time.monotonic() - step_start}
        result_rows.append(row)
        with (attempt / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if len(result_rows) % 32 == 0 or len(result_rows) == len(ordered):
            print(json.dumps({"completed_studies": len(result_rows), "partition": partition,
                              "elapsed_seconds": time.monotonic() - started}), flush=True)
    if next(gold_iterator, None) is not None:
        raise ValueError("unexpected gold archive study")
    write(attempt / "partition-result.json", {
        "schema_version": "rsna-ticket49-convnext-partition-result-v1", "status": "complete",
        "plan_sha256": expected_sha, "job_id": job_id, "partition": partition,
        "study_count": len(result_rows), "features": result_rows,
        "elapsed_seconds": time.monotonic() - started,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "gpu_count": 0, "external_spend_usd": 0, "scientific_model_training": False,
    })
    signal.alarm(0)


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
