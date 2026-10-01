"""Extract one disjoint CPU partition of frozen pre-layer4 MRI activations."""

from __future__ import annotations

from datetime import datetime, timezone
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


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def run(plan_path, expected_sha, partition):
    started = time.monotonic()
    if digest(plan_path) != expected_sha:
        raise ValueError("Exact prefix extraction plan changed")
    plan = read(plan_path)
    root = Path(plan["run_root"])
    job = os.environ["SLURM_JOB_ID"]
    attempt = root / f"partition-{partition}-job-{job}"
    attempt.mkdir(mode=0o700)
    allocation = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True)
    (attempt / "running-allocation.txt").write_text(allocation)
    fields = dict(token.split("=", 1) for token in allocation.split() if "=" in token)
    expected = {"JobId": job, "JobState": "RUNNING", "Partition": "cluster1-cpu", "NumNodes": "1",
                "NumCPUs": "16", "NumTasks": "1", "CPUs/Task": "16", "MinMemoryNode": "64G",
                "TimeLimit": "03:00:00", "OverSubscribe": "OK", "Requeue": "0", "Restarts": "0"}
    if any(fields.get(key) != value for key, value in expected.items()) or any(
            "gpu" in fields.get(key, "").lower() for key in ("TRES", "AllocTRES", "Gres")):
        raise ValueError("Prefix extraction requires the exact CPU allocation")
    decision = read(plan["decision_path"])
    if (decision["decision"] != "approved" or decision["plan_sha256"] != expected_sha
            or decision["resources"] != plan["resources"]
            or digest(plan["authorization_path"]) != plan["authorization_sha256"]):
        raise ValueError("Exact standing-authorized extraction decision differs")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Prefix extraction watchdog")))
    signal.alarm(plan["resources"]["code_watchdog_seconds_per_job"])
    for row in plan["frozen_files"]:
        path = Path(row["path"])
        if path.is_symlink() or digest(path) != row["sha256"]:
            raise ValueError("Frozen prefix extraction dependency changed")
    for package, version in plan["runtime_versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError("Frozen CPU runtime changed")
    if os.environ.get("PYTHONPATH") != str(Path(plan["source_snapshot"]) / "src"):
        raise ValueError("Immutable extraction source runtime is required")
    archive = Path(plan["source_archive_path"])
    stat = archive.stat()
    if stat.st_size != plan["source_archive_bytes"] or stat.st_mtime_ns != plan["source_archive_mtime_ns"]:
        raise ValueError("Source archive stat identity changed")

    import numpy as np
    import torch
    from rsna_knee.domain import SeriesRecord, StudyRecord
    from rsna_knee.model import ModelConfig, build_study_model
    from rsna_knee.prefix_cache import array_digest, write_prefix_entry
    from rsna_knee.preprocessing import pack_study, prepare_study
    from rsna_knee.real_inference import iter_real_studies, load_integration_manifest
    from rsna_knee.series_cache import SeriesCache
    from rsna_knee.ticket36 import _json_sha256, _study_source_digest

    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise ValueError("Admitted CPU-only PyTorch is required")
    integration = load_integration_manifest(plan["integration_path"])
    if integration.manifest_checksum != plan["integration_manifest_checksum"]:
        raise ValueError("Exact MRI integration identity changed")
    index = read(plan["series_index_path"])["entries"]
    cache = SeriesCache(plan["series_cache_root"], read_only=True)
    studies = {item.study_instance_uid: item for item in integration.studies}
    reference = {row["study_instance_uid"]: row for row in read(plan["reference_feature_manifest_path"])["features"]}
    model = build_study_model(ModelConfig.from_dict(plan["model"]))
    model.load_pretrained_state(torch.load(plan["checkpoint_path"], map_location="cpu", weights_only=True))
    model.requires_grad_(False).eval()
    initial_encoder = {key: value.detach().clone() for key, value in model.encoder.state_dict().items()}
    uids = plan["partitions"][partition]["study_uids"]
    gold_uids = {uid for uid in uids if reference[uid]["source_kind"] == "frozen-rsna-archive"}
    gold_iterator = iter(iter_real_studies(integration, archive, require_complete_series=True,
                                         study_uids=sorted(gold_uids))) if gold_uids else iter(())
    ordered = [uid for uid in uids if uid not in gold_uids]
    ordered.extend(item.study_instance_uid for item in integration.studies if item.study_instance_uid in gold_uids)
    if len(set(ordered)) != len(uids) or set(ordered) != set(uids):
        raise ValueError("Partition source membership differs")
    rows = []
    for uid in ordered:
        study_started = time.monotonic()
        prior = reference[uid]
        manifest = studies[uid]
        if uid in gold_uids:
            loaded = next(gold_iterator)
            study = loaded.study
            if study.study_instance_uid != uid:
                raise ValueError("Gold archive reader order differs")
            prepared_study = prepare_study(study, integration.preprocess_config)
            if prepared_study.failures or not prepared_study.series:
                raise ValueError("Gold MRI preparation failed")
            prepared = prepared_study.series
            source_identity = _json_sha256({
                "archive_fingerprint": integration.archive_fingerprint, "study_instance_uid": uid,
                "series": [{"series_instance_uid": item.series_instance_uid,
                            "entry_names": list(item.source_entry_names),
                            "source_slice_uids": [part.sop_instance_uid for part in item.slices]}
                           for item in sorted(study.series, key=lambda item: item.series_instance_uid)]})
        else:
            if prior["source_kind"] != "ticket35-prepared-series-cache-read-only":
                raise ValueError("Report source kind differs")
            prepared, records = [], []
            for item in manifest.series:
                row = index[item.series_instance_uid]
                if row["study_instance_uid"] != uid or digest(cache.entry_path(row["cache_key"]) / "metadata.json") != row["metadata_sha256"]:
                    raise ValueError("Prepared MRI cache differs from frozen series index")
                prepared.append(cache.load_validated_entry(row["cache_key"], study_instance_uid=uid,
                    series_instance_uid=item.series_instance_uid, expected_source_entry_names=item.entries,
                    config=integration.preprocess_config))
                records.append(SeriesRecord(study_instance_uid=uid, series_instance_uid=item.series_instance_uid,
                    slices=(), anatomical_plane=item.anatomical_plane, fluid_sensitive=item.fluid_sensitive,
                    fat_suppression=item.fat_suppression, diagnostic_status=item.diagnostic_status,
                    source_entry_names=item.entries))
            study = StudyRecord(uid, tuple(records))
            source_identity = _study_source_digest(manifest, index, archive_fingerprint=integration.archive_fingerprint)
        if source_identity != prior["source_identity_sha256"]:
            raise ValueError("Source MRI identity differs from admitted prior features")
        packed = pack_study(study, prepared, integration.preprocess_config, packing_config=integration.packing_config)
        valid = packed.slice_mask & packed.series_mask[:, None]
        count = int(valid.sum())
        if count != plan["valid_slice_counts"][uid]:
            raise ValueError("Actual valid-slice count differs")
        prior_path = Path(plan["reference_feature_root"]) / prior["feature_file"]
        if digest(prior_path) != prior["feature_file_sha256"]:
            raise ValueError("Admitted prior feature file changed")
        with np.load(prior_path, allow_pickle=False) as previous:
            for key, value in {"slice_mask": packed.slice_mask, "series_mask": packed.series_mask,
                               "series_descriptors": packed.series_descriptors}.items():
                if not np.array_equal(previous[key], value):
                    raise ValueError("Packing masks or anatomical descriptors differ")
            previous_values = previous["features"][valid].copy()
        images = torch.from_numpy(packed.images[valid])
        chunks, full_features = [], []
        encoding_started = time.monotonic()
        with torch.inference_mode():
            for offset in range(0, count, 16):
                batch = (images[offset:offset + 16].expand(-1, 3, -1, -1)
                         - model.imagenet_mean) / model.imagenet_std
                prefix = model.encoder.forward_before_layer4(batch)
                if prefix.shape != (min(16, count - offset), 256, 14, 14):
                    raise ValueError("Prefix activation dimensions differ")
                chunks.append(prefix.numpy())
                full_features.append(model.encoder.forward_from_layer4(prefix).numpy())
        prefix_values = np.concatenate(chunks)
        parity_error = float(np.max(np.abs(np.concatenate(full_features) - previous_values)))
        if not np.isfinite(prefix_values).all() or not np.isfinite(parity_error) or parity_error > plan["native_feature_parity_tolerance"]:
            raise ValueError(f"Native frozen encoder feature parity failed: {parity_error}")
        encoding_seconds = time.monotonic() - encoding_started
        metadata = {"study_instance_uid": uid, "feature_identity_sha256": plan["feature_identity_sha256"],
                    "source_identity_sha256": source_identity, "source_kind": prior["source_kind"],
                    "extraction_plan_sha256": expected_sha, "partition": partition, "job_id": job,
                    "sampled_source_indices": [list(item) for item in packed.sampled_source_indices]}
        output = Path(plan["feature_root"]) / (hashlib.sha256(uid.encode()).hexdigest() + ".npz")
        writing_started = time.monotonic()
        checksum = write_prefix_entry(output, prefix=prefix_values, slice_mask=packed.slice_mask,
            series_mask=packed.series_mask, series_descriptors=packed.series_descriptors, metadata=metadata)
        row = {"study_instance_uid": uid, "feature_file": output.name, "feature_file_sha256": checksum,
               "prefix_array_sha256": array_digest(prefix_values), "valid_slices": count,
               "source_identity_sha256": source_identity, "source_kind": prior["source_kind"],
               "partition": partition, "job_id": job, "file_bytes": output.stat().st_size,
               "native_feature_max_abs_error": parity_error, "encoding_seconds": encoding_seconds,
               "write_and_hash_seconds": time.monotonic() - writing_started,
               "elapsed_seconds": time.monotonic() - study_started}
        rows.append(row)
        with (attempt / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        if len(rows) % 32 == 0 or len(rows) == len(ordered):
            print(json.dumps({"partition": partition, "completed_studies": len(rows),
                              "total_studies": len(ordered), "elapsed_seconds": time.monotonic() - started}), flush=True)
    if next(gold_iterator, None) is not None:
        raise ValueError("Unexpected extra gold archive study")
    if any(not torch.equal(value, initial_encoder[key]) for key, value in model.encoder.state_dict().items()):
        raise ValueError("Frozen encoder weights or normalization buffers changed")
    write(attempt / "partition-result.json", {"schema_version": "rsna-ticket57-prefix-partition-result-v1",
        "status": "COMPLETE", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": expected_sha, "job_id": job, "partition": partition, "study_count": len(rows),
        "valid_slices": sum(row["valid_slices"] for row in rows), "features": rows,
        "encoder_parameters_and_buffers_unchanged": True, "labels_read": False, "training": False,
        "elapsed_seconds": time.monotonic() - started,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "gpu_count": 0, "external_spend_usd": 0})
    signal.alarm(0)


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
