"""Compare every retained pixel entry with fresh native source packing."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

from rsna_knee.pixel_cache import load_pixel_entry
from rsna_knee.prefix_cache import array_digest, file_digest
from rsna_knee.training import _array_sha256

from ticket62_mri_pixel_materialize import iter_packed_sources


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main():
    started = time.monotonic()
    root = Path(sys.argv[1]).resolve()
    plan = read(root / "plan.json")
    checksum = file_digest(root / "plan.json")
    job = os.environ["SLURM_JOB_ID"]
    result_path = root / f"partition-0-job-{job}/partition-result.json"
    result = read(result_path)
    assert result["status"] == "COMPLETE" and result["plan_sha256"] == checksum
    assert result["job_id"] == job and result["gpu_count"] == 0 and result["external_spend_usd"] == 0
    assert result["labels_read"] is False and result["training"] is False
    for item in plan["frozen_files"]:
        assert file_digest(Path(item["path"])) == item["sha256"], item["path"]
    uids = plan["partitions"][0]["study_uids"]
    rows = {row["study_instance_uid"]: row for row in result["pixels"]}
    assert len(rows) == len(result["pixels"]) == len(uids) == 4228
    assert set(rows) == set(uids) == set(plan["valid_slice_counts"])
    assert sum(row["valid_slices"] for row in rows.values()) == result["valid_slices"] == 187272
    expected_files = {row["pixel_file"] for row in rows.values()}
    assert {p.name for p in Path(plan["pixel_root"]).glob("*.npz")} == expected_files
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available() and torch.version.cuda is None
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Pixel verifier watchdog")))
    signal.alarm(plan["resources"]["code_watchdog_seconds_per_job"])
    checked, payload_bytes, source_seconds, cache_seconds = 0, 0, 0.0, 0.0
    prior = read(plan["admitted_pilot_plan"])
    prior_result = read(plan["admitted_pilot_result"])
    prior_selected = prior["selected_training_report_uids"]
    replay_images = {}
    entries = []
    for uid, packed, source_identity, reference, native_seconds in iter_packed_sources(plan, uids):
        row = rows[uid]
        path = Path(plan["pixel_root"]) / row["pixel_file"]
        tick = time.monotonic()
        entry = load_pixel_entry(path, expected_sha256=row["pixel_file_sha256"],
            expected_study_uid=uid, expected_identity_sha256=plan["pixel_identity_sha256"],
            expected_source_identity_sha256=source_identity, expected_plan_sha256=checksum)
        images = entry.packed_images()
        cache_read_and_reconstruction = time.monotonic()-tick
        assert np.array_equal(images, packed.images)
        assert np.array_equal(entry.slice_mask, packed.slice_mask)
        assert np.array_equal(entry.series_mask, packed.series_mask)
        assert np.array_equal(entry.series_descriptors, packed.series_descriptors)
        assert entry.metadata["source_kind"] == row["source_kind"] == reference["source_kind"]
        assert entry.metadata["job_id"] == row["job_id"] == job
        assert entry.metadata["sampled_source_indices"] == [list(item) for item in packed.sampled_source_indices]
        assert source_identity == row["source_identity_sha256"] == reference["source_identity_sha256"]
        assert len(entry.pixels) == row["valid_slices"] == plan["valid_slice_counts"][uid]
        assert array_digest(entry.pixels) == row["pixel_array_sha256"]
        payload_bytes += entry.pixels.nbytes
        checked += 1
        source_seconds += native_seconds
        cache_seconds += cache_read_and_reconstruction
        if uid in prior_selected:
            replay_images[uid] = images
        entries.append({**row, "independent_native_pixel_replay_exact": True,
                        "verified_read_reconstruction_seconds": cache_read_and_reconstruction})
        if checked % 128 == 0:
            print(json.dumps({"verified_studies":checked,"elapsed_seconds":time.monotonic()-started}),flush=True)
    assert checked == 4228 and payload_bytes == 37_586_239_488
    assert set(replay_images) == set(prior_selected)
    assert _array_sha256(np.stack([replay_images[uid] for uid in prior_selected])) == prior_result["input_array_sha256"]
    manifest = {"schema_version":"rsna-ticket62-pixel-cohort-manifest-v1", "status":"COMPLETE",
        "plan_sha256":checksum,"pixel_identity_sha256":plan["pixel_identity_sha256"],
        "study_count":checked,"valid_slices":187272,"exact_pixel_payload_bytes":payload_bytes,
        "pixels":sorted(entries,key=lambda row:row["study_instance_uid"])}
    write(root / "pixel-manifest.json", manifest)
    verification = {"status":"COMPLETE","observed_at_utc":datetime.now(timezone.utc).isoformat(),
        "job_id":job,"plan_sha256":checksum,"result_sha256":file_digest(result_path),
        "manifest_sha256":file_digest(root / "pixel-manifest.json"),"study_count":checked,
        "valid_slices":187272,"exact_pixel_payload_bytes":payload_bytes,
        "all_original_source_pixel_replays_exact":True,"all_prior_masks_descriptors_sources_match":True,
        "prior_four_report_full_pixel_replay_exact":True,"labels_read":False,"training":False,
        "source_read_pack_seconds":source_seconds,"all_cache_read_reconstruction_seconds":cache_seconds,
        "maximum_cache_read_reconstruction_seconds":max(r["verified_read_reconstruction_seconds"] for r in entries),
        "elapsed_seconds":time.monotonic()-started,"gpu_count":0,"external_spend_usd":0,
        "actual_terminal_still_required":True,"candidate_promotion":False,"full_training_selected":False}
    write(root / "independent-validation.json", verification)
    signal.alarm(0)
    print(json.dumps(verification),flush=True)


if __name__ == "__main__":
    main()
