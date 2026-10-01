"""Independently check the exact completed paired development artifacts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from public_frontier_validation import REQUIRED_PHASES, validate_public_frontier
from public_meniscus_overlay import ID, TARGETS, apply_meniscus_overlay


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate(root):
    checksum = digest(root / "request.json")
    request, accepted, source, terminal = [read(root / filename) for filename in
        ("request.json", "accepted-launch.json", "remote-source-validation.json", "kernel-terminal.json")]
    if (checksum != "0e490a080a5c51c1aeaaab4e67da5673129d47ebec22a48173eecf0e3fa78ed6"
            or accepted["request_checksum"] != checksum or terminal["status"] != "COMPLETE"
            or terminal["kernel_id"] != accepted["kernel_id"] or terminal["kernel_version"] != accepted["kernel_version"]
            or source["status"] != "PASS" or not source["exact_code_cells"]
            or not source["attachments_canonical_refs_verified"] or not source["private_internet_off"]
            or not source["pinned_docker"] or source["kernel_version"] != accepted["kernel_version"]
            or digest(root / "development.ipynb") != request["notebook_sha256"]
            or digest(root / "kernel-metadata.json") != request["metadata_sha256"]):
        raise ValueError("Exact matching terminal source and immutable request are required")
    remote = read(root / "remote-kernel-capture-v1.json")
    if hashlib.sha256(remote["blob"]["source"].encode()).hexdigest() != source["remote_source_sha256"]:
        raise ValueError("Independently captured remote source changed")
    output = root / "output"
    receipt_path = output / "paired_development.json"
    receipt = read(receipt_path)
    if (receipt["status"] != "COMPLETE" or receipt["study_count"] != 58
            or receipt["anchor_code_cells_unchanged"] != 23 or receipt["specialist_fallback"] != 0
            or not receipt["actual_cpu_mri_parity"] or not receipt["anchor_random_states_preserved"]
            or not receipt["eleven_other_targets_identical"] or not receipt["input_cache_disk_branch"]
            or receipt["input_cache_sha256"] != request["native_cache_sha256"]
            or receipt["input_mask_sha256"] != request["native_mask_sha256"]):
        raise ValueError("Actual native MRI input, specialist and fixed vote contract differs")
    expected_files = {"development-anchor.csv", "development-overlay.csv", "development-transformer.csv",
                      "development-hybrid.csv", "public0033_bag_raw.csv", "public0033_cached_inference_receipt.json"}
    if set(receipt["output_sha256"]) != expected_files:
        raise ValueError("Exact prediction output membership differs")
    for filename in expected_files:
        if digest(output / filename) != receipt["output_sha256"][filename]:
            raise ValueError("Actual prediction or specialist receipt bytes changed")
    frames = [pd.read_csv(output / filename, dtype={ID: str}, float_precision="round_trip") for filename in
        ("development-anchor.csv", "development-overlay.csv", "development-transformer.csv",
         "development-hybrid.csv", "public0033_bag_raw.csv")]
    anchor, overlay, transformer, hybrid, specialist = frames
    uids = anchor[ID].tolist()
    uid_sha = hashlib.sha256("".join(uid + "\n" for uid in uids).encode()).hexdigest()
    if len(uids) != 58 or len(set(uids)) != 58 or uid_sha != request["development_study_uid_sha256"]:
        raise ValueError("Frozen actual study membership or order differs")
    for frame in frames[:4]:
        values = frame[TARGETS].to_numpy()
        if frame[ID].tolist() != uids or frame.columns.tolist() != [ID] + TARGETS or not np.isfinite(values).all():
            raise ValueError("Malformed actual prediction schema, values or study identity")
    specialist_columns = [ID, "Medial Meniscus", "Lateral Meniscus"]
    if (specialist.columns.tolist() != specialist_columns or specialist[ID].tolist() != uids
            or not np.isfinite(specialist[specialist_columns[1:]].to_numpy()).all()):
        raise ValueError("Actual specialist two-target output contract differs")
    specialist_receipt = read(output / "public0033_cached_inference_receipt.json")
    if (specialist_receipt["status"] != "passed" or specialist_receipt["study_count"] != 58
            or specialist_receipt["fallback"] != 0 or specialist_receipt["neutralized_predictions"] != 0
            or specialist_receipt["bundle_manifest_sha256"] != request["specialist_manifest_sha256"]
            or specialist_receipt["runtime_sha256"] != request["specialist_runtime_sha256"]
            or specialist_receipt["checkpoint"]["sha256"] != "e1443bb0f518418a31428cd7eb153f15af4c48c7175baf1f9d21c84f1c2a93e7"
            or not specialist_receipt["checkpoint"]["strict_load"]
            or specialist_receipt["checkpoint"]["state_tensor_count"] != 233
            or specialist_receipt["precision"] != "fp32" or specialist_receipt["autocast"]
            or specialist_receipt["mask_sha256"] != request["native_mask_sha256"]
            or specialist_receipt["raw_output"]["columns"] != specialist_columns
            or specialist_receipt["raw_output"]["sha256"] != digest(output / "public0033_bag_raw.csv")
            or len(specialist_receipt["devices"]) != 2
            or any(device["device_name"] != "Tesla T4" for device in specialist_receipt["devices"])):
        raise ValueError("Actual admitted specialist checkpoint, runtime or GPU execution differs")
    if not overlay.equals(apply_meniscus_overlay(anchor, transformer, hybrid, specialist)):
        raise ValueError("Actual saved predictions differ from the single prospectively fixed vote")
    audit = read(output / "btkd_v559_complete.json")
    blend = read(output / "_coat_raptor_blend_receipt.json")
    paths = {"resgated_top3": "_coat_arm_receipt.json", "global96_top3": "_coat_g96_receipt.json",
             "repairv1_top3": "_coat_repairv1_receipt.json", "d4_swa3": "d4_input/d4.receipt.json"}
    strict = validate_public_frontier(audit, blend, {key: read(output / value) for key, value in paths.items()}, 58)
    phases = audit["phases"]
    required = REQUIRED_PHASES | {name for name in phases if name.startswith("dino_")}
    verified = []
    for name in sorted(required):
        proof = phases[name]
        relative = Path(proof["path"]).relative_to("/kaggle/working")
        path = output / relative
        if digest(path) != proof["sha256"]:
            raise ValueError("Actual component prediction bytes changed")
        with np.load(path, allow_pickle=False) as saved:
            if (saved["study_uids"].astype(str).tolist() != uids or saved["labels"].astype(str).tolist() != TARGETS
                    or saved["values"].shape != (58, 12) or not np.isfinite(saved["values"]).all()):
                raise ValueError("Actual component prediction mapping or values differ")
        verified.append({"component": name, "sha256": digest(path)})
    for log in output.rglob("*.log"):
        if "[dense-fallback]" in log.read_text(errors="replace"):
            raise ValueError("Completed anchor contains a dense input fallback")
    if (output / "submission.csv").exists():
        raise ValueError("Development-only execution must not expose a submission file")
    return {"schema_version": "rsna-public-meniscus-independent-development-validation-v1", "status": "COMPLETE",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(), "request_sha256": checksum,
        "kernel_id": accepted["kernel_id"], "kernel_version": accepted["kernel_version"],
        "terminal_evidence_sha256": digest(root / "kernel-terminal.json"),
        "remote_source_validation_sha256": digest(root / "remote-source-validation.json"),
        "paired_receipt_sha256": digest(receipt_path), "actual_studies_verified": 58,
        "native_cpu_pixel_and_mask_hashes_match": True, "input_cache_disk_branch_verified": True,
        "strict_anchor_execution": strict, "component_prediction_artifacts_verified": verified,
        "fixed_vote_recomputed_exactly": True, "eleven_other_targets_identical": True,
        "assigned_gpu_count": 2, "visible_elapsed_seconds": receipt["elapsed_seconds"],
        "hidden_failure_reproduced": False, "hidden_scale_runtime_verified": False,
        "clean_oof": False, "candidate_promotion": False, "competition_submission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = validate(root)
    with (root / "validation.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "studies_verified": 58,
                      "component_artifacts_verified": len(result["component_prediction_artifacts_verified"])}))


if __name__ == "__main__":
    main()
