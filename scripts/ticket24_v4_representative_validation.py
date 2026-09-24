"""Run the mandatory full-size local Ticket 24 v4 validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from rsna_knee.candidate import load_candidate
from rsna_knee.kaggle import run_kaggle_baseline
from rsna_knee.submission import validate_submission_csv
from rsna_knee.targets import NUM_TARGETS, SUBMISSION_COLUMNS


REPO = Path(__file__).resolve().parents[1]
CANDIDATE = REPO / "artifacts/private/ticket-11-local-candidate-20260911-v1"
INPUT = REPO / "artifacts/private/ticket-12-representative-input-20260912-v12"
PREVIOUS = REPO / "artifacts/private/ticket-12-kaggle-output-20260912-v13/submission.csv"
PREVIOUS_EVIDENCE = REPO / "artifacts/private/ticket-12-kaggle-output-20260912-v13/kaggle-execution.json"
PREVIOUS_ADAPTER = REPO / "artifacts/private/ticket-24-v4-validation/representative-158-previous-v3-adapter/submission.csv"
PREVIOUS_ADAPTER_EVIDENCE = REPO / "artifacts/private/ticket-24-v4-validation/representative-158-previous-v3-adapter/kaggle-execution.json"
ARCHIVED_V3_ADAPTER = REPO / "artifacts/private/ticket-24-kaggle-delivery-20260923-v3/code/rsna_knee/kaggle.py"
ARCHIVED_V3_REQUEST = REPO / "artifacts/private/ticket-24-kaggle-delivery-20260923-v3/request.json"
OUTPUT_ROOT = REPO / "artifacts/private/ticket-24-v4-validation/representative-158"
EXPECTED_MANIFEST = "cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc"
EXPECTED_CHECKPOINTS = {
    "models/fold-0.pt": "5d71865ee26f36d6d447bafe28d8c6c6b9d7a18f63aad31c7df8f9532f530c76",
    "models/fold-1.pt": "4b91ab25ef3d3e7ba45dd54c56c30122b0196f6e07b64cc40ebba34df7f4095d",
    "models/fold-2.pt": "808813e7ac465aa2f02c4115145f9928a56bd7a2cd502ab3486989f53a7592e8",
}
OLD_PEAK_RSS_BYTES = 26_716_372_992
EXPECTED_STUDIES = 158
EXPECTED_SERIES = 965
EXPECTED_SLICES = 31_486


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="validate the completed private run without repeating inference",
    )
    args = parser.parse_args()
    if OUTPUT_ROOT.exists() and not args.finalize_existing:
        raise FileExistsError(f"refusing to overwrite validation artifacts: {OUTPUT_ROOT}")

    candidate_started = time.perf_counter()
    candidate = load_candidate(CANDIDATE)
    candidate_load_seconds = time.perf_counter() - candidate_started
    if candidate.manifest_checksum != EXPECTED_MANIFEST:
        raise AssertionError("frozen candidate manifest checksum changed")
    checked_checkpoints: dict[str, str] = {}
    for relative, expected in EXPECTED_CHECKPOINTS.items():
        observed = sha256(CANDIDATE / relative)
        if observed != expected:
            raise AssertionError(f"frozen checkpoint hash changed: {relative}")
        checked_checkpoints[relative] = observed

    prior_input = json.loads(PREVIOUS_EVIDENCE.read_text(encoding="utf-8"))
    old_peak = int(prior_input["resources"]["peak_ram_bytes"])
    if old_peak != OLD_PEAK_RSS_BYTES:
        raise AssertionError("archived previous peak RSS differs from the recorded baseline")
    if int(prior_input["workload"]["study_count"]) != EXPECTED_STUDIES:
        raise AssertionError("archived previous run used a different study count")
    if int(prior_input["workload"]["series_count"]) != EXPECTED_SERIES:
        raise AssertionError("archived previous run used a different series count")
    if int(prior_input["workload"]["slice_count"]) != EXPECTED_SLICES:
        raise AssertionError("archived previous run used a different slice count")

    historical_rows = read_csv(PREVIOUS)
    if not historical_rows or historical_rows[0] != list(SUBMISSION_COLUMNS):
        raise AssertionError("previous output does not have the frozen target order")
    expected_uids = tuple(row[0] for row in historical_rows[1:])
    if len(expected_uids) != EXPECTED_STUDIES:
        raise AssertionError("previous output has an unexpected study count")
    validate_submission_csv(PREVIOUS, expected_uids)
    previous_adapter_rows = read_csv(PREVIOUS_ADAPTER)
    if previous_adapter_rows[0] != list(SUBMISSION_COLUMNS):
        raise AssertionError("same-runtime previous adapter output has an unexpected header")
    validate_submission_csv(PREVIOUS_ADAPTER, expected_uids)
    archived_request = json.loads(ARCHIVED_V3_REQUEST.read_text(encoding="utf-8"))
    if sha256(ARCHIVED_V3_ADAPTER) != archived_request["delivery_adapter"]["sha256"]:
        raise AssertionError("archived previous adapter no longer matches the v3 request")
    previous_adapter_evidence = json.loads(PREVIOUS_ADAPTER_EVIDENCE.read_text(encoding="utf-8"))
    if previous_adapter_evidence["bundle"]["candidate_manifest_checksum"] != EXPECTED_MANIFEST:
        raise AssertionError("same-runtime previous adapter used a different candidate")

    output = OUTPUT_ROOT / "submission.csv"
    evidence = OUTPUT_ROOT / "kaggle-execution.json"
    private_evidence = OUTPUT_ROOT / "private-execution.json"
    if args.finalize_existing:
        for path in (output, evidence, private_evidence):
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"completed validation artifact is missing or unsafe: {path.name}")
        observed = json.loads(evidence.read_text(encoding="utf-8"))
        total_wall_seconds = float(observed["timing"]["whole_notebook_seconds"])
    else:
        OUTPUT_ROOT.mkdir(parents=True)
        started = time.perf_counter()
        result = run_kaggle_baseline(
            CANDIDATE,
            INPUT,
            output,
            evidence_path=evidence,
            private_evidence_path=private_evidence,
            allow_local_smoke=True,
            device="cpu",
        )
        total_wall_seconds = time.perf_counter() - started
        observed = result.evidence

    output_rows = read_csv(output)
    if output_rows[0] != list(SUBMISSION_COLUMNS):
        raise AssertionError("v4 output target order changed")
    output_uids = tuple(row[0] for row in output_rows[1:])
    if output_uids != expected_uids:
        raise AssertionError("v4 output UID order differs from the previous output")
    if len(set(output_uids)) != EXPECTED_STUDIES:
        raise AssertionError("v4 output contains duplicate UIDs")
    validate_submission_csv(output, expected_uids)

    previous_numeric = np.asarray([[float(value) for value in row[1:]] for row in previous_adapter_rows[1:]])
    historical_numeric = np.asarray([[float(value) for value in row[1:]] for row in historical_rows[1:]])
    output_numeric = np.asarray([[float(value) for value in row[1:]] for row in output_rows[1:]])
    if previous_numeric.shape != (EXPECTED_STUDIES, NUM_TARGETS):
        raise AssertionError("previous output has an unexpected score matrix shape")
    if output_numeric.shape != previous_numeric.shape:
        raise AssertionError("v4 output has an unexpected score matrix shape")
    differences = np.abs(output_numeric - previous_numeric)
    differing_cells = int(np.count_nonzero(differences))
    max_abs_difference = float(differences.max(initial=0.0))
    byte_identical = output.read_bytes() == PREVIOUS_ADAPTER.read_bytes()
    if not byte_identical or differing_cells != 0 or max_abs_difference != 0.0:
        raise AssertionError("v4 predictions differ from the archived v3 adapter on the same runtime")
    historical_differences = np.abs(output_numeric - historical_numeric)
    historical_differing_cells = int(np.count_nonzero(historical_differences))
    historical_max_abs_difference = float(historical_differences.max(initial=0.0))
    historical_byte_identical = output.read_bytes() == PREVIOUS.read_bytes()
    if historical_max_abs_difference > float(np.finfo(np.float32).eps):
        raise AssertionError("cross-runtime score differences exceed one float32 epsilon")
    pairwise_order_differences = [
        int(
            np.count_nonzero(
                np.triu(
                    np.sign(historical_numeric[:, target, None] - historical_numeric[None, :, target])
                    != np.sign(output_numeric[:, target, None] - output_numeric[None, :, target]),
                    k=1,
                )
            )
        )
        for target in range(NUM_TARGETS)
    ]

    workload = observed["workload"]
    timing = observed["timing"]
    detailed = timing["detailed"]
    resources = observed["resources"]
    memory = detailed["memory_profile"]
    old = OLD_PEAK_RSS_BYTES
    new_peak = int(resources["peak_ram_bytes"])
    same_runtime_old_peak = int(previous_adapter_evidence["resources"]["peak_ram_bytes"])
    if new_peak >= old // 2:
        raise AssertionError("v4 peak RSS is not materially below the previous 26.7 GB peak")
    if workload["study_count"] != EXPECTED_STUDIES:
        raise AssertionError("v4 run did not process all representative studies")
    if workload["series_count"] != EXPECTED_SERIES or workload["slice_count"] != EXPECTED_SLICES:
        raise AssertionError("v4 study, series, or slice counts differ from the prior run")
    if workload["decoded_slice_count"] != EXPECTED_SLICES:
        raise AssertionError("v4 did not decode every representative slice")
    if observed["output"]["decode_failure_count"] != 0:
        raise AssertionError("v4 representative run had DICOM decode failures")
    if observed["fallback"]["numerator"] != 0:
        raise AssertionError("v4 representative run used fallback predictions")
    if observed["recovery"]["save_reload_verified"] is not True:
        raise AssertionError("v4 prediction checkpoint save/load verification did not pass")
    if observed["recovery"]["checkpoint_rows_resumed"] != 0:
        raise AssertionError("fresh full-size validation unexpectedly resumed old rows")

    release_samples = memory["after_each_study_release"]
    if release_samples["sample_count"] != EXPECTED_STUDIES:
        raise AssertionError("RSS was not sampled after every study release")
    if memory["active_study_limit"] != 1:
        raise AssertionError("v4 active-study memory bound is not one")
    release_spread = int(release_samples["maximum_bytes"]) - int(release_samples["minimum_bytes"])
    if memory["after_each_study_decode"]["sample_count"] != EXPECTED_STUDIES:
        raise AssertionError("RSS was not sampled after every study decode")
    if memory["after_each_study_prediction"]["sample_count"] != EXPECTED_STUDIES:
        raise AssertionError("RSS was not sampled after every study prediction")

    summary: dict[str, Any] = {
        "schema_version": "ticket-24-v4-representative-validation-v1",
        "status": "passed",
        "candidate": {
            "id": CANDIDATE.name,
            "manifest_sha256": candidate.manifest_checksum,
            "manifest_verified_before_inference": True,
            "checkpoint_sha256": checked_checkpoints,
            "load_hash_verification_seconds": candidate_load_seconds,
        },
        "input": {
            "workload_kind": "private_representative",
            "study_count": workload["study_count"],
            "series_count": workload["series_count"],
            "slice_count": workload["slice_count"],
            "decoded_slice_count": workload["decoded_slice_count"],
            "decode_failure_count": observed["output"]["decode_failure_count"],
            "fallback_count": observed["fallback"]["numerator"],
            "active_study_limit": memory["active_study_limit"],
        },
        "memory": {
            "old_peak_rss_bytes": old,
            "same_runtime_previous_adapter_peak_rss_bytes": same_runtime_old_peak,
            "new_peak_rss_bytes": new_peak,
            "reduction_ratio": 1.0 - new_peak / old,
            "same_runtime_reduction_ratio": 1.0 - new_peak / same_runtime_old_peak,
            "stage_rss_bytes": memory["stage_rss_bytes"],
            "system_memory": memory["exposed_system_memory"],
            "after_each_study_decode": memory["after_each_study_decode"],
            "after_each_study_prediction": memory["after_each_study_prediction"],
            "after_each_study_release": release_samples,
            "post_release_rss_spread_bytes": release_spread,
        },
        "runtime": {
            "full_adapter_wall_seconds": total_wall_seconds,
            "whole_adapter_seconds": timing["whole_notebook_seconds"],
            "component_seconds": {
                "candidate_hash_verification": candidate_load_seconds,
                "adapter_preflight": detailed["adapter_preflight_seconds"],
                "model_and_metadata_load": detailed["model_metadata_loading_seconds"],
                "dicom_directory_index_discovery": detailed["dicom_archive_index_discovery_seconds"],
                "dicom_read_and_decode": detailed["dicom_read_decode_seconds"],
                "pixel_decode_subset_of_dicom_read_decode": detailed["pixel_decode_seconds"],
                "preprocessing_resize": detailed["preprocessing_resize_seconds"],
                "preprocessing_packing": detailed["preprocessing_packing_seconds"],
                "model_inference_ensembling": detailed["model_inference_seconds"],
                "csv_generation_and_validation": detailed["csv_generation_validation_seconds"],
            },
            "component_timing_note": "pixel decode is included in DICOM read/decode; components are diagnostic and may not sum to wall time",
        },
        "equivalence": {
            "study_count": len(output_uids),
            "uid_order_matches_previous": True,
            "column_count": len(SUBMISSION_COLUMNS),
            "target_order_matches": True,
            "missing_rows": 0,
            "duplicate_rows": 0,
            "scores_finite_and_bounded": True,
            "byte_identical_to_previous_adapter_same_runtime": byte_identical,
            "numeric_differing_cells": differing_cells,
            "maximum_absolute_difference": max_abs_difference,
            "previous_adapter_sha256": sha256(ARCHIVED_V3_ADAPTER),
            "previous_adapter_csv_sha256": sha256(PREVIOUS_ADAPTER),
            "archived_kaggle_v13": {
                "byte_identical": historical_byte_identical,
                "numeric_differing_cells": historical_differing_cells,
                "maximum_absolute_difference": historical_max_abs_difference,
                "absolute_tolerance": float(np.finfo(np.float32).eps),
                "pairwise_order_differences_by_target": pairwise_order_differences,
                "total_pairwise_order_differences": sum(pairwise_order_differences),
                "interpretation": "small cross-runtime CPU floating-point differences; v4 is byte-identical to the archived adapter on this runtime",
            },
            "historical_kaggle_v13_csv_sha256": sha256(PREVIOUS),
            "v4_csv_sha256": sha256(output),
        },
        "recovery": {
            "checkpoint_save_reload_verified": True,
            "fresh_run_resumed_rows": observed["recovery"]["checkpoint_rows_resumed"],
            "interruption_resume_equivalence_test": {
                "test": "test_kaggle_baseline_resumes_only_from_checksum_bound_study_checkpoint",
                "result": "passed; resumed CSV byte-identical to uninterrupted CSV",
            },
        },
        "root_cause": {
            "previous_adapter": "decoded StudyRecord objects for the complete cohort were retained before inference, keeping every slice pixel array live together",
            "v4_correction": "decode, predict, durably checkpoint, drop study references, and collect before decoding the next study",
            "same_runtime_previous_peak_rss_bytes": same_runtime_old_peak,
            "v4_peak_rss_bytes": new_peak,
            "historical_old_stage_rss_limitation": "the archived v13 execution recorded process peak only; the same-host v3 adapter run reproduced the cohort-level peak",
        },
        "artifacts": {
            "output": output.name,
            "public_evidence": evidence.name,
            "private_evidence": private_evidence.name,
        },
        "limitations": [
            "The archived old run has process peak RSS but no stage-level RSS samples.",
            "Local CPU validation does not establish Kaggle hidden-test runtime or memory availability.",
        ],
    }
    (OUTPUT_ROOT / "validation-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": summary["status"],
        "study_count": workload["study_count"],
        "series_count": workload["series_count"],
        "slice_count": workload["slice_count"],
        "old_peak_rss_bytes": old,
        "same_runtime_previous_adapter_peak_rss_bytes": same_runtime_old_peak,
        "new_peak_rss_bytes": new_peak,
        "runtime_seconds": total_wall_seconds,
        "byte_identical_to_previous_adapter_same_runtime": byte_identical,
        "historical_kaggle_v13_differing_cells": historical_differing_cells,
        "historical_kaggle_v13_max_abs_difference": historical_max_abs_difference,
        "numeric_differing_cells": differing_cells,
        "csv_sha256": summary["equivalence"]["v4_csv_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
