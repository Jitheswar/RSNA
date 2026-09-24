"""Run Ticket 24 v6's expanded-tree path on the 158-study local cohort."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from rsna_knee.candidate import load_candidate
from rsna_knee.kaggle import run_kaggle_baseline
from rsna_knee.submission import validate_submission_csv
from rsna_knee.targets import NUM_TARGETS, SUBMISSION_COLUMNS
from ticket24_v6_delivery_bootstrap import stage_delivery


REPO = Path(__file__).resolve().parents[1]
V5_BUNDLE = REPO / "artifacts/private/ticket-24-kaggle-bundle-20260924-v5"
V6_BUNDLE = REPO / "artifacts/private/ticket-24-kaggle-bundle-20260924-v6"
V6_NOTEBOOK = REPO / "notebooks/ticket24_hidden_test_delivery_v6.ipynb"
INPUT = REPO / "artifacts/private/ticket-12-representative-input-20260912-v12"
V4_OUTPUT = REPO / "artifacts/private/ticket-24-v4-validation/representative-158/submission.csv"
SAME_RUNTIME_PREVIOUS = REPO / "artifacts/private/ticket-24-v4-validation/representative-158-previous-v3-adapter/submission.csv"
OUTPUT_ROOT = REPO / "artifacts/private/ticket-24-v6-validation/representative-158"
EXPECTED_MANIFEST = "cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc"
EXPECTED_STUDIES = 158
EXPECTED_SERIES = 965
EXPECTED_SLICES = 31_486
MAX_PEAK_RSS_BYTES = 8 * 1024**3
EXPECTED_CHECKPOINTS = {
    "models/fold-0.pt": "5d71865ee26f36d6d447bafe28d8c6c6b9d7a18f63aad31c7df8f9532f530c76",
    "models/fold-1.pt": "4b91ab25ef3d3e7ba45dd54c56c30122b0196f6e07b64cc40ebba34df7f4095d",
    "models/fold-2.pt": "808813e7ac465aa2f02c4115145f9928a56bd7a2cd502ab3486989f53a7592e8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def _make_expanded_mount(destination: Path) -> Path:
    mount = destination / "kaggle-input/datasets/jitheswar/rsna-knee-ticket24-hidden-delivery-v6"
    mount.mkdir(parents=True)
    shutil.copy2(V6_BUNDLE / "request.json", mount / "request.json")
    shutil.copy2(V6_NOTEBOOK, mount / V6_NOTEBOOK.name)
    for name in ("candidate", "code", "runtime"):
        shutil.copytree(V6_BUNDLE / name, mount / name, copy_function=shutil.copy2)
    return mount


def main() -> None:
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(f"refusing to overwrite validation artifacts: {OUTPUT_ROOT}")
    for path in (INPUT, V4_OUTPUT, SAME_RUNTIME_PREVIOUS, V5_BUNDLE, V6_BUNDLE, V6_NOTEBOOK):
        if path.is_symlink() or not path.exists():
            raise FileNotFoundError(f"required validation input is missing or unsafe: {path}")

    request = json.loads((V6_BUNDLE / "request.json").read_text(encoding="utf-8"))
    request_checksum = request["request_checksum"]
    notebook_sha256 = sha256(V6_NOTEBOOK)
    adapter_path = V6_BUNDLE / "code/rsna_knee/kaggle.py"
    adapter_sha256 = sha256(adapter_path)
    if request.get("candidate", {}).get("manifest_checksum") != EXPECTED_MANIFEST:
        raise AssertionError("v6 request does not bind the frozen Ticket 11 candidate")
    if request.get("delivery_notebook", {}).get("sha256") != notebook_sha256:
        raise AssertionError("v6 request does not bind the prepared notebook")
    if request.get("delivery_adapter", {}).get("sha256") != adapter_sha256:
        raise AssertionError("v6 request does not bind the inference adapter")
    if sha256(V5_BUNDLE / "code/rsna_knee/kaggle.py") != adapter_sha256:
        raise AssertionError("v6 adapter differs from the historical v5 inference adapter")
    if sha256(REPO / "src/rsna_knee/kaggle.py") != adapter_sha256:
        raise AssertionError("local validation adapter differs from the request-bound delivery adapter")

    OUTPUT_ROOT.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="ticket24-v6-expanded-", dir=OUTPUT_ROOT.parent) as temporary_name:
        temporary_root = Path(temporary_name)
        mount = _make_expanded_mount(temporary_root)
        temporary, delivery_root, package_audit = stage_delivery(
            mount,
            mount / "request.json",
            request,
            expected_request_checksum=request_checksum,
            expected_candidate_manifest_checksum=EXPECTED_MANIFEST,
            working_root=temporary_root,
        )
        if temporary is not None or delivery_root != mount:
            raise AssertionError("v6 expanded-tree inference did not use the verified mounted files directly")
        if package_audit.get("mount_mode") != "expanded_trees":
            raise AssertionError("v6 representative inference did not take the expanded-tree path")
        if package_audit.get("mounted_file_count") != 77:
            raise AssertionError("v6 inference mount does not contain the expected 77 files")

        output = OUTPUT_ROOT / "submission.csv"
        evidence = OUTPUT_ROOT / "kaggle-execution.json"
        private_evidence = OUTPUT_ROOT / "private-execution.json"
        started = time.perf_counter()
        result = run_kaggle_baseline(
            delivery_root / "candidate",
            INPUT,
            output,
            evidence_path=evidence,
            private_evidence_path=private_evidence,
            delivery_request_checksum=request_checksum,
            delivery_notebook_sha256=notebook_sha256,
            delivery_adapter_sha256=adapter_sha256,
            delivery_notebook_path=delivery_root / V6_NOTEBOOK.name,
            delivery_adapter_path=delivery_root / "code/rsna_knee/kaggle.py",
            allow_local_smoke=True,
            device="cpu",
        )
        wall_seconds = time.perf_counter() - started

    observed = result.evidence
    workload = observed["workload"]
    resources = observed["resources"]
    output_record = observed["output"]
    fallback = observed["fallback"]
    recovery = observed["recovery"]
    memory = observed["timing"]["detailed"]["memory_profile"]
    for key, expected in (
        ("study_count", EXPECTED_STUDIES),
        ("series_count", EXPECTED_SERIES),
        ("slice_count", EXPECTED_SLICES),
        ("decoded_slice_count", EXPECTED_SLICES),
    ):
        if workload.get(key) != expected:
            raise AssertionError(f"v6 workload {key} is {workload.get(key)!r}, expected {expected}")
    if output_record.get("decode_failure_count") != 0:
        raise AssertionError("v6 representative inference had DICOM decode failures")
    if fallback.get("numerator") != 0:
        raise AssertionError("v6 representative inference used fallback predictions")
    if recovery.get("save_reload_verified") is not True or recovery.get("checkpoint_rows_resumed") != 0:
        raise AssertionError("v6 representative inference did not complete a fresh verified run")
    if memory.get("active_study_limit") != 1:
        raise AssertionError("v6 inference did not keep the one-study memory bound")
    for name in ("after_each_study_decode", "after_each_study_prediction", "after_each_study_release"):
        if memory[name].get("sample_count") != EXPECTED_STUDIES:
            raise AssertionError(f"v6 memory profile did not sample every study at {name}")
    peak_rss_bytes = int(resources["peak_ram_bytes"])
    if peak_rss_bytes <= 0 or peak_rss_bytes > MAX_PEAK_RSS_BYTES:
        raise AssertionError(f"v6 peak RSS {peak_rss_bytes} exceeds the 8 GiB bound")

    output_rows = read_csv(output)
    v4_rows = read_csv(V4_OUTPUT)
    previous_rows = read_csv(SAME_RUNTIME_PREVIOUS)
    if output_rows[0] != list(SUBMISSION_COLUMNS) or v4_rows[0] != list(SUBMISSION_COLUMNS):
        raise AssertionError("v6 or v4 output target order changed")
    expected_uids = tuple(row[0] for row in v4_rows[1:])
    output_uids = tuple(row[0] for row in output_rows[1:])
    if len(expected_uids) != EXPECTED_STUDIES or output_uids != expected_uids:
        raise AssertionError("v6 output study identifiers or row order differ from v4")
    if len(set(output_uids)) != EXPECTED_STUDIES:
        raise AssertionError("v6 output contains duplicate study identifiers")
    validate_submission_csv(output, expected_uids)
    if output.read_bytes() != V4_OUTPUT.read_bytes():
        raise AssertionError("v6 output is not byte-identical to the validated v4 predictions")
    if output.read_bytes() != SAME_RUNTIME_PREVIOUS.read_bytes():
        raise AssertionError("v6 output is not byte-identical to the same-runtime previous-adapter predictions")
    if len(previous_rows) != EXPECTED_STUDIES + 1:
        raise AssertionError("same-runtime previous adapter output has an unexpected row count")

    candidate = load_candidate(V6_BUNDLE / "candidate")
    checked_checkpoints = {
        relative: sha256(V6_BUNDLE / "candidate" / relative)
        for relative in EXPECTED_CHECKPOINTS
    }
    if candidate.manifest_checksum != EXPECTED_MANIFEST or checked_checkpoints != EXPECTED_CHECKPOINTS:
        raise AssertionError("v6 candidate manifest or checkpoint hashes differ from the frozen candidate")
    summary: dict[str, Any] = {
        "schema_version": "ticket-24-v6-representative-validation-v1",
        "status": "passed",
        "request": {
            "request_id": request["request_id"],
            "request_sha256": request_checksum,
            "request_file_sha256": sha256(V6_BUNDLE / "request.json"),
            "notebook_sha256": notebook_sha256,
            "adapter_sha256": adapter_sha256,
        },
        "package_mount": package_audit,
        "candidate": {
            "manifest_sha256": candidate.manifest_checksum,
            "checkpoint_sha256": checked_checkpoints,
        },
        "workload": {
            "study_count": workload["study_count"],
            "series_count": workload["series_count"],
            "slice_count": workload["slice_count"],
            "decoded_slice_count": workload["decoded_slice_count"],
            "decode_failure_count": output_record["decode_failure_count"],
            "fallback_count": fallback["numerator"],
            "active_study_limit": memory["active_study_limit"],
        },
        "memory": {
            "peak_rss_bytes": peak_rss_bytes,
            "peak_rss_gib": peak_rss_bytes / 1024**3,
            "required_maximum_bytes": MAX_PEAK_RSS_BYTES,
            "memory_profile": memory,
        },
        "runtime": {
            "wall_seconds": wall_seconds,
            "adapter_whole_notebook_seconds": observed["timing"]["whole_notebook_seconds"],
            "components": observed["timing"]["detailed"],
        },
        "equivalence": {
            "v4_csv_sha256": sha256(V4_OUTPUT),
            "v6_csv_sha256": sha256(output),
            "same_runtime_previous_csv_sha256": sha256(SAME_RUNTIME_PREVIOUS),
            "byte_identical_to_v4": True,
            "byte_identical_to_same_runtime_previous_adapter": True,
            "uid_order_matches": True,
            "target_order_matches": True,
            "output_rows": len(output_rows) - 1,
            "output_columns": len(output_rows[0]),
            "numeric_score_count": EXPECTED_STUDIES * NUM_TARGETS,
        },
        "artifacts": {
            "submission": output.name,
            "execution_evidence": evidence.name,
            "private_execution_evidence": private_evidence.name,
            "validation_summary": "validation-summary.json",
        },
        "scope": "local representative-data inference through the exact expanded v6 mount; no Kaggle upload, kernel execution, or competition submission",
    }
    (OUTPUT_ROOT / "validation-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": summary["status"],
        "request_id": request["request_id"],
        "studies": workload["study_count"],
        "peak_rss_bytes": peak_rss_bytes,
        "wall_seconds": round(wall_seconds, 3),
        "byte_identical_to_v4": True,
        "csv_sha256": summary["equivalence"]["v6_csv_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
