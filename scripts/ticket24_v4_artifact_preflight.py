"""Verify the private, approval-gated Ticket 24 v4 delivery artifacts."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from rsna_knee.candidate import load_candidate
from rsna_knee.kaggle import _verify_delivery_code_mirror, load_kaggle_delivery


REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "artifacts/private/ticket-24-kaggle-bundle-20260924-v4"
REQUEST_ROOT = REPO / "artifacts/private/ticket-24-kaggle-delivery-20260924-v4"
KERNEL_ROOT = REPO / "artifacts/private/ticket-24-kaggle-kernel-20260924-v4"
VALIDATION_ROOT = REPO / "artifacts/private/ticket-24-v4-validation"
VERIFICATION_PATH = VALIDATION_ROOT / "v4-artifact-verification.json"
V3_FAILURE_PATH = VALIDATION_ROOT / "v3-failure-status.json"
EXPECTED_REQUEST_ID = "ticket-24-ticket-11-local-candidate-20260924-hidden-v4"
EXPECTED_MANIFEST = "cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc"
EXPECTED_CHECKPOINTS = {
    "models/fold-0.pt": "5d71865ee26f36d6d447bafe28d8c6c6b9d7a18f63aad31c7df8f9532f530c76",
    "models/fold-1.pt": "4b91ab25ef3d3e7ba45dd54c56c30122b0196f6e07b64cc40ebba34df7f4095d",
    "models/fold-2.pt": "808813e7ac465aa2f02c4115145f9928a56bd7a2cd502ab3486989f53a7592e8",
}
V3_REQUEST_SHA256 = "f95b22401d553e4b3fff73f1282faf8d046584098eea23a9ebe87a907597bff5"
V3_NOTEBOOK_SHA256 = "6a4249075d75eb8cefcc88abf9bdbd7268b69f597d8ded7d8dd27cec5c02b927"
V3_ADAPTER_SHA256 = "c672f553d37f737f3894e1ccc3006f0490786cbe60e68ba07113f82edd691bc1"
V3_SUBMISSION_REF = "56500900"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    request_path = BUNDLE / "request.json"
    notebook_path = BUNDLE / "ticket24_hidden_test_delivery_v4.ipynb"
    adapter_path = BUNDLE / "code/rsna_knee/kaggle.py"
    candidate_root = BUNDLE / "candidate"
    wheels = tuple(sorted((BUNDLE / "runtime").glob("*.whl"), key=lambda path: path.name))
    request = load_kaggle_delivery(
        BUNDLE,
        candidate_root=candidate_root,
        delivery_notebook_path=notebook_path,
        delivery_adapter_path=adapter_path,
        delivery_runtime_wheel_paths=wheels,
    )
    candidate = load_candidate(candidate_root)
    if candidate.manifest_checksum != EXPECTED_MANIFEST:
        raise AssertionError("v4 bundle does not contain the selected frozen candidate")
    observed_checkpoints = {
        relative: sha256(candidate_root / relative) for relative in EXPECTED_CHECKPOINTS
    }
    if observed_checkpoints != EXPECTED_CHECKPOINTS:
        raise AssertionError("v4 bundle checkpoint hashes differ from the frozen candidate")
    if (REQUEST_ROOT / "request.json").read_bytes() != request_path.read_bytes():
        raise AssertionError("canonical request record differs from the bundled request")
    if request.request_checksum != json.loads(request_path.read_text(encoding="utf-8"))["request_checksum"]:
        raise AssertionError("v4 request checksum did not verify")
    if request.payload["request_id"] != EXPECTED_REQUEST_ID:
        raise AssertionError("v4 request id is inconsistent")
    if request.payload["candidate"]["manifest_checksum"] != EXPECTED_MANIFEST:
        raise AssertionError("v4 request candidate binding is inconsistent")
    execution_input = request.payload["execution_input"]
    if execution_input != {
        "mount_slug": "rsna-knee-abnormality-detection",
        "workload_kind": "hidden_test",
    }:
        raise AssertionError("v4 request does not bind to the official hidden-test input")
    if any(
        field in execution_input
        for field in ("manifest_filename", "manifest_sha256", "test_csv_sha256", "test_series_csv_sha256")
    ):
        raise AssertionError("v4 hidden-test binding pins replaceable hidden files")
    if request.payload["approval"]["approved"] is not False:
        raise AssertionError("v4 request must remain unapproved")
    if request.payload["upload"]["allowed"] is not False or request.payload["upload"]["submission_allowed"] is not False:
        raise AssertionError("v4 upload or submission is not disabled")

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    match = re.search(r"^EXPECTED_REQUEST_ID\s*=\s*['\"]([^'\"]+)['\"]", notebook_source, re.MULTILINE)
    if match is None or match.group(1) != EXPECTED_REQUEST_ID:
        raise AssertionError("notebook request id differs from the v4 request")
    if EXPECTED_MANIFEST not in notebook_source:
        raise AssertionError("notebook does not pin the selected candidate manifest")
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=notebook_path.name)
    if "20260923-hidden-v3" in notebook_source:
        raise AssertionError("v4 notebook retains the historical request id")

    _verify_delivery_code_mirror(candidate_root, BUNDLE / "code", adapter_path)
    kernel_metadata = json.loads((KERNEL_ROOT / "kernel-metadata.json").read_text(encoding="utf-8"))
    if kernel_metadata["id"] != "jitheswar/ticket-24-hidden-test-delivery-v4":
        raise AssertionError("kernel metadata id is inconsistent")
    if kernel_metadata["code_file"] != notebook_path.name:
        raise AssertionError("kernel metadata points at a different notebook")
    if kernel_metadata["is_private"] is not True:
        raise AssertionError("v4 kernel metadata is not private")
    if kernel_metadata["enable_internet"] is not False or kernel_metadata["enable_gpu"] is not False:
        raise AssertionError("v4 kernel metadata must disable internet and GPU")
    if "jitheswar/rsna-knee-ticket24-hidden-delivery-v4" not in kernel_metadata["dataset_sources"]:
        raise AssertionError("kernel metadata is missing the v4 delivery dataset")
    if "jitheswar/rsna-knee-ticket24-approval-v4" not in kernel_metadata["dataset_sources"]:
        raise AssertionError("kernel metadata is missing the planned v4 approval dataset source")

    if V3_REQUEST_SHA256 == request.request_checksum:
        raise AssertionError("v4 must use a new request checksum")
    v3_failure = {
        "schema_version": "ticket-24-v3-failure-status-v1",
        "attempt": "ticket-24 hidden-test delivery v3",
        "status": "failed_during_competition_scoring",
        "failure": "Notebook Out of Memory",
        "failure_source": "owner-provided task instruction dated 2026-09-24",
        "preserved_submission_reference": V3_SUBMISSION_REF,
        "preserved_submission_api_status": "SubmissionStatus.COMPLETE",
        "api_status_interpretation": "submission workflow status did not prove successful scored notebook execution",
        "request_sha256": V3_REQUEST_SHA256,
        "notebook_sha256": V3_NOTEBOOK_SHA256,
        "adapter_sha256": V3_ADAPTER_SHA256,
        "full_scoring_oom_log_archived_locally": False,
        "archived_v3_artifacts_modified": False,
    }
    V3_FAILURE_PATH.write_text(json.dumps(v3_failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def asset(path: Path) -> dict[str, object]:
        return {"sha256": sha256(path), "size_bytes": path.stat().st_size}

    verification: dict[str, Any] = {
        "schema_version": "ticket-24-v4-artifact-verification-v1",
        "status": "passed",
        "request_id": request.payload["request_id"],
        "request_sha256": request.request_checksum,
        "request_status": request.payload["status"],
        "approval_status": request.payload["approval"]["status"],
        "upload_allowed": request.payload["upload"]["allowed"],
        "submission_allowed": request.payload["upload"]["submission_allowed"],
        "execution_input": dict(execution_input),
        "candidate_manifest_sha256": candidate.manifest_checksum,
        "checkpoint_sha256": observed_checkpoints,
        "notebook": asset(notebook_path),
        "adapter": asset(adapter_path),
        "runtime_wheels": {path.name: asset(path) for path in wheels},
        "dataset_metadata": asset(BUNDLE / "dataset-metadata.json"),
        "kernel_metadata": asset(KERNEL_ROOT / "kernel-metadata.json"),
        "kernel_private": True,
        "internet_enabled": False,
        "gpu_enabled": False,
        "code_mirror_verified": True,
        "request_notebook_binding_verified": True,
        "remote_actions_performed": [],
        "owner_approval_present": False,
        "approval_dataset_created": False,
    }
    VERIFICATION_PATH.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": verification["status"],
        "request_id": request.payload["request_id"],
        "request_sha256": request.request_checksum,
        "notebook_sha256": verification["notebook"]["sha256"],
        "adapter_sha256": verification["adapter"]["sha256"],
        "candidate_manifest_sha256": candidate.manifest_checksum,
        "kernel_private": kernel_metadata["is_private"],
        "internet_enabled": kernel_metadata["enable_internet"],
        "gpu_enabled": kernel_metadata["enable_gpu"],
        "upload_allowed": request.payload["upload"]["allowed"],
        "submission_allowed": request.payload["upload"]["submission_allowed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
