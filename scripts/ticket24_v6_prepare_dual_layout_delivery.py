"""Prepare the local, approval-gated Ticket 24 v6 delivery package."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from pprint import pformat
from typing import Any


REPO = Path(__file__).resolve().parents[1]
V5_NOTEBOOK = REPO / "notebooks/ticket24_hidden_test_delivery_v5.ipynb"
V6_NOTEBOOK = REPO / "notebooks/ticket24_hidden_test_delivery_v6.ipynb"
V5_BUNDLE = REPO / "artifacts/private/ticket-24-kaggle-bundle-20260924-v5"
V6_BUNDLE = REPO / "artifacts/private/ticket-24-kaggle-bundle-20260924-v6"
V5_UPLOAD_ROOT = REPO / "artifacts/private/ticket-24-kaggle-upload-20260924-v5"
V6_UPLOAD_ROOT = REPO / "artifacts/private/ticket-24-kaggle-upload-20260924-v6"
V6_REQUEST_ROOT = REPO / "artifacts/private/ticket-24-kaggle-delivery-20260924-v6"
V6_KERNEL_ROOT = REPO / "artifacts/private/ticket-24-kaggle-kernel-20260924-v6"
REMOTE_TREE_EVIDENCE = REPO / "artifacts/private/ticket-24-v6-remote-tree.json"
V6_REQUEST_ID = "ticket-24-ticket-11-local-candidate-20260924-hidden-v6"
V6_DATASET_ID = "jitheswar/rsna-knee-ticket24-hidden-delivery-v6"
V6_APPROVAL_DATASET_ID = "jitheswar/rsna-knee-ticket24-approval-v6"
V6_KERNEL_ID = "jitheswar/ticket-24-hidden-test-delivery-v6"
EXPECTED_MANIFEST = "cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc"
V5_REQUEST_ID = "ticket-24-ticket-11-local-candidate-20260924-hidden-v5"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one {label} site, found {source.count(old)}")
    return source.replace(old, new, 1)


def payload_file_records(bundle: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root_name in ("candidate", "code", "runtime"):
        root = bundle / root_name
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise RuntimeError(f"prepared payload unexpectedly contains a symlink: {path}")
            if path.is_file():
                records.append({
                    "path": path.relative_to(bundle).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_path(path),
                })
    return sorted(records, key=lambda item: item["path"])


def payload_tree_digest(records: list[dict[str, Any]]) -> str:
    rows = "".join(
        f"{record['path']}\t{record['size_bytes']}\t{record['sha256']}\n"
        for record in sorted(records, key=lambda item: item["path"])
    )
    return hashlib.sha256(rows.encode("utf-8")).hexdigest()


def mount_contract(bundle: Path) -> dict[str, Any]:
    records = payload_file_records(bundle)
    manifest = bundle / "candidate/manifest.json"
    return {
        "schema_version": "ticket24-package-layout-v6",
        "supported_modes": ["zip_archives", "expanded_trees"],
        "expanded_payload_file_count": len(records),
        "expanded_mount_file_count": len(records) + 2,
        "expanded_payload_tree_sha256": payload_tree_digest(records),
        "candidate_manifest_sha256": sha256_path(manifest),
        "candidate_manifest_size_bytes": manifest.stat().st_size,
    }


def verify_remote_tree_source(bundle: Path, contract: dict[str, Any]) -> dict[str, Any]:
    remote = json.loads(REMOTE_TREE_EVIDENCE.read_text(encoding="utf-8"))
    remote_records = {item["path"]: item for item in remote["files"]}
    local_records = {item["path"]: item for item in payload_file_records(bundle)}
    common_files = {
        "request.json": bundle / "request.json",
        V5_NOTEBOOK.name: bundle / V5_NOTEBOOK.name,
    }
    for name, path in common_files.items():
        local_records[name] = {"path": name, "size_bytes": path.stat().st_size, "sha256": sha256_path(path)}
    if set(remote_records) != set(local_records):
        raise RuntimeError(
            "prepared v5 expanded tree path set differs from the downloaded remote tree; "
            f"missing={sorted(set(remote_records) - set(local_records))}, "
            f"extra={sorted(set(local_records) - set(remote_records))}"
        )
    mismatches = [
        name
        for name in sorted(remote_records)
        if remote_records[name]["size_bytes"] != local_records[name]["size_bytes"]
        or remote_records[name]["sha256"] != local_records[name]["sha256"]
    ]
    if mismatches:
        raise RuntimeError(f"prepared v5 expanded tree differs byte-for-byte from remote files: {mismatches[:8]}")
    records = payload_file_records(bundle)
    if len(records) != contract["expanded_payload_file_count"]:
        raise RuntimeError("expanded payload count changed after mount contract creation")
    if payload_tree_digest(records) != contract["expanded_payload_tree_sha256"]:
        raise RuntimeError("expanded payload tree digest changed after mount contract creation")
    return {
        "remote_dataset": remote["source_dataset"],
        "remote_file_count": remote["remote_file_count"],
        "remote_full_tree_sha256": remote["tree_sha256"],
        "expanded_payload_file_count": len(records),
        "expanded_payload_tree_sha256": contract["expanded_payload_tree_sha256"],
        "remote_source_byte_for_byte_match": True,
    }


def make_notebook(contract: dict[str, Any]) -> None:
    notebook = json.loads(V5_NOTEBOOK.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = [
        "# Ticket 24 hidden-test delivery v6\n",
        "\n",
        "This delivery-only notebook runs the exact frozen Ticket 11 candidate on the competition hidden-test input supplied by Kaggle at scoring time. It deterministically accepts either the candidate/code/runtime ZIP archives or Kaggle's equivalent expanded candidate/code/runtime directories, verifying the exact bound files in both forms. ZIP files are safely extracted; verified expanded trees are used directly. The model, preprocessing, inference and output semantics remain bound to the frozen Ticket 11 candidate.\n",
        "\n",
        "Internet is disabled. Reports are not an inference input. A separate owner-approval input must authorize this exact v6 request before inference. The competition test files are supplied by Kaggle and are not checksum-pinned because their hidden version is selected at scoring time. The notebook writes `submission.csv` and execution evidence to the working directory.\n",
    ]
    code_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code" and "EXPECTED_REQUEST_ID" in "".join(cell.get("source", []))
    )
    code = "".join(notebook["cells"][code_index]["source"])
    code = replace_once(
        code,
        f"EXPECTED_REQUEST_ID = '{V5_REQUEST_ID}'\nEXPECTED_CANDIDATE_MANIFEST_CHECKSUM = '{EXPECTED_MANIFEST}'\n",
        f"EXPECTED_REQUEST_ID = '{V6_REQUEST_ID}'\n"
        f"EXPECTED_CANDIDATE_MANIFEST_CHECKSUM = '{EXPECTED_MANIFEST}'\n"
        f"EXPECTED_DELIVERY_MOUNT_CONTRACT = {pformat(contract, sort_dicts=True)}\n",
        "request and layout constants",
    )
    old_helper = (REPO / "scripts/ticket24_v5_zip_bootstrap.py").read_text(encoding="utf-8")
    old_helper = old_helper.replace("from __future__ import annotations\n", "", 1)
    new_helper = (REPO / "scripts/ticket24_v6_delivery_bootstrap.py").read_text(encoding="utf-8")
    new_helper = new_helper.replace("from __future__ import annotations\n", "", 1)
    code = replace_once(code, old_helper, new_helper, "v5-to-v6 embedded bootstrap")
    code = replace_once(
        code,
        "        candidate_reference = request_candidate.get('candidate')\n",
        "        candidate_reference = request_candidate.get('candidate')\n"
        "        delivery_layout = request_candidate.get('delivery_layout')\n"
        "        mount_contract = delivery_layout.get('mount_contract') if isinstance(delivery_layout, dict) else None\n",
        "request layout discovery",
    )
    code = replace_once(
        code,
        "        and request_candidate.get('request_id') == EXPECTED_REQUEST_ID\n"
        "        and candidate_reference.get('manifest_checksum') == EXPECTED_CANDIDATE_MANIFEST_CHECKSUM\n",
        "        and request_candidate.get('request_id') == EXPECTED_REQUEST_ID\n"
        "        and candidate_reference.get('manifest_checksum') == EXPECTED_CANDIDATE_MANIFEST_CHECKSUM\n"
        "        and mount_contract == EXPECTED_DELIVERY_MOUNT_CONTRACT\n",
        "request mount contract filter",
    )
    code = replace_once(
        code,
        "delivery_archive_temp, bootstrap_delivery_root, archive_extraction_audit = stage_delivery_archives(\n",
        "delivery_archive_temp, bootstrap_delivery_root, archive_extraction_audit = stage_delivery(\n",
        "dual-layout package staging call",
    )
    code = code.replace("archive_extraction_audit", "delivery_package_audit")
    code = code.replace("delivery_archive_extraction_seconds", "delivery_package_validation_seconds")
    code = code.replace("delivery_archive_started", "delivery_package_started")
    notebook["cells"][code_index]["source"] = code.splitlines(keepends=True)

    inference_cell = next(
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "child_result = subprocess.run" in "".join(cell.get("source", []))
    )
    inference_source = "".join(inference_cell["source"])
    inference_source = replace_once(
        inference_source,
        "finally:\n    delivery_archive_temp.cleanup()\n",
        "finally:\n    if delivery_archive_temp is not None:\n        delivery_archive_temp.cleanup()\n",
        "optional extracted archive cleanup",
    )
    inference_cell["source"] = inference_source.splitlines(keepends=True)
    V6_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def make_request(bundle: Path, contract: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads((V5_BUNDLE / "request.json").read_text(encoding="utf-8"))
    if payload.get("request_id") != V5_REQUEST_ID:
        raise RuntimeError("the source v5 request id is unexpected")
    if payload.get("candidate", {}).get("manifest_checksum") != EXPECTED_MANIFEST:
        raise RuntimeError("the source v5 request is not bound to the frozen Ticket 11 candidate")
    payload["request_id"] = V6_REQUEST_ID
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    notebook_hash = sha256_path(V6_NOTEBOOK)
    payload["delivery_notebook"]["filename"] = V6_NOTEBOOK.name
    payload["delivery_notebook"]["sha256"] = notebook_hash
    payload["delivery_layout"]["notebook"]["relative_path"] = V6_NOTEBOOK.name
    payload["delivery_layout"]["notebook"]["sha256"] = notebook_hash
    payload["delivery_layout"]["mount_contract"] = contract
    payload.pop("request_checksum", None)
    payload["request_checksum"] = canonical_sha256(payload)
    request_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (bundle / "request.json").write_bytes(request_bytes)
    V6_REQUEST_ROOT.mkdir(parents=True)
    (V6_REQUEST_ROOT / "request.json").write_bytes(request_bytes)
    return payload


def _copy_upload_root(bundle: Path) -> None:
    V6_UPLOAD_ROOT.mkdir(parents=True)
    for filename in (
        "dataset-metadata.json",
        "request.json",
        V6_NOTEBOOK.name,
        "candidate.zip",
        "code.zip",
        "runtime.zip",
    ):
        shutil.copy2(bundle / filename, V6_UPLOAD_ROOT / filename)


def prepare() -> dict[str, Any]:
    outputs = (V6_NOTEBOOK, V6_BUNDLE, V6_UPLOAD_ROOT, V6_REQUEST_ROOT, V6_KERNEL_ROOT)
    existing = [str(path) for path in outputs if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(f"refusing to replace existing v6 artifacts: {existing}")
    if not REMOTE_TREE_EVIDENCE.is_file():
        raise FileNotFoundError(f"missing read-only remote tree evidence: {REMOTE_TREE_EVIDENCE}")

    contract = mount_contract(V5_BUNDLE)
    remote_tree = verify_remote_tree_source(V5_BUNDLE, contract)
    shutil.copytree(V5_BUNDLE, V6_BUNDLE, copy_function=shutil.copy2)
    old_notebook = V6_BUNDLE / V5_NOTEBOOK.name
    old_notebook.unlink()
    make_notebook(contract)
    payload = make_request(V6_BUNDLE, contract)
    shutil.copy2(V6_NOTEBOOK, V6_BUNDLE / V6_NOTEBOOK.name)

    metadata_path = V6_BUNDLE / "dataset-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["id"] = V6_DATASET_ID
    metadata["title"] = "RSNA Knee Ticket 24 Hidden-Test Delivery v6"
    metadata["description"] = (
        "Private, approval-gated v6 delivery bundle for the official RSNA Knee hidden-test input. "
        "It contains the unchanged frozen Ticket 11 candidate and supports both safely verified ZIP "
        "archives and Kaggle's equivalent expanded directory layout."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _copy_upload_root(V6_BUNDLE)

    V6_KERNEL_ROOT.mkdir(parents=True)
    kernel_metadata = {
        "id": V6_KERNEL_ID,
        "title": "Ticket 24 Hidden-Test Delivery v6",
        "code_file": V6_NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [V6_DATASET_ID, V6_APPROVAL_DATASET_ID],
    }
    (V6_KERNEL_ROOT / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    if payload.get("upload", {}).get("allowed") is not False or payload.get("upload", {}).get("submission_allowed") is not False:
        raise RuntimeError("v6 request unexpectedly enables upload or competition submission")
    return {
        "status": "prepared_locally_pending_owner_approval",
        "request_id": payload["request_id"],
        "request_sha256": payload["request_checksum"],
        "notebook": str(V6_NOTEBOOK.relative_to(REPO)),
        "notebook_sha256": sha256_path(V6_NOTEBOOK),
        "candidate_manifest_sha256": payload["candidate"]["manifest_checksum"],
        "candidate_manifest_file_sha256": contract["candidate_manifest_sha256"],
        "expanded_payload_file_count": contract["expanded_payload_file_count"],
        "expanded_mount_file_count": contract["expanded_mount_file_count"],
        "expanded_payload_tree_sha256": contract["expanded_payload_tree_sha256"],
        "remote_tree": remote_tree,
        "upload_allowed": payload["upload"]["allowed"],
        "submission_allowed": payload["upload"]["submission_allowed"],
        "private_kernel": kernel_metadata["is_private"],
        "internet_enabled": kernel_metadata["enable_internet"],
        "gpu_enabled": kernel_metadata["enable_gpu"],
    }


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2, sort_keys=True))
