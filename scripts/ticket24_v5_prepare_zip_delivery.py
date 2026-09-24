"""Prepare a new local Ticket 24 request for the archive-backed Kaggle layout."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
NOTEBOOK_V4 = REPO / "notebooks/ticket24_hidden_test_delivery_v4.ipynb"
NOTEBOOK_V5 = REPO / "notebooks/ticket24_hidden_test_delivery_v5.ipynb"
V4_BUNDLE = REPO / "artifacts/private/ticket-24-kaggle-bundle-20260924-v4"
V5_BUNDLE = REPO / "artifacts/private/ticket-24-kaggle-bundle-20260924-v5"
V5_UPLOAD_ROOT = REPO / "artifacts/private/ticket-24-kaggle-upload-20260924-v5"
V5_REQUEST_ROOT = REPO / "artifacts/private/ticket-24-kaggle-delivery-20260924-v5"
V5_KERNEL_ROOT = REPO / "artifacts/private/ticket-24-kaggle-kernel-20260924-v5"
V5_REQUEST_ID = "ticket-24-ticket-11-local-candidate-20260924-hidden-v5"
V5_KERNEL_ID = "jitheswar/ticket-24-hidden-test-delivery-v5"
V5_DATASET_ID = "jitheswar/rsna-knee-ticket24-hidden-delivery-v5"
V5_APPROVAL_DATASET_ID = "jitheswar/rsna-knee-ticket24-approval-v5"
EXPECTED_MANIFEST = "cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one {label} site")
    return source.replace(old, new, 1)


def make_notebook() -> None:
    notebook = json.loads(NOTEBOOK_V4.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = [
        "# Ticket 24 hidden-test delivery v5\n",
        "\n",
        "This delivery-only notebook runs the exact frozen Ticket 11 candidate on the competition hidden-test input supplied by Kaggle at scoring time. It consumes the private Kaggle CLI archives `candidate.zip`, `code.zip`, and `runtime.zip`, validates every expected file and checksum, and stages them under a bounded temporary directory before use. The model, preprocessing, inference and output semantics remain bound to the frozen Ticket 11 candidate.\n",
        "\n",
        "Internet is disabled. Reports are not an inference input. A separate owner-approval input must authorize this exact v5 request before inference. The competition test files are supplied by Kaggle and are not checksum-pinned because their hidden version is selected at scoring time. The notebook writes `submission.csv` and execution evidence to the working directory.\n",
    ]
    code_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code" and "EXPECTED_REQUEST_ID" in "".join(cell.get("source", []))
    )
    code = "".join(notebook["cells"][code_index]["source"])
    code = replace_once(
        code,
        "EXPECTED_REQUEST_ID = 'ticket-24-ticket-11-local-candidate-20260924-hidden-v4'",
        f"EXPECTED_REQUEST_ID = '{V5_REQUEST_ID}'",
        "request id",
    )
    code = replace_once(
        code,
        "from pathlib import Path\n\nnotebook_started_at = time.perf_counter()\n",
        "from pathlib import Path, PurePosixPath\n\nnotebook_started_at = time.perf_counter()\n",
        "pathlib import",
    )
    helper_source = (REPO / "scripts/ticket24_v5_zip_bootstrap.py").read_text(encoding="utf-8")
    helper_source = helper_source.replace("from __future__ import annotations\n", "", 1)
    helper_anchor = "input_root = Path('/kaggle/input')\n"
    code = replace_once(code, helper_anchor, helper_source + "\n" + helper_anchor, "archive helper insertion")
    code = replace_once(
        code,
        "bootstrap_request_path, expected_request_checksum, bootstrap_request = request_matches[0]\n"
        "bootstrap_delivery_root = bootstrap_request_path.parent\n"
        "bootstrap_candidate_root = bootstrap_delivery_root / 'candidate'\n",
        "bootstrap_request_path, expected_request_checksum, bootstrap_request = request_matches[0]\n"
        "bootstrap_source_root = bootstrap_request_path.parent\n"
        "delivery_archive_started = time.perf_counter()\n"
        "delivery_archive_temp, bootstrap_delivery_root, archive_extraction_audit = stage_delivery_archives(\n"
        "    bootstrap_source_root,\n"
        "    bootstrap_request_path,\n"
        "    bootstrap_request,\n"
        "    expected_request_checksum=expected_request_checksum,\n"
        "    expected_candidate_manifest_checksum=EXPECTED_CANDIDATE_MANIFEST_CHECKSUM,\n"
        "    working_root=Path('/kaggle/working'),\n"
        ")\n"
        "delivery_archive_extraction_seconds = time.perf_counter() - delivery_archive_started\n"
        "bootstrap_candidate_root = bootstrap_delivery_root / 'candidate'\n",
        "archive staging setup",
    )
    code = replace_once(
        code,
        "wheel_sha256 = hashlib.sha256(wheel_path.read_bytes()).hexdigest()",
        "wheel_sha256 = _sha256_path(wheel_path)",
        "streaming wheel hash",
    )
    code = replace_once(
        code,
        "    'input_root': str(input_root),\n",
        "    'input_root': str(input_root),\n"
        "    'delivery_root': str(bootstrap_delivery_root),\n"
        "    'archive_extraction_audit': archive_extraction_audit,\n",
        "delivery root config",
    )
    code = replace_once(
        code,
        "        'offline_runtime_probe_seconds': offline_runtime_probe_seconds,\n",
        "        'offline_runtime_probe_seconds': offline_runtime_probe_seconds,\n"
        "        'delivery_archive_extraction_seconds': delivery_archive_extraction_seconds,\n",
        "archive timing record",
    )
    code = replace_once(
        code,
        "    config['input_root'],\n"
        "    expected_request_checksum=config['expected_request_checksum'],\n",
        "    config['input_root'],\n"
        "    expected_request_checksum=config['expected_request_checksum'],\n"
        "    delivery_root_override=config['delivery_root'],\n",
        "extracted delivery discovery",
    )
    notebook["cells"][code_index]["source"] = code.splitlines(keepends=True)
    inference_cell = next(
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "child_result = subprocess.run" in "".join(cell.get("source", []))
    )
    inference_source = "".join(inference_cell["source"])
    inference_source = replace_once(
        inference_source,
        "child_result = subprocess.run([sys.executable, str(inference_script)], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)\n",
        "try:\n"
        "    child_result = subprocess.run([sys.executable, str(inference_script)], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)\n"
        "finally:\n"
        "    delivery_archive_temp.cleanup()\n",
        "archive cleanup",
    )
    inference_cell["source"] = inference_source.splitlines(keepends=True)
    NOTEBOOK_V5.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def make_v5_request(bundle: Path) -> dict[str, object]:
    payload = json.loads((V4_BUNDLE / "request.json").read_text(encoding="utf-8"))
    if payload.get("request_id") != "ticket-24-ticket-11-local-candidate-20260924-hidden-v4":
        raise RuntimeError("the source v4 request id is unexpected")
    if payload.get("candidate", {}).get("manifest_checksum") != EXPECTED_MANIFEST:
        raise RuntimeError("the source v4 request is not bound to the frozen Ticket 11 candidate")
    adapter_path = REPO / "src/rsna_knee/kaggle.py"
    adapter_sha256 = sha256(adapter_path)
    notebook_sha256 = sha256(NOTEBOOK_V5)
    payload["request_id"] = V5_REQUEST_ID
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["delivery_notebook"]["filename"] = NOTEBOOK_V5.name
    payload["delivery_notebook"]["sha256"] = notebook_sha256
    payload["delivery_layout"]["notebook"]["relative_path"] = NOTEBOOK_V5.name
    payload["delivery_layout"]["notebook"]["sha256"] = notebook_sha256
    payload["delivery_adapter"]["sha256"] = adapter_sha256
    payload["delivery_layout"]["adapter"]["sha256"] = adapter_sha256
    payload.pop("request_checksum", None)
    payload["request_checksum"] = canonical_sha256(payload)
    bundle_request = bundle / "request.json"
    bundle_request.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    request_root = V5_REQUEST_ROOT
    request_root.mkdir(parents=True, exist_ok=False)
    (request_root / "request.json").write_bytes(bundle_request.read_bytes())
    return payload


def main() -> None:
    for output in (NOTEBOOK_V5, V5_BUNDLE, V5_UPLOAD_ROOT, V5_REQUEST_ROOT, V5_KERNEL_ROOT):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to replace existing v5 artifact: {output}")
    make_notebook()
    shutil.copytree(V4_BUNDLE, V5_BUNDLE, copy_function=shutil.copy2)
    old_notebook = V5_BUNDLE / "ticket24_hidden_test_delivery_v4.ipynb"
    if old_notebook.exists():
        old_notebook.unlink()
    shutil.copy2(NOTEBOOK_V5, V5_BUNDLE / NOTEBOOK_V5.name)

    delivery_code = V5_BUNDLE / "code/rsna_knee/kaggle.py"
    delivery_code.chmod(delivery_code.stat().st_mode | 0o200)
    shutil.copyfile(REPO / "src/rsna_knee/kaggle.py", delivery_code)

    metadata_path = V5_BUNDLE / "dataset-metadata.json"
    dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset_metadata["id"] = V5_DATASET_ID
    dataset_metadata["title"] = "RSNA Knee Ticket 24 Hidden-Test Delivery v5"
    dataset_metadata["description"] = (
        "Private, approval-gated v5 delivery bundle for the official RSNA Knee hidden-test input. "
        "It contains the exact frozen Ticket 11 candidate, request-bound notebook and adapter, "
        "and pinned offline runtime wheels packaged in checksum-verified ZIP archives."
    )
    metadata_path.write_text(json.dumps(dataset_metadata, indent=2) + "\n", encoding="utf-8")

    payload = make_v5_request(V5_BUNDLE)
    for directory in ("candidate", "code", "runtime"):
        archive_path = shutil.make_archive(
            str(V5_BUNDLE / directory),
            "zip",
            root_dir=str(V5_BUNDLE / directory),
        )
        if Path(archive_path).name != f"{directory}.zip":
            raise RuntimeError(f"Kaggle-style archive has an unexpected name: {archive_path}")

    V5_UPLOAD_ROOT.mkdir(parents=True, exist_ok=False)
    upload_files = [
        "dataset-metadata.json",
        "request.json",
        NOTEBOOK_V5.name,
        "candidate.zip",
        "code.zip",
        "runtime.zip",
    ]
    for filename in upload_files:
        shutil.copy2(V5_BUNDLE / filename, V5_UPLOAD_ROOT / filename)

    V5_KERNEL_ROOT.mkdir(parents=True, exist_ok=False)
    kernel_metadata = {
        "id": V5_KERNEL_ID,
        "title": "Ticket 24 Hidden-Test Delivery v5",
        "code_file": NOTEBOOK_V5.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [V5_DATASET_ID, V5_APPROVAL_DATASET_ID],
    }
    (V5_KERNEL_ROOT / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": "prepared_locally_pending_owner_approval",
        "request_id": payload["request_id"],
        "request_sha256": payload["request_checksum"],
        "notebook": str(NOTEBOOK_V5.relative_to(REPO)),
        "notebook_sha256": sha256(NOTEBOOK_V5),
        "adapter_sha256": sha256(REPO / "src/rsna_knee/kaggle.py"),
        "candidate_manifest_sha256": payload["candidate"]["manifest_checksum"],
        "bundle": str(V5_BUNDLE.relative_to(REPO)),
        "upload_root": str(V5_UPLOAD_ROOT.relative_to(REPO)),
        "upload_data_files": sorted(path.name for path in V5_UPLOAD_ROOT.iterdir() if path.name != "dataset-metadata.json"),
        "private_kernel": kernel_metadata["is_private"],
        "internet_enabled": kernel_metadata["enable_internet"],
        "gpu_enabled": kernel_metadata["enable_gpu"],
        "dataset_sources": kernel_metadata["dataset_sources"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
