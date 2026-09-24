"""Validate Ticket 24 v5 against the top-level ZIP layout mounted by Kaggle."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import stat
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Any

from rsna_knee.candidate import load_candidate
from rsna_knee.kaggle import load_kaggle_delivery
from ticket24_v5_zip_bootstrap import stage_delivery_archives


REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "artifacts/private/ticket-24-kaggle-bundle-20260924-v5"
V4_BUNDLE = REPO / "artifacts/private/ticket-24-kaggle-bundle-20260924-v4"
UPLOAD_ROOT = REPO / "artifacts/private/ticket-24-kaggle-upload-20260924-v5"
NOTEBOOK = REPO / "notebooks/ticket24_hidden_test_delivery_v5.ipynb"
ADAPTER = REPO / "src/rsna_knee/kaggle.py"
EXPECTED_MANIFEST = "cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc"
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


def file_map(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, sha256(path))
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def copy_mount(source: Path, destination: Path, *, link_archives: bool = False) -> None:
    destination.mkdir(parents=True)
    for name in ("request.json", NOTEBOOK.name):
        shutil.copy2(source / name, destination / name)
    for name in ("candidate.zip", "code.zip", "runtime.zip"):
        source_path = source / name
        destination_path = destination / name
        if link_archives:
            os.link(source_path, destination_path)
        else:
            shutil.copy2(source_path, destination_path)


def _run_rejected_archive_case(mount_root: Path, work_root: Path, label: str, entries: list[zipfile.ZipInfo | tuple[str, bytes]]) -> None:
    case_root = mount_root.parent / f"malformed-{label}"
    copy_mount(mount_root, case_root, link_archives=True)
    archive_path = case_root / "candidate.zip"
    archive_path.unlink()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for entry in entries:
                if isinstance(entry, tuple):
                    archive.writestr(entry[0], entry[1])
                else:
                    archive.writestr(entry, b"target")
    before = set(work_root.iterdir())
    try:
        stage_delivery_archives(
            case_root,
            case_root / "request.json",
            json.loads((case_root / "request.json").read_text(encoding="utf-8")),
            expected_request_checksum=json.loads((case_root / "request.json").read_text(encoding="utf-8"))["request_checksum"],
            expected_candidate_manifest_checksum=EXPECTED_MANIFEST,
            working_root=work_root,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError(f"unsafe archive was accepted: {label}")
    if set(work_root.iterdir()) != before:
        raise AssertionError(f"failed archive extraction left temporary files behind: {label}")
    if (case_root.parent / "escape.txt").exists():
        raise AssertionError(f"archive escaped its extraction root: {label}")


def validate_static_inference_identity(request: dict[str, Any]) -> dict[str, object]:
    candidate = load_candidate(BUNDLE / "candidate")
    if candidate.manifest_checksum != EXPECTED_MANIFEST:
        raise AssertionError("v5 candidate manifest differs from the frozen v4 candidate")
    if file_map(BUNDLE / "candidate") != file_map(V4_BUNDLE / "candidate"):
        raise AssertionError("v5 changed a frozen candidate file")
    for relative, expected in EXPECTED_CHECKPOINTS.items():
        if sha256(BUNDLE / "candidate" / relative) != expected:
            raise AssertionError(f"frozen checkpoint hash changed: {relative}")
    if request["candidate"] != json.loads((V4_BUNDLE / "request.json").read_text(encoding="utf-8"))["candidate"]:
        raise AssertionError("v5 request changed the frozen candidate binding")

    v4_adapter_path = V4_BUNDLE / "code/rsna_knee/kaggle.py"
    old_tree = ast.parse(v4_adapter_path.read_text(encoding="utf-8"))
    new_tree = ast.parse(ADAPTER.read_text(encoding="utf-8"))
    old_functions = {node.name: ast.dump(node, include_attributes=False) for node in old_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    new_functions = {node.name: ast.dump(node, include_attributes=False) for node in new_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for function_name in ("run_kaggle_baseline", "_predict_studies", "_run_candidate_inference"):
        if function_name in old_functions and old_functions[function_name] != new_functions.get(function_name):
            raise AssertionError(f"v5 changed prediction path function {function_name}")
    if old_functions.get("run_kaggle_baseline") != new_functions.get("run_kaggle_baseline"):
        raise AssertionError("v5 changed run_kaggle_baseline")

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"{NOTEBOOK.name}#cell{index}", "exec")
    return {
        "candidate_manifest_sha256": candidate.manifest_checksum,
        "candidate_file_count": len(file_map(BUNDLE / "candidate")),
        "checkpoint_sha256": EXPECTED_CHECKPOINTS,
        "run_kaggle_baseline_ast_unchanged_from_v4": True,
        "notebook_code_cells_compile": True,
    }


def main() -> None:
    request = json.loads((BUNDLE / "request.json").read_text(encoding="utf-8"))
    notebook_sha256 = sha256(NOTEBOOK)
    adapter_sha256 = sha256(ADAPTER)
    if request.get("delivery_notebook", {}).get("sha256") != notebook_sha256:
        raise AssertionError("v5 request does not bind the exact notebook bytes")
    if request.get("delivery_adapter", {}).get("sha256") != adapter_sha256:
        raise AssertionError("v5 request does not bind the exact adapter bytes")
    if request.get("candidate", {}).get("manifest_checksum") != EXPECTED_MANIFEST:
        raise AssertionError("v5 request does not bind the frozen candidate")

    static_identity = validate_static_inference_identity(request)
    expected_mount_files = {
        "request.json",
        NOTEBOOK.name,
        "candidate.zip",
        "code.zip",
        "runtime.zip",
    }
    expected_upload_files = expected_mount_files | {"dataset-metadata.json"}
    if {path.name for path in UPLOAD_ROOT.iterdir()} != expected_upload_files:
        raise AssertionError("upload root contains unexpected files or directories")
    metadata = json.loads((UPLOAD_ROOT / "dataset-metadata.json").read_text(encoding="utf-8"))
    if metadata.get("id") != "jitheswar/rsna-knee-ticket24-hidden-delivery-v5":
        raise AssertionError("upload root metadata does not name the exact v5 delivery dataset")
    with tempfile.TemporaryDirectory(prefix="ticket24-v5-preflight-") as temporary_name:
        temporary_root = Path(temporary_name)
        mount_root = temporary_root / "kaggle-input/datasets/jitheswar/rsna-knee-ticket24-hidden-delivery-v5"
        copy_mount(UPLOAD_ROOT, mount_root)
        if {path.name for path in mount_root.iterdir()} != expected_mount_files:
            raise AssertionError("synthetic Kaggle mount does not have the exact top-level archive layout")
        working_root = temporary_root / "kaggle-working"
        working_root.mkdir()
        staged_temporary, extracted_root, audit = stage_delivery_archives(
            mount_root,
            mount_root / "request.json",
            request,
            expected_request_checksum=request["request_checksum"],
            expected_candidate_manifest_checksum=EXPECTED_MANIFEST,
            working_root=working_root,
        )
        try:
            for name in ("candidate", "code", "runtime"):
                if file_map(extracted_root / name) != file_map(BUNDLE / name):
                    raise AssertionError(f"staged {name} tree differs from its source bundle")
            if (extracted_root / "request.json").read_bytes() != (BUNDLE / "request.json").read_bytes():
                raise AssertionError("staged request differs from the exact request")
            if sha256(extracted_root / NOTEBOOK.name) != notebook_sha256:
                raise AssertionError("staged notebook differs from its request binding")
            wheel_paths = [extracted_root / record["relative_path"] for record in request["delivery_runtime"]["offline_wheels"]]
            loaded = load_kaggle_delivery(
                extracted_root,
                candidate_root=extracted_root / "candidate",
                delivery_notebook_path=extracted_root / NOTEBOOK.name,
                delivery_adapter_path=extracted_root / "code/rsna_knee/kaggle.py",
                delivery_runtime_wheel_paths=wheel_paths,
            )
            if loaded.request_checksum != request["request_checksum"]:
                raise AssertionError("staged request failed the adapter's exact request binding")
            if loaded.payload["candidate"]["manifest_checksum"] != EXPECTED_MANIFEST:
                raise AssertionError("staged adapter resolved a different candidate")
        finally:
            staged_temporary.cleanup()
        if any(working_root.iterdir()):
            raise AssertionError("temporary delivery tree remained after cleanup")

        traversal = [("../escape.txt", b"blocked")]
        duplicate = [("manifest.json", b"first"), ("manifest.json", b"second")]
        symlink = zipfile.ZipInfo("candidate-link")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        _run_rejected_archive_case(mount_root, working_root, "traversal", traversal)
        _run_rejected_archive_case(mount_root, working_root, "duplicate", duplicate)
        _run_rejected_archive_case(mount_root, working_root, "symlink", [symlink])

    print(json.dumps({
        "status": "passed",
        "mount_layout": sorted(expected_mount_files),
        "request_sha256": sha256(BUNDLE / "request.json"),
        "request_checksum": request["request_checksum"],
        "notebook_sha256": notebook_sha256,
        "adapter_sha256": adapter_sha256,
        "candidate": static_identity,
        "archive_audit": audit,
        "adapter_binding_validation": "passed",
        "unsafe_archives_rejected": ["path traversal", "duplicate member", "symbolic link"],
        "temporary_tree_cleanup": "passed",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
