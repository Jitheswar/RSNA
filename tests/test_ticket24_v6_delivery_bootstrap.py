from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ticket24_v6_delivery_bootstrap import _json_sha256, stage_delivery


MOUNT_CONTRACT = {
    "schema_version": "ticket24-package-layout-v6",
    "supported_modes": ["zip_archives", "expanded_trees"],
    "expanded_payload_file_count": 6,
    "expanded_mount_file_count": 8,
    "expanded_payload_tree_sha256": "",
    "candidate_manifest_sha256": "",
    "candidate_manifest_size_bytes": 0,
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tree_sha256(files: dict[str, bytes]) -> str:
    rows = "".join(
        f"{name}\t{len(content)}\t{_sha256(content)}\n"
        for name, content in sorted(files.items())
    )
    return _sha256(rows.encode("utf-8"))


def _write_zip(root: Path, name: str, tree_root: Path) -> None:
    with zipfile.ZipFile(root / name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(tree_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(tree_root).as_posix())


def _make_mount(root: Path, *, mode: str = "expanded_trees") -> tuple[Path, dict[str, Any]]:
    root.mkdir(parents=True)
    notebook = b"{\"cells\":[],\"nbformat\":4}\n"
    (root / "delivery.ipynb").write_bytes(notebook)

    candidate_tree = root / "candidate"
    code_tree = root / "code"
    runtime_tree = root / "runtime"
    candidate_tree.mkdir()
    code_tree.mkdir()
    runtime_tree.mkdir()

    payload_files = {
        "candidate/code/rsna_knee/common.py": b"CONST = 1\n",
        "candidate/models/fold-0.bin": b"checkpoint-bytes",
        "code/rsna_knee/common.py": b"CONST = 1\n",
        "code/rsna_knee/kaggle.py": b"def run():\n    return None\n",
        "runtime/test-1.0-py3-none-any.whl": b"wheel-bytes",
    }
    for relative, content in payload_files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    candidate_artifacts = [
        {
            "kind": "shared_inference_code",
            "path": "code/rsna_knee/common.py",
            "sha256": _sha256(payload_files["candidate/code/rsna_knee/common.py"]),
            "size_bytes": len(payload_files["candidate/code/rsna_knee/common.py"]),
        },
        {
            "kind": "model_checkpoint",
            "path": "models/fold-0.bin",
            "sha256": _sha256(payload_files["candidate/models/fold-0.bin"]),
            "size_bytes": len(payload_files["candidate/models/fold-0.bin"]),
        },
    ]
    unsigned_manifest = {"candidate_id": "test-candidate", "artifacts": candidate_artifacts}
    manifest_checksum = _json_sha256(unsigned_manifest)
    manifest = {**unsigned_manifest, "manifest_checksum": manifest_checksum}
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    (candidate_tree / "manifest.json").write_bytes(manifest_bytes)

    package_files = dict(payload_files)
    package_files["candidate/manifest.json"] = manifest_bytes
    contract = dict(MOUNT_CONTRACT)
    contract["expanded_payload_file_count"] = len(package_files)
    contract["expanded_mount_file_count"] = len(package_files) + 2
    contract["expanded_payload_tree_sha256"] = _tree_sha256(package_files)
    contract["candidate_manifest_sha256"] = _sha256(manifest_bytes)
    contract["candidate_manifest_size_bytes"] = len(manifest_bytes)
    adapter = payload_files["code/rsna_knee/kaggle.py"]
    wheel = payload_files["runtime/test-1.0-py3-none-any.whl"]
    request: dict[str, Any] = {
        "schema_version": "rsna-knee-kaggle-delivery-v1",
        "request_id": "ticket-24-test-v6",
        "status": "owner_approval_required",
        "candidate": {
            "candidate_id": "test-candidate",
            "manifest_checksum": manifest_checksum,
        },
        "delivery_notebook": {"filename": "delivery.ipynb", "sha256": _sha256(notebook)},
        "delivery_adapter": {"filename": "kaggle.py", "sha256": _sha256(adapter)},
        "delivery_runtime": {
            "offline_wheels": [{
                "filename": "test-1.0-py3-none-any.whl",
                "relative_path": "runtime/test-1.0-py3-none-any.whl",
                "sha256": _sha256(wheel),
                "size_bytes": len(wheel),
            }],
        },
        "delivery_layout": {"mount_contract": contract},
    }
    request["request_checksum"] = _json_sha256(request)
    (root / "request.json").write_text(json.dumps(request, sort_keys=True), encoding="utf-8")

    if mode == "zip_archives":
        for name in ("candidate", "code", "runtime"):
            _write_zip(root, f"{name}.zip", root / name)
            shutil.rmtree(root / name)
    return root, request


def _stage(root: Path, request: dict[str, Any]):
    working = root.parent / f"{root.name}-working"
    working.mkdir()
    return stage_delivery(
        root,
        root / "request.json",
        request,
        expected_request_checksum=request["request_checksum"],
        expected_candidate_manifest_checksum=request["candidate"]["manifest_checksum"],
        working_root=working,
    )


def test_v6_expanded_tree_layout_is_verified_and_used_in_place(tmp_path: Path) -> None:
    mount, request = _make_mount(tmp_path / "mount")

    temporary, delivery_root, audit = _stage(mount, request)

    assert temporary is None
    assert delivery_root == mount
    assert audit["mount_mode"] == "expanded_trees"
    assert audit["expanded_payload_file_count"] == 6
    assert audit["mounted_file_count"] == 8
    assert not any(mount.parent.glob("ticket24-delivery-*"))


def test_v6_zip_archive_layout_remains_supported(tmp_path: Path) -> None:
    mount, request = _make_mount(tmp_path / "mount", mode="zip_archives")

    temporary, delivery_root, audit = _stage(mount, request)

    try:
        assert temporary is not None
        assert audit["mount_mode"] == "zip_archives"
        assert audit["expanded_payload_file_count"] == 6
        assert delivery_root.is_dir()
    finally:
        if temporary is not None:
            temporary.cleanup()


@pytest.mark.parametrize(
    "case",
    ["missing", "extra", "wrong_hash", "symlink", "path_traversal", "mixed", "case_collision", "hardlink_duplicate"],
)
def test_v6_expanded_tree_rejects_noncanonical_or_unbound_files(tmp_path: Path, case: str) -> None:
    mount, request = _make_mount(tmp_path / "mount")
    if case == "missing":
        (mount / "candidate/models/fold-0.bin").unlink()
    elif case == "extra":
        (mount / "runtime/unexpected.whl").write_bytes(b"extra")
    elif case == "wrong_hash":
        (mount / "code/rsna_knee/kaggle.py").write_bytes(b"changed adapter")
    elif case == "symlink":
        (mount / "candidate/models/fold-0.bin").unlink()
        (mount / "candidate/models/fold-0.bin").symlink_to(mount / "request.json")
    elif case == "path_traversal":
        (mount / "code/..\\escape.py").write_bytes(b"escape")
    elif case == "case_collision":
        (mount / "code/rsna_knee/Kaggle.py").write_bytes(b"case collision")
    elif case == "hardlink_duplicate":
        os.link(mount / "code/rsna_knee/kaggle.py", mount / "code/rsna_knee/duplicate.py")
    elif case == "mixed":
        _write_zip(mount, "candidate.zip", mount / "candidate")

    with pytest.raises(RuntimeError):
        _stage(mount, request)


@pytest.mark.parametrize("case", ["traversal", "duplicate", "symlink"])
def test_v6_zip_layout_retains_safe_extraction_rejections(tmp_path: Path, case: str) -> None:
    mount, request = _make_mount(tmp_path / "mount", mode="zip_archives")
    archive_path = mount / "candidate.zip"
    archive_path.unlink()
    entries: list[tuple[str, bytes] | zipfile.ZipInfo]
    if case == "traversal":
        entries = [("../escape.txt", b"blocked")]
    elif case == "duplicate":
        entries = [("manifest.json", b"first"), ("manifest.json", b"second")]
    else:
        link = zipfile.ZipInfo("candidate-link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        entries = [link]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for entry in entries:
                if isinstance(entry, tuple):
                    archive.writestr(entry[0], entry[1])
                else:
                    archive.writestr(entry, b"target")

    with pytest.raises(RuntimeError):
        _stage(mount, request)
    assert not (mount.parent / "escape.txt").exists()
    assert not any((mount.parent / f"{mount.name}-working").iterdir())
