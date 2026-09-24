"""Verify Ticket 24 v6 bindings and both exact mounted package layouts."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any

from rsna_knee.candidate import load_candidate
from rsna_knee.kaggle import load_kaggle_delivery
from ticket24_v6_delivery_bootstrap import stage_delivery
from ticket24_v6_prepare_dual_layout_delivery import (
    EXPECTED_MANIFEST,
    REMOTE_TREE_EVIDENCE,
    V5_BUNDLE,
    V5_NOTEBOOK,
    V6_BUNDLE,
    V6_DATASET_ID,
    V6_KERNEL_ROOT,
    V6_NOTEBOOK,
    V6_REQUEST_ROOT,
    V6_REQUEST_ID,
    V6_UPLOAD_ROOT,
    mount_contract,
    payload_file_records,
    payload_tree_digest,
    verify_remote_tree_source,
)


REPO = Path(__file__).resolve().parents[1]
EXPECTED_V5_REQUEST = "d91d5a520e22b0d8f206e4f2929fc4978b81986f038fa0fc54d20f356062808a"
EXPECTED_V5_NOTEBOOK = "3c54a47cdc4b6e6634de577dcadcbbd759e111a1d8ff771a20a06e744682aecb"
EXPECTED_V5_ADAPTER = "8dd46cf3fea1509303376f1e10f8fee8990a932dce7c4c6da2a45c0188a60043"
EXPECTED_V5_ZIPS = {
    "candidate.zip": "7760cb50e9e4a9e8d41514a113de3e9e1300e2d08f145b4862391cf49a1d1fcd",
    "code.zip": "0954d4510bdcd49726faa492832f0c27362f09a969529f5694bee8686bc94f8b",
    "runtime.zip": "a59e5c0887153ac467060fa8c37241545fef16006991da060190d01444e3a589",
}
CHECKPOINTS = {
    "models/fold-0.pt": "5d71865ee26f36d6d447bafe28d8c6c6b9d7a18f63aad31c7df8f9532f530c76",
    "models/fold-1.pt": "4b91ab25ef3d3e7ba45dd54c56c30122b0196f6e07b64cc40ebba34df7f4095d",
    "models/fold-2.pt": "808813e7ac465aa2f02c4115145f9928a56bd7a2cd502ab3486989f53a7592e8",
}
EVIDENCE_PATH = REPO / "artifacts/private/ticket-24-v6-artifact-preflight.json"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_map(root: Path) -> dict[str, dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"artifact tree is not a regular directory: {root}")
    output: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"artifact tree contains a symlink: {path}")
        if path.is_file():
            output[path.relative_to(root).as_posix()] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
    return output


def tree_digest(records: dict[str, dict[str, Any]]) -> str:
    rows = "".join(
        f"{name}\t{records[name]['size_bytes']}\t{records[name]['sha256']}\n"
        for name in sorted(records)
    )
    return hashlib.sha256(rows.encode("utf-8")).hexdigest()


def _copy_tree_as_mount(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_symlink():
            raise RuntimeError(f"source package contains a symlink: {item}")
        if item.is_dir():
            target.mkdir(exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(item, target, follow_symlinks=False)
            except OSError:
                shutil.copy2(item, target)
        else:
            raise RuntimeError(f"source package contains a special entry: {item}")
    for directory in sorted((p for p in destination.rglob("*") if p.is_dir()), reverse=True):
        directory.chmod(0o555)
    destination.chmod(0o555)


def _copy_mount_file(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination)
    destination.chmod(0o444)


def _mount_file_map(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"simulated Kaggle mount contains a symlink: {path}")
        if path.is_file():
            records[path.relative_to(root).as_posix()] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        elif not path.is_dir():
            raise RuntimeError(f"simulated Kaggle mount contains a special entry: {path}")
    return records


def _validate_static_bindings() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_path(V5_NOTEBOOK) != EXPECTED_V5_NOTEBOOK:
        raise AssertionError("historical v5 notebook hash changed")
    v5_request = json.loads((V5_BUNDLE / "request.json").read_text(encoding="utf-8"))
    if v5_request.get("request_checksum") != EXPECTED_V5_REQUEST:
        raise AssertionError("historical v5 request checksum changed")
    if v5_request.get("delivery_adapter", {}).get("sha256") != EXPECTED_V5_ADAPTER:
        raise AssertionError("historical v5 adapter hash changed")
    for name, expected in EXPECTED_V5_ZIPS.items():
        if sha256_path(V5_BUNDLE / name) != expected:
            raise AssertionError(f"historical v5 archive changed: {name}")

    request = json.loads((V6_BUNDLE / "request.json").read_text(encoding="utf-8"))
    unsigned = dict(request)
    checksum = unsigned.pop("request_checksum", None)
    calculated = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if request.get("request_id") != V6_REQUEST_ID or checksum != calculated:
        raise AssertionError("v6 request id or canonical checksum is invalid")
    if request.get("candidate", {}).get("manifest_checksum") != EXPECTED_MANIFEST:
        raise AssertionError("v6 request does not bind the frozen candidate")
    if request.get("delivery_notebook", {}).get("filename") != V6_NOTEBOOK.name:
        raise AssertionError("v6 request names a different delivery notebook")
    notebook_hash = sha256_path(V6_NOTEBOOK)
    if request["delivery_notebook"].get("sha256") != notebook_hash:
        raise AssertionError("v6 request does not bind the notebook bytes")
    notebook_layout = request.get("delivery_layout", {}).get("notebook", {})
    if notebook_layout != {"relative_path": V6_NOTEBOOK.name, "sha256": notebook_hash}:
        raise AssertionError("v6 request layout does not bind the notebook path and bytes")
    request_bytes = (V6_BUNDLE / "request.json").read_bytes()
    for path in (
        V6_REQUEST_ROOT / "request.json",
        V6_UPLOAD_ROOT / "request.json",
    ):
        if path.read_bytes() != request_bytes:
            raise AssertionError(f"prepared v6 request copy differs from the canonical request: {path}")
    notebook_bytes = V6_NOTEBOOK.read_bytes()
    for path in (V6_BUNDLE / V6_NOTEBOOK.name, V6_UPLOAD_ROOT / V6_NOTEBOOK.name):
        if path.read_bytes() != notebook_bytes:
            raise AssertionError(f"prepared v6 notebook copy differs from the bound notebook: {path}")
    if request.get("delivery_adapter", {}).get("sha256") != EXPECTED_V5_ADAPTER:
        raise AssertionError("v6 request changed the delivery adapter")
    if request.get("upload", {}).get("allowed") is not False:
        raise AssertionError("v6 request enables upload")
    if request.get("upload", {}).get("submission_allowed") is not False:
        raise AssertionError("v6 request enables competition submission")
    if request.get("approval", {}).get("approved") is not False:
        raise AssertionError("v6 request is not pending owner approval")
    if (V6_REQUEST_ID, request["request_checksum"]) == (v5_request["request_id"], v5_request["request_checksum"]):
        raise AssertionError("v6 request accidentally reuses v5 identity")

    candidate = load_candidate(V6_BUNDLE / "candidate")
    if candidate.manifest_checksum != EXPECTED_MANIFEST:
        raise AssertionError("v6 candidate manifest differs from the frozen manifest")
    for relative, expected in CHECKPOINTS.items():
        if sha256_path(V6_BUNDLE / "candidate" / relative) != expected:
            raise AssertionError(f"frozen checkpoint changed: {relative}")
    for name in ("candidate", "code", "runtime"):
        if file_map(V6_BUNDLE / name) != file_map(V5_BUNDLE / name):
            raise AssertionError(f"v6 changed historical v5 payload tree: {name}")
    for name, expected in EXPECTED_V5_ZIPS.items():
        if sha256_path(V6_BUNDLE / name) != expected:
            raise AssertionError(f"v6 changed historical package archive: {name}")

    expected_contract = mount_contract(V5_BUNDLE)
    if request.get("delivery_layout", {}).get("mount_contract") != expected_contract:
        raise AssertionError("v6 request mount contract differs from the prepared expanded tree")
    remote_tree = verify_remote_tree_source(V5_BUNDLE, expected_contract)

    notebook_json = json.loads(V6_NOTEBOOK.read_text(encoding="utf-8"))
    code_sources = []
    for index, cell in enumerate(notebook_json.get("cells", [])):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            compile(source, f"{V6_NOTEBOOK.name}#cell{index}", "exec")
            if "EXPECTED_REQUEST_ID" in source:
                code_sources.append(source)
    if len(code_sources) != 1:
        raise AssertionError("v6 notebook does not have one request-bound bootstrap cell")
    helper = (REPO / "scripts/ticket24_v6_delivery_bootstrap.py").read_text(encoding="utf-8")
    helper = helper.replace("from __future__ import annotations\n", "", 1)
    if code_sources[0].count(helper) != 1:
        raise AssertionError("notebook inline bootstrap differs from the reviewed v6 helper")
    if "stage_delivery(\n" not in code_sources[0] or "EXPECTED_DELIVERY_MOUNT_CONTRACT" not in code_sources[0]:
        raise AssertionError("v6 notebook does not select the exact dual-layout contract")
    if "delivery_archive_temp is not None" not in "".join(
        "".join(cell.get("source", [])) for cell in notebook_json["cells"] if cell.get("cell_type") == "code"
    ):
        raise AssertionError("v6 notebook does not guard cleanup for direct expanded trees")

    v5_adapter = ast.parse((V5_BUNDLE / "code/rsna_knee/kaggle.py").read_text(encoding="utf-8"))
    v6_adapter = ast.parse((V6_BUNDLE / "code/rsna_knee/kaggle.py").read_text(encoding="utf-8"))
    def function_nodes(tree: ast.Module) -> dict[str, str]:
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
    old_functions = function_nodes(v5_adapter)
    new_functions = function_nodes(v6_adapter)
    for name in ("run_kaggle_baseline", "_predict_studies", "_run_candidate_inference"):
        if old_functions.get(name) != new_functions.get(name):
            raise AssertionError(f"v6 changed model prediction behavior: {name}")

    kernel_metadata = json.loads((V6_KERNEL_ROOT / "kernel-metadata.json").read_text(encoding="utf-8"))
    if kernel_metadata.get("is_private") is not True:
        raise AssertionError("prepared v6 kernel metadata is not private")
    if kernel_metadata.get("enable_internet") is not False or kernel_metadata.get("enable_gpu") is not False:
        raise AssertionError("prepared v6 kernel metadata enables internet or GPU")
    if kernel_metadata.get("code_file") != V6_NOTEBOOK.name:
        raise AssertionError("v6 kernel metadata names the wrong notebook")
    if kernel_metadata.get("dataset_sources") != [V6_DATASET_ID, "jitheswar/rsna-knee-ticket24-approval-v6"]:
        raise AssertionError("v6 kernel metadata names unexpected datasets")

    runtime_paths = [V6_BUNDLE / wheel["relative_path"] for wheel in request["delivery_runtime"]["offline_wheels"]]
    loaded = load_kaggle_delivery(
        V6_BUNDLE,
        candidate_root=V6_BUNDLE / "candidate",
        delivery_notebook_path=V6_NOTEBOOK,
        delivery_adapter_path=V6_BUNDLE / "code/rsna_knee/kaggle.py",
        delivery_runtime_wheel_paths=runtime_paths,
    )
    if loaded.request_checksum != request["request_checksum"]:
        raise AssertionError("delivery loader returned a different v6 checksum")
    return request, {"remote_tree": remote_tree, "notebook_code_cells_compile": True}


def _preflight_mounts(request: dict[str, Any]) -> dict[str, Any]:
    contract = request["delivery_layout"]["mount_contract"]
    with tempfile.TemporaryDirectory(prefix="ticket24-v6-preflight-", dir=V6_BUNDLE.parent) as temporary_name:
        temporary_root = Path(temporary_name)
        expanded_mount = temporary_root / "kaggle-input/datasets/jitheswar/rsna-knee-ticket24-hidden-delivery-v6"
        expanded_mount.mkdir(parents=True)
        _copy_mount_file(V6_BUNDLE / "request.json", expanded_mount / "request.json")
        _copy_mount_file(V6_NOTEBOOK, expanded_mount / V6_NOTEBOOK.name)
        for name in ("candidate", "code", "runtime"):
            _copy_tree_as_mount(V6_BUNDLE / name, expanded_mount / name)
        expanded_records = _mount_file_map(expanded_mount)
        if len(expanded_records) != 77 or len(expanded_records) != contract["expanded_mount_file_count"]:
            raise AssertionError(f"expanded mount has {len(expanded_records)} files, expected 77")
        expected_payload_paths = {
            record["path"] for record in payload_file_records(V6_BUNDLE)
        }
        expected_paths = expected_payload_paths | {"request.json", V6_NOTEBOOK.name}
        if set(expanded_records) != expected_paths:
            raise AssertionError("expanded mount does not contain the exact 77 expected relative paths")
        expected_mount_records = {
            record["path"]: {"size_bytes": record["size_bytes"], "sha256": record["sha256"]}
            for record in payload_file_records(V6_BUNDLE)
        }
        for name, path in (
            ("request.json", V6_BUNDLE / "request.json"),
            (V6_NOTEBOOK.name, V6_NOTEBOOK),
        ):
            expected_mount_records[name] = {"size_bytes": path.stat().st_size, "sha256": sha256_path(path)}
        if expanded_records != expected_mount_records:
            raise AssertionError("expanded mount file sizes or SHA-256 values differ from the prepared v6 files")
        expanded_tree_digest = tree_digest(expanded_records)

        working = temporary_root / "expanded-working"
        working.mkdir()
        temporary, resolved_root, audit = stage_delivery(
            expanded_mount,
            expanded_mount / "request.json",
            request,
            expected_request_checksum=request["request_checksum"],
            expected_candidate_manifest_checksum=EXPECTED_MANIFEST,
            working_root=working,
        )
        if temporary is not None or resolved_root != expanded_mount:
            raise AssertionError("expanded trees were copied instead of used directly")
        if audit.get("mount_mode") != "expanded_trees":
            raise AssertionError("expanded mount preflight selected the wrong mode")
        if audit.get("mounted_file_count") != 77:
            raise AssertionError("expanded mount audit does not report 77 entries")
        if audit.get("expanded_payload_tree_sha256") != contract["expanded_payload_tree_sha256"]:
            raise AssertionError("expanded mount payload tree digest differs from the request")
        if any(working.iterdir()):
            raise AssertionError("expanded-tree verification wrote staged files")

        zip_mount = temporary_root / "kaggle-input/datasets/jitheswar/zip-mode"
        zip_mount.mkdir(parents=True)
        _copy_mount_file(V6_BUNDLE / "request.json", zip_mount / "request.json")
        _copy_mount_file(V6_NOTEBOOK, zip_mount / V6_NOTEBOOK.name)
        for archive in ("candidate.zip", "code.zip", "runtime.zip"):
            shutil.copy2(V6_UPLOAD_ROOT / archive, zip_mount / archive)
            (zip_mount / archive).chmod(0o444)
        if len(list(zip_mount.iterdir())) != 5:
            raise AssertionError("ZIP mount does not have exactly five top-level entries")
        zip_working = temporary_root / "zip-working"
        zip_working.mkdir()
        staged_temporary, staged_root, zip_audit = stage_delivery(
            zip_mount,
            zip_mount / "request.json",
            request,
            expected_request_checksum=request["request_checksum"],
            expected_candidate_manifest_checksum=EXPECTED_MANIFEST,
            working_root=zip_working,
        )
        try:
            if staged_temporary is None or zip_audit.get("mount_mode") != "zip_archives":
                raise AssertionError("ZIP mount was not safely staged")
            for name in ("candidate", "code", "runtime"):
                if file_map(staged_root / name) != file_map(V6_BUNDLE / name):
                    raise AssertionError(f"ZIP-staged {name} tree differs from its prepared source")
        finally:
            if staged_temporary is not None:
                staged_temporary.cleanup()

        return {
            "expanded_mount_file_count": len(expanded_records),
            "expanded_mount_tree_sha256": expanded_tree_digest,
            "expanded_payload_tree_sha256": audit["expanded_payload_tree_sha256"],
            "expanded_mode_used_directories_in_place": True,
            "zip_mount_top_level_entry_count": len(list(zip_mount.iterdir())),
            "zip_mode_staged_payload_equivalent": True,
            "zip_mode_audit": zip_audit,
        }


def main() -> None:
    if EVIDENCE_PATH.exists():
        raise FileExistsError(f"refusing to replace existing preflight evidence: {EVIDENCE_PATH}")
    request, static = _validate_static_bindings()
    mounts = _preflight_mounts(request)
    report = {
        "status": "passed",
        "request_id": request["request_id"],
        "request_sha256": request["request_checksum"],
        "request_file_sha256": sha256_path(V6_BUNDLE / "request.json"),
        "notebook_sha256": request["delivery_notebook"]["sha256"],
        "adapter_sha256": request["delivery_adapter"]["sha256"],
        "candidate_manifest_sha256": request["candidate"]["manifest_checksum"],
        "candidate_manifest_file_sha256": request["delivery_layout"]["mount_contract"]["candidate_manifest_sha256"],
        "expanded_payload_file_count": request["delivery_layout"]["mount_contract"]["expanded_payload_file_count"],
        "expanded_payload_tree_sha256": request["delivery_layout"]["mount_contract"]["expanded_payload_tree_sha256"],
        "upload_allowed": request["upload"]["allowed"],
        "submission_allowed": request["upload"]["submission_allowed"],
        "static_binding_checks": static,
        "mount_preflight": mounts,
    }
    EVIDENCE_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
