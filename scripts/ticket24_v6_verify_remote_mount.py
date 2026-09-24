"""Verify Kaggle's uploaded v6 files and stage the exact remote mount locally."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ticket24_v6_delivery_bootstrap import _extract_delivery_archive, stage_delivery
from ticket24_v6_prepare_dual_layout_delivery import (
    EXPECTED_MANIFEST,
    V6_BUNDLE,
    V6_DATASET_ID,
    V6_NOTEBOOK,
    V6_UPLOAD_ROOT,
    payload_file_records,
)


REPO = Path(__file__).resolve().parents[1]
KAGGLE_CLI = Path("/home/ch.jitheswar/.venvs/kaggle-cli-2.2.4/bin/kaggle")
EVIDENCE_PATH = REPO / "artifacts/private/ticket-24-v6-remote-mount-preflight.json"
OUTER_ZIP_LIMITS = {
    "max_archive_bytes": 250_000_000,
    "max_entries": 256,
    "max_files": 96,
    "max_total_bytes": 220_000_000,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_records(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"downloaded remote mount contains a symbolic link: {path}")
        if path.is_file():
            records[path.relative_to(root).as_posix()] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        elif not path.is_dir():
            raise RuntimeError(f"downloaded remote mount contains a special entry: {path}")
    return records


def tree_sha256(records: dict[str, dict[str, Any]]) -> str:
    rows = "".join(
        f"{name}\t{records[name]['size_bytes']}\t{records[name]['sha256']}\n"
        for name in sorted(records)
    )
    return hashlib.sha256(rows.encode("utf-8")).hexdigest()


def _run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, check=True, capture_output=True, text=True)


def _expected_mounts() -> dict[str, dict[str, dict[str, Any]]]:
    request = V6_BUNDLE / "request.json"
    notebook = V6_NOTEBOOK
    common = {
        "request.json": {"size_bytes": request.stat().st_size, "sha256": sha256(request)},
        notebook.name: {"size_bytes": notebook.stat().st_size, "sha256": sha256(notebook)},
    }
    expanded = {
        record["path"]: {"size_bytes": record["size_bytes"], "sha256": record["sha256"]}
        for record in payload_file_records(V6_BUNDLE)
    }
    expanded.update(common)
    zipped = dict(common)
    for name in ("candidate.zip", "code.zip", "runtime.zip"):
        path = V6_UPLOAD_ROOT / name
        zipped[name] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    return {"expanded_trees": expanded, "zip_archives": zipped}


def main() -> None:
    if EVIDENCE_PATH.exists() or EVIDENCE_PATH.is_symlink():
        raise FileExistsError(f"refusing to overwrite remote-layout evidence: {EVIDENCE_PATH}")
    if not KAGGLE_CLI.is_file() or KAGGLE_CLI.is_symlink():
        raise FileNotFoundError(f"authenticated Kaggle CLI is unavailable: {KAGGLE_CLI}")

    request = json.loads((V6_BUNDLE / "request.json").read_text(encoding="utf-8"))
    expected_mounts = _expected_mounts()
    listing_result = _run([
        str(KAGGLE_CLI), "datasets", "files", V6_DATASET_ID,
        "--page-size", "200", "--format", "json",
    ])
    remote_listing = json.loads(listing_result.stdout)
    if not isinstance(remote_listing, list):
        raise RuntimeError("Kaggle dataset file listing is not a JSON list")
    observed_listing: dict[str, dict[str, Any]] = {}
    for item in remote_listing:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise RuntimeError("Kaggle dataset file listing contains a malformed record")
        name = item["name"]
        if name in observed_listing:
            raise RuntimeError(f"Kaggle dataset file listing repeats a path: {name}")
        observed_listing[name] = {"size_bytes": int(item["size"])}
    modes = [
        mode for mode, expected in expected_mounts.items()
        if set(observed_listing) == set(expected)
        and all(observed_listing[name]["size_bytes"] == expected[name]["size_bytes"] for name in expected)
    ]
    if len(modes) != 1:
        missing_by_mode = {
            mode: sorted(set(expected) - set(observed_listing))
            for mode, expected in expected_mounts.items()
        }
        extra_by_mode = {
            mode: sorted(set(observed_listing) - set(expected))
            for mode, expected in expected_mounts.items()
        }
        raise RuntimeError(
            "uploaded Kaggle file set or sizes do not match either exact v6 mount form; "
            f"files={len(observed_listing)}, missing={missing_by_mode}, extra={extra_by_mode}"
        )
    mode = modes[0]

    with tempfile.TemporaryDirectory(prefix="ticket24-v6-remote-verify-", dir=EVIDENCE_PATH.parent) as temporary_name:
        temporary_root = Path(temporary_name)
        download_root = temporary_root / "download"
        download_root.mkdir()
        _run([
            str(KAGGLE_CLI), "datasets", "download", V6_DATASET_ID,
            "--path", str(download_root),
        ])
        downloaded_archives = sorted(download_root.glob("*.zip"))
        if len(downloaded_archives) != 1 or downloaded_archives[0].is_symlink():
            raise RuntimeError("Kaggle download did not produce one regular dataset archive")
        downloaded_archive = downloaded_archives[0]
        extracted_root = temporary_root / "remote-files"
        downloaded_records, _directories = _extract_delivery_archive(
            downloaded_archive,
            extracted_root,
            limits=OUTER_ZIP_LIMITS,
        )
        expected_records = expected_mounts[mode]
        missing = sorted(set(expected_records) - set(downloaded_records))
        extra = sorted(set(downloaded_records) - set(expected_records))
        if missing or extra:
            raise RuntimeError(f"downloaded remote file set differs from the verified listing: missing={missing}, extra={extra}")
        for name, expected in expected_records.items():
            if downloaded_records[name]["size_bytes"] != expected["size_bytes"]:
                raise RuntimeError(f"downloaded remote file size differs from the prepared v6 file: {name}")
            if downloaded_records[name]["sha256"] != expected["sha256"]:
                raise RuntimeError(f"downloaded remote file SHA-256 differs from the prepared v6 file: {name}")
        if set(downloaded_records) != set(observed_listing):
            raise RuntimeError("downloaded remote files do not match the authenticated Kaggle listing")

        working_root = temporary_root / "staging"
        working_root.mkdir()
        temporary, delivery_root, audit = stage_delivery(
            extracted_root,
            extracted_root / "request.json",
            request,
            expected_request_checksum=request["request_checksum"],
            expected_candidate_manifest_checksum=EXPECTED_MANIFEST,
            working_root=working_root,
        )
        try:
            if audit.get("mount_mode") != mode:
                raise RuntimeError("remote v6 mount selected a different layout than the authenticated listing")
            if audit.get("expanded_payload_tree_sha256") != request["delivery_layout"]["mount_contract"]["expanded_payload_tree_sha256"]:
                raise RuntimeError("remote v6 payload digest differs from the request's layout contract")
            if mode == "expanded_trees":
                if temporary is not None or delivery_root != extracted_root:
                    raise RuntimeError("remote expanded-tree form was not used directly")
                if audit.get("mounted_file_count") != 77:
                    raise RuntimeError("remote expanded-tree form does not contain the expected 77 files")
            elif temporary is None:
                raise RuntimeError("remote archive form was not staged through safe extraction")
        finally:
            if temporary is not None:
                temporary.cleanup()

        report = {
            "status": "passed",
            "dataset_id": V6_DATASET_ID,
            "remote_mode": mode,
            "remote_file_count": len(downloaded_records),
            "remote_mount_tree_sha256": tree_sha256(downloaded_records),
            "request_id": request["request_id"],
            "request_sha256": request["request_checksum"],
            "notebook_sha256": request["delivery_notebook"]["sha256"],
            "expanded_payload_tree_sha256": audit["expanded_payload_tree_sha256"],
            "remote_file_hashes_match_prepared_v6": True,
            "remote_listing_matches_download": True,
            "verified_stage_audit": audit,
            "kernel_execution_allowed_after_this_preflight": True,
        }
        EVIDENCE_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
