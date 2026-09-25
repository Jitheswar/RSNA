"""Produce a private, bounded metadata-only inventory for known campaign roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence, cast

from rsna_knee.image_diagnostic import (
    DiagnosticError,
    STORAGE_SCAN_MAX_ENTRIES,
    STORAGE_SCAN_MAX_SECONDS,
    _storage_snapshot,
    _write_private_new_json,
)


def _parse_root(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z0-9._-]+", label) or not raw_path:
        raise argparse.ArgumentTypeError("roots must use LABEL=/absolute/path")
    path = Path(raw_path)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("root paths must be absolute")
    return label, path


def run_inventory(roots: Sequence[tuple[str, Path]], archive_path: Path, output_path: Path) -> dict[str, object]:
    if not roots or len({label for label, _ in roots}) != len(roots):
        raise DiagnosticError("storage inventory requires unique, named roots")
    resolved_roots = tuple(path.resolve() for _, path in roots)
    manifest = [{"label": label, "path": str(path)} for (label, _), path in zip(roots, resolved_roots, strict=True)]
    snapshot = _storage_snapshot(resolved_roots, archive_path=archive_path)
    if snapshot["archive_in_measured_roots"] is not True:
        raise DiagnosticError("the retained archive is outside the declared storage roots")
    root_metrics = cast(Mapping[str, Mapping[str, object]], snapshot["root_metrics"])
    output: dict[str, object] = {
        "schema_version": "rsna-ticket29-campaign-storage-inventory-v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": snapshot["measurement"],
        "measurement_method": "logical file size from lstat-style metadata only; file contents were not opened; symlink targets were not followed",
        "root_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "root_metrics": [
            {
                "label": label,
                "logical_bytes": root_metrics[f"root_{index}"].get("logical_bytes"),
                "file_entries": root_metrics[f"root_{index}"].get("file_entries"),
            }
            for index, (label, _) in enumerate(roots, start=1)
        ],
        "root_count": len(roots),
        "measured_roots_logical_bytes": snapshot["measured_roots_logical_bytes"],
        "source_archive_bytes": snapshot["archive_bytes"],
        "source_archive_included_once_in_roots": True,
        "soft_budget_bytes": snapshot["soft_budget_bytes"],
        "projected_headroom_within_measured_roots_bytes": snapshot["projected_headroom_before_run_bytes"],
        "shared_filesystem_available_bytes": snapshot["shared_filesystem_available_bytes"],
        "shared_free_space_is_project_headroom": False,
        "scan_limits_per_root": {
            "max_entries": STORAGE_SCAN_MAX_ENTRIES,
            "max_seconds": STORAGE_SCAN_MAX_SECONDS,
        },
        "excluded_scope": "home-directory entries outside the explicit RSNA and Ticket 13 through Ticket 30 roots; shared filesystem metadata and capacity are not project usage",
    }
    _write_private_new_json(output_path, output)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=_parse_root, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_inventory(args.root, args.archive, args.output)
    except (OSError, DiagnosticError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": "completed",
                "root_count": result["root_count"],
                "measured_roots_logical_bytes": result["measured_roots_logical_bytes"],
                "source_archive_bytes_included_once": result["source_archive_bytes"],
                "root_manifest_sha256": result["root_manifest_sha256"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
