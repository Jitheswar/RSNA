"""Supersede only the unlaunched Ticket70 source envelope; never run a kernel."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
PARENT = PROJECT / "artifacts/private/ticket70-scored-anchor-gold58-v1"
DEST = PROJECT / "artifacts/private/ticket70-scored-anchor-gold58-v2"
BASELINE = PROJECT / "artifacts/private/public-notebook-adoption/scored-baseline-v1"
AUDIT = PROJECT / "artifacts/private/ticket70-asset-lineage-audit-v1"
PARENT_REQUEST_SHA = "e447295cda5201165327da9410c85f55f2815aa7dea07ee013a473f435c2eb71"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_metadata(metadata, datasets, cutoff, docker_image):
    """Pin observed ready versions only when their updates predate the baseline."""
    refs = metadata["dataset_sources"]
    if (len(refs) != len(set(refs)) or len(datasets) != len(refs)
            or {item["ref"] for item in datasets} != set(refs)):
        raise ValueError("Exact attachment membership required")
    if not re.fullmatch(r"gcr\.io/kaggle-private-byod/python@sha256:[0-9a-f]{64}", docker_image):
        raise ValueError("Exact saved container digest required")
    by_ref = {item["ref"]: item for item in datasets}
    pins = []
    for ref in refs:
        data = by_ref[ref]
        version = data["currentVersionNumber"]
        if type(version) is not int or version < 1:
            raise ValueError("Explicit positive dataset version required")
        latest = [row for row in data["versions"] if row["versionNumber"] == version]
        if len(latest) != 1 or latest[0]["status"] != "Ready":
            raise ValueError("Unique ready dataset version required")
        if (datetime.fromisoformat(data["lastUpdated"]) > cutoff
                or datetime.fromisoformat(latest[0]["creationDate"]) > cutoff):
            raise ValueError("Dataset update postdates the scored baseline")
        pins.append(f"{ref}/{version}")
    result = dict(metadata)
    result.update(dataset_sources=pins, docker_image=docker_image,
                  docker_image_pinning_type="original")
    return result


def inputs():
    if digest(PARENT / "request.json") != PARENT_REQUEST_SHA:
        raise ValueError("Original prepared request changed")
    parent = json.loads((PARENT / "request.json").read_text())
    validation = json.loads((PARENT / "preparation-validation.json").read_text())
    if validation["status"] != "PREPARATION_VALIDATED" or validation["request_sha256"] != PARENT_REQUEST_SHA:
        raise ValueError("Original independent preparation proof differs")
    if digest(PARENT / "development.ipynb") != parent["notebook_sha256"]:
        raise ValueError("Original source notebook changed")
    for name, expected in parent["frozen_files"].items():
        if digest(PARENT / name) != expected:
            raise ValueError("Original frozen source changed")
    metadata = json.loads((PARENT / "kernel-metadata.json").read_text())
    if digest(PARENT / "kernel-metadata.json") != parent["metadata_sha256"]:
        raise ValueError("Original attachment metadata changed")
    capture = json.loads((BASELINE / "remote-version-2.json").read_text())
    if digest(BASELINE / "remote-version-2.json") != parent["remote_capture_sha256"]:
        raise ValueError("Saved scored container/source capture changed")
    source_request = json.loads((BASELINE / "request.json").read_text())
    cutoff = min(datetime.fromisoformat(source_request["created_at_utc"]),
                 datetime.fromisoformat(capture["metadata"]["lastRunTime"]))
    summary = json.loads((AUDIT / "current-dataset-summary.json").read_text())
    data = []
    files = {}
    for row in summary:
        name = row["ref"].replace("/", "--") + "--current.json"
        path = AUDIT / name
        if digest(path) != row["metadata_sha256"]:
            raise ValueError("Observed dataset metadata changed")
        item = json.loads(path.read_text())
        if item["ref"] != row["ref"] or item["id"] != row["identity"]["id"]:
            raise ValueError("Observed dataset identity differs")
        data.append(item)
        files[name] = digest(path)
    file_summary = json.loads((AUDIT / "pinned-file-summary.json").read_text())
    if len(file_summary) != len(data) or {row["ref"] for row in file_summary} != {row["ref"] for row in data}:
        raise ValueError("Pinned file inventory membership differs")
    selected = set()
    by_ref = {item["ref"]: item for item in data}
    for row in file_summary:
        name = row["ref"].replace("/", "--") + "--pinned-files.json"
        if digest(AUDIT / name) != row["metadata_sha256"] or row["version"] != by_ref[row["ref"]]["currentVersionNumber"]:
            raise ValueError("Pinned file inventory changed")
        files[name] = digest(AUDIT / name)
        selected.update(entry["name"] for entry in row["selected_files"])
    if not {"raptor_ft_coatnet_v5_full_swa.pt", "raptor_ft_coatnet_v4_full.pt"}.issubset(selected):
        raise ValueError("Exact original Raptor filenames are missing")
    for name in ("current-dataset-summary.json", "pinned-file-summary.json"):
        files[name] = digest(AUDIT / name)
    expected = pinned_metadata(metadata, data, cutoff, capture["metadata"]["dockerImage"])
    expected.update(id="jitheswar/rsna-exact-scored-anchor-gold58-v2")
    return parent, expected, files, cutoff


def prepare():
    if DEST.exists():
        raise SystemExit("Immutable pinned-envelope preparation already exists")
    parent, metadata, files, cutoff = inputs()
    DEST.mkdir()
    for name in ("development.ipynb", "NOTICE.md", "LICENSE-APACHE-2.0.txt"):
        shutil.copyfile(PARENT / name, DEST / name)
    shutil.copytree(PARENT / "source-snapshot", DEST / "source-snapshot")
    shutil.copyfile(Path(__file__), DEST / "source-snapshot" / Path(__file__).name)
    (DEST / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    request = dict(parent)
    request.update(schema_version="rsna-ticket70-exact-scored-anchor-pinned-envelope-v2",
        created_at_utc=datetime.now(timezone.utc).isoformat(), kernel_id=metadata["id"],
        parent_request_sha256=PARENT_REQUEST_SHA, supersedes=str(PARENT),
        metadata_sha256=digest(DEST / "kernel-metadata.json"),
        dataset_version_pins=metadata["dataset_sources"], docker_image=metadata["docker_image"],
        historical_input_cutoff_utc=cutoff.isoformat(), asset_lineage_files=files,
        asset_lineage_basis="Authenticated current version histories and file inventories predate the earliest saved baseline run/request; latest-version attachment lineage is inferred, not historical byte hashes",
        resource_decision_status="Prepared pinned envelope only; no cap amendment, reservation, launch or scoring",
        launch_gate=parent["launch_gate"] + "; verify remote explicit version/container settings and mounted input/weight hashes before accepting predictions",
        frozen_files={str(p.relative_to(DEST)):digest(p) for p in (DEST / "source-snapshot").iterdir()})
    (DEST / "request.json").write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    (DEST / "request-checksum.txt").write_text(digest(DEST / "request.json") + "\n")
    print(json.dumps({"request_sha256":digest(DEST / "request.json"), "root":str(DEST), "launched":False}))


def verify():
    parent, metadata, files, cutoff = inputs()
    request = json.loads((DEST / "request.json").read_text())
    if digest(DEST / "request.json") != (DEST / "request-checksum.txt").read_text().strip():
        raise ValueError("Pinned request bytes changed")
    if json.loads((DEST / "kernel-metadata.json").read_text()) != metadata or digest(DEST / "kernel-metadata.json") != request["metadata_sha256"]:
        raise ValueError("Explicit pinned source envelope differs")
    if (digest(DEST / "development.ipynb") != parent["notebook_sha256"]
            or request["notebook_sha256"] != parent["notebook_sha256"]
            or request["source_cells"] != parent["source_cells"]
            or request["dataset_version_pins"] != metadata["dataset_sources"]
            or request["kernel_id"] != metadata["id"]
            or request["docker_image"] != metadata["docker_image"]
            or request["asset_lineage_files"] != files
            or request["historical_input_cutoff_utc"] != cutoff.isoformat()
            or request["parent_request_sha256"] != PARENT_REQUEST_SHA):
        raise ValueError("Pinned preparation changed recipe or lineage")
    for key in ("assigned_gpu_count", "session_timeout_seconds", "execution_watchdog_seconds",
                "proposed_reserved_gpu_seconds", "maximum_launch_attempts", "model_fitting",
                "competition_submission", "final_selection", "external_spend_usd"):
        if request[key] != parent[key]:
            raise ValueError("Pinned envelope changes scientific/resource scope")
    for name, expected in request["frozen_files"].items():
        if digest(DEST / name) != expected:
            raise ValueError("Pinned frozen source changed")
    record = dict(status="PINNED_ENVELOPE_PREPARATION_VALIDATED", request_sha256=digest(DEST / "request.json"),
        parent_request_sha256=PARENT_REQUEST_SHA, unchanged_notebook_bytes=True, dataset_versions_verified=7,
        saved_container_pinned=True, historical_weight_byte_parity_proven=False,
        original_mri_predictions_verified=False, launched=False, resource_decision_admitted=False)
    path = DEST / "preparation-validation.json"
    serialized = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != serialized:
        raise ValueError("Immutable envelope validation differs")
    if not path.exists(): path.write_text(serialized)
    print(json.dumps(record))


if __name__ == "__main__":
    if not sys.argv[1:]: prepare()
    elif sys.argv[1:] == ["--verify"]: verify()
    else: raise SystemExit("Use no arguments to prepare, or --verify")
