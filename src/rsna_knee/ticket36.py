"""Expanded-cohort, three-fold Ticket 36 report-supervision comparison.

The workflow freezes the data, label and compute identities before loading the
pretrained encoder.  Feature materialization is restartable by Ticket 35 image
batch; training then reuses the existing fold trainer and OOF comparator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import torch

from .archive import atomic_json_write
from .domain import SeriesRecord, StudyRecord
from .evaluation import OOFEvaluation
from .feature_cache import FEATURE_CACHE_SCHEMA_VERSION
from .folds import FOLD_IDS, FoldManifest, load_fold_manifest
from .image_variant import (
    TICKET31_RESNET18_VARIANT,
    image_model_training_config,
)
from .model import ANATOMICAL_POOLING_ID, build_study_model
from .preprocessing import PreparedSeries, pack_study, prepare_study
from .pretrained import (
    RESNET18_ARTIFACT_BYTES,
    RESNET18_DOWNLOAD_URL,
    RESNET18_SHA256,
    RESNET18_SOURCE_REVISION,
    TICKET36_RESOURCE_SCOPE,
    load_resnet18_weights,
    load_resource_approval,
)
from .real_inference import (
    IntegrationManifest,
    IntegrationSeries,
    IntegrationStudy,
    iter_real_studies,
    load_integration_manifest,
    write_integration_manifest,
)
from .resnet18 import RESNET18_FEATURE_DIM, RESNET18_MODEL_ID, ResNet18StudyModel
from .series_cache import SeriesCache
from .supervision import AdmittedFoldSupervision
from .targets import TARGET_COLUMNS
from .ticket34 import (
    TICKET30_ADMISSION_POLICY_SHA256,
    TICKET30_AUDITED_ARTIFACT_SHA256,
    TICKET30_SOURCE_ID,
    TICKET34_ADDITIVE_DESIGN,
    TICKET34_WEAK_SAMPLING_POLICY,
    TICKET34_WEAK_BATCH_SEED_OFFSET,
    Ticket34Error,
    admit_ticket30_fold_supervision,
)
from .ticket35 import (
    TICKET35_BATCH_EVIDENCE_SCHEMA_VERSION,
    Ticket35Error,
    capture_ticket35_storage_ledger,
    validate_ticket35_membership,
)
from .training import (
    CpuPreflightResult,
    PreflightError,
    SupervisionProvenance,
    TrainingConfig,
    TrainingExample,
    TrainingResult,
    load_training_checkpoint,
    require_cpu_preflight,
    run_cpu_preflight,
    train_one_fold,
)


PROJECT_ROOT = Path(
    os.environ.get("RSNA_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
RUN_ROOT = PROJECT_ROOT / "artifacts/private/ticket36-expanded-supervision"
PLAN_PATH = RUN_ROOT / "ticket36-run-plan.json"
FEATURE_ROOT = RUN_ROOT / "features"
INTEGRATION_PATH = RUN_ROOT / "private/expanded-integration-manifest.json"
SERIES_INDEX_PATH = RUN_ROOT / "private/ticket35-series-index.json"
FOLD_MANIFEST_PATH = PROJECT_ROOT / "artifacts/private/ticket-04-real-v2/private/fold-manifest.json"
REFERENCE_INTEGRATION_PATH = (
    PROJECT_ROOT / "artifacts/private/ticket-05-real-training-input/private/integration-manifest.json"
)
TICKET35_ROOT = PROJECT_ROOT / "artifacts/private/ticket35-expanded-image-preparation"
TICKET35_MEMBERSHIP_PATH = TICKET35_ROOT / "membership-v2.json"
TICKET35_SERIES_CACHE = TICKET35_ROOT / "series-cache"
TICKET30_ARTIFACT = Path(
    "/home/ch.jitheswar/ticket-30-report-audit-20260925/private/admitted-report-labels-v4.json"
)
SOURCE_ARCHIVE = PROJECT_ROOT / "rsna-knee-abnormality-detection.zip"
WEIGHTS_PATH = PROJECT_ROOT / "artifacts/private/ticket31-resnet18/weights/resnet18-f37072fd.pth"
APPROVAL_PATH = PROJECT_ROOT / ".scratch/rsna-leaderboard/resource-approvals/ticket36-resnet18-imagenet1k-v1.json"

TICKET36_PLAN_SCHEMA = "rsna-ticket36-expanded-supervision-plan-v1"
TICKET36_FEATURE_BATCH_SCHEMA = "rsna-ticket36-frozen-feature-batch-v1"
TICKET36_FEATURE_MANIFEST_SCHEMA = "rsna-ticket36-frozen-feature-manifest-v1"
TICKET36_APPROVAL_SCHEMA = "rsna-ticket36-resource-approval-v1"
TICKET36_CODE_SCHEMA = "rsna-ticket36-source-manifest-v1"
TICKET36_WEAK_FEATURE_SOURCE = TICKET30_SOURCE_ID
TICKET36_IMAGE_SIZE = (224, 224)
TICKET36_REPRESENTATION_ID = "ticket31-resnet18-224-whole-field-v1"
TICKET36_BOOTSTRAP_RESAMPLES = 2000
TICKET36_BOOTSTRAP_SEED = 20260926
_RESOURCE_REQUIREMENTS: Mapping[str, object] = {
    "partition": "cluster1-cpu",
    "nodes": 1,
    "ntasks": 1,
    "cpus_per_task": 16,
    "memory_gb": 64,
    "wall_time_minutes": 360,
    "exclusive": False,
}
_REQUESTED_RESOURCES: Mapping[str, object] = {
    "requested_partition": "cluster1-cpu",
    "requested_nodes": 1,
    "requested_cpus": 16,
    "requested_memory_gb": 64,
    "requested_wall_time_hours": 6,
    "exclusive": False,
    "maximum_concurrent_jobs": 2,
    "allocated_gpu_count": 0,
    "external_spend": 0,
}


class Ticket36Error(ValueError):
    """Raised when the frozen Ticket 36 cohort or execution is inconsistent."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uid_set_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Ticket36Error(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _read_json(path: str | Path, field: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Ticket36Error(f"could not read {field}: {path}") from exc
    return _mapping(value, field)


def _private_directory(path: Path) -> None:
    if path.is_symlink():
        raise Ticket36Error(f"private output directory must not be a symlink: {path}")
    existed = path.exists()
    if existed and not path.is_dir():
        raise Ticket36Error(f"private output path must be a directory: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if existed and path.stat().st_mode & 0o077:
        raise Ticket36Error(f"private output directory has group or world access: {path}")


def _write_private_json(path: Path, payload: Mapping[str, object]) -> None:
    _private_directory(path.parent)
    if path.is_symlink():
        raise Ticket36Error(f"private JSON destination must not be a symlink: {path}")
    if path.exists():
        if not path.is_file() or _read_json(path, path.name) != payload:
            raise Ticket36Error(f"refusing to replace different frozen evidence: {path}")
        return
    atomic_json_write(path, dict(payload))
    path.chmod(0o600)


def _replace_private_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically replace a private, explicitly mutable measurement record."""

    _private_directory(path.parent)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise Ticket36Error(f"mutable private JSON destination is unsafe: {path}")
    atomic_json_write(path, dict(payload), overwrite=True)
    path.chmod(0o600)


def _write_private_text(path: Path, value: str) -> None:
    _private_directory(path.parent)
    if path.is_symlink():
        raise Ticket36Error(f"private text destination must not be a symlink: {path}")
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != value:
            raise Ticket36Error(f"refusing to replace different frozen text evidence: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        path.chmod(0o600)
    except FileExistsError:
        if not path.is_file() or path.read_text(encoding="utf-8") != value:
            raise Ticket36Error(f"concurrent writer produced different text evidence: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _replace_private_text(path: Path, value: str) -> None:
    """Atomically replace a private report that is refreshed after finalization."""

    _private_directory(path.parent)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise Ticket36Error(f"mutable private text destination is unsafe: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_checksum_payload(
    payload: Mapping[str, object], checksum_field: str, description: str
) -> str:
    checksum = payload.get(checksum_field)
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise Ticket36Error(f"{description} has no SHA-256 checksum")
    unsealed = dict(payload)
    unsealed.pop(checksum_field, None)
    if _json_sha256(unsealed) != checksum:
        raise Ticket36Error(f"{description} checksum does not match its contents")
    return checksum


def _code_manifest(source_root: Path | None = None) -> dict[str, object]:
    root = (source_root or PROJECT_ROOT).resolve()
    source_files = sorted((root / "src/rsna_knee").glob("*.py"))
    source_files.extend(
        [
            root / "slurm/materialize_ticket36_features.sbatch",
            root / "slurm/run_ticket36_comparison.sbatch",
        ]
    )
    rows = []
    for path in source_files:
        if path.is_symlink() or not path.is_file():
            raise Ticket36Error(f"Ticket 36 source file is missing or unsafe: {path}")
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "schema_version": TICKET36_CODE_SCHEMA,
        "files": rows,
    }
    payload["source_manifest_sha256"] = _json_sha256(payload)
    return payload


def _create_frozen_source_snapshot() -> tuple[dict[str, object], Path]:
    """Copy the exact Python and Slurm source tree used by this experiment."""

    source_manifest = _code_manifest(PROJECT_ROOT)
    manifest_sha256 = str(source_manifest["source_manifest_sha256"])
    snapshot_parent = RUN_ROOT / "private/source-snapshots"
    _private_directory(snapshot_parent)
    snapshot_root = snapshot_parent / manifest_sha256
    if snapshot_root.is_symlink():
        raise Ticket36Error("Ticket 36 source snapshot must not be a symlink")
    if snapshot_root.exists():
        if not snapshot_root.is_dir() or _code_manifest(snapshot_root) != source_manifest:
            raise Ticket36Error("existing Ticket 36 source snapshot has different contents")
        return source_manifest, snapshot_root

    staging_root = Path(
        tempfile.mkdtemp(prefix=".ticket36-source-", dir=snapshot_parent)
    )
    try:
        for row in cast(Sequence[Mapping[str, object]], source_manifest["files"]):
            relative_path = Path(str(row["path"]))
            source = PROJECT_ROOT / relative_path
            destination = staging_root / relative_path
            if source.is_symlink() or not source.is_file():
                raise Ticket36Error("Ticket 36 source changed while it was being snapshotted")
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(source, destination)
            destination.chmod(0o600)
        if _code_manifest(PROJECT_ROOT) != source_manifest:
            raise Ticket36Error("Ticket 36 source files changed while the snapshot was being created")
        if _code_manifest(staging_root) != source_manifest:
            raise Ticket36Error("Ticket 36 source snapshot does not match its frozen manifest")
        os.replace(staging_root, snapshot_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    snapshot_root.chmod(0o700)
    return source_manifest, snapshot_root


def _make_training_config(integration: IntegrationManifest) -> TrainingConfig:
    config = image_model_training_config(
        TICKET31_RESNET18_VARIANT,
        preprocessing_config=integration.preprocess_config,
        packing_config=integration.packing_config,
        pooling_id=ANATOMICAL_POOLING_ID,
    )
    return replace(
        config,
        seed=20260911,
        batch_order_seed=20260912,
        batch_size=2,
        successful_optimizer_steps=100,
        device="cpu",
        selection_policy="final_checkpoint",
    )


def _approval_payload(plan: Mapping[str, object]) -> Mapping[str, object]:
    plan_checksum = _verify_checksum_payload(plan, "plan_sha256", "Ticket 36 plan")
    approval = _read_json(APPROVAL_PATH, "Ticket 36 resource approval")
    expected_resources = {
        "partition": "cluster1-cpu",
        "nodes": 1,
        "tasks": 1,
        "cpus_per_task": 16,
        "memory_gib": 64,
        "walltime_seconds": 21600,
        "maximum_concurrent_jobs": 2,
        "gpu_count": 0,
        "external_spend_usd": 0,
    }
    resources = _mapping(approval.get("resources"), "approval.resources")
    if (
        approval.get("schema_version") != TICKET36_APPROVAL_SCHEMA
        or approval.get("scope") != TICKET36_RESOURCE_SCOPE
        or approval.get("decision") != "approved"
        or approval.get("approval_status") != "approved_by_project_owner"
        or approval.get("plan_sha256") != plan_checksum
        or approval.get("checkpoint_path") != str(WEIGHTS_PATH.resolve())
        or approval.get("download_url") != RESNET18_DOWNLOAD_URL
        or approval.get("checkpoint_sha256") != RESNET18_SHA256
        or approval.get("checkpoint_bytes") != RESNET18_ARTIFACT_BYTES
        or approval.get("ticket30_artifact_sha256") != TICKET30_AUDITED_ARTIFACT_SHA256
        or approval.get("ticket35_membership_checksum") != plan.get("ticket35_membership_checksum")
        or approval.get("fold_manifest_checksum") != plan.get("fold_manifest_checksum")
        or approval.get("integration_manifest_checksum") != plan.get("integration_manifest_checksum")
        or approval.get("feature_manifest_identity_fingerprint")
        != plan.get("feature_identity_fingerprint")
        or approval.get("external_spend_usd") != 0
        or approval.get("gpu_count") != 0
        or approval.get("network_during_execution") is not False
        or approval.get("new_weight_download") is not False
        or approval.get("candidate_promotion") is not False
        or approval.get("final_candidate_freeze") is not False
        or approval.get("release_actions") != "blocked"
        or dict(resources) != expected_resources
        or not isinstance(approval.get("approved_by"), str)
        or not isinstance(approval.get("approved_at"), str)
    ):
        raise Ticket36Error("Ticket 36 approval does not match the exact frozen CPU-only scope")
    loaded = load_resource_approval(APPROVAL_PATH)
    if loaded.get("scope") != TICKET36_RESOURCE_SCOPE:
        raise Ticket36Error("pretrained checkpoint approval does not name Ticket 36")
    if (
        not WEIGHTS_PATH.is_file()
        or WEIGHTS_PATH.is_symlink()
        or WEIGHTS_PATH.stat().st_size != RESNET18_ARTIFACT_BYTES
        or _sha256_file(WEIGHTS_PATH) != RESNET18_SHA256
    ):
        raise Ticket36Error("the already acquired ResNet-18 checkpoint failed exact verification")
    return approval


def _load_plan(path: Path = PLAN_PATH) -> Mapping[str, object]:
    plan = _read_json(path, "Ticket 36 run plan")
    if plan.get("schema_version") != TICKET36_PLAN_SCHEMA or plan.get("ticket") != 36:
        raise Ticket36Error("the run plan is not the frozen Ticket 36 plan")
    _verify_checksum_payload(plan, "plan_sha256", "Ticket 36 plan")
    if (
        plan.get("source_version") != "rsna-2026-08-04"
        or plan.get("fold_manifest_checksum") is None
        or plan.get("ticket30_artifact_sha256") != TICKET30_AUDITED_ARTIFACT_SHA256
    ):
        raise Ticket36Error("Ticket 36 frozen input identity is unsupported")
    expected_code = _mapping(plan.get("source_code_manifest"), "Ticket 36 source code manifest")
    manifest_sha256 = expected_code.get("source_manifest_sha256")
    if not isinstance(manifest_sha256, str):
        raise Ticket36Error("Ticket 36 source code manifest has no source identity")
    snapshot_root = RUN_ROOT / "private/source-snapshots" / manifest_sha256
    if (
        plan.get("source_snapshot_path") != str(snapshot_root.resolve())
        or not snapshot_root.is_dir()
        or snapshot_root.is_symlink()
        or _code_manifest(snapshot_root) != dict(expected_code)
        or (
            os.environ.get("RSNA_TICKET36_SOURCE_ROOT") is not None
            and Path(os.environ["RSNA_TICKET36_SOURCE_ROOT"]).resolve()
            != snapshot_root.resolve()
        )
        or (
            os.environ.get("RSNA_CODE_REVISION") is not None
            and os.environ["RSNA_CODE_REVISION"]
            != f"ticket36-source-{manifest_sha256}"
        )
    ):
        raise Ticket36Error("Ticket 36 frozen source snapshot is missing or changed")
    return plan


def _verify_slurm_allocation() -> None:
    torch.set_num_threads(16)
    expected = {
        "SLURM_JOB_PARTITION": "cluster1-cpu",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "16",
    }
    if not os.environ.get("SLURM_JOB_ID"):
        raise PreflightError("Ticket 36 real-data work must run in Slurm")
    if any(os.environ.get(name) != value for name, value in expected.items()):
        raise PreflightError("Ticket 36 Slurm allocation differs from the frozen CPU request")
    if os.environ.get("SLURM_MEM_PER_NODE", "").strip().upper() not in {"64G", "65536"}:
        raise PreflightError("Ticket 36 Slurm allocation does not provide exactly 64 GiB")
    if os.environ.get("SLURM_TIMELIMIT", "").strip() not in {"360", "06:00", "06:00:00"}:
        raise PreflightError("Ticket 36 Slurm allocation does not provide the six-hour limit")
    if os.environ.get("SLURM_JOB_EXCLUSIVE", "0").strip().lower() in {"1", "true", "yes"}:
        raise PreflightError("Ticket 36 allocation must not request exclusive-node use")


def _read_membership(path: Path = TICKET35_MEMBERSHIP_PATH) -> Mapping[str, object]:
    return validate_ticket35_membership(_read_json(path, "Ticket 35 membership"))


def _validate_t35_batch_evidence(
    membership: Mapping[str, object], batch_index: int
) -> tuple[Mapping[str, object], IntegrationManifest, Path, tuple[str, ...]]:
    batches = membership.get("batches")
    if not isinstance(batches, list) or batch_index < 0 or batch_index >= len(batches):
        raise Ticket36Error("Ticket 36 feature batch is outside the frozen Ticket 35 plan")
    planned = _mapping(batches[batch_index], "Ticket 35 batch")
    if planned.get("batch_index") != batch_index:
        raise Ticket36Error("Ticket 35 batch indices are not canonical")
    batch_dir = TICKET35_ROOT / "batches" / f"batch-{batch_index}"
    evidence_paths = sorted(batch_dir.glob("job-*-inference-evidence.json"))
    completed: list[tuple[Mapping[str, object], Path]] = []
    for evidence_path in evidence_paths:
        evidence = _read_json(evidence_path, "Ticket 35 batch inference evidence")
        if evidence.get("status") not in {"completed", "completed_with_exclusions"}:
            continue
        if evidence.get("schema_version") != TICKET35_BATCH_EVIDENCE_SCHEMA_VERSION:
            continue
        if evidence.get("ticket35_membership_checksum") != membership.get("manifest_checksum"):
            continue
        if evidence.get("batch_index") != batch_index:
            continue
        completed.append((evidence, evidence_path))
    if len(completed) != 1:
        raise Ticket36Error(
            f"Ticket 35 batch {batch_index} must have exactly one terminal inference record"
        )
    evidence, evidence_path = completed[0]
    manifest_path_value = evidence.get("integration_manifest_path")
    if not isinstance(manifest_path_value, str):
        raise Ticket36Error("Ticket 35 batch evidence has no integration manifest path")
    manifest_path = Path(manifest_path_value)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise Ticket36Error("Ticket 35 batch integration manifest is missing or unsafe")
    integration = load_integration_manifest(manifest_path)
    expected_uids_value = planned.get("study_uids")
    if not isinstance(expected_uids_value, list):
        raise Ticket36Error("Ticket 35 batch study membership is malformed")
    expected_uids = tuple(str(item) for item in expected_uids_value)
    study_rows_value = evidence.get("studies")
    if not isinstance(study_rows_value, list):
        raise Ticket36Error("Ticket 35 batch evidence has no per-study input reconciliation")
    study_rows = tuple(_mapping(item, "Ticket 35 per-study evidence") for item in study_rows_value)
    observed_uids = tuple(
        str(row.get("study_instance_uid"))
        for row in study_rows
        if isinstance(row.get("study_instance_uid"), str)
    )
    if (
        len(observed_uids) != len(study_rows)
        or set(observed_uids) != set(expected_uids)
        or len(observed_uids) != len(set(observed_uids))
    ):
        raise Ticket36Error("Ticket 35 per-study evidence does not reconcile with batch membership")
    eligible_uids: list[str] = []
    excluded_uids: list[str] = []
    for row in study_rows:
        uid = str(row["study_instance_uid"])
        exclusions = row.get("input_exclusions")
        complete_input = row.get("complete_input")
        if not isinstance(exclusions, list) or not isinstance(complete_input, bool):
            raise Ticket36Error("Ticket 35 per-study input completeness evidence is malformed")
        if complete_input and not exclusions:
            eligible_uids.append(uid)
        else:
            excluded_uids.append(uid)
    expected_status = "completed_with_exclusions" if excluded_uids else "completed"
    if evidence.get("status") != expected_status:
        raise Ticket36Error("Ticket 35 batch status disagrees with per-study input exclusions")
    decoded_slice_count = evidence.get("decoded_slice_count")
    expected_slice_count = evidence.get("expected_slice_count")
    summed_decoded = sum(int(cast(Any, row.get("decoded_slice_count", 0))) for row in study_rows)
    summed_expected = sum(int(cast(Any, row.get("expected_slice_count", 0))) for row in study_rows)
    if (
        integration.study_uids != expected_uids
        or evidence.get("integration_manifest_checksum") != integration.manifest_checksum
        or evidence.get("study_count") != len(expected_uids)
        or evidence.get("prediction_count") != len(expected_uids)
        or evidence.get("series_count") != planned.get("series_count")
        or decoded_slice_count != summed_decoded
        or expected_slice_count != summed_expected
        or expected_slice_count != planned.get("effective_frame_count")
        or not 0 <= int(cast(Any, decoded_slice_count)) <= int(cast(Any, expected_slice_count))
        or (not excluded_uids and decoded_slice_count != expected_slice_count)
        or integration.archive_fingerprint
        != _mapping(membership.get("source_identity"), "Ticket 35 source identity").get("archive_fingerprint")
    ):
        raise Ticket36Error(f"Ticket 35 batch {batch_index} evidence does not reconcile")
    return evidence, integration, evidence_path, tuple(sorted(eligible_uids))


def _verify_frozen_t35_batch_evidence(
    plan: Mapping[str, object],
    batch_index: int,
    evidence: Mapping[str, object],
    integration: IntegrationManifest,
    evidence_path: Path,
) -> None:
    """Require live Ticket 35 files to retain the identities frozen in the plan."""

    rows = plan.get("ticket35_batch_evidence")
    if not isinstance(rows, list) or batch_index < 0 or batch_index >= len(rows):
        raise Ticket36Error("Ticket 36 plan is missing frozen Ticket 35 batch evidence")
    frozen = _mapping(rows[batch_index], "frozen Ticket 35 batch evidence")
    integration_path_value = evidence.get("integration_manifest_path")
    if not isinstance(integration_path_value, str):
        raise Ticket36Error("Ticket 35 batch evidence has no integration manifest path")
    integration_path = Path(integration_path_value)
    if (
        frozen.get("batch_index") != batch_index
        or frozen.get("inference_evidence_path") != str(evidence_path.resolve())
        or frozen.get("inference_evidence_sha256") != _sha256_file(evidence_path)
        or frozen.get("integration_manifest_path") != str(integration_path.resolve())
        or frozen.get("integration_manifest_file_sha256") != _sha256_file(integration_path)
        or frozen.get("integration_manifest_checksum") != integration.manifest_checksum
        or frozen.get("study_count") != len(integration.studies)
        or frozen.get("prediction_count") != evidence.get("prediction_count")
        or frozen.get("series_count") != evidence.get("series_count")
        or frozen.get("decoded_slice_count") != evidence.get("decoded_slice_count")
    ):
        raise Ticket36Error("Ticket 35 batch evidence changed after the Ticket 36 plan was frozen")


def _scan_ticket35_series_cache(
    report_series: Mapping[str, IntegrationSeries],
    preprocess_config: Any,
) -> dict[str, object]:
    if not TICKET35_SERIES_CACHE.is_dir() or TICKET35_SERIES_CACHE.is_symlink():
        raise Ticket36Error("Ticket 35 prepared-series cache is missing or unsafe")
    cache = SeriesCache(TICKET35_SERIES_CACHE)
    rows: dict[str, object] = {}
    for metadata_path in sorted(cache.entries_root.glob("*/*/metadata.json")):
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise Ticket36Error("Ticket 35 cache contains a linked or non-regular metadata file")
        metadata = cache._read_metadata(metadata_path.parent)
        if metadata.get("state") != "complete":
            continue
        source = _mapping(metadata.get("source"), "series cache source")
        study_uid = source.get("study_instance_uid")
        series_uid = source.get("series_instance_uid")
        if not isinstance(study_uid, str) or not isinstance(series_uid, str):
            raise Ticket36Error("Ticket 35 cache source identity is malformed")
        if series_uid not in report_series:
            continue
        expected = report_series[series_uid]
        slices = source.get("slices")
        if not isinstance(slices, list):
            raise Ticket36Error("Ticket 35 cache source slices are malformed")
        source_entries = []
        for item in slices:
            row = _mapping(item, "cached source slice")
            source_identity = row.get("source_identity")
            content_digest = row.get("content_sha256")
            sop_uid = row.get("sop_instance_uid")
            if not all(isinstance(value, str) and value for value in (source_identity, content_digest, sop_uid)):
                raise Ticket36Error("Ticket 35 cache source slice identity is incomplete")
            if len(cast(str, content_digest)) != 64:
                raise Ticket36Error("Ticket 35 cache source content digest is invalid")
            source_entries.append(
                {
                    "sop_instance_uid": sop_uid,
                    "source_identity": source_identity,
                    "content_sha256": content_digest,
                }
            )
        source_entries.sort(key=lambda item: str(item["sop_instance_uid"]))
        expected_entries = tuple(sorted(expected.entries))
        observed_entries = tuple(str(item["source_identity"]) for item in source_entries)
        array = _mapping(metadata.get("array"), "series cache array")
        array_path = metadata_path.parent / str(array.get("filename", ""))
        if (
            study_uid != expected.study_instance_uid
            or metadata.get("preprocessing") != preprocess_config.cache_config()
            or observed_entries != expected_entries
            or metadata_path.parent.name != metadata.get("cache_key")
            or not array_path.is_file()
            or array_path.is_symlink()
            or array_path.stat().st_size != array.get("bytes")
            or len(source_entries) != len(expected.entries)
        ):
            raise Ticket36Error(f"Ticket 35 cache identity differs for series {series_uid}")
        row_payload: dict[str, object] = {
            "study_instance_uid": study_uid,
            "series_instance_uid": series_uid,
            "cache_key": metadata_path.parent.name,
            "metadata_sha256": _sha256_file(metadata_path),
            "array_sha256": array.get("sha256"),
            "array_bytes": array.get("bytes"),
            "source_payload_sha256": _json_sha256(source_entries),
            "source_entry_names_sha256": _json_sha256(list(observed_entries)),
            "entry_count": len(source_entries),
        }
        if series_uid in rows:
            raise Ticket36Error("Ticket 35 cache repeats a source series")
        rows[series_uid] = row_payload
    if set(rows) != set(report_series):
        raise Ticket36Error(
            f"Ticket 35 series cache does not cover the expanded cohort: missing {len(set(report_series) - set(rows))}"
        )
    index: dict[str, object] = {
        "schema_version": "rsna-ticket36-ticket35-series-index-v1",
        "source_cache_root": str(TICKET35_SERIES_CACHE.resolve()),
        "series_count": len(rows),
        "entries": dict(sorted(rows.items())),
    }
    index["index_sha256"] = _json_sha256(index)
    return index


def _freeze_feature_batches(
    batch_rows: Sequence[Mapping[str, object]],
    gold_uids: Sequence[str],
    expected_report_uids: Sequence[str],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, row in enumerate(batch_rows):
        uids = row.get("study_uids")
        if not isinstance(uids, list):
            raise Ticket36Error("Ticket 35 batch study set is malformed")
        result.append(
            {
                "batch_index": index,
                "kind": "report_only",
                "study_uids": [str(uid) for uid in uids],
                "ticket35_batch_index": index,
            }
        )
    result.append(
        {
            "batch_index": len(result),
            "kind": "reference_gold",
            "study_uids": sorted(gold_uids),
            "ticket35_batch_index": None,
        }
    )
    all_uids = [
        str(uid)
        for batch in result
        for uid in cast(Sequence[object], batch["study_uids"])
    ]
    if (
        set(all_uids) != set(expected_report_uids) | set(gold_uids)
        or len(all_uids) != len(set(all_uids))
    ):
        raise Ticket36Error("Ticket 36 materialization batches do not exactly cover its feature cohort")
    return result


def freeze_ticket36_plan() -> Mapping[str, object]:
    """Freeze all scientific inputs and resource limits before feature extraction."""

    _private_directory(RUN_ROOT)
    _private_directory(RUN_ROOT / "private")
    _private_directory(FEATURE_ROOT)
    _private_directory(RUN_ROOT / "compute")
    membership = _read_membership()
    fold_manifest = load_fold_manifest(FOLD_MANIFEST_PATH)
    base = load_integration_manifest(REFERENCE_INTEGRATION_PATH)
    if fold_manifest.superseded:
        raise Ticket36Error("Ticket 36 cannot use a superseded fold manifest")
    source_identity = _mapping(membership.get("source_identity"), "Ticket 35 source identity")
    frozen_inputs = {
        "fold_manifest_checksum": fold_manifest.manifest_checksum,
        "reference_integration_manifest_checksum": base.manifest_checksum,
        "ticket35_membership_checksum": membership.get("manifest_checksum"),
        "ticket35_membership_file_sha256": _sha256_file(TICKET35_MEMBERSHIP_PATH),
        "ticket30_artifact_sha256": _sha256_file(TICKET30_ARTIFACT),
        "source_fingerprint": source_identity.get("archive_fingerprint"),
        "source_version": source_identity.get("source_version"),
    }
    if (
        source_identity.get("fold_manifest_checksum") != fold_manifest.manifest_checksum
        or source_identity.get("integration_manifest_sha256") != _sha256_file(REFERENCE_INTEGRATION_PATH)
        or source_identity.get("label_artifact_sha256") != TICKET30_AUDITED_ARTIFACT_SHA256
        or frozen_inputs["ticket30_artifact_sha256"] != TICKET30_AUDITED_ARTIFACT_SHA256
        or base.archive_fingerprint != source_identity.get("archive_fingerprint")
        or base.source_version != source_identity.get("source_version")
        or not SOURCE_ARCHIVE.is_file()
    ):
        raise Ticket36Error("Ticket 36 source, fold or label lineage differs from Ticket 35")
    if (
        not WEIGHTS_PATH.is_file()
        or WEIGHTS_PATH.is_symlink()
        or WEIGHTS_PATH.stat().st_size != RESNET18_ARTIFACT_BYTES
        or _sha256_file(WEIGHTS_PATH) != RESNET18_SHA256
    ):
        raise Ticket36Error("the existing ResNet-18 checkpoint is not the exact frozen artifact")
    expected_gold_uids = {
        item.study_instance_uid for item in fold_manifest.gold_memberships
    }
    gold_studies = tuple(
        item for item in base.studies if item.study_instance_uid in expected_gold_uids
    )
    if {item.study_instance_uid for item in gold_studies} != expected_gold_uids:
        raise Ticket36Error("reference integration is missing one or more fold-manifest gold studies")
    batch_rows_value = membership.get("batches")
    if not isinstance(batch_rows_value, list) or len(batch_rows_value) != 131:
        raise Ticket36Error("Ticket 36 requires the frozen 131-batch Ticket 35 image cohort")
    report_studies: list[IntegrationStudy] = []
    report_series: dict[str, IntegrationSeries] = {}
    first_report_manifest: IntegrationManifest | None = None
    batch_inputs: list[dict[str, object]] = []
    feature_batch_rows: list[dict[str, object]] = []
    input_excluded_report_uids: list[str] = []
    observed_ticket35_report_uids: list[str] = []
    eligible_report_decoded_slices = 0
    for index in range(len(batch_rows_value)):
        evidence, integration, evidence_path, eligible_uids = _validate_t35_batch_evidence(
            membership, index
        )
        if first_report_manifest is None:
            first_report_manifest = integration
        if (
            integration.geometry_policy != base.geometry_policy
            or integration.decoder_policy != base.decoder_policy
            or integration.packing_config != base.packing_config
            or integration.preprocess_config.image_size != TICKET36_IMAGE_SIZE
            or integration.preprocess_config.representation_id != TICKET36_REPRESENTATION_ID
            or integration.source_version != base.source_version
            or integration.archive_fingerprint != base.archive_fingerprint
        ):
            raise Ticket36Error("Ticket 35 image batch does not match the frozen ResNet representation")
        planned_batch = _mapping(batch_rows_value[index], "Ticket 35 batch")
        planned_uids_value = planned_batch.get("study_uids")
        if not isinstance(planned_uids_value, list):
            raise Ticket36Error("Ticket 35 batch study membership is malformed")
        planned_uids = tuple(str(uid) for uid in planned_uids_value)
        if integration.study_uids != planned_uids:
            raise Ticket36Error("Ticket 35 batch integration does not match its complete planned cohort")
        eligible_uid_set = set(eligible_uids)
        observed_ticket35_report_uids.extend(planned_uids)
        input_excluded_report_uids.extend(sorted(set(planned_uids) - eligible_uid_set))
        feature_batch_rows.append({"study_uids": list(eligible_uids)})
        eligible_report_decoded_slices += sum(
            int(cast(Any, row.get("decoded_slice_count", 0)))
            for row_value in cast(Sequence[object], evidence.get("studies", ()))
            for row in (_mapping(row_value, "Ticket 35 per-study evidence"),)
            if row.get("study_instance_uid") in eligible_uid_set
        )
        for study in integration.studies:
            if study.study_instance_uid in expected_gold_uids:
                raise Ticket36Error("Ticket 35 report image batch overlaps reference gold membership")
            if study.study_instance_uid not in eligible_uid_set:
                continue
            report_studies.append(study)
            for series in study.series:
                if series.series_instance_uid in report_series:
                    raise Ticket36Error("Ticket 35 batch manifests repeat a source series")
                report_series[series.series_instance_uid] = series
        manifest_path = Path(str(evidence["integration_manifest_path"]))
        batch_inputs.append(
            {
                "batch_index": index,
                "inference_evidence_path": str(evidence_path.resolve()),
                "inference_evidence_sha256": _sha256_file(evidence_path),
                "integration_manifest_path": str(manifest_path.resolve()),
                "integration_manifest_file_sha256": _sha256_file(manifest_path),
                "integration_manifest_checksum": integration.manifest_checksum,
                "study_count": len(integration.studies),
                "feature_study_count": len(eligible_uids),
                "input_excluded_study_count": len(planned_uids) - len(eligible_uids),
                "input_excluded_study_uid_sha256": _uid_set_sha256(
                    tuple(sorted(set(planned_uids) - eligible_uid_set))
                ),
                "inference_status": evidence.get("status"),
                "prediction_count": evidence.get("prediction_count"),
                "series_count": evidence.get("series_count"),
                "decoded_slice_count": evidence.get("decoded_slice_count"),
            }
        )
    if first_report_manifest is None:
        raise Ticket36Error("Ticket 35 has no report image batches")
    image_ready_reports = tuple(
        sorted(
            item.study_instance_uid for item in report_studies
        )
    )
    ticket35_planned_reports = tuple(
        str(uid)
        for uid in cast(
            Sequence[object],
            _mapping(membership.get("cohort"), "Ticket 35 cohort").get(
                "image_ready_report_study_uids", ()
            ),
        )
    )
    if (
        set(observed_ticket35_report_uids) != set(ticket35_planned_reports)
        or len(observed_ticket35_report_uids) != len(set(observed_ticket35_report_uids))
    ):
        raise Ticket36Error("completed Ticket 35 batches do not cover the frozen report image cohort")
    if set(image_ready_reports) != set(ticket35_planned_reports) - set(input_excluded_report_uids):
        raise Ticket36Error("Ticket 36 complete-input feature cohort does not reconcile with Ticket 35 exclusions")
    if len({item.study_instance_uid for item in report_studies}) != len(report_studies):
        raise Ticket36Error("Ticket 35 report image batches repeat studies")
    combined_preprocess = first_report_manifest.preprocess_config
    expanded = replace(
        base,
        preprocess_config=combined_preprocess,
        studies=tuple(
            sorted(
                (*gold_studies, *report_studies),
                key=lambda item: item.study_instance_uid,
            )
        ),
        manifest_checksum="",
    )
    if INTEGRATION_PATH.exists():
        existing = load_integration_manifest(INTEGRATION_PATH)
        if replace(existing, manifest_checksum="") != expanded:
            raise Ticket36Error("refusing to replace a different frozen Ticket 36 integration manifest")
    else:
        write_integration_manifest(expanded, INTEGRATION_PATH)
        INTEGRATION_PATH.chmod(0o600)
    expanded = load_integration_manifest(INTEGRATION_PATH)
    if set(expanded.study_uids) != expected_gold_uids | set(image_ready_reports):
        raise Ticket36Error("expanded Ticket 36 integration manifest has the wrong image cohort")

    report_series_map = {
        series.series_instance_uid: series
        for study in report_studies
        for series in study.series
    }
    series_index = _scan_ticket35_series_cache(
        report_series_map,
        expanded.preprocess_config,
    )
    _write_private_json(SERIES_INDEX_PATH, series_index)
    fold_cohorts: list[dict[str, object]] = []
    for fold in FOLD_IDS:
        full = admit_ticket30_fold_supervision(
            fold_manifest,
            fold,
            TICKET30_ARTIFACT,
            expected_artifact_sha256=TICKET30_AUDITED_ARTIFACT_SHA256,
        )
        if full.source_id != TICKET36_WEAK_FEATURE_SOURCE:
            raise Ticket36Error("Ticket 30 admitted labels do not match the frozen report source")
        image_scoped = full.for_image_studies(expanded.study_uids)
        held_out = [
            item for item in fold_manifest.gold_memberships if item.fold == fold
        ]
        target_support = []
        for index, target in enumerate(TARGET_COLUMNS):
            values = [
                item.reference_labels[index]
                for item in held_out
                if item.reference_labels[index] in (0, 1)
            ]
            target_support.append(
                {
                    "target": target,
                    "positive": sum(value == 1 for value in values),
                    "negative": sum(value == 0 for value in values),
                }
            )
        fold_cohorts.append(
            {
                "fold": fold,
                "reference_training_studies": len(image_scoped.reference_study_uids),
                "held_out_gold_studies": len(held_out),
                "report_image_studies": len(
                    set(image_ready_reports)
                    & {
                        item.study_instance_uid
                        for item in fold_manifest.report_only_memberships
                        if item.fold != fold
                    }
                ),
                "admitted_report_image_studies": len(
                    image_scoped.admitted_report_only_study_uids
                ),
                "admitted_report_target_entries": image_scoped.supervised_entry_count
                - sum(
                    item.supervised_entry_count
                    for item in image_scoped.resolved
                    if item.source_kind == "reference"
                ),
                "held_out_target_support": target_support,
                "held_out_study_uid_sha256": _uid_set_sha256(
                    tuple(item.study_instance_uid for item in held_out)
                ),
                "admitted_report_uid_sha256": _uid_set_sha256(
                    image_scoped.admitted_report_only_study_uids
                ),
                "image_excluded_report_only_studies": len(
                    image_scoped.excluded_image_report_only_study_uids
                ),
            }
        )
    config = _make_training_config(expanded)
    feature_batches = _freeze_feature_batches(
        feature_batch_rows,
        tuple(expected_gold_uids),
        image_ready_reports,
    )
    feature_identity: dict[str, object] = {
        "source_version": expanded.source_version,
        "source_fingerprint": expanded.archive_fingerprint,
        "integration_manifest_checksum": expanded.manifest_checksum,
        "fold_manifest_checksum": fold_manifest.manifest_checksum,
        "source_membership_sha256": _json_sha256(tuple(sorted(expanded.study_uids))),
        "encoder_id": config.encoder_id,
        "encoder_weights_sha256": config.encoder_weights_sha256,
        "preprocessing": expanded.preprocess_config.cache_config(),
        "packing": expanded.packing_config.to_dict(),
        "sampling_policy": "geometry-ordered-rounded-uniform-v1",
        "output_representation": "resnet18-adaptive-average-pool-512-float32-v1",
        "dtype": "float32",
        "ticket35_membership_checksum": membership.get("manifest_checksum"),
        "ticket35_series_index_sha256": series_index.get("index_sha256"),
        "ticket30_artifact_sha256": TICKET30_AUDITED_ARTIFACT_SHA256,
        "ticket30_admission_policy_sha256": TICKET30_ADMISSION_POLICY_SHA256,
        "feature_cohort_uid_sha256": _uid_set_sha256(expanded.study_uids),
    }
    identity_fingerprint = _json_sha256(
        {"schema_version": FEATURE_CACHE_SCHEMA_VERSION, "identity": feature_identity}
    )
    code_manifest, source_snapshot_root = _create_frozen_source_snapshot()
    if (config.max_series, config.max_slices) != (16, 8):
        raise Ticket36Error("Ticket 36 frozen ResNet configuration requires 16 by 8 packing")
    storage_ledger = capture_ticket35_storage_ledger(
        (
            ("rsna-project", PROJECT_ROOT),
            ("ticket30-audit", TICKET30_ARTIFACT.parent),
        ),
        SOURCE_ARCHIVE,
    )
    _write_private_json(RUN_ROOT / "private/storage-ledger.json", storage_ledger)
    current_measured_bytes = int(cast(Any, storage_ledger["measured_roots_logical_bytes"]))
    feature_bytes = (
        len(expanded.study_uids)
        * config.max_series
        * config.max_slices
        * RESNET18_FEATURE_DIM
        * 4
    )
    checkpoint_bytes = 6 * 50_000_000
    storage_estimate = {
        "current_measured_roots_bytes": current_measured_bytes,
        "source_archive_bytes_counted_once": storage_ledger.get("source_archive_bytes"),
        "additional_feature_bytes_uncompressed": feature_bytes,
        "additional_checkpoint_bytes_estimate": checkpoint_bytes,
        "maximum_atomic_feature_temporary_bytes_per_job": config.max_series
        * config.max_slices
        * RESNET18_FEATURE_DIM
        * 4,
        "maximum_simultaneous_feature_temporary_bytes": 2
        * config.max_series
        * config.max_slices
        * RESNET18_FEATURE_DIM
        * 4,
        "maximum_concurrent_jobs": 2,
        "soft_budget_bytes": 2_000_000_000_000,
        "projected_bytes_estimate": current_measured_bytes
        + feature_bytes
        + checkpoint_bytes
        + (2 * config.max_series * config.max_slices * RESNET18_FEATURE_DIM * 4),
        "measurement_ledger_path": str((RUN_ROOT / "private/storage-ledger.json").resolve()),
    }
    if int(storage_estimate["projected_bytes_estimate"]) > int(storage_estimate["soft_budget_bytes"]):
        raise Ticket36Error("Ticket 36 projected storage exceeds the campaign soft budget")
    plan: dict[str, object] = {
        "schema_version": TICKET36_PLAN_SCHEMA,
        "ticket": 36,
        "status": "frozen_before_feature_extraction_and_evaluation",
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_identity": "ticket36-expanded-ticket16-astra-vs-reference-resnet18-three-fold-v1",
        "source_version": expanded.source_version,
        "source_fingerprint": expanded.archive_fingerprint,
        "fold_manifest_checksum": fold_manifest.manifest_checksum,
        "reference_integration_manifest_checksum": base.manifest_checksum,
        "integration_manifest_path": str(INTEGRATION_PATH.resolve()),
        "integration_manifest_checksum": expanded.manifest_checksum,
        "integration_manifest_file_sha256": _sha256_file(INTEGRATION_PATH),
        "ticket35_membership_checksum": membership.get("manifest_checksum"),
        "ticket35_membership_file_sha256": _sha256_file(TICKET35_MEMBERSHIP_PATH),
        "ticket35_completed_batch_count": len(batch_inputs),
        "ticket35_batch_evidence": batch_inputs,
        "ticket35_series_index_path": str(SERIES_INDEX_PATH.resolve()),
        "ticket35_series_index_sha256": series_index.get("index_sha256"),
        "ticket30_artifact_path": str(TICKET30_ARTIFACT.resolve()),
        "ticket30_artifact_sha256": TICKET30_AUDITED_ARTIFACT_SHA256,
        "ticket30_admission_policy_sha256": feature_identity["ticket30_admission_policy_sha256"],
        "model": {
            "model_id": config.model_id,
            "encoder_id": config.encoder_id,
            "encoder_weights_sha256": config.encoder_weights_sha256,
            "encoder_source_revision": RESNET18_SOURCE_REVISION,
            "pooling_id": config.pooling_id,
            "encoder_frozen": True,
            "image_size": list(config.image_size),
            "representation_id": config.preprocess_config().representation_id,
            "checkpoint_path": str(WEIGHTS_PATH.resolve()),
            "checkpoint_sha256": RESNET18_SHA256,
            "checkpoint_bytes": RESNET18_ARTIFACT_BYTES,
        },
        "cohort": {
            "gold_reference_studies": len(expected_gold_uids),
            "ticket35_planned_report_studies": len(ticket35_planned_reports),
            "image_ready_report_studies": len(image_ready_reports),
            "input_excluded_report_studies": len(input_excluded_report_uids),
            "input_excluded_report_uid_sha256": _uid_set_sha256(
                tuple(sorted(input_excluded_report_uids))
            ),
            "total_feature_studies": len(expanded.study_uids),
            "image_ready_report_series": len(report_series),
            "image_ready_report_decoded_slices": eligible_report_decoded_slices,
            "supervised_report_target_entries_across_folds": sum(
                int(cast(Any, item["admitted_report_target_entries"]))
                for item in fold_cohorts
            ),
            "folds": fold_cohorts,
            "gold_uid_sha256": _uid_set_sha256(tuple(expected_gold_uids)),
            "report_uid_sha256": _uid_set_sha256(image_ready_reports),
            "all_feature_uid_sha256": _uid_set_sha256(expanded.study_uids),
            "target_order": list(TARGET_COLUMNS),
        },
        "training_configuration": config.to_dict(),
        "training_configuration_fingerprint": config.config_fingerprint,
        "training_design": TICKET34_ADDITIVE_DESIGN,
        "report_source_id": TICKET36_WEAK_FEATURE_SOURCE,
        "weak_sampling_policy": TICKET34_WEAK_SAMPLING_POLICY,
        "weak_batch_seed_offset": TICKET34_WEAK_BATCH_SEED_OFFSET,
        "paired_bootstrap": {
            "unit": "study",
            "resamples": TICKET36_BOOTSTRAP_RESAMPLES,
            "seed": TICKET36_BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "undefined_resamples": "count, never impute",
        },
        "decision_policy": {
            "advance_to_separate_candidate_review": (
                "pooled report-minus-reference macro AUC delta is positive and positive folds "
                "outnumber negative folds; a wide interval remains inconclusive evidence"
            ),
            "reject_expanded_report_supervision": (
                "pooled delta is negative, the paired 95% interval is entirely below zero, "
                "and negative folds outnumber positive folds"
            ),
            "otherwise": "inconclusive; retain both source arms without candidate promotion",
            "release_gate": "candidate freeze, publication, redistribution and release remain blocked",
        },
        "feature_identity": feature_identity,
        "feature_identity_fingerprint": identity_fingerprint,
        "feature_batches": feature_batches,
        "source_snapshot_path": str(source_snapshot_root.resolve()),
        "execution": {
            "feature_materialization": "one restartable Slurm array element per Ticket 35 batch plus one gold batch",
            "training": "one CPU Slurm job, six fold/arm runs sequentially",
            "partition": "cluster1-cpu",
            "nodes": 1,
            "cpus_per_task": 16,
            "memory_gib": 64,
            "wall_time_limit": "06:00:00 per allocation",
            "maximum_concurrent_jobs": 2,
            "gpu_count": 0,
            "external_spend_usd": 0,
            "network_during_execution": False,
            "new_weight_download": False,
            "feature_cache_reads": "Ticket 35 prepared-series cache read-only and checksum-validated",
            "feature_cache_writes": "Ticket 36-only frozen features under the Ticket 36 private run root",
            "storage_estimate": storage_estimate,
            "resource_approval_path": str(APPROVAL_PATH.resolve()),
            "source_code_manifest_sha256": code_manifest["source_manifest_sha256"],
        },
        "source_code_manifest": code_manifest,
        "retained_comparator": {
            "ticket32_reference_only_pooled_macro_auc": 0.518125,
            "ticket32_reference_only_paired_delta_vs_mechanics": 0.053812,
            "ticket32_paired_95_interval": [-0.004524, 0.115394],
            "ticket34_fold0_report_delta": 0.083534,
            "ticket34_fold0_paired_95_interval": [-0.038625, 0.209216],
            "ticket34_role": "positive but inconclusive one-fold pilot, not promotion evidence",
        },
        "release_eligibility": {
            "candidate_promotion": "not authorized by Ticket 36 experiment scope",
            "final_candidate_freeze": "blocked",
            "winner_publication": "blocked",
            "redistribution_or_release": "blocked",
        },
    }
    plan["plan_sha256"] = _json_sha256(plan)
    _write_private_json(RUN_ROOT / "private/source-code-manifest.json", code_manifest)
    _write_private_text(
        RUN_ROOT / "private/code-revision.env",
        "export RSNA_CODE_REVISION=ticket36-source-"
        + str(code_manifest["source_manifest_sha256"])
        + "\nexport RSNA_TICKET36_SOURCE_ROOT="
        + str(source_snapshot_root.resolve())
        + "\n",
    )
    _write_private_json(PLAN_PATH, plan)
    return plan


def _load_series_index(path: Path = SERIES_INDEX_PATH) -> Mapping[str, object]:
    index = _read_json(path, "Ticket 36 series index")
    _verify_checksum_payload(index, "index_sha256", "Ticket 36 series index")
    if index.get("schema_version") != "rsna-ticket36-ticket35-series-index-v1":
        raise Ticket36Error("Ticket 36 series index schema is unsupported")
    return index


def _feature_path(study_uid: str) -> Path:
    key = hashlib.sha256(study_uid.encode("utf-8")).hexdigest()
    return FEATURE_ROOT / f"{key}.npz"


def _load_feature_file(
    path: Path,
    *,
    study_uid: str,
    batch_index: int,
    feature_identity_fingerprint: str,
) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise Ticket36Error(f"frozen feature file is missing or unsafe: {path}")
    try:
        with np.load(path, allow_pickle=False) as payload:
            features = np.asarray(payload["features"], dtype=np.float32)
            slice_mask = np.asarray(payload["slice_mask"], dtype=bool)
            series_mask = np.asarray(payload["series_mask"], dtype=bool)
            descriptors = np.asarray(payload["series_descriptors"], dtype=np.float32)
            metadata = json.loads(str(np.asarray(payload["metadata"]).item()))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise Ticket36Error(f"could not decode frozen features for {study_uid}") from exc
    row = _mapping(metadata, "frozen feature metadata")
    expected_checksums = {
        "features_sha256": _array_sha256(features),
        "slice_mask_sha256": _array_sha256(slice_mask.astype(np.uint8)),
        "series_mask_sha256": _array_sha256(series_mask.astype(np.uint8)),
        "series_descriptors_sha256": _array_sha256(descriptors),
    }
    source_digest = row.get("source_identity_sha256")
    if (
        row.get("study_instance_uid") != study_uid
        or row.get("batch_index") != batch_index
        or row.get("feature_identity_fingerprint") != feature_identity_fingerprint
        or row.get("source_kind")
        not in {"ticket35-prepared-series-cache-read-only", "frozen-rsna-archive"}
        or not isinstance(source_digest, str)
        or len(source_digest) != 64
        or any(character not in "0123456789abcdef" for character in source_digest)
        or any(row.get(key) != value for key, value in expected_checksums.items())
        or features.shape != (16, 8, RESNET18_FEATURE_DIM)
        or slice_mask.shape != (16, 8)
        or series_mask.shape != (16,)
        or descriptors.ndim != 2
        or descriptors.shape[0] != 16
        or not slice_mask.any()
        or not series_mask.any()
        or not np.isfinite(features).all()
        or not np.isfinite(descriptors).all()
    ):
        raise Ticket36Error(f"frozen feature identity or array checks failed for {study_uid}")
    return {
        "study_instance_uid": study_uid,
        "features": features,
        "slice_mask": slice_mask,
        "series_mask": series_mask,
        "series_descriptors": descriptors,
        "metadata": dict(row),
        "file_sha256": _sha256_file(path),
        "path": path,
    }


def _write_feature_file(
    *,
    study_uid: str,
    batch_index: int,
    feature_identity_fingerprint: str,
    features: np.ndarray,
    slice_mask: np.ndarray,
    series_mask: np.ndarray,
    series_descriptors: np.ndarray,
    source_kind: str,
    source_identity_sha256: str,
) -> Mapping[str, object]:
    destination = _feature_path(study_uid)
    metadata: dict[str, object] = {
        "study_instance_uid": study_uid,
        "batch_index": batch_index,
        "feature_identity_fingerprint": feature_identity_fingerprint,
        "source_kind": source_kind,
        "source_identity_sha256": source_identity_sha256,
        "features_sha256": _array_sha256(features),
        "slice_mask_sha256": _array_sha256(slice_mask.astype(np.uint8)),
        "series_mask_sha256": _array_sha256(series_mask.astype(np.uint8)),
        "series_descriptors_sha256": _array_sha256(series_descriptors),
    }
    if destination.exists():
        row = _load_feature_file(
            destination,
            study_uid=study_uid,
            batch_index=batch_index,
            feature_identity_fingerprint=feature_identity_fingerprint,
        )
        if row["metadata"] != metadata:
            raise Ticket36Error(f"existing frozen feature differs from this batch: {study_uid}")
        return {
            "study_instance_uid": study_uid,
            "feature_file": destination.name,
            "feature_file_sha256": row["file_sha256"],
            "features_sha256": metadata["features_sha256"],
            "source_identity_sha256": source_identity_sha256,
            "source_kind": source_kind,
        }
    _private_directory(destination.parent)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                features=np.asarray(features, dtype=np.float32),
                slice_mask=np.asarray(slice_mask, dtype=np.uint8),
                series_mask=np.asarray(series_mask, dtype=np.uint8),
                series_descriptors=np.asarray(series_descriptors, dtype=np.float32),
                metadata=np.asarray(_canonical_json(metadata)),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        destination.chmod(0o600)
    except FileExistsError as exc:
        raise Ticket36Error(f"another worker published the same feature file: {study_uid}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "study_instance_uid": study_uid,
        "feature_file": destination.name,
        "feature_file_sha256": _sha256_file(destination),
        "features_sha256": metadata["features_sha256"],
        "source_identity_sha256": source_identity_sha256,
        "source_kind": source_kind,
    }


def _study_source_digest(
    study: IntegrationStudy,
    series_index: Mapping[str, object] | None = None,
    *,
    archive_fingerprint: str,
) -> str:
    rows: list[dict[str, object]] = []
    for series in sorted(study.series, key=lambda item: item.series_instance_uid):
        cache_row: Mapping[str, object] | None = None
        if series_index is not None:
            cache_row = _mapping(series_index.get(series.series_instance_uid), "series index row")
        rows.append(
            {
                "series_instance_uid": series.series_instance_uid,
                "entries": list(series.entries),
                "prepared_source_payload_sha256": (
                    cache_row.get("source_payload_sha256") if cache_row is not None else None
                ),
                "prepared_array_sha256": cache_row.get("array_sha256") if cache_row is not None else None,
            }
        )
    return _json_sha256(
        {
            "archive_fingerprint": archive_fingerprint,
            "study_instance_uid": study.study_instance_uid,
            "series": rows,
        }
    )


def _packed_features(
    model: ResNet18StudyModel,
    study: StudyRecord,
    prepared: Sequence[PreparedSeries],
    integration: IntegrationManifest,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    packed = pack_study(
        study,
        prepared,
        integration.preprocess_config,
        packing_config=integration.packing_config,
    )
    if packed.series_descriptors is None:
        raise Ticket36Error("Ticket 36 ResNet packing requires anatomical series descriptors")
    with torch.inference_mode():
        features = model.encode_one_study(
            torch.from_numpy(np.asarray(packed.images, dtype=np.float32)),
            torch.from_numpy(np.asarray(packed.slice_mask, dtype=bool)),
            torch.from_numpy(np.asarray(packed.series_mask, dtype=bool)),
        ).detach().cpu().numpy().astype(np.float32, copy=False)
    return (
        features,
        np.asarray(packed.slice_mask, dtype=bool),
        np.asarray(packed.series_mask, dtype=bool),
        np.asarray(packed.series_descriptors, dtype=np.float32),
    )


def _materialize_report_batch(
    *,
    batch: Mapping[str, object],
    batch_index: int,
    integration: IntegrationManifest,
    model: ResNet18StudyModel,
    series_index: Mapping[str, object],
    feature_identity_fingerprint: str,
) -> list[Mapping[str, object]]:
    cache = SeriesCache(TICKET35_SERIES_CACHE)
    cache_rows = _mapping(series_index.get("entries"), "series index entries")
    results = []
    study_by_uid = {item.study_instance_uid: item for item in integration.studies}
    uids = cast(Sequence[str], batch.get("study_uids", ()))
    if tuple(uids) != integration.study_uids:
        raise Ticket36Error("Ticket 36 report batch differs from its source integration manifest")
    for uid in uids:
        study_manifest = study_by_uid[uid]
        series_records: list[SeriesRecord] = []
        prepared: list[PreparedSeries] = []
        for item in study_manifest.series:
            index_row = _mapping(cache_rows.get(item.series_instance_uid), "Ticket 35 series index row")
            if index_row.get("study_instance_uid") != uid:
                raise Ticket36Error("Ticket 35 series cache row belongs to another study")
            series_records.append(
                SeriesRecord(
                    study_instance_uid=uid,
                    series_instance_uid=item.series_instance_uid,
                    slices=(),
                    anatomical_plane=item.anatomical_plane,
                    fluid_sensitive=item.fluid_sensitive,
                    fat_suppression=item.fat_suppression,
                    diagnostic_status=item.diagnostic_status,
                    source_entry_names=item.entries,
                )
            )
            prepared.append(
                cache.load_validated_entry(
                    str(index_row["cache_key"]),
                    study_instance_uid=uid,
                    series_instance_uid=item.series_instance_uid,
                    expected_source_entry_names=item.entries,
                    config=integration.preprocess_config,
                )
            )
        study = StudyRecord(uid, tuple(series_records))
        features, slice_mask, series_mask, descriptors = _packed_features(
            model, study, prepared, integration
        )
        results.append(
            _write_feature_file(
                study_uid=uid,
                batch_index=batch_index,
                feature_identity_fingerprint=feature_identity_fingerprint,
                features=features,
                slice_mask=slice_mask,
                series_mask=series_mask,
                series_descriptors=descriptors,
                source_kind="ticket35-prepared-series-cache-read-only",
                source_identity_sha256=_study_source_digest(
                    study_manifest,
                    cache_rows,
                    archive_fingerprint=integration.archive_fingerprint,
                ),
            )
        )
    return results


def _materialize_gold_batch(
    *,
    batch: Mapping[str, object],
    batch_index: int,
    integration: IntegrationManifest,
    model: ResNet18StudyModel,
    feature_identity_fingerprint: str,
) -> list[Mapping[str, object]]:
    uids = tuple(cast(Sequence[str], batch.get("study_uids", ())))
    if not SOURCE_ARCHIVE.is_file():
        raise Ticket36Error("the frozen RSNA source archive is missing")
    rows = []
    seen: list[str] = []
    for loaded in iter_real_studies(
        integration,
        SOURCE_ARCHIVE,
        require_complete_series=True,
        study_uids=uids,
    ):
        study = loaded.study
        prepared_study = prepare_study(study, integration.preprocess_config)
        if prepared_study.failures or not prepared_study.series:
            raise Ticket36Error(
                f"gold study {study.study_instance_uid} has rejected or missing image series"
            )
        features, slice_mask, series_mask, descriptors = _packed_features(
            model, study, prepared_study.series, integration
        )
        source_identity = _json_sha256(
            {
                "archive_fingerprint": integration.archive_fingerprint,
                "study_instance_uid": study.study_instance_uid,
                "series": [
                    {
                        "series_instance_uid": item.series_instance_uid,
                        "entry_names": list(item.source_entry_names),
                        "source_slice_uids": [slice_item.sop_instance_uid for slice_item in item.slices],
                    }
                    for item in sorted(study.series, key=lambda item: item.series_instance_uid)
                ],
            }
        )
        rows.append(
            _write_feature_file(
                study_uid=study.study_instance_uid,
                batch_index=batch_index,
                feature_identity_fingerprint=feature_identity_fingerprint,
                features=features,
                slice_mask=slice_mask,
                series_mask=series_mask,
                series_descriptors=descriptors,
                source_kind="frozen-rsna-archive",
                source_identity_sha256=source_identity,
            )
        )
        seen.append(study.study_instance_uid)
    if tuple(sorted(seen)) != tuple(sorted(uids)):
        raise Ticket36Error("Ticket 36 gold feature batch did not return every declared study")
    return rows


def materialize_feature_batch(batch_index: int) -> Mapping[str, object]:
    """Materialize one bounded Ticket 35 batch using the frozen encoder."""

    _verify_slurm_allocation()
    plan = _load_plan()
    _approval_payload(plan)
    membership = _read_membership()
    integration = load_integration_manifest(INTEGRATION_PATH)
    if integration.manifest_checksum != plan.get("integration_manifest_checksum"):
        raise Ticket36Error("Ticket 36 expanded integration manifest changed after plan freeze")
    batches = plan.get("feature_batches")
    if not isinstance(batches, list) or batch_index < 0 or batch_index >= len(batches):
        raise Ticket36Error("Ticket 36 feature batch index is outside the frozen plan")
    batch = _mapping(batches[batch_index], "Ticket 36 feature batch")
    kind = batch.get("kind")
    batch_integration: IntegrationManifest | None = None
    if kind == "report_only":
        ticket35_index = int(cast(Any, batch.get("ticket35_batch_index")))
        t35_evidence, batch_integration, t35_evidence_path, eligible_uids = (
            _validate_t35_batch_evidence(membership, ticket35_index)
        )
        _verify_frozen_t35_batch_evidence(
            plan,
            ticket35_index,
            t35_evidence,
            batch_integration,
            t35_evidence_path,
        )
        expected_report_uids = tuple(cast(Sequence[str], batch.get("study_uids", ())))
        if eligible_uids != expected_report_uids:
            raise Ticket36Error("Ticket 36 report feature batch differs from complete Ticket 35 inputs")
    evidence_path = FEATURE_ROOT / f"batch-{batch_index:03d}.json"
    if evidence_path.exists():
        evidence = _read_json(evidence_path, "completed Ticket 36 feature batch evidence")
        _verify_checksum_payload(evidence, "batch_evidence_sha256", "Ticket 36 feature batch evidence")
        rows = evidence.get("features")
        expected_uids = set(cast(Sequence[str], batch.get("study_uids", ())))
        if (
            evidence.get("schema_version") != TICKET36_FEATURE_BATCH_SCHEMA
            or evidence.get("batch_index") != batch_index
            or evidence.get("batch_identity_sha256") != _json_sha256(batch)
            or evidence.get("feature_identity_fingerprint") != plan.get("feature_identity_fingerprint")
            or evidence.get("study_count") != len(expected_uids)
            or not isinstance(rows, list)
            or {str(_mapping(row, "Ticket 36 batch feature row").get("study_instance_uid")) for row in rows}
            != expected_uids
        ):
            raise Ticket36Error("existing Ticket 36 feature batch evidence has a different identity")
        for row_value in cast(Sequence[object], rows):
            row = _mapping(row_value, "Ticket 36 batch feature row")
            uid = str(row.get("study_instance_uid"))
            path = _feature_path(uid)
            expected_kind = (
                "ticket35-prepared-series-cache-read-only"
                if batch.get("kind") == "report_only"
                else "frozen-rsna-archive"
            )
            feature = _load_feature_file(
                path,
                study_uid=uid,
                batch_index=batch_index,
                feature_identity_fingerprint=str(plan["feature_identity_fingerprint"]),
            )
            if (
                feature["file_sha256"] != row.get("feature_file_sha256")
                or path.name != row.get("feature_file")
                or _mapping(feature["metadata"], "Ticket 36 feature metadata").get("source_kind")
                != expected_kind
            ):
                raise Ticket36Error("completed Ticket 36 feature file checksum changed")
        return evidence

    if kind == "report_only":
        expected_uids = tuple(cast(Sequence[str], batch.get("study_uids", ())))
        if batch_integration is None:
            raise Ticket36Error("Ticket 36 report feature batch has no validated Ticket 35 manifest")
        if expected_uids:
            filtered_integration = replace(
                batch_integration,
                studies=tuple(
                    study
                    for study in batch_integration.studies
                    if study.study_instance_uid in set(expected_uids)
                ),
                manifest_checksum="",
            )
            series_index = _load_series_index()
            config = _make_training_config(integration)
            state = load_resnet18_weights(WEIGHTS_PATH, APPROVAL_PATH)
            model = build_study_model(config.model_config())
            if not isinstance(model, ResNet18StudyModel):
                raise Ticket36Error("Ticket 36 model builder returned an unsupported model")
            model.load_pretrained_state(state)
            model.eval()
            model.to(torch.device("cpu"))
            rows = _materialize_report_batch(
                batch=batch,
                batch_index=batch_index,
                integration=filtered_integration,
                model=model,
                series_index=series_index,
                feature_identity_fingerprint=str(plan["feature_identity_fingerprint"]),
            )
        else:
            rows = []
    elif kind == "reference_gold":
        gold_integration = IntegrationManifest(
            source_version=integration.source_version,
            archive_fingerprint=integration.archive_fingerprint,
            geometry_policy=integration.geometry_policy,
            decoder_policy=integration.decoder_policy,
            preprocess_config=integration.preprocess_config,
            studies=tuple(
                item
                for item in integration.studies
                if item.study_instance_uid in set(cast(Sequence[str], batch.get("study_uids", ())))
            ),
            source_split=integration.source_split,
            packing_config=integration.packing_config,
            source_manifest_checksum=integration.source_manifest_checksum,
        )
        if len(gold_integration.studies) != len(cast(Sequence[object], batch.get("study_uids", ()))):
            raise Ticket36Error("Ticket 36 gold feature batch differs from the frozen gold membership")
        config = _make_training_config(integration)
        state = load_resnet18_weights(WEIGHTS_PATH, APPROVAL_PATH)
        model = build_study_model(config.model_config())
        if not isinstance(model, ResNet18StudyModel):
            raise Ticket36Error("Ticket 36 model builder returned an unsupported model")
        model.load_pretrained_state(state)
        model.eval()
        model.to(torch.device("cpu"))
        rows = _materialize_gold_batch(
            batch=batch,
            batch_index=batch_index,
            integration=gold_integration,
            model=model,
            feature_identity_fingerprint=str(plan["feature_identity_fingerprint"]),
        )
    else:
        raise Ticket36Error("Ticket 36 feature batch has an unsupported cohort kind")

    expected_uids = tuple(cast(Sequence[str], batch.get("study_uids", ())))
    if tuple(sorted(str(item["study_instance_uid"]) for item in rows)) != tuple(sorted(expected_uids)):
        raise Ticket36Error("Ticket 36 batch features do not reconcile with frozen study membership")
    evidence: dict[str, object] = {
        "schema_version": TICKET36_FEATURE_BATCH_SCHEMA,
        "batch_index": batch_index,
        "batch_identity_sha256": _json_sha256(batch),
        "feature_identity_fingerprint": plan.get("feature_identity_fingerprint"),
        "integration_manifest_checksum": integration.manifest_checksum,
        "ticket35_membership_checksum": membership.get("manifest_checksum"),
        "kind": kind,
        "study_count": len(rows),
        "features": sorted(rows, key=lambda item: str(item["study_instance_uid"])),
        "held_out_data_used_for_feature_adaptation": False,
        "encoder_adapted": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    evidence["batch_evidence_sha256"] = _json_sha256(evidence)
    _write_private_json(evidence_path, evidence)
    return evidence


def seal_feature_manifest() -> Mapping[str, object]:
    """Validate every batch and bind all feature bytes into one immutable manifest."""

    plan = _load_plan()
    _approval_payload(plan)
    batches = plan.get("feature_batches")
    if not isinstance(batches, list):
        raise Ticket36Error("Ticket 36 feature batch plan is missing")
    feature_entries: list[dict[str, object]] = []
    batch_evidence: list[dict[str, object]] = []
    for batch_index, batch_value in enumerate(batches):
        batch = _mapping(batch_value, "Ticket 36 feature batch")
        evidence_path = FEATURE_ROOT / f"batch-{batch_index:03d}.json"
        evidence = _read_json(evidence_path, "Ticket 36 feature batch evidence")
        _verify_checksum_payload(evidence, "batch_evidence_sha256", "Ticket 36 feature batch evidence")
        if (
            evidence.get("schema_version") != TICKET36_FEATURE_BATCH_SCHEMA
            or evidence.get("batch_index") != batch_index
            or evidence.get("batch_identity_sha256") != _json_sha256(batch)
            or evidence.get("feature_identity_fingerprint") != plan.get("feature_identity_fingerprint")
            or evidence.get("study_count") != len(cast(Sequence[object], batch.get("study_uids", ())))
        ):
            raise Ticket36Error(f"Ticket 36 feature batch {batch_index} does not match the frozen plan")
        rows = evidence.get("features")
        if not isinstance(rows, list):
            raise Ticket36Error("Ticket 36 batch feature rows are malformed")
        expected_uids = set(cast(Sequence[str], batch.get("study_uids", ())))
        if {str(_mapping(row, "feature evidence row").get("study_instance_uid")) for row in rows} != expected_uids:
            raise Ticket36Error("Ticket 36 batch does not contain exactly the frozen study membership")
        for row_value in rows:
            row = _mapping(row_value, "Ticket 36 feature evidence row")
            uid = str(row.get("study_instance_uid"))
            path = _feature_path(uid)
            feature = _load_feature_file(
                path,
                study_uid=uid,
                batch_index=batch_index,
                feature_identity_fingerprint=str(plan["feature_identity_fingerprint"]),
            )
            expected_source_kind = (
                "ticket35-prepared-series-cache-read-only"
                if batch.get("kind") == "report_only"
                else "frozen-rsna-archive"
            )
            metadata = _mapping(feature["metadata"], "Ticket 36 feature metadata")
            if (
                feature.get("file_sha256") != row.get("feature_file_sha256")
                or path.name != row.get("feature_file")
                or row.get("source_kind") != expected_source_kind
                or row.get("source_identity_sha256") != metadata.get("source_identity_sha256")
                or row.get("features_sha256") != metadata.get("features_sha256")
            ):
                raise Ticket36Error(f"Ticket 36 feature file checksum differs for {uid}")
            feature_entries.append(
                {
                    "study_instance_uid": uid,
                    "batch_index": batch_index,
                    "feature_file": str(row.get("feature_file")),
                    "feature_file_sha256": row.get("feature_file_sha256"),
                    "features_sha256": row.get("features_sha256"),
                    "source_identity_sha256": row.get("source_identity_sha256"),
                    "source_kind": row.get("source_kind"),
                }
            )
        batch_evidence.append(
            {
                "batch_index": batch_index,
                "path": str(evidence_path.resolve()),
                "sha256": _sha256_file(evidence_path),
                "study_count": len(rows),
            }
        )
    feature_entries.sort(key=lambda item: str(item["study_instance_uid"]))
    expected_uids = {
        str(uid)
        for batch in batches
        for uid in cast(Sequence[object], _mapping(batch, "Ticket 36 feature batch").get("study_uids", ()))
    }
    if {str(item["study_instance_uid"]) for item in feature_entries} != expected_uids:
        raise Ticket36Error("Ticket 36 feature manifest does not cover every planned study")
    manifest: dict[str, object] = {
        "schema_version": TICKET36_FEATURE_MANIFEST_SCHEMA,
        "ticket": 36,
        "feature_identity": plan.get("feature_identity"),
        "identity_fingerprint": plan.get("feature_identity_fingerprint"),
        "feature_study_count": len(feature_entries),
        "feature_uid_sha256": _uid_set_sha256(tuple(sorted(expected_uids))),
        "batch_evidence": batch_evidence,
        "features": feature_entries,
        "held_out_features_used_for_prediction_only": True,
        "encoder_adapted_on_held_out_data": False,
        "ticket35_prepared_series_cache_access": "read-only, per-entry source and array checksums verified during extraction",
    }
    manifest["feature_manifest_sha256"] = _json_sha256(manifest)
    manifest_path = FEATURE_ROOT / "feature-manifest.json"
    _write_private_json(manifest_path, manifest)
    return manifest


def _load_feature_manifest(plan: Mapping[str, object]) -> Mapping[str, object]:
    path = FEATURE_ROOT / "feature-manifest.json"
    if not path.is_file():
        return seal_feature_manifest()
    manifest = _read_json(path, "Ticket 36 feature manifest")
    _verify_checksum_payload(manifest, "feature_manifest_sha256", "Ticket 36 feature manifest")
    if (
        manifest.get("schema_version") != TICKET36_FEATURE_MANIFEST_SCHEMA
        or manifest.get("identity_fingerprint") != plan.get("feature_identity_fingerprint")
    ):
        raise Ticket36Error("Ticket 36 feature manifest identity differs from the frozen plan")
    return manifest


def _load_training_features(
    plan: Mapping[str, object], manifest: Mapping[str, object]
) -> dict[str, Mapping[str, object]]:
    rows = manifest.get("features")
    if not isinstance(rows, list):
        raise Ticket36Error("Ticket 36 feature manifest has no study rows")
    result: dict[str, Mapping[str, object]] = {}
    for row_value in rows:
        row = _mapping(row_value, "Ticket 36 feature manifest row")
        uid = str(row.get("study_instance_uid"))
        batch_index = int(cast(Any, row.get("batch_index")))
        feature = _load_feature_file(
            _feature_path(uid),
            study_uid=uid,
            batch_index=batch_index,
            feature_identity_fingerprint=str(plan["feature_identity_fingerprint"]),
        )
        if feature.get("file_sha256") != row.get("feature_file_sha256"):
            raise Ticket36Error(f"Ticket 36 feature file differs from its sealed manifest: {uid}")
        if uid in result:
            raise Ticket36Error("Ticket 36 feature manifest repeats a study")
        result[uid] = feature
    return result


def _reference_example(
    uid: str,
    feature: Mapping[str, object],
    targets: Sequence[int | None],
    *,
    fold: int,
    fold_manifest_checksum: str,
    training_uids: Sequence[str],
    heldout_uids: Sequence[str],
) -> TrainingExample:
    provenance = SupervisionProvenance(
        source_kind="reference",
        source_id=f"fold-manifest:{fold_manifest_checksum}",
        fold=fold,
        held_out_excluded=True,
        eligible_study_uids=tuple(sorted(training_uids)),
        excluded_study_uids=tuple(sorted(heldout_uids)),
    )
    return TrainingExample.from_features(
        uid,
        cast(np.ndarray, feature["features"]),
        cast(np.ndarray, feature["slice_mask"]),
        cast(np.ndarray, feature["series_mask"]),
        cast(np.ndarray, feature["series_descriptors"]),
        targets,
        provenance=provenance,
    )


def _weak_example(
    uid: str,
    feature: Mapping[str, object],
    supervision: AdmittedFoldSupervision,
) -> TrainingExample:
    payload = supervision.provenance_for(uid)
    provenance = SupervisionProvenance(
        source_kind=str(payload["source_kind"]),
        source_id=str(payload["source_id"]),
        fold=int(cast(Any, payload["fold"])),
        held_out_excluded=bool(payload["held_out_excluded"]),
        eligible_study_uids=tuple(cast(Sequence[str], payload["eligible_study_uids"])),
        excluded_study_uids=tuple(cast(Sequence[str], payload["excluded_study_uids"])),
        weak_label_version=str(payload["weak_label_version"]),
        mixture_policy=str(payload["mixture_policy"]),
        source_report_ref=str(payload["source_report_ref"]),
        source_report_sha256=str(payload["source_report_sha256"]),
        source_disagreement_count=int(cast(Any, payload["source_disagreement_count"])),
    )
    return TrainingExample.from_features(
        uid,
        cast(np.ndarray, feature["features"]),
        cast(np.ndarray, feature["slice_mask"]),
        cast(np.ndarray, feature["series_mask"]),
        cast(np.ndarray, feature["series_descriptors"]),
        supervision.by_study[uid].targets,
        provenance=provenance,
    )


def _load_single_heldout_study(
    integration: IntegrationManifest,
    fold_manifest: FoldManifest,
    fold: int,
) -> Mapping[str, StudyRecord]:
    uid = min(
        item.study_instance_uid
        for item in fold_manifest.gold_memberships
        if item.fold == fold
    )
    loaded = tuple(
        iter_real_studies(
            integration,
            SOURCE_ARCHIVE,
            require_complete_series=True,
            study_uids=(uid,),
        )
    )
    if len(loaded) != 1 or loaded[0].study.study_instance_uid != uid:
        raise Ticket36Error("the offline lifecycle study is not the declared held-out member")
    return {uid: loaded[0].study}


def _completed_run(run_dir: Path) -> bool:
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        return False
    run = _read_json(run_path, "completed Ticket 36 fold run")
    if run.get("status") != "completed_fixed_budget_training":
        raise Ticket36Error(f"existing Ticket 36 run is not complete: {run_dir}")
    return True


def _run_arm(
    *,
    arm: str,
    fold: int,
    studies: Mapping[str, StudyRecord],
    fold_manifest: FoldManifest,
    integration: IntegrationManifest,
    config: TrainingConfig,
    feature_examples: tuple[
        Sequence[TrainingExample], Sequence[TrainingExample], Sequence[TrainingExample]
    ],
    supervision: AdmittedFoldSupervision | None,
    feature_evidence: Mapping[str, object],
    preflight: CpuPreflightResult,
    plan: Mapping[str, object],
) -> TrainingResult | None:
    run_dir = RUN_ROOT / ("reference-only" if arm == "reference" else "report-supervision") / f"fold-{fold}"
    compute_path = run_dir.parent / f"fold-{fold}.compute.json"
    public_path = run_dir.parent / f"fold-{fold}.public.json"
    if _completed_run(run_dir):
        if not compute_path.is_file() or (arm == "report" and not public_path.is_file()):
            raise Ticket36Error("completed Ticket 36 run is missing compute or report-lineage evidence")
        return None
    _private_directory(run_dir.parent)
    resume_path = None
    latest = run_dir / "checkpoints/latest.pt"
    if run_dir.exists():
        if latest.is_file():
            resume_path = latest
        elif any(run_dir.iterdir()):
            raise Ticket36Error("incomplete Ticket 36 run has no resumable latest checkpoint")
    cpu_started = resource.getrusage(resource.RUSAGE_SELF)
    started = time.perf_counter()
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_restart_count = os.environ.get("SLURM_RESTART_COUNT", "0")
    attempt_id = (
        f"{slurm_job_id}-restart-{slurm_restart_count}"
        if slurm_job_id
        else "ticket36-local"
    )
    result = train_one_fold(
        studies,
        fold_manifest,
        fold,
        run_dir,
        config=config,
        source_version=integration.source_version,
        source_fingerprint=integration.archive_fingerprint,
        cache_fingerprint=_json_sha256(
            {
                "feature_identity_fingerprint": plan["feature_identity_fingerprint"],
                "feature_manifest_sha256": feature_evidence["feature_manifest_sha256"],
            }
        ),
        integration_manifest_checksum=integration.manifest_checksum,
        supervision=supervision,
        ticket34_additive_design=True,
        preflight=preflight,
        require_preflight=True,
        allow_local_smoke=False,
        requested_resources=_REQUESTED_RESOURCES,
        pretrained_weights_path=WEIGHTS_PATH,
        resource_approval_path=APPROVAL_PATH,
        precomputed_feature_examples=feature_examples,
        precomputed_feature_evidence=feature_evidence,
        resume_from=resume_path,
        execution_attempt_id=attempt_id,
    )
    ended = resource.getrusage(resource.RUSAGE_SELF)
    elapsed = time.perf_counter() - started
    attempt_compute: dict[str, object] = {
        "schema_version": "rsna-ticket36-run-compute-attempt-v1",
        "run_id": result.run_id,
        "fold": fold,
        "arm": arm,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_restart_count": int(slurm_restart_count),
        "elapsed_seconds": elapsed,
        "user_cpu_seconds": ended.ru_utime - cpu_started.ru_utime,
        "system_cpu_seconds": ended.ru_stime - cpu_started.ru_stime,
        "peak_rss_kb": ended.ru_maxrss,
    }
    attempts_root = run_dir / "attempts"
    _private_directory(attempts_root)
    _write_private_json(attempts_root / f"ticket36-compute-{attempt_id}.json", attempt_compute)
    if compute_path.exists():
        prior = _read_json(compute_path, "Ticket 36 aggregate run compute")
        previous_elapsed = float(cast(Any, prior.get("elapsed_seconds", 0)))
        previous_user = float(cast(Any, prior.get("user_cpu_seconds", 0)))
        previous_system = float(cast(Any, prior.get("system_cpu_seconds", 0)))
        elapsed += previous_elapsed
        user_cpu = ended.ru_utime - cpu_started.ru_utime + previous_user
        system_cpu = ended.ru_stime - cpu_started.ru_stime + previous_system
        peak_rss = max(int(cast(Any, prior.get("peak_rss_kb", 1))), int(ended.ru_maxrss))
    else:
        user_cpu = ended.ru_utime - cpu_started.ru_utime
        system_cpu = ended.ru_stime - cpu_started.ru_stime
        peak_rss = ended.ru_maxrss
    compute: dict[str, object] = {
        "schema_version": "rsna-knee-observed-compute-v1",
        "elapsed_seconds": elapsed,
        "user_cpu_seconds": user_cpu,
        "system_cpu_seconds": system_cpu,
        "peak_rss_kb": peak_rss,
        "attempt_count": len(list(attempts_root.glob("ticket36-compute-*.json"))),
        "attempt_compute_files": sorted(
            path.name for path in attempts_root.glob("ticket36-compute-*.json")
        ),
    }
    _write_private_json(compute_path, compute)
    if result.final_checkpoint_path is None or result.latest_checkpoint_path is None:
        raise Ticket36Error("Ticket 36 fold run did not save reloadable final and latest checkpoints")
    load_training_checkpoint(
        result.final_checkpoint_path,
        model=result.model,
        expected_config=config,
        expected_fold=fold,
    )
    if arm == "report":
        assert supervision is not None
        public_evidence = {
            "schema_version": "rsna-ticket36-report-training-evidence-v1",
            "ticket": 36,
            "fold": fold,
            "fold_manifest_checksum": fold_manifest.manifest_checksum,
            "preflight_passed": preflight.passed,
            "report_extraction_semantic_digest": TICKET30_AUDITED_ARTIFACT_SHA256,
            "frozen_report_artifact_sha256": TICKET30_AUDITED_ARTIFACT_SHA256,
            "supervision_admission": supervision.to_public_dict(),
            "full_fold_supervision_admission": admit_ticket30_fold_supervision(
                fold_manifest,
                fold,
                TICKET30_ARTIFACT,
                expected_artifact_sha256=TICKET30_AUDITED_ARTIFACT_SHA256,
            ).to_public_dict(),
            "feature_manifest_sha256": feature_evidence.get("feature_manifest_sha256"),
            "held_out_inputs_excluded_from_adaptation": True,
            "raw_report_text_read": False,
        }
        _write_private_json(public_path, public_evidence)
    return result


def _read_materialization_compute(path: Path) -> Mapping[str, object] | None:
    """Read GNU-time output, including its nonzero-exit preamble when present."""

    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise Ticket36Error(f"could not read Ticket 36 materialization compute: {path}") from exc
    if not raw.strip():
        return None
    lines = raw.splitlines()
    timer_start: int | None = None
    timer_payload: Mapping[str, object] | None = None
    try:
        whole_payload = json.loads(raw)
    except json.JSONDecodeError:
        whole_payload = None
    if isinstance(whole_payload, Mapping):
        timer_start = 0
        timer_payload = cast(Mapping[str, object], whole_payload)
    candidate_starts = range(len(lines) - 1, -1, -1) if timer_payload is None else ()
    for index in candidate_starts:
        if not lines[index].lstrip().startswith("{"):
            continue
        try:
            candidate = json.loads("\n".join(lines[index:]))
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, Mapping):
            timer_start = index
            timer_payload = cast(Mapping[str, object], candidate)
            break
    if timer_payload is None or timer_start is None:
        raise Ticket36Error(f"Ticket 36 materialization timer is not valid JSON: {path}")
    row = dict(timer_payload)
    if row.get("schema_version") != "rsna-ticket36-materialization-compute-v1":
        raise Ticket36Error(f"Ticket 36 materialization timer schema is unsupported: {path}")
    exit_status = row.get("process_exit_status")
    if exit_status is None:
        exit_status = 0
        for line in lines[:timer_start]:
            prefix = line.strip()
            if prefix.startswith("Command exited with non-zero status "):
                try:
                    exit_status = int(prefix.rsplit(" ", 1)[1])
                except ValueError as exc:
                    raise Ticket36Error("Ticket 36 timer has a malformed process exit status") from exc
                break
            if prefix.startswith("Command terminated by signal "):
                try:
                    exit_status = 128 + int(prefix.rsplit(" ", 1)[1])
                except ValueError as exc:
                    raise Ticket36Error("Ticket 36 timer has a malformed termination signal") from exc
                break
    if not isinstance(exit_status, int) or isinstance(exit_status, bool) or exit_status < 0:
        raise Ticket36Error(f"Ticket 36 materialization exit status is invalid: {path}")
    for field in ("elapsed_seconds", "user_cpu_seconds", "system_cpu_seconds"):
        try:
            value = float(cast(Any, row.get(field)))
        except (TypeError, ValueError, OverflowError) as exc:
            raise Ticket36Error(f"Ticket 36 materialization timer has invalid {field}: {path}") from exc
        if not math.isfinite(value) or value < 0:
            raise Ticket36Error(f"Ticket 36 materialization timer has invalid {field}: {path}")
        row[field] = value
    try:
        peak_rss = int(cast(Any, row.get("peak_rss_kb")))
    except (TypeError, ValueError, OverflowError) as exc:
        raise Ticket36Error(f"Ticket 36 materialization timer has invalid peak_rss_kb: {path}") from exc
    if peak_rss < 0:
        raise Ticket36Error(f"Ticket 36 materialization timer has invalid peak_rss_kb: {path}")
    row["peak_rss_kb"] = peak_rss
    row["process_exit_status"] = exit_status
    return row


def _execution_compute(plan: Mapping[str, object]) -> Mapping[str, object]:
    compute_root = RUN_ROOT / "compute"
    allocation_root = RUN_ROOT / "allocations"
    materialization_attempts = []
    unmeasured_materialization_paths = []
    feature_measured_keys = set()
    for path in sorted(compute_root.glob("materialize-*.json")):
        row = _read_materialization_compute(path)
        if row is None:
            unmeasured_materialization_paths.append(path.name)
            continue
        materialization_attempts.append({"path": path.name, **dict(row)})
        feature_measured_keys.add(path.stem.removeprefix("materialize-"))
    batch_evidence_count = len(list(FEATURE_ROOT.glob("batch-*.json")))
    fold_compute = []
    total_training_elapsed = 0.0
    total_training_cpu = 0.0
    max_training_rss = 0
    for arm in ("reference-only", "report-supervision"):
        for fold in FOLD_IDS:
            path = RUN_ROOT / arm / f"fold-{fold}.compute.json"
            row = _read_json(path, "Ticket 36 fold compute")
            elapsed = float(cast(Any, row.get("elapsed_seconds", 0)))
            cpu_seconds = float(cast(Any, row.get("user_cpu_seconds", 0))) + float(
                cast(Any, row.get("system_cpu_seconds", 0)
            ))
            total_training_elapsed += elapsed
            total_training_cpu += cpu_seconds
            max_training_rss = max(max_training_rss, int(cast(Any, row.get("peak_rss_kb", 0))))
            fold_compute.append({"arm": arm, "fold": fold, **dict(row)})
    feature_elapsed = sum(float(cast(Any, row.get("elapsed_seconds", 0))) for row in materialization_attempts)
    feature_cpu = sum(
        float(cast(Any, row.get("user_cpu_seconds", 0)))
        + float(cast(Any, row.get("system_cpu_seconds", 0)))
        for row in materialization_attempts
    )
    feature_allocation_keys: dict[str, str] = {}
    for path in sorted(allocation_root.glob("feature-*.json")):
        allocation = _read_json(path, "Ticket 36 feature allocation")
        job_id = str(allocation.get("job_id", path.stem.removeprefix("feature-")))
        task_id = str(allocation.get("array_task_id", ""))
        restart_value = allocation.get("restart_count")
        if not task_id:
            raise Ticket36Error("Ticket 36 feature allocation is missing its array task id")
        allocation_key = (
            f"{task_id}-{job_id}"
            if restart_value is None
            else f"{task_id}-{job_id}-restart-{int(cast(Any, restart_value))}"
        )
        feature_allocation_keys[allocation_key] = job_id
    training_allocations = []
    for path in sorted(allocation_root.glob("training-*.json")):
        allocation = _read_json(path, "Ticket 36 training allocation")
        job_id = str(allocation.get("job_id", path.stem.removeprefix("training-")))
        restart_count = int(cast(Any, allocation.get("restart_count", 0)))
        compute_path = compute_root / f"training-job-{job_id}-restart-{restart_count}.json"
        compute = (
            _read_json(compute_path, "Ticket 36 training allocation compute")
            if compute_path.is_file()
            else None
        )
        training_allocations.append(
            {
                "allocation_path": path.name,
                "job_id": job_id,
                "restart_count": restart_count,
                "compute": dict(compute) if compute is not None else None,
            }
        )
    training_allocation_compute = [
        cast(Mapping[str, object], row["compute"])
        for row in training_allocations
        if isinstance(row.get("compute"), Mapping)
    ]
    training_allocation_elapsed = sum(
        float(cast(Any, row.get("elapsed_seconds", 0)))
        for row in training_allocation_compute
    )
    training_allocation_cpu = sum(
        float(cast(Any, row.get("user_cpu_seconds", 0)))
        + float(cast(Any, row.get("system_cpu_seconds", 0)))
        for row in training_allocation_compute
    )
    return {
        "schema_version": "rsna-ticket36-observed-compute-v1",
        "feature_materialization": {
            "batch_evidence_count": batch_evidence_count,
            "planned_batch_count": len(cast(Sequence[object], plan.get("feature_batches", ()))),
            "attempt_count": len(feature_allocation_keys),
            "measured_attempt_count": len(materialization_attempts),
            "unmeasured_attempt_paths": unmeasured_materialization_paths,
            "unaccounted_allocation_job_ids": sorted(
                job_id
                for allocation_key, job_id in feature_allocation_keys.items()
                if allocation_key not in feature_measured_keys
            ),
            "measured_nonzero_exit_attempt_count": sum(
                row.get("process_exit_status") != 0 for row in materialization_attempts
            ),
            "elapsed_seconds_sum_across_attempts": feature_elapsed,
            "cpu_seconds_sum_across_attempts": feature_cpu,
            "attempts": materialization_attempts,
        },
        "training": {
            "run_count": len(fold_compute),
            "elapsed_seconds_sum_across_runs": total_training_elapsed,
            "cpu_seconds_sum_across_runs": total_training_cpu,
            "peak_rss_kb_max": max_training_rss,
            "runs": fold_compute,
            "slurm_allocation_attempt_count": len(training_allocations),
            "slurm_allocation_measured_attempt_count": len(training_allocation_compute),
            "slurm_allocation_unaccounted_job_ids": [
                f"{row['job_id']}-restart-{row['restart_count']}"
                for row in training_allocations
                if not isinstance(row.get("compute"), Mapping)
            ],
            "slurm_allocation_elapsed_seconds_sum": training_allocation_elapsed,
            "slurm_allocation_cpu_seconds_sum": training_allocation_cpu,
            "slurm_allocations": training_allocations,
        },
        "combined_cpu_seconds_sum": feature_cpu + training_allocation_cpu,
        "external_spend_usd": 0,
        "gpu_count": 0,
    }


def _write_candidate_review_metadata(
    *,
    arm: str,
    run_paths: Sequence[Path],
    evaluation: OOFEvaluation,
    code_revision: str,
    config: TrainingConfig,
    integration: IntegrationManifest,
    decision: str,
) -> Mapping[str, object]:
    """Write candidate-loader-compatible OOF and checkpoint references without freezing a candidate."""

    review_root = RUN_ROOT / "candidate-review"
    _private_directory(review_root)
    evidence_path = review_root / f"{arm}-oof-evidence.json"
    _write_private_json(evidence_path, evaluation.to_public_dict())
    checkpoint_rows: list[dict[str, object]] = []
    for run_path in run_paths:
        run_payload = _read_json(run_path / "run.json", "candidate-review source run")
        checkpoint_path = run_path / "checkpoints/final.pt"
        if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
            raise Ticket36Error("candidate-review arm is missing a reloadable final checkpoint")
        checkpoint_rows.append(
            {
                "fold": run_payload.get("fold"),
                "run_id": run_payload.get("run_id"),
                "checkpoint_kind": "final_selected",
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": _sha256_file(checkpoint_path),
            }
        )
    if tuple(sorted(int(cast(Any, row["fold"])) for row in checkpoint_rows)) != FOLD_IDS:
        raise Ticket36Error("candidate-review metadata must contain one final checkpoint per fold")
    metadata: dict[str, object] = {
        "schema_version": "rsna-ticket36-candidate-review-metadata-v1",
        "ticket": 36,
        "arm": arm,
        "decision": decision,
        "selected_for_separate_candidate_review": (
            (
                arm == "report-supervision"
                and decision == "advance_report_supervision_to_separate_candidate_review"
            )
            or (
                arm == "reference-only"
                and decision == "reject_expanded_report_supervision"
            )
        ),
        "candidate_freeze_performed": False,
        "final_candidate_freeze": "blocked",
        "source_runs": [str(path.resolve()) for path in run_paths],
        "final_checkpoints": checkpoint_rows,
        "oof_evidence_path": str(evidence_path.resolve()),
        "oof_evidence_sha256": _sha256_file(evidence_path),
        "code_revision": code_revision,
        "source_version": integration.source_version,
        "source_fingerprint": integration.archive_fingerprint,
        "integration_manifest_checksum": integration.manifest_checksum,
        "fold_manifest_checksum": evaluation.fold_manifest_checksum,
        "model": config.model_config().to_dict(),
        "preprocessing": config.preprocess_config().cache_config(),
        "packing": config.packing_config().to_dict(),
    }
    metadata_path = review_root / f"{arm}-metadata.json"
    _write_private_json(metadata_path, metadata)
    return {"path": str(metadata_path.resolve()), "sha256": _sha256_file(metadata_path)}


def _write_execution_summary(
    plan: Mapping[str, object],
    execution_compute: Mapping[str, object],
    decision_payload: Mapping[str, object],
) -> None:
    """Refresh measured allocation costs in the decision and readable report."""

    _replace_private_json(RUN_ROOT / "observed-compute.json", execution_compute)
    _replace_private_json(RUN_ROOT / "promotion-decision.json", decision_payload)
    markdown_path = RUN_ROOT / "comparison.md"
    if not markdown_path.is_file():
        raise Ticket36Error("Ticket 36 comparison report is missing during cost accounting")
    markdown = markdown_path.read_text(encoding="utf-8")
    marker = "\n## Feature materialization and total observed compute\n"
    if marker in markdown:
        markdown = markdown.split(marker, 1)[0]
    feature = _mapping(execution_compute.get("feature_materialization"), "feature compute totals")
    training = _mapping(execution_compute.get("training"), "training compute totals")
    cohort = _mapping(plan.get("cohort"), "Ticket 36 cohort")
    decision = str(decision_payload.get("decision"))
    rationale = str(decision_payload.get("rationale"))
    lines = [
        "",
        "## Feature materialization and total observed compute",
        "",
        (
            f"Frozen feature cohorts: {int(cast(Any, cohort['total_feature_studies'])):,} studies "
            f"({int(cast(Any, cohort['gold_reference_studies']))} gold and "
            f"{int(cast(Any, cohort['image_ready_report_studies'])):,} complete-input report studies). "
            f"Ticket 35 reported {int(cast(Any, cohort['ticket35_planned_report_studies'])):,} planned report studies; "
            f"{int(cast(Any, cohort['input_excluded_report_studies']))} with incomplete image inputs were excluded. "
            f"Feature extraction used {len(cast(Sequence[object], plan['feature_batches']))} restartable batches. "
            "The Ticket 35 prepared-series cache was read-only; the ResNet-18 encoder was frozen, "
            "and held-out features were used for prediction and evaluation only."
        ),
        "",
        (
            f"Feature materialization measured {float(cast(Any, feature['elapsed_seconds_sum_across_attempts'])):.2f} wall seconds "
            f"and {float(cast(Any, feature['cpu_seconds_sum_across_attempts'])):.2f} CPU seconds across "
            f"{int(cast(Any, feature['measured_attempt_count']))} timed allocation attempts "
            f"({int(cast(Any, feature['batch_evidence_count']))}/{int(cast(Any, feature['planned_batch_count']))} batches sealed)."
        ),
        "",
        (
            f"Training fold-run timers sum to {float(cast(Any, training['elapsed_seconds_sum_across_runs'])):.2f} wall seconds "
            f"and {float(cast(Any, training['cpu_seconds_sum_across_runs'])):.2f} CPU seconds. "
            f"The outer Slurm training allocation measured {float(cast(Any, training['slurm_allocation_elapsed_seconds_sum'])):.2f} wall seconds "
            f"and {float(cast(Any, training['slurm_allocation_cpu_seconds_sum'])):.2f} CPU seconds "
            f"across {int(cast(Any, training['slurm_allocation_measured_attempt_count']))} timed allocation attempt(s). "
            "Outer allocation totals include process startup and comparison work; they are reported separately from per-fold timers."
        ),
        "",
        (
            f"Combined measured CPU time is {float(cast(Any, execution_compute['combined_cpu_seconds_sum'])):.2f} seconds, "
            "using feature-array CPU time plus outer training-allocation CPU time to avoid double-counting the per-fold timers. "
            "Execution used CPU only and zero external spend."
        ),
        "",
        "## Promotion decision",
        "",
        f"**{decision}** — {rationale}.",
        "",
        "Ticket 36 authorizes this bounded experiment only. Candidate freeze, publication, redistribution and release remain blocked pending their separate gate.",
        "",
    ]
    _replace_private_text(
        markdown_path,
        markdown.rstrip() + "\n" + "\n".join(lines),
    )


def finalize_training_allocation_compute(compute_path: Path) -> Mapping[str, object]:
    """Include the enclosing Slurm training allocation in the final cost record."""

    plan = _load_plan()
    _approval_payload(plan)
    expected_root = (RUN_ROOT / "compute").resolve()
    resolved = compute_path.resolve()
    if resolved.parent != expected_root or not resolved.name.startswith("training-job-"):
        raise Ticket36Error("Ticket 36 training compute path is outside the frozen run root")
    compute = _read_json(resolved, "Ticket 36 completed training allocation compute")
    if compute.get("schema_version") != "rsna-ticket36-training-job-compute-v1":
        raise Ticket36Error("Ticket 36 training allocation compute schema is unsupported")
    decision_path = RUN_ROOT / "promotion-decision.json"
    if not decision_path.is_file():
        raise Ticket36Error("Ticket 36 cannot account training compute before all comparison outputs exist")
    decision = dict(_read_json(decision_path, "Ticket 36 promotion decision"))
    execution_compute = _execution_compute(plan)
    decision["resource_cost"] = execution_compute
    decision["training_job_compute_path"] = str(resolved)
    decision["training_job_compute_sha256"] = _sha256_file(resolved)
    _write_execution_summary(plan, execution_compute, decision)
    return execution_compute


def run_ticket36_comparison() -> Mapping[str, object]:
    """Train all six matched runs and generate the complete OOF comparison."""

    _verify_slurm_allocation()
    plan = _load_plan()
    approval = _approval_payload(plan)
    feature_manifest = _load_feature_manifest(plan)
    fold_manifest = load_fold_manifest(FOLD_MANIFEST_PATH)
    integration = load_integration_manifest(INTEGRATION_PATH)
    if (
        fold_manifest.manifest_checksum != plan.get("fold_manifest_checksum")
        or integration.manifest_checksum != plan.get("integration_manifest_checksum")
    ):
        raise Ticket36Error("Ticket 36 frozen fold or integration manifest changed")
    config = _make_training_config(integration)
    if config.to_dict() != plan.get("training_configuration"):
        raise Ticket36Error("Ticket 36 training configuration changed after plan freeze")
    features_by_uid = _load_training_features(plan, feature_manifest)
    if set(features_by_uid) != set(integration.study_uids):
        raise Ticket36Error("Ticket 36 frozen features do not cover the exact merged image cohort")
    preflight_attempt = (
        f"{os.environ['SLURM_JOB_ID']}-restart-"
        f"{os.environ.get('SLURM_RESTART_COUNT', '0')}"
    )
    preflight = run_cpu_preflight(
        output_path=RUN_ROOT / f"cpu-preflight-{preflight_attempt}.json",
        require_isolated_python=True,
        require_slurm=True,
        required_slurm_resources=_RESOURCE_REQUIREMENTS,
    )
    require_cpu_preflight(preflight, require_slurm=True, require_isolated_python=True)
    if not preflight.passed:
        raise PreflightError("Ticket 36 CPU preflight did not pass")
    feature_evidence = {
        "identity": plan.get("feature_identity"),
        "identity_fingerprint": plan.get("feature_identity_fingerprint"),
        "feature_manifest_sha256": feature_manifest.get("feature_manifest_sha256"),
        "feature_count": feature_manifest.get("feature_study_count"),
        "feature_cache_schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "source_ticket35_series_cache": str(TICKET35_SERIES_CACHE.resolve()),
        "source_ticket35_cache_access": "read-only",
        "read_only": True,
    }
    outputs: dict[str, list[Path]] = {"reference": [], "report": []}
    completed_runs: list[dict[str, object]] = []
    gold_by_uid = {
        item.study_instance_uid: item for item in fold_manifest.gold_memberships
    }
    for fold in FOLD_IDS:
        supervision = admit_ticket30_fold_supervision(
            fold_manifest,
            fold,
            TICKET30_ARTIFACT,
            expected_artifact_sha256=TICKET30_AUDITED_ARTIFACT_SHA256,
        ).for_image_studies(integration.study_uids)
        heldout_gold_uids = tuple(
            item.study_instance_uid
            for item in fold_manifest.gold_memberships
            if item.fold == fold
        )
        heldout_partition_uids = tuple(fold_manifest.held_out_study_uids(fold))
        training_gold = tuple(
            item.study_instance_uid
            for item in fold_manifest.gold_memberships
            if item.fold != fold
        )
        reference_examples = tuple(
            _reference_example(
                uid,
                features_by_uid[uid],
                gold_by_uid[uid].reference_labels,
                fold=fold,
                fold_manifest_checksum=fold_manifest.manifest_checksum,
                training_uids=training_gold,
                heldout_uids=heldout_partition_uids,
            )
            for uid in training_gold
        )
        heldout_examples = tuple(
            _reference_example(
                uid,
                features_by_uid[uid],
                gold_by_uid[uid].reference_labels,
                fold=fold,
                fold_manifest_checksum=fold_manifest.manifest_checksum,
                training_uids=training_gold,
                heldout_uids=heldout_partition_uids,
            )
            for uid in heldout_gold_uids
        )
        weak_examples = tuple(
            _weak_example(uid, features_by_uid[uid], supervision)
            for uid in supervision.admitted_report_only_study_uids
        )
        if set(item.study_instance_uid for item in reference_examples) != set(training_gold):
            raise Ticket36Error(f"fold {fold} reference exposure differs from the frozen gold partition")
        if set(item.study_instance_uid for item in heldout_examples) != set(heldout_gold_uids):
            raise Ticket36Error(f"fold {fold} held-out feature membership differs from the fold manifest")
        if set(item.study_instance_uid for item in weak_examples) != set(
            supervision.admitted_report_only_study_uids
        ):
            raise Ticket36Error(f"fold {fold} report admission differs from its image-scoped labels")
        studies = _load_single_heldout_study(integration, fold_manifest, fold)
        reference_result = _run_arm(
            arm="reference",
            fold=fold,
            studies=studies,
            fold_manifest=fold_manifest,
            integration=integration,
            config=config,
            feature_examples=(reference_examples, heldout_examples, ()),
            supervision=None,
            feature_evidence=feature_evidence,
            preflight=preflight,
            plan=plan,
        )
        report_result = _run_arm(
            arm="report",
            fold=fold,
            studies=studies,
            fold_manifest=fold_manifest,
            integration=integration,
            config=config,
            feature_examples=(reference_examples, heldout_examples, weak_examples),
            supervision=supervision,
            feature_evidence=feature_evidence,
            preflight=preflight,
            plan=plan,
        )
        for arm, result in (("reference", reference_result), ("report", report_result)):
            run_dir = RUN_ROOT / ("reference-only" if arm == "reference" else "report-supervision") / f"fold-{fold}"
            if not (run_dir / "run.json").is_file() or not (run_dir / "checkpoints/final.pt").is_file():
                raise Ticket36Error(f"fold {fold} {arm} run did not complete with a reloadable checkpoint")
            outputs[arm].append(run_dir)
            run_payload = _read_json(run_dir / "run.json", "Ticket 36 run")
            completed_runs.append(
                {
                    "fold": fold,
                    "arm": arm,
                    "run_id": run_payload.get("run_id"),
                    "checkpoint_sha256": _sha256_file(run_dir / "checkpoints/final.pt"),
                    "completed_this_attempt": result is not None,
                }
            )

    from .supervision_comparison import (
        compare_supervision_oof,
        render_supervision_comparison_markdown,
    )

    comparison = compare_supervision_oof(
        outputs["reference"],
        outputs["report"],
        fold_manifest,
        bootstrap_resamples=TICKET36_BOOTSTRAP_RESAMPLES,
        bootstrap_seed=TICKET36_BOOTSTRAP_SEED,
        bootstrap_confidence_level=0.95,
        require_additive_gold_exposure=True,
        additive_design_id=TICKET34_ADDITIVE_DESIGN,
        weak_sampling_policy=TICKET34_WEAK_SAMPLING_POLICY,
        report_ticket_number=36,
    )
    comparison_path = RUN_ROOT / "comparison.json"
    markdown_path = RUN_ROOT / "comparison.md"
    expected_comparison = comparison.to_public_dict()
    expected_markdown = render_supervision_comparison_markdown(comparison)
    if (
        comparison_path.is_symlink()
        or markdown_path.is_symlink()
        or (comparison_path.exists() and not comparison_path.is_file())
        or (markdown_path.exists() and not markdown_path.is_file())
    ):
        raise Ticket36Error("existing Ticket 36 comparison outputs are unsafe")
    summary_marker = "\n## Feature materialization and total observed compute\n"
    if comparison_path.exists():
        existing_comparison = _read_json(comparison_path, "existing Ticket 36 comparison")
        if existing_comparison != expected_comparison:
            raise Ticket36Error("existing Ticket 36 comparison JSON disagrees with completed fold runs")
    else:
        atomic_json_write(comparison_path, expected_comparison)
        comparison_path.chmod(0o600)
    if markdown_path.exists():
        existing_markdown = markdown_path.read_text(encoding="utf-8")
        existing_markdown_base = existing_markdown.split(summary_marker, 1)[0]
        if existing_markdown_base != expected_markdown:
            raise Ticket36Error("existing Ticket 36 comparison Markdown disagrees with completed fold runs")
    else:
        # The pair cannot be renamed atomically. Publish either missing member
        # after validating its existing counterpart so a retry repairs a stop
        # between the two writes without hiding conflicting output.
        _replace_private_text(markdown_path, expected_markdown)
    paired = comparison.macro_auc_delta_interval
    if (
        comparison.pooled_macro_auc_delta is not None
        and comparison.pooled_macro_auc_delta > 0
        and comparison.positive_folds > comparison.negative_folds
    ):
        decision = "advance_report_supervision_to_separate_candidate_review"
        rationale = (
            "the pooled point estimate is positive and more folds favor report supervision; "
            "this advances the arm for review without claiming a demonstrated gain"
        )
    elif (
        comparison.pooled_macro_auc_delta is not None
        and comparison.pooled_macro_auc_delta < 0
        and paired.upper is not None
        and paired.upper < 0
        and comparison.negative_folds > comparison.positive_folds
    ):
        decision = "reject_expanded_report_supervision"
        rationale = "the pooled delta and paired interval are negative, with more folds favoring reference-only"
    else:
        decision = "inconclusive_retain_both_arms_without_promotion"
        rationale = "the paired uncertainty does not establish a consistent report-supervision benefit or harm"
    execution_compute = _execution_compute(plan)
    _replace_private_json(RUN_ROOT / "observed-compute.json", execution_compute)
    candidate_review = {
        "reference-only": _write_candidate_review_metadata(
            arm="reference-only",
            run_paths=outputs["reference"],
            evaluation=comparison.gold_only,
            code_revision=comparison.code_revision,
            config=config,
            integration=integration,
            decision=decision,
        ),
        "report-supervision": _write_candidate_review_metadata(
            arm="report-supervision",
            run_paths=outputs["report"],
            evaluation=comparison.report_supervised,
            code_revision=comparison.code_revision,
            config=config,
            integration=integration,
            decision=decision,
        ),
    }
    decision_payload: dict[str, object] = {
        "schema_version": "rsna-ticket36-promotion-decision-v1",
        "ticket": 36,
        "decision": decision,
        "rationale": rationale,
        "interpretation": comparison.interpretation,
        "paired_delta": comparison.pooled_macro_auc_delta,
        "paired_95_interval": [paired.lower, paired.upper],
        "undefined_resamples": paired.undefined_resamples,
        "resamples": paired.resamples,
        "positive_folds": comparison.positive_folds,
        "negative_folds": comparison.negative_folds,
        "within_fold_macro_auc": {
            "reference_only": comparison.gold_only.mean_fold_macro_auc,
            "report_supervision": comparison.report_supervised.mean_fold_macro_auc,
            "reference_fold_sd": comparison.gold_only.fold_macro_auc_std,
            "report_fold_sd": comparison.report_supervised.fold_macro_auc_std,
        },
        "resource_cost": execution_compute,
        "approval_scope": approval.get("scope"),
        "approval_sha256": _sha256_file(APPROVAL_PATH),
        "candidate_promotion_authorized": False,
        "final_candidate_freeze": "blocked",
        "release_actions": "blocked",
        "completed_runs": completed_runs,
        "comparison_path": str(comparison_path.resolve()),
        "candidate_review_metadata": candidate_review,
        "feature_manifest_sha256": feature_manifest.get("feature_manifest_sha256"),
    }
    _write_execution_summary(plan, execution_compute, decision_payload)
    return decision_payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze", help="freeze exact inputs and training/resource policy")
    materialize = subparsers.add_parser("materialize", help="materialize one restartable feature batch")
    materialize.add_argument("--batch-index", type=int, required=True)
    subparsers.add_parser("seal-features", help="verify and seal every feature batch")
    subparsers.add_parser("run", help="train all matched fold arms and write comparison")
    finalize_compute = subparsers.add_parser(
        "finalize-compute", help="include the outer Slurm training allocation in cost evidence"
    )
    finalize_compute.add_argument("--training-job-compute-path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze_ticket36_plan()
        elif args.command == "materialize":
            result = materialize_feature_batch(args.batch_index)
        elif args.command == "seal-features":
            result = seal_feature_manifest()
        elif args.command == "finalize-compute":
            result = finalize_training_allocation_compute(args.training_job_compute_path)
        else:
            result = run_ticket36_comparison()
    except (Ticket36Error, Ticket34Error, Ticket35Error, PreflightError, OSError, ValueError) as exc:
        print(f"Ticket 36 failed: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
