"""Bounded, training-only real-image and optimizer diagnostic for Ticket 29."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import stat
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from getpass import getuser
from pathlib import Path
from typing import Mapping, Sequence, cast

import numpy as np
import torch

from .archive import atomic_json_write
from .evaluation import roc_auc_score
from .folds import FoldManifest, FoldMembership, load_fold_manifest
from .inventory import identify_archive
from .model import HierarchicalStudyModel
from .real_inference import (
    IntegrationManifest,
    LoadedStudy,
    iter_real_studies,
    load_integration_manifest,
)
from .targets import TARGET_COLUMNS
from .training import (
    TrainingConfig,
    TrainingDataError,
    TrainingExample,
    build_fold_training_examples,
    load_training_checkpoint,
    masked_bce_loss,
    require_cpu_preflight,
    run_cpu_preflight,
    seed_everything,
    train_fixed_budget,
)


DIAGNOSTIC_SCHEMA_VERSION = "rsna-knee-real-image-diagnostic-v1"
PLAN_SCHEMA_VERSION = "rsna-knee-diagnostic-plan-v1"
CAMPAIGN_GPU_BUDGET_HOURS = 20.0
FOLD_ID = 0
COHORT_SIZE = 8
SHORT_STEPS = 100
MID_STEPS = 250
LONG_STEPS = 500
CURVE_CHECKPOINT_INTERVAL_STEPS = 2
DIAGNOSTIC_SEED = 20260925
BATCH_ORDER_SEED = 20260926
SLURM_RESOURCES: dict[str, object] = {
    "partition": "cluster1-cpu",
    "nodes": 1,
    "ntasks": 1,
    "cpus_per_task": 8,
    "memory_gb": 32,
    "wall_time_minutes": 360,
    "exclusive": False,
}
SUCCESS_CRITERIA: dict[str, int | float | bool] = {
    "cohort_studies": COHORT_SIZE,
    "minimum_valid_series_per_study": 1,
    "minimum_valid_slices_per_study": 3,
    "minimum_nonconstant_source_slice_fraction": 0.80,
    "gradient_global_norm_minimum": 1e-8,
    "relative_parameter_l2_change_minimum": 1e-8,
    "long_run_loss_fraction_of_training_prevalence_control_maximum": 0.50,
    "long_run_supported_target_macro_auc_minimum": 0.85,
    "long_run_supported_targets_minimum": 12,
    "auc_is_training_only": True,
    "generalization_claim": False,
}


class DiagnosticError(RuntimeError):
    """Raised when input provenance, training scope, or the bounded run is invalid."""


class GPURequestRejected(DiagnosticError):
    """Raised by the campaign ledger before a mocked or real GPU request is admitted."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"could not read {description}") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"{description} must be a JSON object")
    return value


def _write_private_new_json(path: Path, value: Mapping[str, object]) -> None:
    """Publish an identifier-bearing plan once, with private mode and no overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _selection_members(selection_path: Path) -> tuple[set[str], set[str]]:
    selection = _read_object(selection_path, "source selection report")
    gold = selection.get("gold_study_uids")
    report_only = selection.get("report_only_study_uids")
    if not isinstance(gold, list) or not isinstance(report_only, list):
        raise DiagnosticError("source selection report has no prepared-cohort membership")
    if any(not isinstance(uid, str) or not uid.strip() for uid in (*gold, *report_only)):
        raise DiagnosticError("source selection report contains invalid study identities")
    gold_set = set(cast(list[str], gold))
    report_set = set(cast(list[str], report_only))
    if len(gold_set) != len(gold) or len(report_set) != len(report_only):
        raise DiagnosticError("source selection report contains duplicate study identities")
    if gold_set.intersection(report_set):
        raise DiagnosticError("source selection report overlaps reference and report-only studies")
    return gold_set, report_set


def _greedy_training_cohort(fold_manifest: FoldManifest, *, fold: int, size: int) -> tuple[str, ...]:
    """Choose a diverse reference-only subset using labels and identifiers only."""

    if size < 2:
        raise DiagnosticError("training diagnostic cohort must include at least two studies")
    eligible = tuple(
        item for item in fold_manifest.gold_memberships if item.fold != fold
    )
    if len(eligible) < size:
        raise DiagnosticError("corrected fold has too few reference-training studies")
    selected: list[FoldMembership] = []
    remaining = list(eligible)

    def score(candidate: FoldMembership) -> tuple[int, int, str]:
        rows = [*selected, candidate]
        support = 0
        balance = 0
        for target_index in range(len(TARGET_COLUMNS)):
            positive = sum(item.reference_labels[target_index] == 1 for item in rows)
            negative = sum(item.reference_labels[target_index] == 0 for item in rows)
            support += int(positive > 0 and negative > 0)
            balance += min(positive, negative)
        uid = candidate.study_instance_uid
        tie_break = hashlib.sha256(f"ticket-29-cohort-v1:{uid}".encode("utf-8")).hexdigest()
        return support, balance, tie_break

    while len(selected) < size:
        candidate = max(remaining, key=score)
        selected.append(candidate)
        remaining.remove(candidate)
    ids = tuple(item.study_instance_uid for item in selected)
    if len(set(ids)) != size:
        raise DiagnosticError("frozen diagnostic cohort contains duplicate study identities")
    if set(ids).intersection(fold_manifest.held_out_study_uids(fold)):
        raise DiagnosticError("frozen diagnostic cohort includes held-out studies")
    memberships = {item.study_instance_uid: item for item in eligible}
    for target_index in range(len(TARGET_COLUMNS)):
        labels = [memberships[uid].reference_labels[target_index] for uid in ids]
        if 0 not in labels or 1 not in labels:
            raise DiagnosticError("diagnostic cohort must support both classes for all targets")
    return ids


def freeze_diagnostic_plan(
    *,
    integration_manifest_path: str | Path,
    fold_manifest_path: str | Path,
    selection_report_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Freeze exact data, fold, cohort, checks, and CPU/GPU bounds before pixel reads."""

    integration_path = Path(integration_manifest_path)
    fold_path = Path(fold_manifest_path)
    selection_path = Path(selection_report_path)
    integration = load_integration_manifest(integration_path)
    fold_manifest = load_fold_manifest(fold_path)
    if fold_manifest.superseded or fold_manifest.generation_version != 2:
        raise DiagnosticError("Ticket 29 requires the corrected version-2 fold manifest")
    if fold_manifest.fold_ids != (0, 1, 2) or FOLD_ID not in fold_manifest.fold_ids:
        raise DiagnosticError("corrected fold manifest does not declare all three expected folds")
    if (
        integration.source_version != fold_manifest.source_version
        or integration.archive_fingerprint != fold_manifest.source_fingerprint
        or integration.source_manifest_checksum != fold_manifest.source_manifest_checksum
    ):
        raise DiagnosticError("integration and corrected fold source identities do not agree")
    if integration.preprocess_config.image_size != (128, 128):
        raise DiagnosticError("Ticket 29 must use the selected 128-pixel training input identity")
    if tuple(TARGET_COLUMNS) != (
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    ):
        raise DiagnosticError("canonical twelve-target order has changed")

    selected_gold, selected_report_only = _selection_members(selection_path)
    integration_ids = set(integration.study_uids)
    fold_gold = {item.study_instance_uid for item in fold_manifest.gold_memberships}
    fold_report_only = {item.study_instance_uid for item in fold_manifest.report_only_memberships}
    if len(fold_manifest.memberships) != 4407 or len(fold_gold) != 58 or len(fold_report_only) != 4349:
        raise DiagnosticError("corrected source fold manifest no longer has the reconciled 4,407-study cohort")
    if len(integration_ids) != 158 or len(selected_gold) != 58 or len(selected_report_only) != 100:
        raise DiagnosticError("selected image integration no longer has the 158-study scope")
    if integration_ids != selected_gold | selected_report_only or fold_gold != selected_gold:
        raise DiagnosticError("prepared integration membership disagrees with selected source cohorts")
    cohort = _greedy_training_cohort(fold_manifest, fold=FOLD_ID, size=COHORT_SIZE)
    if not set(cohort).issubset(integration_ids):
        raise DiagnosticError("frozen reference cohort is not covered by the prepared image integration")
    uid_to_membership = {
        item.study_instance_uid: item for item in fold_manifest.gold_memberships
    }
    label_counts = [
        {
            "positive": sum(uid_to_membership[uid].reference_labels[index] == 1 for uid in cohort),
            "negative": sum(uid_to_membership[uid].reference_labels[index] == 0 for uid in cohort),
        }
        for index in range(len(TARGET_COLUMNS))
    ]
    selection_payload = _read_object(selection_path, "source selection report")
    storage_estimate = selection_payload.get("storage_estimate")
    if not isinstance(storage_estimate, dict):
        raise DiagnosticError("source selection report has no storage projection")
    source_archive_bytes = storage_estimate.get("source_archive_bytes")
    if not isinstance(source_archive_bytes, int) or isinstance(source_archive_bytes, bool):
        raise DiagnosticError("source selection report has no source archive byte count")

    payload: dict[str, object] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "ticket": "29",
        "source": {
            "source_version": integration.source_version,
            "archive_fingerprint": integration.archive_fingerprint,
            "source_manifest_checksum": integration.source_manifest_checksum,
            "source_archive_bytes": source_archive_bytes,
            "integration_manifest_checksum": integration.manifest_checksum,
            "integration_manifest_file_sha256": _file_sha256(integration_path),
            "integration_studies": len(integration.study_uids),
            "prepared_reference_studies": len(selected_gold),
            "prepared_report_only_studies": len(selected_report_only),
            "full_source_fold_studies": len(fold_manifest.memberships),
            "full_source_reference_studies": len(fold_gold),
            "full_source_report_only_studies": len(fold_report_only),
            "selection_report_sha256": _file_sha256(selection_path),
            "preprocessing": integration.preprocess_config.cache_config(),
            "packing": integration.packing_config.to_dict(),
            "reusable_cache": {
                "status": "no cache artifact supplied; read original archive through existing decoder and preprocessing path",
                "cache_identity": None,
            },
        },
        "fold": {
            "fold_manifest_checksum": fold_manifest.manifest_checksum,
            "fold_manifest_file_sha256": _file_sha256(fold_path),
            "schema_version": fold_manifest.schema_version,
            "generation_version": fold_manifest.generation_version,
            "fold_ids": list(fold_manifest.fold_ids),
            "selected_fold": FOLD_ID,
            "split_unit": fold_manifest.grouping.status,
            "training_study_count": len(
                [item for item in fold_manifest.gold_memberships if item.fold != FOLD_ID]
            ),
            "held_out_study_count": len(
                [item for item in fold_manifest.gold_memberships if item.fold == FOLD_ID]
            ),
        },
        "targets": list(TARGET_COLUMNS),
        "cohort": {
            "selection_policy": "greedy maximum two-class target coverage, then class balance, then stable SHA-256 tie break; labels only",
            "study_uids": list(cohort),
            "membership": "reference-labeled fold-0 training only",
            "target_support": label_counts,
            "cohort_membership_sha256": _json_sha256(sorted(cohort)),
        },
        "training": {
            "model": "existing mechanics baseline via train_fixed_budget",
            "steps": [0, SHORT_STEPS, MID_STEPS, LONG_STEPS],
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "schedule": "constant",
            "batch_size": 2,
            "augmentation": "none",
            "seed": DIAGNOSTIC_SEED,
            "batch_order_seed": BATCH_ORDER_SEED,
            "training_prevalence_control": "constant logits at the selected training-cohort per-target prevalence, clipped to [1e-6,1-1e-6]",
            "held_out_data_read": False,
        },
        "success_criteria": SUCCESS_CRITERIA,
        "resources": {
            "external_spend_usd": 0,
            "cpu_slurm_request": dict(SLURM_RESOURCES),
            "gpu_request": None,
            "gpu_budget_hours": CAMPAIGN_GPU_BUDGET_HOURS,
            "cpu_wall_time_is_not_gpu_allocation": True,
        },
        "storage_policy": {
            "soft_budget_bytes": 2_000_000_000_000,
            "preserve_original_archive": True,
            "write_decoded_image_cache": False,
            "write_prepared_pixel_cache": False,
            "expected_persistent_diagnostic_outputs": "logs, checksums, report, and atomic training checkpoints only",
        },
    }
    payload["plan_sha256"] = _json_sha256(payload)
    destination = Path(output_path)
    _write_private_new_json(destination, payload)
    return payload


def load_diagnostic_plan(path: str | Path) -> dict[str, object]:
    plan = _read_object(Path(path), "frozen diagnostic plan")
    checksum = plan.get("plan_sha256")
    without_checksum = dict(plan)
    without_checksum.pop("plan_sha256", None)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION or checksum != _json_sha256(without_checksum):
        raise DiagnosticError("frozen diagnostic plan checksum or schema is invalid")
    return plan


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DiagnosticError(f"frozen plan {name} must be an object")
    return cast(Mapping[str, object], value)


def _as_int(value: object) -> int:
    """Convert a value already checked by the artifact boundary."""

    return int(cast(int | float | str, value))


def _as_float(value: object) -> float:
    """Convert a value already checked by the artifact boundary."""

    return float(cast(int | float | str, value))


def summarize_gpu_attempts(
    attempts: Sequence[Mapping[str, object]],
    *,
    budget_hours: float = CAMPAIGN_GPU_BUDGET_HOURS,
) -> dict[str, object]:
    """Count allocated GPU time while keeping submit-to-start queue time separate."""

    if not math.isfinite(budget_hours) or budget_hours <= 0:
        raise ValueError("GPU budget must be a positive finite number of hours")
    allocated_gpu_seconds = 0.0
    queue_wait_seconds = 0.0
    allocated_attempt_count = 0
    for index, attempt in enumerate(attempts):
        gpus = attempt.get("allocated_gpu_count", 0)
        seconds = attempt.get("allocated_seconds", 0.0)
        wait = attempt.get("queue_wait_seconds", 0.0)
        for name, value in (("allocated_gpu_count", gpus), ("allocated_seconds", seconds), ("queue_wait_seconds", wait)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"GPU attempt {index} has invalid {name}")
        if cast(int | float, gpus) > 0 and cast(int | float, seconds) > 0:
            allocated_attempt_count += 1
            allocated_gpu_seconds += _as_float(gpus) * _as_float(seconds)
        queue_wait_seconds += _as_float(wait)
    allocated_hours = allocated_gpu_seconds / 3600.0
    remaining = max(0.0, budget_hours - allocated_hours)
    return {
        "budget_hours": budget_hours,
        "allocated_attempt_count": allocated_attempt_count,
        "allocated_gpu_hours": allocated_hours,
        "queue_wait_seconds": queue_wait_seconds,
        "remaining_gpu_hours": remaining,
        "within_budget": allocated_hours <= budget_hours,
    }


def admit_mock_gpu_request(
    attempts: Sequence[Mapping[str, object]],
    *,
    active_or_pending_requests: int,
    proposed_gpu_count: int,
    proposed_wall_time_seconds: int,
    accounting_complete: bool = True,
    budget_hours: float = CAMPAIGN_GPU_BUDGET_HOURS,
) -> dict[str, object]:
    """Apply the same single-request and reserved-hours gate at a scheduler seam."""

    if not accounting_complete:
        raise GPURequestRejected("campaign allocation history is incomplete")
    if active_or_pending_requests != 0:
        raise GPURequestRejected("only one active or pending GPU request is allowed")
    if proposed_gpu_count <= 0 or proposed_wall_time_seconds <= 0:
        raise GPURequestRejected("a proposed GPU request must have positive GPU count and wall time")
    if proposed_gpu_count > 1:
        raise GPURequestRejected("the campaign full-job ceiling is one GPU")
    ledger = summarize_gpu_attempts(attempts, budget_hours=budget_hours)
    reservation_hours = proposed_gpu_count * proposed_wall_time_seconds / 3600.0
    if not ledger["within_budget"] or reservation_hours > _as_float(ledger["remaining_gpu_hours"]):
        raise GPURequestRejected("proposed maximum allocation would exceed the campaign GPU-hour budget")
    return {
        **ledger,
        "proposed_reservation_hours": reservation_hours,
        "admitted": True,
    }


STORAGE_SCAN_MAX_ENTRIES = 250_000
STORAGE_SCAN_MAX_SECONDS = 120.0


def _measure_tree(root: Path) -> dict[str, int]:
    """Boundedly measure logical bytes without following symlinks or reading contents."""

    if not root.exists() or not root.is_dir():
        raise DiagnosticError("declared storage root is unavailable")
    total = 0
    files = 0
    entries_seen = 0
    started = time.monotonic()
    stack = [root]
    while stack:
        if time.monotonic() - started > STORAGE_SCAN_MAX_SECONDS:
            raise DiagnosticError("bounded storage metadata scan exceeded its time limit")
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    entries_seen += 1
                    if entries_seen > STORAGE_SCAN_MAX_ENTRIES:
                        raise DiagnosticError("bounded storage metadata scan exceeded its entry limit")
                    if time.monotonic() - started > STORAGE_SCAN_MAX_SECONDS:
                        raise DiagnosticError("bounded storage metadata scan exceeded its time limit")
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        raise DiagnosticError("storage tree changed during measurement") from exc
                    if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                        stack.append(Path(entry.path))
                    else:
                        total += int(info.st_size)
                        files += 1
        except OSError as exc:
            raise DiagnosticError("could not measure declared storage root") from exc
    return {"logical_bytes": total, "file_entries": files}


def _storage_snapshot(roots: Sequence[Path], *, archive_path: Path) -> dict[str, object]:
    resolved = tuple(path.resolve() for path in roots)
    if len(set(resolved)) != len(resolved):
        raise DiagnosticError("storage roots must be distinct")
    for index, root in enumerate(resolved):
        if any(root == other or root in other.parents or other in root.parents for other in resolved[index + 1 :]):
            raise DiagnosticError("storage roots must not overlap")
    root_metrics = {f"root_{index + 1}": _measure_tree(root) for index, root in enumerate(resolved)}
    project_bytes = sum(int(item["logical_bytes"]) for item in root_metrics.values())
    archive_bytes = archive_path.stat().st_size
    soft_budget = 2_000_000_000_000
    filesystem = os.statvfs(resolved[0])
    return {
        "measurement": "Slurm compute-node metadata walk bounded to 250,000 entries and 120 seconds per root; logical file sizes; no content hashing",
        "root_metrics": root_metrics,
        "measured_roots_logical_bytes": project_bytes,
        "archive_bytes": archive_bytes,
        "archive_in_measured_roots": any(
            archive_path.resolve() == root or root in archive_path.resolve().parents for root in resolved
        ),
        "soft_budget_bytes": soft_budget,
        "projected_headroom_before_run_bytes": max(0, soft_budget - project_bytes),
        "projected_additional_peak_reserve_bytes": 64_000_000,
        "projected_peak_bytes_with_reserve": project_bytes + 64_000_000,
        "projected_peak_within_soft_budget": project_bytes + 64_000_000 <= soft_budget,
        "shared_filesystem_available_bytes": filesystem.f_bavail * filesystem.f_frsize,
        "shared_free_space_is_project_headroom": False,
    }


def _owner_gpu_queue_snapshot() -> dict[str, object]:
    """Read only the current owner's GPU queue state, without retaining job IDs."""

    try:
        completed = subprocess.run(
            ["squeue", "-h", "-u", getuser(), "-o", "%P|%T"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return {"status": "unavailable", "owner_active_or_pending_requests": None}
    if completed.returncode != 0:
        return {"status": "unavailable", "owner_active_or_pending_requests": None}
    gpu_rows = []
    for line in completed.stdout.splitlines():
        partition, _, state = line.strip().partition("|")
        if partition.startswith("cluster1-gpu") and state:
            gpu_rows.append({"partition": partition, "state": state})
    return {
        "status": "queried",
        "owner_active_or_pending_requests": len(gpu_rows),
        "active_or_pending_partition_states": gpu_rows,
    }


def _slurm_time_limit_minutes(value: str | None) -> int | None:
    """Normalize common Slurm time-limit spellings to minutes."""

    if value is None:
        return None
    raw = value.strip()
    if raw.isdigit():
        minutes = int(raw)
        return minutes if minutes > 0 else None
    days = 0
    clock = raw
    if "-" in raw:
        day_text, clock = raw.split("-", 1)
        if not day_text.isdigit():
            return None
        days = int(day_text)
    parts = clock.split(":")
    if len(parts) == 2:
        hour_text, minute_text = parts
        second_text = "0"
    elif len(parts) == 3:
        hour_text, minute_text, second_text = parts
    else:
        return None
    if not all(part.isdigit() for part in (hour_text, minute_text, second_text)):
        return None
    hours, minutes, seconds = int(hour_text), int(minute_text), int(second_text)
    if seconds >= 60 or minutes >= 60 or (days and hours >= 24) or seconds != 0:
        return None
    total = days * 24 * 60 + hours * 60 + minutes
    return total if total > 0 else None


def _resolve_slurm_time_limit(job_id: str, reported_value: str | None) -> str:
    """Read Slurm's active job record when the batch environment omits its limit."""

    if reported_value is not None and reported_value.strip():
        return reported_value.strip()
    try:
        job_record = subprocess.run(
            ["scontrol", "show", "job", job_id],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DiagnosticError("CPU diagnostic could not query the active Slurm job TimeLimit") from exc
    if job_record.returncode != 0:
        raise DiagnosticError("CPU diagnostic could not query the active Slurm job TimeLimit")
    fields: dict[str, str] = {}
    for field in job_record.stdout.split():
        name, separator, value = field.partition("=")
        if separator:
            fields[name] = value.strip()
    if fields.get("JobId") != job_id:
        raise DiagnosticError("Slurm job record did not identify the active CPU diagnostic job")
    if fields.get("TimeLimit"):
        return fields["TimeLimit"]
    raise DiagnosticError("CPU diagnostic could not resolve the active Slurm job TimeLimit")


def _check_slurm_context() -> dict[str, object]:
    expected = SLURM_RESOURCES
    environment: dict[str, object] = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "nodes": os.environ.get("SLURM_JOB_NUM_NODES"),
        "tasks": os.environ.get("SLURM_NTASKS"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "memory_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "time_limit": os.environ.get("SLURM_TIMELIMIT"),
        "exclusive": os.environ.get("SLURM_JOB_EXCLUSIVE", "0"),
        "gpu_request_fields": {
            key: os.environ.get(key)
            for key in ("SLURM_JOB_GRES", "SLURM_GPUS", "SLURM_JOB_GPUS")
            if os.environ.get(key)
        },
    }
    if (
        not isinstance(environment["job_id"], str)
        or not environment["job_id"].isdigit()
        or environment["partition"] != expected["partition"]
    ):
        raise DiagnosticError("real-image diagnostic requires its declared cluster1-cpu Slurm allocation")
    if environment["gpu_request_fields"]:
        raise DiagnosticError("CPU diagnostic allocation unexpectedly exposes a GPU request")
    expected_mem = {"32G", "32768"}
    if str(environment["memory_per_node"]).upper() not in expected_mem:
        raise DiagnosticError("CPU diagnostic memory allocation differs from the frozen request")
    environment["time_limit"] = _resolve_slurm_time_limit(
        cast(str, environment["job_id"]), cast(str | None, environment["time_limit"])
    )
    observed_wall_minutes = _slurm_time_limit_minutes(cast(str, environment["time_limit"]))
    if (
        environment["nodes"] != "1"
        or environment["tasks"] != "1"
        or environment["cpus_per_task"] != "8"
        or observed_wall_minutes != expected["wall_time_minutes"]
        or str(environment["exclusive"]).lower() in {"1", "true", "yes"}
    ):
        raise DiagnosticError(
            "CPU diagnostic Slurm allocation exceeds or differs from the frozen request "
            f"(nodes={environment['nodes']!r}, tasks={environment['tasks']!r}, "
            f"cpus_per_task={environment['cpus_per_task']!r}, "
            f"time_limit={environment['time_limit']!r}, exclusive={environment['exclusive']!r})"
        )
    # The shared training preflight expects a populated, minute-valued Slurm
    # environment field. Preserve the scheduler's original value in `environment`
    # while passing the verified equivalent to that existing preflight.
    os.environ["SLURM_TIMELIMIT"] = str(observed_wall_minutes)
    return environment


def _verify_manifest_identity(
    plan: Mapping[str, object],
    integration: IntegrationManifest,
    fold_manifest: FoldManifest,
    integration_manifest_path: Path,
    fold_manifest_path: Path,
    selection_report_path: Path,
) -> tuple[tuple[str, ...], dict[str, object]]:
    source = _mapping(plan.get("source"), "source")
    fold = _mapping(plan.get("fold"), "fold")
    cohort = _mapping(plan.get("cohort"), "cohort")
    training_plan = _mapping(plan.get("training"), "training")
    resources = _mapping(plan.get("resources"), "resources")
    storage_policy = _mapping(plan.get("storage_policy"), "storage_policy")
    selected = cohort.get("study_uids")
    if (
        not isinstance(selected, list)
        or len(selected) != COHORT_SIZE
        or any(not isinstance(uid, str) or not uid.strip() for uid in selected)
    ):
        raise DiagnosticError("frozen training cohort is malformed")
    ids = tuple(cast(list[str], selected))
    if len(set(ids)) != COHORT_SIZE:
        raise DiagnosticError("frozen training cohort repeats study identities")
    selected_gold, selected_report = _selection_members(selection_report_path)
    fold_gold = {item.study_instance_uid for item in fold_manifest.gold_memberships}
    fold_report = {item.study_instance_uid for item in fold_manifest.report_only_memberships}
    source_fold_count = len(fold_manifest.memberships)
    training_count = sum(item.fold != FOLD_ID for item in fold_manifest.gold_memberships)
    held_out_count = sum(item.fold == FOLD_ID for item in fold_manifest.gold_memberships)
    selection_payload = _read_object(selection_report_path, "source selection report")
    selection_storage = _mapping(selection_payload.get("storage_estimate"), "selection storage estimate")
    expected_training_plan: dict[str, object] = {
        "model": "existing mechanics baseline via train_fixed_budget",
        "steps": [0, SHORT_STEPS, MID_STEPS, LONG_STEPS],
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "schedule": "constant",
        "batch_size": 2,
        "augmentation": "none",
        "seed": DIAGNOSTIC_SEED,
        "batch_order_seed": BATCH_ORDER_SEED,
        "training_prevalence_control": "constant logits at the selected training-cohort per-target prevalence, clipped to [1e-6,1-1e-6]",
        "held_out_data_read": False,
    }
    expected_storage_policy = {
        "soft_budget_bytes": 2_000_000_000_000,
        "preserve_original_archive": True,
        "write_decoded_image_cache": False,
        "write_prepared_pixel_cache": False,
        "expected_persistent_diagnostic_outputs": "logs, checksums, report, and atomic training checkpoints only",
    }
    if (
        plan.get("ticket") != "29"
        or plan.get("targets") != list(TARGET_COLUMNS)
        or source.get("source_version") != integration.source_version
        or source.get("archive_fingerprint") != integration.archive_fingerprint
        or source.get("integration_manifest_checksum") != integration.manifest_checksum
        or source.get("integration_manifest_file_sha256") != _file_sha256(integration_manifest_path)
        or source.get("source_manifest_checksum") != integration.source_manifest_checksum
        or source.get("integration_studies") != len(integration.study_uids)
        or source.get("prepared_reference_studies") != len(selected_gold)
        or source.get("prepared_report_only_studies") != len(selected_report)
        or source.get("full_source_fold_studies") != source_fold_count
        or source.get("full_source_reference_studies") != len(fold_gold)
        or source.get("full_source_report_only_studies") != len(fold_report)
        or source.get("source_archive_bytes") != selection_storage.get("source_archive_bytes")
        or source.get("preprocessing") != integration.preprocess_config.cache_config()
        or source.get("packing") != integration.packing_config.to_dict()
        or source.get("reusable_cache")
        != {
            "status": "no cache artifact supplied; read original archive through existing decoder and preprocessing path",
            "cache_identity": None,
        }
        or fold.get("fold_manifest_checksum") != fold_manifest.manifest_checksum
        or fold.get("fold_manifest_file_sha256") != _file_sha256(fold_manifest_path)
        or fold.get("schema_version") != fold_manifest.schema_version
        or fold.get("generation_version") != fold_manifest.generation_version
        or fold.get("fold_ids") != list(fold_manifest.fold_ids)
        or fold.get("selected_fold") != FOLD_ID
        or fold.get("split_unit") != fold_manifest.grouping.status
        or fold.get("training_study_count") != training_count
        or fold.get("held_out_study_count") != held_out_count
        or cohort.get("selection_policy")
        != "greedy maximum two-class target coverage, then class balance, then stable SHA-256 tie break; labels only"
        or cohort.get("membership") != "reference-labeled fold-0 training only"
        or cohort.get("cohort_membership_sha256") != _json_sha256(sorted(ids))
        or training_plan != expected_training_plan
        or plan.get("success_criteria") != SUCCESS_CRITERIA
        or resources
        != {
            "external_spend_usd": 0,
            "cpu_slurm_request": dict(SLURM_RESOURCES),
            "gpu_request": None,
            "gpu_budget_hours": CAMPAIGN_GPU_BUDGET_HOURS,
            "cpu_wall_time_is_not_gpu_allocation": True,
        }
        or storage_policy != expected_storage_policy
        or source.get("selection_report_sha256") != _file_sha256(selection_report_path)
    ):
        raise DiagnosticError("frozen plan, resource policy, or real input manifests changed after planning")
    if not selected_gold.issubset(set(integration.study_uids)) or not selected_report.issubset(set(integration.study_uids)):
        raise DiagnosticError("prepared image membership changed after the plan was frozen")
    if not selected_gold.issubset(fold_gold) or not selected_report.issubset(fold_report):
        raise DiagnosticError("prepared labels no longer reconcile with corrected source folds")
    if set(integration.study_uids) != selected_gold | selected_report or fold_gold != selected_gold:
        raise DiagnosticError("prepared image membership disagrees with frozen source membership")
    membership_by_uid = {item.study_instance_uid: item for item in fold_manifest.gold_memberships}
    training = {item.study_instance_uid for item in fold_manifest.gold_memberships if item.fold != FOLD_ID}
    if any(uid not in training or uid not in selected_gold for uid in ids):
        raise DiagnosticError("frozen diagnostic cohort is not reference-only fold-0 training data")
    if set(ids).intersection(fold_manifest.held_out_study_uids(FOLD_ID)):
        raise DiagnosticError("frozen diagnostic cohort intersects the held-out fold")
    if [dict(item) for item in cast(list[dict[str, object]], cohort.get("target_support", []))] != [
        {
            "positive": sum(membership_by_uid[uid].reference_labels[index] == 1 for uid in ids),
            "negative": sum(membership_by_uid[uid].reference_labels[index] == 0 for uid in ids),
        }
        for index in range(len(TARGET_COLUMNS))
    ]:
        raise DiagnosticError("frozen target support no longer agrees with the corrected folds")
    return ids, {
        "integration_studies": len(integration.study_uids),
        "source_studies": len(fold_manifest.memberships),
        "prepared_reference_studies": len(selected_gold),
        "prepared_report_only_studies": len(selected_report),
        "cohort_training_studies": len(ids),
        "cohort_held_out_studies": 0,
        "target_columns": list(TARGET_COLUMNS),
    }


def _pixel_and_geometry_summary(loaded: Sequence[LoadedStudy]) -> tuple[dict[str, object], dict[str, object]]:
    series_status: Counter[str] = Counter()
    plane_counts: Counter[str] = Counter()
    sensitivity_counts: Counter[str] = Counter()
    fat_suppression_counts: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    expected_files = 0
    expected_accepted_files = 0
    decoded_slices = 0
    finite_slices = 0
    nonconstant_slices = 0
    padding_tagged_slices = 0
    studies_with_input = 0
    per_study: list[dict[str, object]] = []
    for item in loaded:
        if item.study.series:
            studies_with_input += 1
        study_nonconstant = 0
        study_slices = 0
        for result in item.series_results:
            series_status[result.status] += 1
            expected_files += result.expected_slice_count
            decoded_slices += result.decoded_slice_count
            if result.status in {"accepted", "accepted_partial"}:
                expected_accepted_files += result.expected_slice_count
            descriptor = result.manifest_series
            plane_counts[descriptor.anatomical_plane or "unknown"] += 1
            sensitivity_counts[
                "unknown" if descriptor.fluid_sensitive is None else str(descriptor.fluid_sensitive).lower()
            ] += 1
            fat_suppression_counts[
                "unknown" if descriptor.fat_suppression is None else str(descriptor.fat_suppression).lower()
            ] += 1
        for series in item.study.series:
            for image in series.slices:
                pixels = np.asarray(image.pixel_array)
                decoded_slices += 0  # decoded total is taken from manifest load results above.
                study_slices += 1
                shape = f"{pixels.shape[0]}x{pixels.shape[1]}" if pixels.ndim == 2 else "non_2d"
                shapes[shape] += 1
                if pixels.ndim == 2 and np.isfinite(pixels).all():
                    finite_slices += 1
                    if float(np.max(pixels)) > float(np.min(pixels)):
                        nonconstant_slices += 1
                        study_nonconstant += 1
                if image.padding_value is not None:
                    padding_tagged_slices += 1
        per_study.append(
            {
                "study_instance_uid": item.study.study_instance_uid,
                "accepted_series": len(item.study.series),
                "source_slices": study_slices,
                "nonconstant_slices": study_nonconstant,
            }
        )
    total_series = sum(series_status.values())
    nonconstant_fraction = nonconstant_slices / max(1, finite_slices)
    summary: dict[str, object] = {
        "studies_with_accepted_series": studies_with_input,
        "expected_series": total_series,
        "series_status": dict(sorted(series_status.items())),
        "expected_source_slice_entries": expected_files,
        "expected_source_slice_entries_in_accepted_series": expected_accepted_files,
        "decoded_slices_in_accepted_series": sum(
            len(series.slices) for study in loaded for series in study.study.series
        ),
        "finite_slices": finite_slices,
        "nonconstant_finite_slices": nonconstant_slices,
        "nonconstant_fraction_of_finite_slices": nonconstant_fraction,
        "padding_value_tagged_slices": padding_tagged_slices,
        "source_dimension_counts": dict(sorted(shapes.items())),
        "anatomical_plane_series_counts": dict(sorted(plane_counts.items())),
        "fluid_sensitive_series_counts": dict(sorted(sensitivity_counts.items())),
        "fat_suppression_series_counts": dict(sorted(fat_suppression_counts.items())),
        "study_identity_alignment": all(
            item.study.study_instance_uid
            in {entry.manifest_series.study_instance_uid for entry in item.series_results}
            for item in loaded
        ),
    }
    private_detail: dict[str, object] = {
        "aggregate": summary,
        "per_study": per_study,
        "failures": [
            {
                "study_instance_uid": study.study.study_instance_uid,
                "series_instance_uid": failure.series_instance_uid,
                "reason": failure.reason,
            }
            for study in loaded
            for failure in study.failures
        ],
    }
    return summary, private_detail


def _example_summary(examples: Sequence[TrainingExample], *, expected_ids: Sequence[str]) -> dict[str, object]:
    if tuple(item.study_instance_uid for item in examples) != tuple(expected_ids):
        raise DiagnosticError("training examples do not preserve the frozen study order")
    mask_counts = np.sum(np.stack([item.supervision_mask for item in examples]), axis=0)
    positives = np.sum(
        np.stack([item.targets * item.supervision_mask for item in examples]), axis=0
    ).astype(int)
    negatives = mask_counts.astype(int) - positives
    finite_pixels = all(np.isfinite(item.images).all() for item in examples)
    valid_studies = 0
    valid_slice_count = 0
    valid_series_count = 0
    per_study_valid_slice_counts: list[int] = []
    for item in examples:
        series_mask = np.asarray(item.series_mask, dtype=bool)
        slice_mask = np.asarray(item.slice_mask, dtype=bool)
        if series_mask.any() and slice_mask.any():
            valid_studies += 1
        valid_series_count += int(series_mask.sum())
        study_valid_slices = int(slice_mask.sum())
        per_study_valid_slice_counts.append(study_valid_slices)
        valid_slice_count += study_valid_slices
    return {
        "training_examples": len(examples),
        "valid_studies_with_series_and_slices": valid_studies,
        "valid_series_mask_count": valid_series_count,
        "valid_slice_mask_count": valid_slice_count,
        "minimum_valid_slice_count_per_study": min(per_study_valid_slice_counts, default=0),
        "packed_pixels_finite": bool(finite_pixels),
        "supervised_target_entries": int(mask_counts.sum()),
        "target_support": [
            {"target": target, "positive": int(positives[index]), "negative": int(negatives[index])}
            for index, target in enumerate(TARGET_COLUMNS)
        ],
    }


def _example_tensors(examples: Sequence[TrainingExample]) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(np.stack([item.images for item in examples])),
        torch.from_numpy(np.stack([item.slice_mask for item in examples])),
        torch.from_numpy(np.stack([item.series_mask for item in examples])),
        torch.from_numpy(np.stack([item.targets for item in examples])),
        torch.from_numpy(np.stack([item.supervision_mask for item in examples])),
    )


def _evaluate(model: HierarchicalStudyModel, examples: Sequence[TrainingExample]) -> dict[str, object]:
    model.eval()
    score_rows: list[np.ndarray] = []
    loss_values: list[float] = []
    with torch.no_grad():
        for start in range(0, len(examples), 2):
            batch = examples[start : start + 2]
            images, slice_mask, series_mask, targets, supervision_mask = _example_tensors(batch)
            logits = model(images, slice_mask, series_mask)
            loss = masked_bce_loss(logits, targets, supervision_mask)
            if loss is None or not bool(torch.isfinite(loss)):
                raise DiagnosticError("training-only evaluation produced a missing or non-finite loss")
            loss_values.append(float(loss.item()))
            score_rows.extend(torch.sigmoid(logits).cpu().numpy())
    scores = np.stack(score_rows)
    labels = np.stack([item.targets for item in examples])
    mask = np.stack([item.supervision_mask for item in examples]).astype(bool)
    target_results: list[dict[str, object]] = []
    defined_aucs: list[float] = []
    for index, target in enumerate(TARGET_COLUMNS):
        selected = mask[:, index]
        auc = roc_auc_score(labels[selected, index].astype(int), scores[selected, index])
        if auc is not None:
            defined_aucs.append(float(auc))
        target_results.append(
            {
                "target": target,
                "auc": auc,
                "positive": int(labels[selected, index].sum()),
                "negative": int(selected.sum() - labels[selected, index].sum()),
            }
        )
    return {
        "training_cohort_mean_masked_bce": float(np.average(loss_values)),
        "supported_target_macro_auc": float(np.mean(defined_aucs)) if defined_aucs else None,
        "supported_targets": len(defined_aucs),
        "target_metrics": target_results,
        "used_training_studies_only": True,
    }


def _prevalence_control(examples: Sequence[TrainingExample]) -> dict[str, object]:
    labels = np.stack([item.targets for item in examples])
    mask = np.stack([item.supervision_mask for item in examples]).astype(bool)
    logit_scores: list[float] = []
    for target_index in range(len(TARGET_COLUMNS)):
        selected = mask[:, target_index]
        prevalence = float(labels[selected, target_index].mean())
        clipped = min(1.0 - 1e-6, max(1e-6, prevalence))
        logit = math.log(clipped / (1.0 - clipped))
        logit_scores.extend(
            [
                float(torch.nn.functional.binary_cross_entropy_with_logits(
                    torch.tensor(logit), torch.tensor(float(label))
                ).item())
                for label in labels[selected, target_index]
            ]
        )
    return {
        "constant_prevalence_masked_bce": float(np.mean(logit_scores)),
        "study_count": len(examples),
        "used_training_studies_only": True,
    }


def _gradient_summary(model: HierarchicalStudyModel, examples: Sequence[TrainingExample]) -> dict[str, object]:
    model.zero_grad(set_to_none=True)
    images, slice_mask, series_mask, targets, supervision_mask = _example_tensors(examples[:2])
    loss = masked_bce_loss(model(images, slice_mask, series_mask), targets, supervision_mask)
    if loss is None or not bool(torch.isfinite(loss)):
        raise DiagnosticError("initial training batch has no finite supervised loss")
    loss.backward()
    squared_norm = 0.0
    finite = True
    nonzero_tensors = 0
    tensors_with_grad = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            finite = False
            continue
        gradient = parameter.grad.detach()
        tensors_with_grad += 1
        finite = finite and bool(torch.isfinite(gradient).all())
        squared_norm += float(torch.sum(gradient.double() ** 2).item())
        nonzero_tensors += int(bool(torch.any(gradient != 0)))
    model.zero_grad(set_to_none=True)
    return {
        "initial_batch_masked_bce": float(loss.detach().item()),
        "all_parameter_gradients_present_and_finite": finite and tensors_with_grad > 0,
        "parameter_tensors_with_gradient": tensors_with_grad,
        "nonzero_gradient_tensors": nonzero_tensors,
        "global_gradient_l2_norm": math.sqrt(squared_norm),
    }


def _relative_parameter_change(initial: Mapping[str, torch.Tensor], final: Mapping[str, torch.Tensor]) -> dict[str, float]:
    difference_squared = 0.0
    initial_squared = 0.0
    for name, initial_tensor in initial.items():
        final_tensor = final[name].detach().cpu()
        delta = final_tensor.double() - initial_tensor.detach().cpu().double()
        difference_squared += float(torch.sum(delta.square()).item())
        initial_squared += float(torch.sum(initial_tensor.detach().cpu().double().square()).item())
    initial_norm = math.sqrt(initial_squared)
    delta_norm = math.sqrt(difference_squared)
    return {
        "initial_parameter_l2_norm": initial_norm,
        "parameter_change_l2_norm": delta_norm,
        "relative_parameter_change_l2": delta_norm / max(initial_norm, 1e-30),
    }


def _check_inputs(summary: Mapping[str, object], examples: Mapping[str, object]) -> dict[str, bool]:
    statuses = cast(Mapping[str, int], summary["series_status"])
    total_expected = _as_int(summary["expected_series"])
    accepted = int(statuses.get("accepted", 0))
    excluded = int(statuses.get("excluded_non_diagnostic", 0))
    minimum_slices_ok = _as_int(examples["valid_studies_with_series_and_slices"]) == COHORT_SIZE
    checks = {
        "series_study_identities_align_with_loaded_studies": summary.get("study_identity_alignment") is True,
        "all_frozen_studies_have_accepted_series": _as_int(summary["studies_with_accepted_series"]) == COHORT_SIZE,
        "all_expected_series_accounted_for": accepted + excluded == total_expected,
        "all_packed_pixels_finite": examples["packed_pixels_finite"] is True,
        "all_frozen_studies_have_valid_series_and_slices": minimum_slices_ok,
        "all_frozen_studies_meet_minimum_valid_slice_count": _as_int(
            examples.get("minimum_valid_slice_count_per_study", 0)
        ) >= int(SUCCESS_CRITERIA["minimum_valid_slices_per_study"]),
        "source_pixels_meet_nonconstant_fraction": _as_float(summary["nonconstant_fraction_of_finite_slices"]) >= float(SUCCESS_CRITERIA["minimum_nonconstant_source_slice_fraction"]),
        "all_twelve_targets_have_two_class_support": all(
            _as_int(item["positive"]) > 0 and _as_int(item["negative"]) > 0
            for item in cast(Sequence[Mapping[str, object]], examples["target_support"])
        ),
        "all_expected_reference_entries_supervised": _as_int(examples["supervised_target_entries"]) == COHORT_SIZE * len(TARGET_COLUMNS),
    }
    return checks


def _write_public_report(path: Path, evidence: Mapping[str, object]) -> None:
    if path.exists():
        raise DiagnosticError("refusing to replace existing public Ticket 29 evidence")
    source = _mapping(evidence.get("source"), "evidence source")
    fold = _mapping(evidence.get("fold"), "evidence fold")
    pixel = _mapping(evidence.get("real_pixel_diagnostic"), "pixel diagnostic")
    examples = _mapping(evidence.get("training_input_diagnostic"), "training input diagnostic")
    training = _mapping(evidence.get("training"), "training result")
    curves = cast(Sequence[Mapping[str, object]], training.get("learning_curve", []))
    resource = _mapping(evidence.get("resources"), "resource record")
    storage = _mapping(evidence.get("storage"), "storage ledger")
    queue = _mapping(resource.get("gpu_campaign_accounting"), "GPU campaign accounting")
    checks = cast(Mapping[str, bool], evidence.get("input_checks", {}))
    series_status = _mapping(pixel.get("series_status"), "pixel series status")
    lines = [
        "# Ticket 29 real-image learning diagnostic",
        "",
        f"Recorded: {evidence.get('recorded_at_local', 'unknown')} (Asia/Kolkata).",
        "This report contains aggregate evidence only. Study, series, and source-entry identifiers and pixels remain in the private run directory.",
        "",
        "## Source and fold reconciliation",
        "",
        f"- Source version: `{source.get('source_version')}`; archive fingerprint: `{source.get('archive_fingerprint')}`.",
        f"- Integration identity: `{source.get('integration_manifest_checksum')}`; it covers {source.get('integration_studies')} prepared studies ({source.get('prepared_reference_studies')} reference and {source.get('prepared_report_only_studies')} report-only).",
        f"- Corrected folds: `{fold.get('fold_manifest_checksum')}`; version {fold.get('generation_version')}, three folds, {source.get('source_studies')} source studies ({source.get('full_source_reference_studies')} reference and {source.get('full_source_report_only_studies')} report-only).",
        f"- Canonical target order: {', '.join(cast(Sequence[str], evidence.get('targets', [])))}.",
        f"- Frozen cohort: {examples.get('training_examples')} fold-{fold.get('selected_fold')} reference-training studies; held-out studies read: 0. It was selected from labels alone, before pixel reads.",
        f"- Input/cache identity: existing `{source.get('preprocess_schema')}` 128-pixel preparation policy; no prepared-image cache was supplied or written. Raw archive membership and manifest checksums bind this run.",
        "",
        "## Real image and label checks",
        "",
        f"- Real DICOM studies loaded: {pixel.get('studies_with_accepted_series')}/{examples.get('training_examples')}; accepted series: {series_status.get('accepted', 0)}, explicitly excluded non-diagnostic series: {series_status.get('excluded_non_diagnostic', 0)}, declared series: {pixel.get('expected_series')}; decoded slices in accepted series: {pixel.get('decoded_slices_in_accepted_series')}/{pixel.get('expected_source_slice_entries_in_accepted_series')}.",
        f"- Finite slices: {pixel.get('finite_slices')}; nonconstant finite slices: {pixel.get('nonconstant_finite_slices')} ({_as_float(pixel.get('nonconstant_fraction_of_finite_slices', 0.0)):.3f}). Source dimensions: `{pixel.get('source_dimension_counts')}`.",
        f"- Anatomical-plane coverage by series: `{pixel.get('anatomical_plane_series_counts')}`; fluid-sensitive states: `{pixel.get('fluid_sensitive_series_counts')}`; fat-suppression states: `{pixel.get('fat_suppression_series_counts')}`.",
        f"- Packed valid masks: {examples.get('valid_studies_with_series_and_slices')} studies with input, {examples.get('valid_series_mask_count')} valid series, {examples.get('valid_slice_mask_count')} valid slices; minimum per study: {examples.get('minimum_valid_slice_count_per_study')}. All packed pixels finite: `{examples.get('packed_pixels_finite')}`.",
        f"- Supervision: {examples.get('supervised_target_entries')}/96 reference target entries aligned across the frozen cohort. Per-target positive/negative support: `{[(item.get('target'), item.get('positive'), item.get('negative')) for item in cast(Sequence[Mapping[str, object]], examples.get('target_support', []))]}`.",
        f"- Input checks: `{dict(checks)}`.",
        "",
        "## Training-only optimization evidence",
        "",
        f"- Model path: existing mechanics baseline; Adam 0.001, constant schedule, no augmentation, seed {training.get('seed')}, {training.get('optimizer_steps')} bounded steps. The control is the selected cohort's constant training-prevalence loss.",
        f"- Prevalence-control masked BCE: {training.get('prevalence_control_bce')}; initial loss: {training.get('initial_loss')}; initial gradient L2 norm: {training.get('initial_gradient_l2_norm')}; nonzero gradient tensors: {training.get('nonzero_gradient_tensors')}/{training.get('gradient_tensors')}.",
        f"- Relative parameter L2 change: {training.get('relative_parameter_change_l2')}; final training-cohort loss: {training.get('final_loss')}; final supported-target training macro AUC: {training.get('final_supported_target_macro_auc')} across {training.get('final_supported_targets')} targets.",
        "- Full training-cohort curves (0, 100, 250, and 500 optimizer steps):",
        "",
        "| Steps | Training-only masked BCE | Supported-target training macro AUC |",
        "| ---: | ---: | ---: |",
    ]
    for point in curves:
        loss = point.get("training_cohort_mean_masked_bce")
        auc = point.get("supported_target_macro_auc")
        lines.append(f"| {point.get('steps')} | {loss} | {auc} |")
    lines.extend(
        [
            "",
            f"- Optimization competence demonstrated: **{training.get('optimization_competence_demonstrated')}**. This is memorization evidence on the selected training examples; it is not held-out generalization or a campaign score.",
            "- Frozen success criteria: finite/nonzero gradient norm ≥1e-8; relative parameter change ≥1e-8; final masked BCE ≤50% of the training-prevalence control; supported-target training macro AUC ≥0.85 across all 12 targets. No held-out outcomes were used.",
            "",
            "## Resource accounting and storage",
            "",
            f"- CPU Slurm job: `{resource.get('slurm_job_id')}` on `{resource.get('slurm_node')}`, partition `{resource.get('partition')}`, {resource.get('cpus')} CPUs, {resource.get('memory_gb')} GB, nonexclusive, six-hour limit. Actual elapsed time: {resource.get('elapsed_seconds')} s; CPU queue wait: {resource.get('cpu_queue_wait_seconds')} s.",
            f"- GPU: no request or allocation for Ticket 29; Ticket 29 allocated GPU attempts/hours: 0/0.0. Current owner active/pending GPU requests at pre-submit snapshot: {queue.get('owner_active_or_pending_requests')}. Campaign cap: {queue.get('budget_hours')} GPU-hours; total remaining is `{queue.get('remaining_gpu_hours')}` because historical Slurm accounting is `{queue.get('historical_accounting_status')}`. Queue wait is reported separately from allocated time. The mocked scheduler gate rejected over-budget and concurrent requests.",
            f"- Retained archive: {storage.get('archive_bytes')} bytes; no raw or prepared-image copy was created. Added durable Ticket 29 outputs: {storage.get('ticket29_added_bytes')} bytes; peak temporary write reserve: {storage.get('temporary_write_peak_reserve_bytes')} bytes.",
            f"- Measured declared project roots before/after: {storage.get('measured_roots_before_bytes')} / {storage.get('measured_roots_after_bytes')} logical bytes; projected peak with 64 MB reserve: {storage.get('projected_peak_with_reserve_bytes')} / 2,000,000,000,000 bytes. Shared free space: {storage.get('shared_filesystem_available_bytes')} bytes and is not project headroom.",
            f"- Storage scan scope: `{storage.get('measurement_scope')}`. {storage.get('scope_limitation')}",
            "",
            "## Disposition and limitations",
            "",
            f"- Diagnostic disposition: `{training.get('disposition')}`. The next representation experiment is `{training.get('dependent_experiment_status')}` based on this optimization result.",
            "- No report text, images, study identifiers, or per-study predictions are published here. The private plan, checksums, profile, detailed training log, and checkpoints are retained under the ignored private artifact root.",
            "- No GPU run was authorized or required; this evidence does not establish GPU throughput. Project storage beyond the measured repository and Ticket 29 worktree roots, and total campaign GPU remaining hours, remain unverified because Slurm accounting is unavailable.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_real_diagnostic(
    *,
    plan_path: str | Path,
    integration_manifest_path: str | Path,
    fold_manifest_path: str | Path,
    selection_report_path: str | Path,
    source_archive_path: str | Path,
    run_directory: str | Path,
    project_roots: Sequence[str | Path],
    public_evidence_path: str | Path,
    submit_epoch: float | None = None,
) -> dict[str, object]:
    """Execute the frozen real-data diagnostic in the exact CPU Slurm shape."""

    slurm = _check_slurm_context()
    output = Path(run_directory)
    if output.exists() and any(output.iterdir()):
        raise DiagnosticError("refusing to overwrite a non-empty private diagnostic directory")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    started_at = time.time()
    plan = load_diagnostic_plan(plan_path)
    integration = load_integration_manifest(integration_manifest_path)
    fold_manifest = load_fold_manifest(fold_manifest_path)
    selected_uids, cohort_reconciliation = _verify_manifest_identity(
        plan,
        integration,
        fold_manifest,
        Path(integration_manifest_path),
        Path(fold_manifest_path),
        Path(selection_report_path),
    )
    roots = tuple(Path(value) for value in project_roots)
    archive_path = Path(source_archive_path)
    preflight = run_cpu_preflight(
        output_path=output / "cpu-preflight.json",
        require_isolated_python=True,
        require_slurm=True,
        required_slurm_resources=SLURM_RESOURCES,
    )
    require_cpu_preflight(preflight, require_slurm=True, require_isolated_python=True)
    if not preflight.passed:
        raise DiagnosticError("existing CPU preflight did not pass in the frozen Slurm allocation")
    if not archive_path.is_file() or archive_path.stat().st_size != _mapping(plan.get("source"), "source").get("source_archive_bytes"):
        raise DiagnosticError("source archive size changed after the diagnostic plan was frozen")
    actual_archive = identify_archive(archive_path, source_version=integration.source_version)
    if actual_archive.fingerprint != integration.archive_fingerprint:
        raise DiagnosticError("source archive identity differs from the frozen integration manifest")

    storage_before = _storage_snapshot(roots, archive_path=archive_path)
    load_started = time.perf_counter()
    loaded = tuple(
        iter_real_studies(
            integration,
            archive_path,
            require_complete_series=True,
            study_uids=selected_uids,
        )
    )
    load_seconds = time.perf_counter() - load_started
    if tuple(item.study.study_instance_uid for item in loaded) != tuple(
        item.study_instance_uid for item in integration.studies if item.study_instance_uid in set(selected_uids)
    ):
        raise DiagnosticError("real loader did not return the exact selected integration rows")
    pixel_summary, pixel_private = _pixel_and_geometry_summary(loaded)
    _write_private_new_json(
        output / "real-input-summary.json",
        {
            "plan_sha256": plan["plan_sha256"],
            "aggregate": pixel_summary,
            "private_detail": pixel_private,
        },
    )
    studies_by_uid = {item.study.study_instance_uid: item.study for item in loaded}
    examples = build_fold_training_examples(
        studies_by_uid,
        fold_manifest,
        FOLD_ID,
        preprocess_config=integration.preprocess_config,
        packing_config=integration.packing_config,
        allow_partial_inputs=False,
        selected_training_study_uids=selected_uids,
    )
    example_summary = _example_summary(examples, expected_ids=selected_uids)
    input_checks = _check_inputs(pixel_summary, example_summary)
    cohort = _mapping(plan.get("cohort"), "cohort")
    _write_private_new_json(
        output / "training-input-summary.json",
        {
            "plan_sha256": plan["plan_sha256"],
            "cohort_membership_sha256": cohort["cohort_membership_sha256"],
            "training_input": example_summary,
        },
    )
    _write_private_new_json(output / "input-checks.json", {"checks": input_checks})
    if not all(input_checks.values()):
        raise DiagnosticError("real input and label gates failed; optimizer run was not started")
    if any(
        example.provenance.fold != FOLD_ID
        or not example.provenance.held_out_excluded
        or example.provenance.source_kind != "reference"
        for example in examples
    ):
        raise DiagnosticError("training examples do not have reference-only training provenance")

    config = TrainingConfig(
        image_size=integration.preprocess_config.image_size,
        channels=(8, 16, 32),
        max_slices=integration.packing_config.max_slices,
        max_series=integration.packing_config.max_series,
        batch_size=2,
        successful_optimizer_steps=LONG_STEPS,
        optimizer="Adam",
        learning_rate=0.001,
        schedule="constant",
        seed=DIAGNOSTIC_SEED,
        batch_order_seed=BATCH_ORDER_SEED,
        checkpoint_interval_steps=CURVE_CHECKPOINT_INTERVAL_STEPS,
        checkpoint_interval_seconds=900.0,
        augmentation="none",
        device="cpu",
        selection_policy="final_checkpoint",
        preprocessing_config=integration.preprocess_config,
    )
    identity: dict[str, object] = {
        "campaign": "rsna-leaderboard-zero-spend-2026-09-25",
        "ticket": "29",
        "diagnostic_plan_sha256": plan["plan_sha256"],
        "cohort_membership_sha256": _mapping(plan.get("cohort"), "cohort")["cohort_membership_sha256"],
        "source_version": integration.source_version,
        "archive_fingerprint": integration.archive_fingerprint,
        "source_manifest_checksum": integration.source_manifest_checksum,
        "integration_manifest_checksum": integration.manifest_checksum,
        "fold_manifest_checksum": fold_manifest.manifest_checksum,
        "fold": FOLD_ID,
        "target_columns": list(TARGET_COLUMNS),
        "preprocessing": integration.preprocess_config.cache_config(),
        "packing": integration.packing_config.to_dict(),
        "supervision": "reference-only tiny training cohort; no held-out or report-derived labels",
    }
    seed_everything(config.seed)
    model = HierarchicalStudyModel(config.model_config())
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    initial_summary = _evaluate(model, examples)
    gradient = _gradient_summary(model, examples)
    prevalence = _prevalence_control(examples)
    optimizer_started = time.perf_counter()
    result = train_fixed_budget(
        examples,
        config,
        model=model,
        artifact_dir=output / "training",
        fold=FOLD_ID,
        identity=identity,
        requested_resources={
            "partition": SLURM_RESOURCES["partition"],
            "nodes": SLURM_RESOURCES["nodes"],
            "ntasks": SLURM_RESOURCES["ntasks"],
            "cpus_per_task": SLURM_RESOURCES["cpus_per_task"],
            "memory_gb": SLURM_RESOURCES["memory_gb"],
            "wall_time_hours": 6,
            "exclusive": False,
            "allocated_gpu_count": 0,
        },
    )
    optimizer_seconds = time.perf_counter() - optimizer_started
    if result.successful_optimizer_steps != LONG_STEPS or result.final_checkpoint_path is None:
        raise DiagnosticError("fixed-budget mechanics training did not complete the frozen 500 steps")
    change = _relative_parameter_change(initial_state, result.model.state_dict())

    checkpoints: dict[int, tuple[Path, str]] = {LONG_STEPS: (result.final_checkpoint_path, "final_selected")}
    for path in sorted((output / "training" / "checkpoints" / "history").glob("*.pt")):
        loaded_checkpoint = load_training_checkpoint(path)
        progress = _mapping(loaded_checkpoint.get("progress"), "checkpoint progress")
        step = progress.get("successful_optimizer_steps")
        if isinstance(step, int) and step in {SHORT_STEPS, MID_STEPS}:
            checkpoints[step] = (path, "latest_resumable")
    missing = {SHORT_STEPS, MID_STEPS, LONG_STEPS}.difference(checkpoints)
    if missing:
        raise DiagnosticError("training checkpoints do not preserve every frozen learning-curve step")
    curve: list[dict[str, object]] = [
        {
            "steps": 0,
            **initial_summary,
            "checkpoint_sha256": None,
        }
    ]
    for step in (SHORT_STEPS, MID_STEPS, LONG_STEPS):
        checkpoint_path, checkpoint_kind = checkpoints[step]
        evaluation_model = HierarchicalStudyModel(config.model_config())
        payload = load_training_checkpoint(
            checkpoint_path,
            model=evaluation_model,
            expected_config=config,
            expected_fold=FOLD_ID,
            expected_identity=identity,
            expected_checkpoint_kind=checkpoint_kind,
        )
        curve.append(
            {
                "steps": step,
                **_evaluate(evaluation_model, examples),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
                "checkpoint_progress": _mapping(payload.get("progress"), "checkpoint progress").get(
                    "successful_optimizer_steps"
                ),
            }
        )
    final_curve = curve[-1]
    final_loss = _as_float(final_curve["training_cohort_mean_masked_bce"])
    baseline_loss = _as_float(prevalence["constant_prevalence_masked_bce"])
    gradient_norm = _as_float(gradient["global_gradient_l2_norm"])
    relative_change = _as_float(change["relative_parameter_change_l2"])
    supported_targets = _as_int(final_curve["supported_targets"])
    final_auc = float(cast(float, final_curve["supported_target_macro_auc"]))
    optimization_competence = bool(
        gradient["all_parameter_gradients_present_and_finite"]
        and gradient_norm >= float(SUCCESS_CRITERIA["gradient_global_norm_minimum"])
        and relative_change >= float(SUCCESS_CRITERIA["relative_parameter_l2_change_minimum"])
        and final_loss <= baseline_loss * float(SUCCESS_CRITERIA["long_run_loss_fraction_of_training_prevalence_control_maximum"])
        and supported_targets >= int(SUCCESS_CRITERIA["long_run_supported_targets_minimum"])
        and final_auc >= float(SUCCESS_CRITERIA["long_run_supported_target_macro_auc_minimum"])
    )
    elapsed_seconds = time.time() - started_at
    submit_wait = max(0.0, started_at - submit_epoch) if submit_epoch is not None else None
    gpu_queue = _owner_gpu_queue_snapshot()
    storage_after = _storage_snapshot(roots, archive_path=archive_path)
    archive_resolved = archive_path.resolve()
    before_by_root = cast(Mapping[str, Mapping[str, int]], storage_before["root_metrics"])
    after_by_root = cast(Mapping[str, Mapping[str, int]], storage_after["root_metrics"])
    added_by_root = {
        name: max(0, int(after_by_root[name]["logical_bytes"]) - int(before_by_root[name]["logical_bytes"]))
        for name in before_by_root
    }
    gpu_accounting = summarize_gpu_attempts([])
    gpu_accounting.update(
        {
            "ticket29_allocated_attempts": 0,
            "ticket29_allocated_gpu_hours": 0.0,
            "ticket29_gpu_queue_wait_seconds": None,
            "owner_active_or_pending_requests": gpu_queue["owner_active_or_pending_requests"],
            "owner_queue_snapshot": gpu_queue,
            "single_request_constraint_satisfied": gpu_queue["owner_active_or_pending_requests"] == 0,
            "historical_accounting_status": "unavailable: sacct persistent database connection refused",
            "remaining_gpu_hours": None,
            "remaining_gpu_hours_reason": "historical campaign attempts cannot be fully queried",
        }
    )
    evidence: dict[str, object] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "ticket": "29",
        "status": "completed_real_training_only_diagnostic",
        "recorded_at_local": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "plan_sha256": plan["plan_sha256"],
        "source": {
            "source_version": integration.source_version,
            "archive_fingerprint": integration.archive_fingerprint,
            "source_archive_bytes": archive_path.stat().st_size,
            "source_archive_fingerprint_reverified": actual_archive.fingerprint,
            "source_manifest_checksum": integration.source_manifest_checksum,
            "integration_manifest_checksum": integration.manifest_checksum,
            "integration_studies": len(integration.study_uids),
            "prepared_reference_studies": 58,
            "prepared_report_only_studies": 100,
            "full_source_reference_studies": 58,
            "full_source_report_only_studies": 4349,
            "source_studies": 4407,
            "preprocess_schema": integration.preprocess_config.cache_config().get("schema_version"),
            "preprocessing": integration.preprocess_config.cache_config(),
            "packing": integration.packing_config.to_dict(),
            "reusable_cache_status": "no cache used or written; source archive decoded through the existing path",
        },
        "fold": {
            "fold_manifest_checksum": fold_manifest.manifest_checksum,
            "selected_fold": FOLD_ID,
            "generation_version": fold_manifest.generation_version,
            "fold_ids": list(fold_manifest.fold_ids),
            "split_unit": fold_manifest.grouping.status,
            **cohort_reconciliation,
        },
        "targets": list(TARGET_COLUMNS),
        "real_pixel_diagnostic": pixel_summary,
        "training_input_diagnostic": example_summary,
        "private_pixel_detail": pixel_private,
        "input_checks": input_checks,
        "training": {
            "optimizer_steps": result.successful_optimizer_steps,
            "checkpoint_interval_steps": CURVE_CHECKPOINT_INTERVAL_STEPS,
            "seed": DIAGNOSTIC_SEED,
            "batch_order_seed": BATCH_ORDER_SEED,
            "initial_loss": initial_summary["training_cohort_mean_masked_bce"],
            "initial_gradient_l2_norm": gradient_norm,
            "gradient_tensors": gradient["parameter_tensors_with_gradient"],
            "nonzero_gradient_tensors": gradient["nonzero_gradient_tensors"],
            "all_parameter_gradients_present_and_finite": gradient["all_parameter_gradients_present_and_finite"],
            "prevalence_control_bce": baseline_loss,
            **change,
            "final_loss": final_loss,
            "final_supported_target_macro_auc": final_auc,
            "final_supported_targets": supported_targets,
            "learning_curve": curve,
            "training_wall_seconds": optimizer_seconds,
            "training_run_id": result.run_id,
            "training_data_fingerprint": result.data_fingerprint,
            "semantic_fingerprint": result.semantic_fingerprint_value,
            "final_checkpoint_sha256": _file_sha256(result.final_checkpoint_path),
            "checkpoint_directory_file_bytes": sum(
                item.stat().st_size
                for item in (output / "training" / "checkpoints").rglob("*")
                if item.is_file()
            ),
            "optimization_competence_demonstrated": optimization_competence,
            "disposition": "optimization_competence_demonstrated" if optimization_competence else "optimization_competence_not_demonstrated",
            "dependent_experiment_status": "unblocked_by_optimization_diagnostic" if optimization_competence else "blocked_pending_diagnosis_or_repair",
            "held_out_studies_used_for_training_or_selection": 0,
            "generalization_claim": False,
        },
        "resources": {
            "slurm_job_id": slurm["job_id"],
            "slurm_node": os.environ.get("SLURM_JOB_NODELIST"),
            "partition": slurm["partition"],
            "cpus": int(cast(str, slurm["cpus_per_task"])),
            "memory_gb": 32,
            "exclusive": False,
            "time_limit": slurm["time_limit"],
            "elapsed_seconds": elapsed_seconds,
            "cpu_queue_wait_seconds": submit_wait,
            "cpu_preflight_passed": preflight.passed,
            "cpu_preflight": preflight.to_dict(),
            "external_spend_usd": 0,
            "gpu_campaign_accounting": gpu_accounting,
        },
        "storage": {
            "measurement_scope": "canonical repository plus isolated Ticket 29 worktree, including their private artifact trees and local environment; no unrelated home or shared-filesystem trees",
            "scope_limitation": "historical external home-directory experiment folders are outside the measured roots; the private root aggregate is not claimed as whole-home usage",
            "archive_bytes": _as_int(storage_before["archive_bytes"]),
            "measured_roots_before_bytes": _as_int(storage_before["measured_roots_logical_bytes"]),
            "measured_roots_after_bytes": _as_int(storage_after["measured_roots_logical_bytes"]),
            "projected_peak_with_reserve_bytes": _as_int(storage_before["measured_roots_logical_bytes"]) + 64_000_000,
            "projected_peak_within_soft_budget": _as_int(storage_before["measured_roots_logical_bytes"]) + 64_000_000 <= 2_000_000_000_000,
            "soft_budget_bytes": 2_000_000_000_000,
            "projected_headroom_before_run_bytes": _as_int(storage_before["projected_headroom_before_run_bytes"]),
            "shared_filesystem_available_bytes": _as_int(storage_before["shared_filesystem_available_bytes"]),
            "shared_free_space_is_project_headroom": False,
            "ticket29_added_bytes": sum(added_by_root.values()),
            "ticket29_added_bytes_by_measured_root": added_by_root,
            "temporary_write_peak_reserve_bytes": 64_000_000,
            "extracted_original_copy_created": False,
            "prepared_image_cache_created": False,
            "protected_artifacts_deleted": False,
            "storage_before": storage_before,
            "storage_after": storage_after,
            "archive_location_was_in_measured_roots": any(
                archive_resolved == root or root in archive_resolved.parents
                for root in (path.resolve() for path in roots)
            ),
        },
        "execution": {
            "hostname": socket.gethostname(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "python_isolated": sys.prefix != sys.base_prefix,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "source_load_seconds": load_seconds,
            "public_evidence_path": "docs/evidence/ticket-29-image-diagnostic.md",
        },
        "durable_input_evidence": {
            "real_input_summary_path": "real-input-summary.json",
            "real_input_summary_sha256": _file_sha256(output / "real-input-summary.json"),
            "training_input_summary_path": "training-input-summary.json",
            "training_input_summary_sha256": _file_sha256(output / "training-input-summary.json"),
            "input_checks_path": "input-checks.json",
            "input_checks_sha256": _file_sha256(output / "input-checks.json"),
        },
    }
    evidence_path = output / "private-evidence.json"
    atomic_json_write(evidence_path, evidence)
    _write_public_report(Path(public_evidence_path), evidence)
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="freeze the exact cohort and numerical plan")
    freeze.add_argument("--integration-manifest", type=Path, required=True)
    freeze.add_argument("--fold-manifest", type=Path, required=True)
    freeze.add_argument("--selection-report", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="run the frozen real-data diagnostic in Slurm")
    for option in ("plan", "integration-manifest", "fold-manifest", "selection-report", "source-archive", "run-directory", "public-evidence"):
        run.add_argument(f"--{option}", type=Path, required=True)
    run.add_argument("--project-root", type=Path, action="append", required=True)
    run.add_argument("--submit-epoch", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze_diagnostic_plan(
                integration_manifest_path=args.integration_manifest,
                fold_manifest_path=args.fold_manifest,
                selection_report_path=args.selection_report,
                output_path=args.output,
            )
            print(json.dumps({"status": "plan_frozen", "plan_sha256": result["plan_sha256"], "cohort_size": COHORT_SIZE, "fold": FOLD_ID, "target_count": len(TARGET_COLUMNS)}, indent=2))
            return 0
        result = run_real_diagnostic(
            plan_path=args.plan,
            integration_manifest_path=args.integration_manifest,
            fold_manifest_path=args.fold_manifest,
            selection_report_path=args.selection_report,
            source_archive_path=args.source_archive,
            run_directory=args.run_directory,
            project_roots=args.project_root,
            public_evidence_path=args.public_evidence,
            submit_epoch=args.submit_epoch,
        )
    except (DiagnosticError, OSError, RuntimeError, ValueError, TrainingDataError) as exc:
        print(f"ticket-29 diagnostic failed: {exc}", file=sys.stderr)
        return 1
    public = {
        "status": result["status"],
        "plan_sha256": result["plan_sha256"],
        "slurm_job_id": _mapping(result.get("resources"), "resources").get("slurm_job_id"),
        "optimization_competence_demonstrated": _mapping(result.get("training"), "training").get("optimization_competence_demonstrated"),
        "training_loss": _mapping(result.get("training"), "training").get("final_loss"),
        "training_auc": _mapping(result.get("training"), "training").get("final_supported_target_macro_auc"),
        "evidence": "docs/evidence/ticket-29-image-diagnostic.md",
    }
    print(json.dumps(public, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAMPAIGN_GPU_BUDGET_HOURS",
    "COHORT_SIZE",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "DiagnosticError",
    "GPURequestRejected",
    "LONG_STEPS",
    "MID_STEPS",
    "PLAN_SCHEMA_VERSION",
    "SHORT_STEPS",
    "SUCCESS_CRITERIA",
    "admit_mock_gpu_request",
    "freeze_diagnostic_plan",
    "load_diagnostic_plan",
    "main",
    "summarize_gpu_attempts",
]
