"""Fold-safe planning and restartable image preparation for Ticket 35.

The private membership manifest records which report-only studies have
admitted supervision in each consuming fold, which of those studies have
inventory-complete images, and how they are divided into deterministic
batches.  Actual image access stays behind :mod:`real_inference` and the
checksum-validating :class:`~rsna_knee.series_cache.SeriesCache` boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import torch

from .archive import atomic_json_write
from .dicom import DecoderPolicy
from .folds import FOLD_IDS, load_fold_manifest, load_inventory_source, source_manifest_checksum
from .inference import InferenceEngine
from .model import StudyModel
from .preprocessing import PackingConfig, PreprocessConfig, SeriesFailure, pack_study
from .real_inference import (
    IntegrationManifest,
    LoadedStudy,
    build_integration_manifest,
    iter_real_studies,
    load_integration_manifest,
    _verify_archive_source_identity,
)
from .series_cache import SeriesCache, SeriesCacheResult
from .targets import NUM_TARGETS, TARGET_COLUMNS


TICKET35_MEMBERSHIP_SCHEMA_VERSION = "rsna-ticket35-expanded-image-membership-v2"
TICKET35_BATCH_EVIDENCE_SCHEMA_VERSION = "rsna-ticket35-image-batch-evidence-v1"
TICKET35_SOFT_STORAGE_BUDGET_BYTES = 2_000_000_000_000


class Ticket35Error(ValueError):
    """Raised when a frozen Ticket 35 plan or batch is inconsistent."""


class Ticket35StorageError(Ticket35Error):
    """Raised when a batch cannot be admitted under the retained-storage cap."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Ticket35Error(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise Ticket35Error(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _validate_target_vector(value: object, name: str) -> tuple[int | None, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != NUM_TARGETS:
        raise Ticket35Error(f"{name} must contain exactly {NUM_TARGETS} target values")
    result: list[int | None] = []
    for item in value:
        if item not in (None, 0, 1) or isinstance(item, bool):
            raise Ticket35Error(f"{name} values must be 0, 1, or null")
        result.append(cast(int | None, item))
    return tuple(result)


def _sha256_identity(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Ticket35Error(f"{name} must be a lowercase SHA-256 digest")
    return value


def build_ticket35_membership(
    *,
    report_fold_by_uid: Mapping[str, int],
    admitted_targets_by_fold: Mapping[int, Mapping[str, Sequence[int | None]]],
    inventory_series: Sequence[Mapping[str, object]],
    existing_study_uids: Sequence[str],
    held_out_gold_by_fold: Mapping[int, Sequence[str]],
    source_identity: Mapping[str, object],
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    supported_transfer_syntaxes: Sequence[str] = ("1.2.840.10008.1.2.1",),
    minimum_usable_slices: int = 3,
) -> dict[str, object]:
    """Freeze a fold-safe, source-bound and image-aware expansion plan.

    ``admitted_targets_by_fold`` is indexed by the consuming fold.  Every row
    is checked against its immutable report-fold assignment so a report study
    can never be used in the fold where it is held out.  Images are admitted
    only when every inventory series for an eligible study is complete and
    compatible with the declared decoder policy; all other studies retain an
    explicit exclusion reason.
    """

    if batch_size <= 0:
        raise Ticket35Error("batch_size must be positive")
    if len(image_size) != 2 or any(not isinstance(size, int) or size <= 0 for size in image_size):
        raise Ticket35Error("image_size must contain two positive integers")
    if minimum_usable_slices <= 0:
        raise Ticket35Error("minimum_usable_slices must be positive")
    assignments: dict[str, int] = {}
    for raw_uid, raw_fold in report_fold_by_uid.items():
        uid = str(raw_uid).strip()
        if not uid or uid in assignments or raw_fold not in FOLD_IDS:
            raise Ticket35Error("report fold assignments must contain unique IDs and canonical folds")
        assignments[uid] = raw_fold
    if not assignments:
        raise Ticket35Error("report fold assignments must not be empty")

    admission_rows: dict[int, dict[str, tuple[int | None, ...]]] = {}
    for fold in FOLD_IDS:
        raw_rows = admitted_targets_by_fold.get(fold)
        if raw_rows is None:
            raise Ticket35Error(f"admissions are missing consuming fold {fold}")
        rows: dict[str, tuple[int | None, ...]] = {}
        for raw_uid, raw_targets in raw_rows.items():
            uid = str(raw_uid).strip()
            if not uid or uid not in assignments:
                raise Ticket35Error(f"fold {fold} admits a study outside the report fold manifest")
            if assignments[uid] == fold:
                raise Ticket35Error(f"fold {fold} admits report study {uid} from its held-out fold")
            rows[uid] = _validate_target_vector(raw_targets, f"fold {fold} targets for {uid}")
            if not any(value is not None for value in rows[uid]):
                raise Ticket35Error(f"fold {fold} admits a report study without a supervised target")
        admission_rows[fold] = rows

    inventory_by_study: dict[str, list[Mapping[str, object]]] = {}
    seen_series: set[str] = set()
    for row_index, row in enumerate(inventory_series):
        uid = str(row.get("StudyInstanceUID", "")).strip()
        series_uid = str(row.get("SeriesInstanceUID", "")).strip()
        if not uid or not series_uid:
            raise Ticket35Error(f"inventory series row {row_index} lacks study or series identity")
        if series_uid in seen_series:
            raise Ticket35Error(f"inventory repeats SeriesInstanceUID {series_uid}")
        seen_series.add(series_uid)
        inventory_by_study.setdefault(uid, []).append(row)

    existing = {str(uid).strip() for uid in existing_study_uids}
    if any(not uid for uid in existing):
        raise Ticket35Error("existing study identifiers must be non-empty")
    held_out: dict[int, tuple[str, ...]] = {}
    for fold in FOLD_IDS:
        held_out_values = tuple(sorted(str(uid).strip() for uid in held_out_gold_by_fold.get(fold, ())))
        if any(not uid for uid in held_out_values) or len(set(held_out_values)) != len(held_out_values):
            raise Ticket35Error(f"held-out gold study IDs for fold {fold} must be non-empty and unique")
        held_out[fold] = held_out_values

    selected_syntaxes = {str(item).strip() for item in supported_transfer_syntaxes}
    if not selected_syntaxes or "" in selected_syntaxes:
        raise Ticket35Error("supported transfer syntaxes must be non-empty")
    admitted_union = set().union(*(set(rows) for rows in admission_rows.values()))
    no_call = set(assignments) - admitted_union
    image_ready: list[str] = []
    image_exclusions: dict[str, list[str]] = {}
    study_metrics: dict[str, dict[str, object]] = {}
    series_metrics: dict[str, dict[str, object]] = {}

    for uid in sorted(assignments):
        if uid not in admitted_union:
            continue
        image_rows = inventory_by_study.get(uid, [])
        reasons: list[str] = []
        if not image_rows:
            reasons.append("study_missing_from_image_inventory")
        expected_series_count = len(image_rows)
        uncompressed_bytes = 0
        compressed_bytes = 0
        frame_count = 0
        entry_count = 0
        for inventory_item in image_rows:
            series_uid = str(inventory_item["SeriesInstanceUID"])
            is_complete = inventory_item.get("complete") is True
            if not is_complete:
                reasons.append(f"incomplete_series:{series_uid}")
            transfer_counts = _as_mapping(
                inventory_item.get("transfer_syntax_counts", {}), "transfer_syntax_counts"
            )
            syntaxes = {str(value) for value in transfer_counts}
            if not syntaxes or syntaxes - selected_syntaxes:
                reasons.append(f"unsupported_or_unknown_transfer_syntax:{series_uid}")
            expected_frames = _integer(
                inventory_item.get("effective_frame_count", 0), "effective_frame_count"
            )
            if expected_frames < minimum_usable_slices:
                reasons.append(f"below_minimum_usable_slices:{series_uid}")
            entry_names_value = inventory_item.get("entry_names", ())
            if not isinstance(entry_names_value, (tuple, list)):
                reasons.append(f"missing_source_entry_manifest:{series_uid}")
                entries: tuple[str, ...] = ()
            else:
                entries = tuple(str(value) for value in entry_names_value)
                if not entries or any(not value for value in entries):
                    reasons.append(f"missing_source_entry_manifest:{series_uid}")
            uncompressed = _integer(inventory_item.get("uncompressed_bytes", 0), "uncompressed_bytes")
            compressed = _integer(inventory_item.get("compressed_bytes", 0), "compressed_bytes")
            uncompressed_bytes += uncompressed
            compressed_bytes += compressed
            frame_count += expected_frames
            entry_count += len(entries)
            series_metrics[series_uid] = {
                "study_instance_uid": uid,
                "source_entry_count": len(entries),
                "effective_frame_count": expected_frames,
                "uncompressed_source_bytes": uncompressed,
                "compressed_source_bytes": compressed,
                "complete": is_complete,
            }
        unique_reasons = sorted(set(reasons))
        if unique_reasons:
            image_exclusions[uid] = unique_reasons
            continue
        image_ready.append(uid)
        study_metrics[uid] = {
            "assigned_fold": assignments[uid],
            "consuming_folds": sorted(fold for fold, rows in admission_rows.items() if uid in rows),
            "series_count": expected_series_count,
            "source_entry_count": entry_count,
            "effective_frame_count": frame_count,
            "uncompressed_source_bytes": uncompressed_bytes,
            "compressed_source_bytes": compressed_bytes,
            "pixel_cache_bytes": frame_count * image_size[0] * image_size[1] * 4,
            "additional_vs_existing_manifest": uid not in existing,
            "supervised_target_entries_by_fold": {
                str(fold): sum(value is not None for value in admission_rows[fold][uid])
                for fold in FOLD_IDS
                if uid in admission_rows[fold]
            },
        }

    batch_uids = [image_ready[index : index + batch_size] for index in range(0, len(image_ready), batch_size)]
    batches: list[dict[str, object]] = []
    for index, uids in enumerate(batch_uids):
        metrics = [study_metrics[uid] for uid in uids]
        batches.append(
            {
                "batch_index": index,
                "study_uids": uids,
                "series_count": sum(_integer(item.get("series_count"), "study series_count") for item in metrics),
                "source_entry_count": sum(_integer(item.get("source_entry_count"), "study source_entry_count") for item in metrics),
                "effective_frame_count": sum(_integer(item.get("effective_frame_count"), "study effective_frame_count") for item in metrics),
                "uncompressed_source_bytes": sum(_integer(item.get("uncompressed_source_bytes"), "study uncompressed_source_bytes") for item in metrics),
                "compressed_source_bytes": sum(_integer(item.get("compressed_source_bytes"), "study compressed_source_bytes") for item in metrics),
                "pixel_cache_bytes": sum(_integer(item.get("pixel_cache_bytes"), "study pixel_cache_bytes") for item in metrics),
                "additional_study_count": sum(bool(item["additional_vs_existing_manifest"]) for item in metrics),
            }
        )

    fold_payload: list[dict[str, object]] = []
    for fold in FOLD_IDS:
        rows = admission_rows[fold]
        image_ready_rows = {uid: targets for uid, targets in rows.items() if uid in study_metrics}

        def coverage_for(
            target_rows: Mapping[str, tuple[int | None, ...]],
        ) -> list[dict[str, object]]:
            coverage: list[dict[str, object]] = []
            for target_index, target in enumerate(TARGET_COLUMNS):
                target_values = [target_row[target_index] for target_row in target_rows.values()]
                coverage.append(
                    {
                        "target": target,
                        "supervised_entries": sum(value is not None for value in target_values),
                        "positive_entries": sum(value == 1 for value in target_values),
                        "negative_entries": sum(value == 0 for value in target_values),
                    }
                )
            return coverage

        target_coverage = coverage_for(image_ready_rows)
        admitted_target_coverage = coverage_for(rows)
        image_ready_supervised_count = sum(
            _integer(item.get("supervised_entries"), "target supervised_entries")
            for item in target_coverage
        )
        admitted_supervised_count = sum(
            _integer(item.get("supervised_entries"), "target supervised_entries")
            for item in admitted_target_coverage
        )
        fold_payload.append(
            {
                "fold": fold,
                "held_out_gold_study_uids": list(held_out[fold]),
                "admitted_report_only_study_uids": sorted(rows),
                "admitted_report_only_study_count": len(rows),
                "image_ready_report_only_study_count": len(image_ready_rows),
                "supervised_target_entry_count": image_ready_supervised_count,
                "target_coverage": target_coverage,
                "admitted_supervised_target_entry_count_pre_image_exclusion": admitted_supervised_count,
                "admitted_target_coverage_pre_image_exclusion": admitted_target_coverage,
            }
        )

    identity = dict(source_identity)
    required_identities = (
        "source_version",
        "archive_fingerprint",
        "source_manifest_checksum",
        "inventory_sha256",
        "integration_manifest_sha256",
        "fold_manifest_checksum",
        "label_artifact_sha256",
    )
    for name in required_identities:
        if name not in identity or not str(identity[name]).strip():
            raise Ticket35Error(f"source identity is missing {name}")
    for name in ("inventory_sha256", "integration_manifest_sha256", "fold_manifest_checksum", "label_artifact_sha256"):
        _sha256_identity(identity[name], name)

    manifest: dict[str, object] = {
        "schema_version": TICKET35_MEMBERSHIP_SCHEMA_VERSION,
        "source_identity": identity,
        "folds": fold_payload,
        "cohort": {
            "eligible_report_study_count": len(assignments),
            "admitted_report_study_count": len(admitted_union),
            "no_admitted_call_report_study_count": len(no_call),
            "no_admitted_call_report_study_uids": sorted(no_call),
            "image_ready_report_study_count": len(image_ready),
            "image_ready_report_study_uids": image_ready,
            "image_excluded_report_study_count": len(image_exclusions),
            "image_excluded_report_study_uids": sorted(image_exclusions),
            "image_exclusion_reasons": image_exclusions,
            "existing_integration_study_count": len(existing),
            "additional_vs_existing_manifest_count": sum(uid not in existing for uid in image_ready),
            "additional_vs_existing_manifest_study_uids": sorted(uid for uid in image_ready if uid not in existing),
            "image_ready_series_count": sum(_integer(study_metrics[uid].get("series_count"), "series_count") for uid in image_ready),
            "image_ready_source_entry_count": sum(_integer(study_metrics[uid].get("source_entry_count"), "source_entry_count") for uid in image_ready),
            "image_ready_effective_frame_count": sum(_integer(study_metrics[uid].get("effective_frame_count"), "effective_frame_count") for uid in image_ready),
            "image_ready_uncompressed_source_bytes": sum(_integer(study_metrics[uid].get("uncompressed_source_bytes"), "uncompressed_source_bytes") for uid in image_ready),
            "image_ready_compressed_source_bytes": sum(_integer(study_metrics[uid].get("compressed_source_bytes"), "compressed_source_bytes") for uid in image_ready),
            "image_ready_pixel_cache_bytes": sum(_integer(study_metrics[uid].get("pixel_cache_bytes"), "pixel_cache_bytes") for uid in image_ready),
            "supervised_target_entry_count_across_folds": sum(
                _integer(row.get("supervised_target_entry_count"), "fold supervised target entries")
                for row in fold_payload
            ),
        },
        "batch_policy": {
            "batch_size_studies": batch_size,
            "image_size": list(image_size),
            "cache_dtype": "float32",
            "cache_bytes_per_pixel": 4,
            "minimum_usable_slices": minimum_usable_slices,
            "supported_transfer_syntaxes": sorted(selected_syntaxes),
            "storage_estimate_required_before_each_batch": True,
            "soft_budget_bytes": TICKET35_SOFT_STORAGE_BUDGET_BYTES,
        },
        "study_metrics": study_metrics,
        "series_metrics": series_metrics,
        "batches": batches,
        "terminal_disposition": {
            "state": "prepared_plan_only",
            "real_dicom_materialization": "pending_fresh_project_wide_storage_ledger_and_exact_cpu_batch_approval",
            "resnet18_inference": "pending_separate_ticket35_model_use_approval",
            "training_or_scientific_comparison": "not_authorized_by_ticket35_preparation_plan",
        },
    }
    manifest["manifest_checksum"] = _json_sha256(manifest)
    return manifest


def validate_ticket35_membership(value: object) -> Mapping[str, object]:
    """Validate the checksum and structural identity of a frozen plan."""

    payload = _as_mapping(value, "Ticket 35 membership")
    if payload.get("schema_version") != TICKET35_MEMBERSHIP_SCHEMA_VERSION:
        raise Ticket35Error("Ticket 35 membership has an unsupported schema")
    checksum = payload.get("manifest_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise Ticket35Error("Ticket 35 membership is missing its checksum")
    unsealed = dict(payload)
    unsealed.pop("manifest_checksum", None)
    if _json_sha256(unsealed) != checksum:
        raise Ticket35Error("Ticket 35 membership checksum does not match its contents")
    return payload


def write_ticket35_membership(path: str | Path, manifest: Mapping[str, object]) -> Path:
    """Write a private immutable membership plan, accepting identical reruns."""

    payload = dict(validate_ticket35_membership(manifest))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.is_symlink():
        raise Ticket35Error("Ticket 35 plan destination must not be a symlink")
    serialized = json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2) + "\n"
    if destination.exists():
        if not destination.is_file() or json.loads(destination.read_text(encoding="utf-8")) != payload:
            raise Ticket35Error("refusing to replace a different frozen Ticket 35 plan")
        return destination
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        destination.chmod(0o600)
        directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError:
        if not destination.is_file() or json.loads(destination.read_text(encoding="utf-8")) != payload:
            raise Ticket35Error("another writer produced a different frozen Ticket 35 plan")
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def freeze_ticket35_membership(
    fold_manifest_path: str | Path,
    inventory_path: str | Path,
    integration_manifest_path: str | Path,
    label_artifact_path: str | Path,
    output_path: str | Path,
    *,
    expected_label_artifact_sha256: str,
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
) -> dict[str, object]:
    """Build the private plan from the canonical fold, inventory and label inputs."""

    from .ticket34 import admit_ticket30_fold_supervision

    fold_path = Path(fold_manifest_path)
    inventory_file = Path(inventory_path)
    integration_file = Path(integration_manifest_path)
    labels_file = Path(label_artifact_path)
    fold_manifest = load_fold_manifest(fold_path)
    if fold_manifest.superseded:
        raise Ticket35Error("superseded fold manifests cannot drive Ticket 35 membership")
    integration = load_integration_manifest(integration_file)
    if integration.source_manifest_checksum != fold_manifest.source_manifest_checksum:
        raise Ticket35Error("fold and integration source manifest identities differ")
    if integration.source_version != fold_manifest.source_version:
        raise Ticket35Error("fold and integration source versions differ")
    if integration.archive_fingerprint != fold_manifest.source_fingerprint:
        raise Ticket35Error("fold and integration source archive identities differ")
    try:
        inventory_source = load_inventory_source(inventory_file)
    except (OSError, ValueError) as exc:
        raise Ticket35Error("could not load the canonical inventory source") from exc
    if inventory_source.source_version != integration.source_version:
        raise Ticket35Error("inventory and integration source versions differ")
    if inventory_source.source_fingerprint != integration.archive_fingerprint:
        raise Ticket35Error("inventory and integration archive identities differ")
    try:
        inventory_payload = json.loads(inventory_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Ticket35Error("could not read the private full inventory") from exc
    inventory_root = _as_mapping(inventory_payload, "inventory")
    inventory_object = _as_mapping(inventory_root.get("inventory"), "inventory body")
    raw_series = inventory_object.get("series")
    if not isinstance(raw_series, list):
        raise Ticket35Error("inventory series must be a list")
    report_fold_by_uid = {
        item.study_instance_uid: item.fold for item in fold_manifest.report_only_memberships
    }
    admissions: dict[int, dict[str, tuple[int | None, ...]]] = {}
    for fold in FOLD_IDS:
        admitted = admit_ticket30_fold_supervision(
            fold_manifest,
            fold,
            labels_file,
            expected_artifact_sha256=expected_label_artifact_sha256,
        )
        admissions[fold] = {
            uid: admitted.by_study[uid].targets
            for uid in admitted.admitted_report_only_study_uids
        }
    identity = {
        "source_version": integration.source_version,
        "archive_fingerprint": integration.archive_fingerprint,
        "source_manifest_checksum": source_manifest_checksum(inventory_source.source_manifest),
        "archive_size_bytes": _integer(
            _as_mapping(inventory_root.get("archive"), "inventory archive").get("archive_size_bytes"),
            "archive_size_bytes",
            minimum=1,
        ),
        "inventory_sha256": sha256_file(inventory_file),
        "integration_manifest_sha256": sha256_file(integration_file),
        "fold_manifest_checksum": fold_manifest.manifest_checksum,
        "label_artifact_sha256": sha256_file(labels_file),
        "preprocessing_identity": {
            "representation_id": "ticket31-resnet18-224-whole-field-v1",
            "image_size": list(image_size),
            "cache_dtype": "float32",
        },
    }
    if identity["label_artifact_sha256"] != expected_label_artifact_sha256:
        raise Ticket35Error("Ticket 30 label artifact digest differs from its frozen expected identity")
    manifest = build_ticket35_membership(
        report_fold_by_uid=report_fold_by_uid,
        admitted_targets_by_fold=admissions,
        inventory_series=cast(Sequence[Mapping[str, object]], raw_series),
        existing_study_uids=integration.study_uids,
        held_out_gold_by_fold={fold: fold_manifest.held_out_study_uids(fold) for fold in FOLD_IDS},
        source_identity=identity,
        batch_size=batch_size,
        image_size=image_size,
        supported_transfer_syntaxes=integration.decoder_policy.supported_transfer_syntaxes,
        minimum_usable_slices=integration.geometry_policy.minimum_usable_slices,
    )
    write_ticket35_membership(output_path, manifest)
    return manifest


def estimate_batch_storage(
    *,
    current_measured_bytes: int,
    source_archive_bytes: int,
    uncompressed_source_bytes: int,
    effective_frame_count: int,
    image_size: tuple[int, int],
    soft_budget_bytes: int = TICKET35_SOFT_STORAGE_BUDGET_BYTES,
) -> dict[str, int | bool]:
    """Conservatively estimate retained plus temporary growth for one batch.

    The archive is required to be present in the measured roots and is counted
    once.  The peak reserves extracted source bytes, the final float32 pixel
    cache, and one more copy of both for decompression and atomic publication.
    """

    for value, name in (
        (current_measured_bytes, "current_measured_bytes"),
        (source_archive_bytes, "source_archive_bytes"),
        (uncompressed_source_bytes, "uncompressed_source_bytes"),
        (effective_frame_count, "effective_frame_count"),
        (soft_budget_bytes, "soft_budget_bytes"),
    ):
        _integer(value, name)
    if source_archive_bytes > current_measured_bytes:
        raise Ticket35StorageError("measured project bytes do not include the source archive")
    if soft_budget_bytes <= 0 or soft_budget_bytes > TICKET35_SOFT_STORAGE_BUDGET_BYTES:
        raise Ticket35StorageError("storage budget cannot exceed the 2 TB soft budget")
    if len(image_size) != 2 or any(not isinstance(value, int) or value <= 0 for value in image_size):
        raise Ticket35StorageError("image_size must contain two positive integers")
    current_project_bytes = current_measured_bytes - source_archive_bytes
    pixel_cache_bytes = effective_frame_count * image_size[0] * image_size[1] * 4
    temporary_bytes = uncompressed_source_bytes + pixel_cache_bytes
    projected_peak_bytes = (
        current_project_bytes
        + source_archive_bytes
        + uncompressed_source_bytes
        + pixel_cache_bytes
        + temporary_bytes
    )
    result: dict[str, int | bool] = {
        "current_project_bytes": current_project_bytes,
        "source_archive_bytes": source_archive_bytes,
        "uncompressed_source_bytes": uncompressed_source_bytes,
        "pixel_cache_bytes": pixel_cache_bytes,
        "temporary_bytes": temporary_bytes,
        "projected_peak_bytes": projected_peak_bytes,
        "soft_budget_bytes": soft_budget_bytes,
        "within_soft_budget": projected_peak_bytes <= soft_budget_bytes,
        "remaining_headroom_bytes": max(0, soft_budget_bytes - projected_peak_bytes),
    }
    if projected_peak_bytes > soft_budget_bytes:
        raise Ticket35StorageError(
            f"projected batch peak {projected_peak_bytes} bytes exceeds the 2 TB soft budget "
            f"of {soft_budget_bytes} bytes"
        )
    return result


def estimate_planned_batch_storage(
    membership: Mapping[str, object],
    batch_index: int,
    *,
    current_measured_bytes: int,
    source_archive_bytes: int,
) -> dict[str, int | bool]:
    """Recompute a frozen batch estimate from a fresh measured storage ledger."""

    payload = validate_ticket35_membership(membership)
    batches_value = payload.get("batches")
    if not isinstance(batches_value, list) or batch_index < 0 or batch_index >= len(batches_value):
        raise Ticket35Error("batch index is outside the frozen membership plan")
    batch = _as_mapping(batches_value[batch_index], "planned batch")
    policy = _as_mapping(payload.get("batch_policy"), "batch policy")
    image_size_value = policy.get("image_size")
    if not isinstance(image_size_value, list) or len(image_size_value) != 2:
        raise Ticket35Error("frozen batch policy has invalid image_size")
    image_size = (
        _integer(image_size_value[0], "image height", minimum=1),
        _integer(image_size_value[1], "image width", minimum=1),
    )
    frozen_archive_bytes = _as_mapping(payload.get("source_identity"), "source identity").get("archive_size_bytes")
    if frozen_archive_bytes is not None and source_archive_bytes != _integer(
        frozen_archive_bytes, "frozen archive bytes", minimum=1
    ):
        raise Ticket35Error("storage ledger archive byte count differs from frozen source identity")
    return estimate_batch_storage(
        current_measured_bytes=current_measured_bytes,
        source_archive_bytes=source_archive_bytes,
        uncompressed_source_bytes=_integer(batch.get("uncompressed_source_bytes"), "batch uncompressed bytes"),
        effective_frame_count=_integer(batch.get("effective_frame_count"), "batch frame count"),
        image_size=image_size,
        soft_budget_bytes=_integer(policy.get("soft_budget_bytes"), "soft budget", minimum=1),
    )


def capture_ticket35_storage_ledger(
    roots: Sequence[tuple[str, str | Path]],
    archive_path: str | Path,
) -> dict[str, object]:
    """Measure every declared campaign root without opening file contents."""

    from .image_diagnostic import DiagnosticError, _storage_snapshot

    labels = [label for label, _ in roots]
    if not roots or len(set(labels)) != len(labels):
        raise Ticket35StorageError("storage roots must have unique labels")
    resolved_roots = tuple(Path(path).resolve() for _, path in roots)
    archive = Path(archive_path).resolve()
    try:
        snapshot = _storage_snapshot(resolved_roots, archive_path=archive)
    except (DiagnosticError, OSError, ValueError) as exc:
        raise Ticket35StorageError("campaign-wide storage ledger could not be measured") from exc
    if snapshot.get("archive_in_measured_roots") is not True:
        raise Ticket35StorageError("the retained source archive is not counted in the measured roots")
    root_metrics = cast(Mapping[str, Mapping[str, object]], snapshot["root_metrics"])
    return {
        "schema_version": "rsna-ticket35-storage-ledger-v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": snapshot["measurement"],
        "measurement_method": "logical bytes from metadata only; file contents were not opened; symlink targets were not followed",
        "roots": [
            {
                "label": label,
                "path": str(path),
                "logical_bytes": root_metrics[f"root_{index}"].get("logical_bytes"),
                "file_entries": root_metrics[f"root_{index}"].get("file_entries"),
            }
            for index, (label, path) in enumerate(zip(labels, resolved_roots, strict=True), start=1)
        ],
        "measured_roots_logical_bytes": _integer(
            snapshot["measured_roots_logical_bytes"], "measured root bytes"
        ),
        "source_archive_bytes": _integer(snapshot["archive_bytes"], "source archive bytes"),
        "source_archive_included_once_in_roots": True,
        "soft_budget_bytes": TICKET35_SOFT_STORAGE_BUDGET_BYTES,
        "projected_headroom_before_batch_bytes": max(
            0,
            TICKET35_SOFT_STORAGE_BUDGET_BYTES
            - _integer(snapshot["measured_roots_logical_bytes"], "measured root bytes"),
        ),
        "root_manifest_sha256": _json_sha256(
            [{"label": label, "path": str(path)} for label, path in zip(labels, resolved_roots, strict=True)]
        ),
    }


def build_ticket35_batch_integration_manifest(
    inventory_path: str | Path,
    reference_integration_path: str | Path,
    membership: Mapping[str, object],
    batch_index: int,
    output_path: str | Path,
    *,
    current_measured_bytes: int,
    source_archive_bytes: int,
    selection_output_path: str | Path,
) -> IntegrationManifest:
    """Build one source-bound 224-pixel integration manifest after preflight."""

    from dataclasses import replace

    from .image_variant import TICKET31_IMAGE_SIZE, TICKET31_REPRESENTATION_ID

    frozen = validate_ticket35_membership(membership)
    batches_value = frozen.get("batches")
    if not isinstance(batches_value, list) or batch_index < 0 or batch_index >= len(batches_value):
        raise Ticket35Error("batch index is outside the frozen membership plan")
    batch = _as_mapping(batches_value[batch_index], "planned batch")
    study_uids_value = batch.get("study_uids")
    if not isinstance(study_uids_value, list) or not study_uids_value:
        raise Ticket35Error("planned batch has no image-ready studies")
    study_uids = tuple(str(uid) for uid in study_uids_value)
    inventory_file = Path(inventory_path)
    try:
        inventory_root = _as_mapping(json.loads(inventory_file.read_text(encoding="utf-8")), "inventory")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Ticket35Error("could not read the source inventory for the selected batch") from exc
    source_identity = _as_mapping(frozen.get("source_identity"), "source identity")
    if sha256_file(inventory_file) != source_identity.get("inventory_sha256"):
        raise Ticket35Error("source inventory changed after the membership plan was frozen")
    if inventory_root.get("source_version") != source_identity.get("source_version"):
        raise Ticket35Error("source inventory version differs from the frozen membership plan")
    archive_root = _as_mapping(inventory_root.get("archive"), "inventory archive")
    if archive_root.get("fingerprint") != source_identity.get("archive_fingerprint"):
        raise Ticket35Error("source archive fingerprint differs from the frozen membership plan")
    inventory_body = _as_mapping(inventory_root.get("inventory"), "inventory body")
    series_value = inventory_body.get("series")
    if not isinstance(series_value, list):
        raise Ticket35Error("inventory series must be an array")
    selected_studies = set(study_uids)
    selected_series = [
        _as_mapping(row, "inventory series row")
        for row in series_value
        if isinstance(row, Mapping) and row.get("StudyInstanceUID") in selected_studies
    ]
    series_uids = sorted(str(row.get("SeriesInstanceUID", "")) for row in selected_series)
    entry_names = sorted(
        str(entry)
        for row in selected_series
        for entry in cast(Sequence[object], row.get("entry_names", ()))
    )
    estimated = estimate_planned_batch_storage(
        frozen,
        batch_index,
        current_measured_bytes=current_measured_bytes,
        source_archive_bytes=source_archive_bytes,
    )
    storage = {
        "current_project_bytes": int(estimated["current_project_bytes"]),
        "source_archive_bytes": source_archive_bytes,
        "extracted_bytes": int(estimated["uncompressed_source_bytes"]),
        "pixel_cache_bytes": int(estimated["pixel_cache_bytes"]),
        "temporary_bytes": int(estimated["temporary_bytes"]),
        "projected_peak_bytes": int(estimated["projected_peak_bytes"]),
        "soft_budget_bytes": int(estimated["soft_budget_bytes"]),
        "within_soft_budget": bool(estimated["within_soft_budget"]),
        "over_budget_bytes": 0,
    }
    selection = {
        "schema_version": "rsna-knee-extraction-selection-v1",
        "source_version": source_identity["source_version"],
        "archive_fingerprint": source_identity["archive_fingerprint"],
        "seed": 35,
        "requested_report_only_count": len(study_uids),
        "gold_study_uids": [],
        "report_only_study_uids": list(study_uids),
        "selected_series_uids": series_uids,
        "blocked_series_uids": [],
        "entry_names": entry_names,
        "covered_strata": [],
        "unrepresented_strata": [],
        "unsupported_transfer_syntaxes": [],
        "storage_estimate": storage,
        "ready_for_extraction": True,
        "ticket35_membership_checksum": frozen["manifest_checksum"],
        "batch_index": batch_index,
    }
    selection_path = Path(selection_output_path)
    selection_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if selection_path.exists():
        try:
            existing = json.loads(selection_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise Ticket35Error("existing batch selection cannot be read") from exc
        if existing != selection:
            raise Ticket35Error("refusing to replace a batch selection with different storage evidence")
    else:
        atomic_json_write(selection_path, selection)
        selection_path.chmod(0o600)
    reference = load_integration_manifest(reference_integration_path)
    if sha256_file(reference_integration_path) != source_identity.get("integration_manifest_sha256"):
        raise Ticket35Error("reference integration manifest changed after plan freeze")
    preprocessing = replace(
        reference.preprocess_config,
        image_size=TICKET31_IMAGE_SIZE,
        representation_id=TICKET31_REPRESENTATION_ID,
    )
    return build_integration_manifest(
        inventory_file,
        selection_path,
        output_path,
        geometry_policy=reference.geometry_policy,
        decoder_policy=reference.decoder_policy,
        preprocess_config=preprocessing,
        packing_config=reference.packing_config,
    )


class _TrackingSeriesCache(SeriesCache):
    """Track cache activity while retaining the normal integrity boundary."""

    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self.events: list[dict[str, object]] = []
        self.latest_prepared_by_study: dict[str, object] = {}

    def get_or_prepare(self, series: Any, config: PreprocessConfig = PreprocessConfig(), **kwargs: Any) -> SeriesCacheResult:
        try:
            result = super().get_or_prepare(series, config, **kwargs)
        except Exception as exc:
            self.events.append(
                {
                    "study_instance_uid": str(getattr(series, "study_instance_uid", "")),
                    "series_instance_uid": str(getattr(series, "series_instance_uid", "")),
                    "status": "rejected",
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                }
            )
            raise
        self.events.append(
            {
                "study_instance_uid": series.study_instance_uid,
                "series_instance_uid": series.series_instance_uid,
                "cache_key": result.cache_key,
                "status": "hit" if result.hit else result.reason,
                "reason": result.reason,
                "source_fingerprint": result.source_fingerprint,
                "bytes_read": result.bytes_read,
                "bytes_written": result.bytes_written,
                "temporary_bytes": result.temporary_bytes,
            }
        )
        return result

    def prepare_study(self, study: Any, config: PreprocessConfig = PreprocessConfig()) -> Any:
        prepared = super().prepare_study(study, config)
        self.latest_prepared_by_study[study.study_instance_uid] = prepared
        return prepared


def _failure_payload(loaded: LoadedStudy, prediction: object | None) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for loaded_series in loaded.series_results:
        if loaded_series.status not in {"accepted", "accepted_partial"} or loaded_series.failures:
            failures.append(
                {
                    "series_instance_uid": loaded_series.manifest_series.series_instance_uid,
                    "status": loaded_series.status,
                    "expected_slice_count": loaded_series.expected_slice_count,
                    "decoded_slice_count": loaded_series.decoded_slice_count,
                    "reasons": [failure.reason for failure in loaded_series.failures],
                }
            )
    if prediction is not None:
        for failure in cast(Sequence[SeriesFailure], getattr(prediction, "rejected_series", ())):
            failures.append(
                {
                    "series_instance_uid": failure.series_instance_uid,
                    "status": "preprocessing_rejected",
                    "reason": failure.reason,
                }
            )
    return failures


def process_ticket35_studies(
    manifest: IntegrationManifest,
    source: str | Path,
    cache_root: str | Path,
    *,
    model: StudyModel | None = None,
    study_uids: Sequence[str] | None = None,
    device: str | torch.device = "cpu",
    allow_local_smoke: bool = False,
    require_slurm: bool = False,
) -> dict[str, object]:
    """Materialize and optionally infer an exact subset through public boundaries.

    Passing an approved, reconstructed model runs the normal ``InferenceEngine``
    and returns twelve scores per study.  Without a model, the operation is a
    cache-preparation run with the same preprocessing, packing and masks.  The
    latter is suitable for an image-preparation allocation that is still
    waiting for a separate model-use approval.
    """

    if require_slurm:
        job_id = os.environ.get("SLURM_JOB_ID")
        partition = os.environ.get("SLURM_JOB_PARTITION")
        if not job_id or partition != "cluster1-cpu":
            raise Ticket35Error("real Ticket 35 preparation requires a cluster1-cpu Slurm allocation")
    if model is None and allow_local_smoke is False and not os.environ.get("SLURM_JOB_ID"):
        raise Ticket35Error("real Ticket 35 preparation must run in Slurm")
    requested = tuple(study_uids) if study_uids is not None else manifest.study_uids
    if not requested or any(not isinstance(uid, str) or not uid.strip() for uid in requested):
        raise Ticket35Error("study_uids must contain at least one non-empty identifier")
    if len(set(requested)) != len(requested):
        raise Ticket35Error("study_uids must be unique")
    unknown = set(requested) - set(manifest.study_uids)
    if unknown:
        raise Ticket35Error("requested studies are absent from the integration manifest")
    if allow_local_smoke:
        if os.environ.get("SLURM_JOB_ID"):
            raise Ticket35Error("local smoke mode cannot run inside Slurm")
        if Path(source).is_dir() is False:
            raise Ticket35Error("local smoke mode requires a tiny extracted fixture directory")
        if sum(len(item.entries) for study in manifest.studies for item in study.series) > 256:
            raise Ticket35Error("local smoke mode is limited to 256 source entries")
    _verify_archive_source_identity(Path(source), manifest)
    torch.set_num_threads(max(1, min(16, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))))
    cache = _TrackingSeriesCache(cache_root)
    engine = (
        InferenceEngine(
            model,
            manifest.preprocess_config,
            device=device,
            packing_config=manifest.packing_config,
            series_cache=cache,
        )
        if model is not None
        else None
    )
    started = time.perf_counter()
    loaded_by_uid: dict[str, LoadedStudy] = {}
    for loaded in iter_real_studies(
        manifest,
        source,
        require_complete_series=False,
        allow_unverified_directory=allow_local_smoke,
        study_uids=requested,
    ):
        if loaded.study.study_instance_uid in loaded_by_uid:
            raise Ticket35Error("real study iterator repeated an admitted study")
        loaded_by_uid[loaded.study.study_instance_uid] = loaded
    if set(loaded_by_uid) != set(requested):
        raise Ticket35Error("real study iterator returned an incomplete requested cohort")

    records: list[dict[str, object]] = []
    prediction_count = 0
    for uid in requested:
        loaded = loaded_by_uid[uid]
        prediction = None
        if engine is not None:
            prediction = engine.predict_study(
                loaded.study,
                input_failures=tuple(
                    SeriesFailure(
                        item.manifest_series.series_instance_uid,
                        f"{failure.reason}: {failure.detail}",
                    )
                    for item in loaded.series_results
                    for failure in item.failures
                ),
                input_warnings=loaded.warnings,
            )
            prepared = cache.latest_prepared_by_study.get(uid)
        else:
            prepared = cache.prepare_study(loaded.study, manifest.preprocess_config)
        if prepared is None:
            raise Ticket35Error(f"preprocessing produced no study record for {uid}")
        prepared_study = cast(Any, prepared)
        packed = pack_study(
            loaded.study,
            prepared_study.series,
            manifest.preprocess_config,
            packing_config=manifest.packing_config,
        )
        series_mask_count = int(packed.series_mask.sum())
        slice_mask_count = int(packed.slice_mask.sum())
        record: dict[str, object] = {
            "study_instance_uid": uid,
            "manifest_series_count": len(loaded.series_results),
            "decoded_slice_count": sum(item.decoded_slice_count for item in loaded.series_results),
            "expected_slice_count": sum(item.expected_slice_count for item in loaded.series_results),
            "prepared_series_count": len(prepared_study.series),
            "series_mask_count": series_mask_count,
            "slice_mask_count": slice_mask_count,
            "input_exclusions": _failure_payload(loaded, prediction),
            "complete_input": all(item.complete for item in loaded.series_results),
            "used_fallback": bool(getattr(prediction, "used_fallback", False)) if prediction is not None else None,
        }
        if prediction is not None:
            scores = np.asarray(prediction.scores, dtype=np.float64)
            if prediction.study_instance_uid != uid or scores.shape != (NUM_TARGETS,) or not np.isfinite(scores).all():
                raise Ticket35Error(f"prediction identity or score vector is incomplete for {uid}")
            record["prediction_scores"] = [float(value) for value in scores]
            record["valid_series_count"] = prediction.valid_series_count
            prediction_count += 1
        records.append(record)
    if len(records) != len(requested) or {str(item["study_instance_uid"]) for item in records} != set(requested):
        raise Ticket35Error("Ticket 35 output study identities do not match the requested cohort")
    cache_statuses = [str(item.get("status", "")) for item in cache.events]
    return {
        "schema_version": TICKET35_BATCH_EVIDENCE_SCHEMA_VERSION,
        "status": "completed_with_exclusions" if any(item["input_exclusions"] for item in records) else "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity": {
            "source_version": manifest.source_version,
            "archive_fingerprint": manifest.archive_fingerprint,
            "source_manifest_checksum": manifest.source_manifest_checksum,
        },
        "preprocessing": manifest.preprocess_config.cache_config(),
        "study_uids": list(requested),
        "study_count": len(records),
        "series_count": sum(_integer(item.get("manifest_series_count"), "manifest_series_count") for item in records),
        "usable_series_count": sum(_integer(item.get("series_mask_count"), "series_mask_count") for item in records),
        "expected_slice_count": sum(_integer(item.get("expected_slice_count"), "expected_slice_count") for item in records),
        "decoded_slice_count": sum(_integer(item.get("decoded_slice_count"), "decoded_slice_count") for item in records),
        "usable_slice_count": sum(_integer(item.get("slice_mask_count"), "slice_mask_count") for item in records),
        "supervised_target_entry_count": None,
        "elapsed_seconds": time.perf_counter() - started,
        "prediction_count": prediction_count,
        "cache": {
            "hits": sum(value == "hit" for value in cache_statuses),
            "misses": sum(value == "miss" for value in cache_statuses),
            "rebuilds": sum(value == "rebuild" for value in cache_statuses),
            "rejections": sum(value == "rejected" for value in cache_statuses),
            "bytes_read": sum(_integer(item.get("bytes_read", 0), "cache bytes_read") for item in cache.events),
            "bytes_written": sum(_integer(item.get("bytes_written", 0), "cache bytes_written") for item in cache.events),
            "temporary_bytes": sum(_integer(item.get("temporary_bytes", 0), "cache temporary_bytes") for item in cache.events),
            "events": cache.events,
        },
        "studies": records,
    }


def write_ticket35_batch_evidence(path: str | Path, evidence: Mapping[str, object]) -> Path:
    """Write one batch result atomically; an interrupted run can be repeated."""

    if evidence.get("schema_version") != TICKET35_BATCH_EVIDENCE_SCHEMA_VERSION:
        raise Ticket35Error("batch evidence has an unsupported schema")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.is_symlink():
        raise Ticket35Error("batch evidence destination must not be a symlink")
    payload = dict(evidence)
    payload["evidence_checksum"] = _json_sha256(payload)
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise Ticket35Error("existing batch evidence cannot be read") from exc
        if existing != payload:
            raise Ticket35Error("refusing to replace different completed batch evidence")
        return destination
    try:
        atomic_json_write(destination, payload)
    except (OSError, ValueError) as exc:
        raise Ticket35Error("could not atomically publish Ticket 35 batch evidence") from exc
    return destination


__all__ = [
    "TICKET35_BATCH_EVIDENCE_SCHEMA_VERSION",
    "TICKET35_MEMBERSHIP_SCHEMA_VERSION",
    "TICKET35_SOFT_STORAGE_BUDGET_BYTES",
    "Ticket35Error",
    "Ticket35StorageError",
    "build_ticket35_membership",
    "build_ticket35_batch_integration_manifest",
    "capture_ticket35_storage_ledger",
    "estimate_batch_storage",
    "estimate_planned_batch_storage",
    "freeze_ticket35_membership",
    "process_ticket35_studies",
    "sha256_file",
    "validate_ticket35_membership",
    "write_ticket35_batch_evidence",
    "write_ticket35_membership",
]
