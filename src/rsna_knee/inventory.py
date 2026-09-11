"""Private corpus inventory and bounded data-admission primitives.

The inventory path deliberately stops at DICOM headers.  Pixel decoding and
cache creation are separate operations so a complete metadata audit does not
require materialising the corpus or silently turning a bad file into a
missing row.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import socket
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

import pydicom
from pydicom.errors import InvalidDicomError

from .archive import (
    ArchivePathError,
    atomic_json_write,
    extract_batch,
    failed_extraction_report,
    safe_member_path,
    write_extraction_report,
)
from .geometry import GeometryPolicy
from .targets import TARGET_COLUMNS


INVENTORY_SCHEMA_VERSION = "rsna-knee-inventory-v1"
_NON_DIAGNOSTIC_METADATA_TOKENS = frozenset(
    {"scout", "localizer", "localiser", "survey", "pilot", "calibration"}
)


def _has_non_diagnostic_token(values: Iterable[str]) -> bool:
    return any(
        re.search(rf"(?<![\w]){re.escape(token)}(?![\w])", value.casefold())
        for value in values
        for token in _NON_DIAGNOSTIC_METADATA_TOKENS
    )


class InventoryError(ValueError):
    """Base class for invalid or unsafe inventory/admission inputs."""


class SourceManifestError(InventoryError):
    """Raised when the competition source CSVs cannot be reconciled."""


class SelectionError(InventoryError):
    """Raised when a safe representative extraction batch cannot be selected."""


class StorageBudgetExceeded(InventoryError):
    """Raised before extraction when the projected peak exceeds the soft budget."""


@dataclass(frozen=True)
class SourceStudy:
    """One study row from the source metadata, including optional references."""

    study_instance_uid: str
    report_present: bool
    reference_labels: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if len(self.reference_labels) != len(TARGET_COLUMNS):
            raise SourceManifestError("source study reference vectors must contain all competition targets")
        if any(label not in {None, 0, 1} for label in self.reference_labels):
            raise SourceManifestError("source study reference vectors may contain only 0, 1 or missing values")

    @property
    def has_reference_labels(self) -> bool:
        return all(label is not None for label in self.reference_labels)

    @property
    def has_partial_reference_labels(self) -> bool:
        return any(label is not None for label in self.reference_labels) and not self.has_reference_labels


@dataclass(frozen=True)
class SourceSeries:
    """One source series descriptor with independent sequence descriptors."""

    study_instance_uid: str
    series_instance_uid: str
    fluid_sensitive: bool | None
    fat_suppression: bool | None
    anatomical_plane: str | None


@dataclass(frozen=True)
class SourceManifest:
    """Reconciled source CSV metadata for one competition split."""

    split: str
    studies: tuple[SourceStudy, ...]
    series: tuple[SourceSeries, ...]

    def __post_init__(self) -> None:
        study_uids = [study.study_instance_uid for study in self.studies]
        if any(not study.study_instance_uid for study in self.studies):
            raise SourceManifestError("source study metadata contains an empty StudyInstanceUID")
        if any(len(study.reference_labels) != len(TARGET_COLUMNS) for study in self.studies):
            raise SourceManifestError("source study reference vectors must contain all competition targets")
        if len(set(study_uids)) != len(study_uids):
            raise SourceManifestError("source study metadata contains duplicate StudyInstanceUID values")
        series_uids = [item.series_instance_uid for item in self.series]
        if any(not item.series_instance_uid or not item.study_instance_uid for item in self.series):
            raise SourceManifestError("source series metadata contains an empty identifier")
        if len(set(series_uids)) != len(series_uids):
            raise SourceManifestError("source series metadata contains duplicate SeriesInstanceUID values")
        known_studies = set(study_uids)
        orphaned = [item.series_instance_uid for item in self.series if item.study_instance_uid not in known_studies]
        if orphaned:
            raise SourceManifestError(f"source series reference unknown studies: {orphaned[:3]}")

    @property
    def study_count(self) -> int:
        return len(self.studies)

    @property
    def series_count(self) -> int:
        return len(self.series)

    @property
    def gold_study_uids(self) -> tuple[str, ...]:
        return tuple(study.study_instance_uid for study in self.studies if study.has_reference_labels)

    @property
    def report_only_study_uids(self) -> tuple[str, ...]:
        return tuple(
            study.study_instance_uid
            for study in self.studies
            if not study.has_reference_labels and not study.has_partial_reference_labels
        )

    @property
    def partially_labeled_study_uids(self) -> tuple[str, ...]:
        return tuple(study.study_instance_uid for study in self.studies if study.has_partial_reference_labels)

    @property
    def studies_by_uid(self) -> dict[str, SourceStudy]:
        return {study.study_instance_uid: study for study in self.studies}

    @property
    def series_by_uid(self) -> dict[str, SourceSeries]:
        return {item.series_instance_uid: item for item in self.series}

    @property
    def series_by_study(self) -> dict[str, tuple[SourceSeries, ...]]:
        grouped: dict[str, list[SourceSeries]] = {}
        for item in self.series:
            grouped.setdefault(item.study_instance_uid, []).append(item)
        return {study_uid: tuple(items) for study_uid, items in grouped.items()}


@dataclass(frozen=True)
class ArchiveIdentity:
    """Stable source/archive facts recorded with an inventory run."""

    source_version: str
    archive_name: str
    archive_size_bytes: int
    archive_mtime_ns: int
    zip_entry_count: int
    central_directory_sha256: str
    archive_sha256: str | None
    fingerprint: str


@dataclass(frozen=True)
class HeaderInspection:
    """Header-only facts for one archive member; pixels are never loaded."""

    archive_name: str
    expected_study_instance_uid: str
    expected_series_instance_uid: str
    study_instance_uid: str | None
    series_instance_uid: str | None
    sop_instance_uid: str | None
    transfer_syntax_uid: str | None
    frame_count: int | None
    rows: int | None
    columns: int | None
    image_position: tuple[float, float, float] | None
    image_orientation: tuple[float, float, float, float, float, float] | None
    position_status: str
    orientation_status: str
    image_type_values: tuple[str, ...]
    photometric_interpretation: str | None
    bits_allocated: int | None
    bits_stored: int | None
    high_bit: int | None
    pixel_representation: int | None
    samples_per_pixel: int | None
    dicom_series_description: str
    identity_conflicts: tuple[str, ...]


@dataclass(frozen=True)
class HeaderFailure:
    """A DICOM member that was inspected but could not yield a header."""

    archive_name: str
    expected_study_instance_uid: str
    expected_series_instance_uid: str
    reason: str
    detail: str


@dataclass(frozen=True)
class SeriesInventory:
    """Aggregated metadata and failure accounting for one source series."""

    study_instance_uid: str
    series_instance_uid: str
    anatomical_plane: str | None
    fluid_sensitive: bool | None
    fat_suppression: bool | None
    entry_names: tuple[str, ...]
    compressed_bytes: int
    uncompressed_bytes: int
    entry_count: int
    readable_header_count: int
    unreadable_header_count: int
    identity_conflict_count: int
    diagnostic_status: str
    image_type_values: tuple[str, ...]
    orientation_counts: dict[str, int]
    unknown_orientation_count: int
    transfer_syntax_counts: dict[str, int]
    unknown_transfer_syntax_count: int
    frame_count_counts: dict[int, int]
    unknown_frame_count_count: int
    dimension_counts: dict[str, int]
    unknown_dimension_count: int
    photometric_counts: dict[str, int]
    effective_frame_count: int
    geometry_valid_count: int
    geometry_missing_count: int
    geometry_invalid_count: int
    geometry_inconsistent_count: int
    spacing_inconsistent_count: int
    duplicate_position_count: int
    geometry_status: str
    dicom_series_descriptions: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        """Whether this series is safe to admit to a bounded pixel batch."""

        return (
            self.entry_count > 0
            and self.readable_header_count == self.entry_count
            and self.unreadable_header_count == 0
            and self.identity_conflict_count == 0
            and self.unknown_transfer_syntax_count == 0
            and self.unknown_orientation_count == 0
            and self.unknown_frame_count_count == 0
            and self.unknown_dimension_count == 0
            and len(self.dimension_counts) == 1
            and len(self.frame_count_counts) == 1
            and self.geometry_status == "complete"
            and self.diagnostic_status != "excluded_non_diagnostic"
        )

    @property
    def slice_count_bucket(self) -> str:
        count = self.effective_frame_count
        if count == 0:
            return "empty"
        if count <= 8:
            return "1-8"
        if count <= 32:
            return "9-32"
        if count <= 64:
            return "33-64"
        return "65+"

    @property
    def stratum(self) -> "InventoryStratum":
        """Return the auditable class used by representative selection."""

        return InventoryStratum(
            transfer_syntaxes=tuple(sorted(self.transfer_syntax_counts)),
            dimensions=tuple(sorted(self.dimension_counts)),
            frame_counts=tuple(sorted(self.frame_count_counts)),
            orientations=tuple(sorted(self.orientation_counts)),
            geometry_status=self.geometry_status,
            anatomical_plane=self.anatomical_plane,
            fluid_sensitive=self.fluid_sensitive,
            fat_suppression=self.fat_suppression,
            slice_count_bucket=self.slice_count_bucket,
            diagnostic_status=self.diagnostic_status,
            unknown_transfer_syntax=self.unknown_transfer_syntax_count > 0,
            unknown_dimension=self.unknown_dimension_count > 0,
            unknown_frame_count=self.unknown_frame_count_count > 0,
            unknown_orientation=self.unknown_orientation_count > 0,
        )


@dataclass(frozen=True)
class InventoryStratum:
    """A representative-selection class, retaining independent descriptors."""

    transfer_syntaxes: tuple[str, ...]
    dimensions: tuple[str, ...]
    frame_counts: tuple[int, ...]
    orientations: tuple[str, ...]
    geometry_status: str
    anatomical_plane: str | None
    fluid_sensitive: bool | None
    fat_suppression: bool | None
    slice_count_bucket: str
    diagnostic_status: str
    unknown_transfer_syntax: bool
    unknown_dimension: bool
    unknown_frame_count: bool
    unknown_orientation: bool


@dataclass(frozen=True)
class StorageEstimate:
    """Conservative persistent-storage projection for one extraction batch."""

    current_project_bytes: int
    source_archive_bytes: int
    extracted_bytes: int
    pixel_cache_bytes: int
    temporary_bytes: int
    projected_peak_bytes: int
    soft_budget_bytes: int

    @property
    def over_budget_bytes(self) -> int:
        return max(0, self.projected_peak_bytes - self.soft_budget_bytes)

    @property
    def within_soft_budget(self) -> bool:
        return self.projected_peak_bytes <= self.soft_budget_bytes


@dataclass(frozen=True)
class ExtractionSelection:
    """Private, identifier-bearing plan for a bounded representative batch."""

    seed: int
    source_version: str
    archive_fingerprint: str
    requested_report_only_count: int
    gold_study_uids: tuple[str, ...]
    report_only_study_uids: tuple[str, ...]
    selected_series_uids: tuple[str, ...]
    blocked_series_uids: tuple[str, ...]
    entry_names: tuple[str, ...]
    covered_strata: tuple[InventoryStratum, ...]
    unrepresented_strata: tuple[InventoryStratum, ...]
    unsupported_transfer_syntaxes: tuple[str, ...]
    storage_estimate: StorageEstimate

    @property
    def study_uids(self) -> tuple[str, ...]:
        return self.gold_study_uids + self.report_only_study_uids

    @property
    def ready_for_extraction(self) -> bool:
        return (
            not self.blocked_series_uids
            and not self.unsupported_transfer_syntaxes
            and self.storage_estimate.within_soft_budget
        )

    def to_private_dict(self) -> dict[str, object]:
        """Return the identifier-bearing selection plan for a private path."""

        return {
            "schema_version": "rsna-knee-extraction-selection-v1",
            "source_version": self.source_version,
            "archive_fingerprint": self.archive_fingerprint,
            "seed": self.seed,
            "requested_report_only_count": self.requested_report_only_count,
            "gold_study_uids": list(self.gold_study_uids),
            "report_only_study_uids": list(self.report_only_study_uids),
            "selected_series_uids": list(self.selected_series_uids),
            "blocked_series_uids": list(self.blocked_series_uids),
            "entry_names": list(self.entry_names),
            "covered_strata": [_stratum_to_dict(item) for item in self.covered_strata],
            "unrepresented_strata": [_stratum_to_dict(item) for item in self.unrepresented_strata],
            "unsupported_transfer_syntaxes": list(self.unsupported_transfer_syntaxes),
            "storage_estimate": _storage_to_dict(self.storage_estimate),
            "ready_for_extraction": self.ready_for_extraction,
        }

    def to_public_dict(self) -> dict[str, object]:
        """Return selection counts and coverage without study or file identifiers."""

        return {
            "schema_version": "rsna-knee-extraction-selection-v1",
            "source_version": self.source_version,
            "archive_fingerprint": self.archive_fingerprint,
            "seed": self.seed,
            "requested_report_only_count": self.requested_report_only_count,
            "counts": {
                "gold_studies": len(self.gold_study_uids),
                "report_only_studies": len(self.report_only_study_uids),
                "selected_series": len(self.selected_series_uids),
                "blocked_series": len(self.blocked_series_uids),
                "selected_files": len(self.entry_names),
                "covered_strata": len(self.covered_strata),
                "unrepresented_strata": len(self.unrepresented_strata),
                "unsupported_transfer_syntaxes": len(self.unsupported_transfer_syntaxes),
            },
            "coverage": {
                "covered_strata": [_stratum_to_dict(item) for item in self.covered_strata],
                "unrepresented_strata": [_stratum_to_dict(item) for item in self.unrepresented_strata],
                "unsupported_transfer_syntaxes": list(self.unsupported_transfer_syntaxes),
            },
            "storage_estimate": _storage_to_dict(self.storage_estimate),
            "ready_for_extraction": self.ready_for_extraction,
        }


@dataclass(frozen=True)
class InventoryReport:
    """Reconciled full-split header inventory and its private diagnostics."""

    schema_version: str
    source_version: str
    split: str
    archive_identity: ArchiveIdentity
    geometry_policy: GeometryPolicy
    source_manifest: SourceManifest
    archive_data_file_count: int
    archive_dicom_file_count: int
    readable_header_count: int
    unreadable_headers: tuple[HeaderFailure, ...]
    malformed_archive_entries: tuple[str, ...]
    duplicate_archive_entries: tuple[str, ...]
    extra_series: tuple[tuple[str, str], ...]
    missing_series: tuple[tuple[str, str], ...]
    identity_conflict_count: int
    actual_study_uids: tuple[str, ...]
    series: tuple[SeriesInventory, ...]
    expected_study_count: int | None
    expected_gold_count: int | None

    @property
    def study_count(self) -> int:
        return self.source_manifest.study_count

    @property
    def gold_study_count(self) -> int:
        return len(self.source_manifest.gold_study_uids)

    @property
    def report_only_study_count(self) -> int:
        return len(self.source_manifest.report_only_study_uids)

    @property
    def series_count(self) -> int:
        return self.source_manifest.series_count

    @property
    def actual_study_count(self) -> int:
        return len(self.actual_study_uids)

    @property
    def unreadable_header_count(self) -> int:
        return len(self.unreadable_headers)

    @property
    def series_by_uid(self) -> dict[str, SeriesInventory]:
        return {item.series_instance_uid: item for item in self.series}

    @property
    def source_counts_match_expectations(self) -> bool:
        study_matches = self.expected_study_count is None or self.study_count == self.expected_study_count
        gold_matches = self.expected_gold_count is None or self.gold_study_count == self.expected_gold_count
        return study_matches and gold_matches

    @property
    def full_corpus_reconciled(self) -> bool:
        """Return true only after path, header and source membership checks pass."""

        source_studies = set(self.source_manifest.studies_by_uid)
        observed_series_entries = sum(item.entry_count for item in self.series)
        return (
            self.source_counts_match_expectations
            and not self.source_manifest.partially_labeled_study_uids
            and self.actual_study_count == self.study_count
            and set(self.actual_study_uids) == source_studies
            and observed_series_entries == self.archive_dicom_file_count
            and self.archive_dicom_file_count == self.archive_data_file_count
            and not self.unreadable_headers
            and not self.malformed_archive_entries
            and not self.duplicate_archive_entries
            and not self.extra_series
            and not self.missing_series
            and self.identity_conflict_count == 0
            and all(item.is_complete for item in self.series)
        )

    @property
    def transfer_syntax_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for item in self.series:
            counts.update(item.transfer_syntax_counts)
            if item.unknown_transfer_syntax_count:
                counts["unknown"] += item.unknown_transfer_syntax_count
        return dict(counts)

    @property
    def geometry_status_counts(self) -> dict[str, int]:
        return dict(Counter(item.geometry_status for item in self.series))

    @property
    def dimension_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for item in self.series:
            counts.update(item.dimension_counts)
            if item.unknown_dimension_count:
                counts["unknown"] += item.unknown_dimension_count
        return dict(counts)

    @property
    def frame_count_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for item in self.series:
            counts.update({str(key): value for key, value in item.frame_count_counts.items()})
            if item.unknown_frame_count_count:
                counts["unknown"] += item.unknown_frame_count_count
        return dict(counts)

    @property
    def orientation_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for item in self.series:
            counts.update(item.orientation_counts)
            if item.unknown_orientation_count:
                counts["unknown"] += item.unknown_orientation_count
        return dict(counts)

    @property
    def slice_count_bucket_counts(self) -> dict[str, int]:
        return dict(Counter(item.slice_count_bucket for item in self.series))

    @property
    def descriptor_counts(self) -> dict[str, dict[str, int]]:
        planes: Counter[str] = Counter()
        fluid: Counter[str] = Counter()
        fat: Counter[str] = Counter()
        for item in self.series:
            planes[item.anatomical_plane or "unknown"] += 1
            fluid[str(item.fluid_sensitive).lower() if item.fluid_sensitive is not None else "unknown"] += 1
            fat[str(item.fat_suppression).lower() if item.fat_suppression is not None else "unknown"] += 1
        return {
            "anatomical_plane": dict(sorted(planes.items())),
            "fluid_sensitive": dict(sorted(fluid.items())),
            "fat_suppression": dict(sorted(fat.items())),
        }

    def to_public_dict(self) -> dict[str, object]:
        """Return an aggregate-only representation safe for public artifacts."""

        return {
            "schema_version": self.schema_version,
            "source_version": self.source_version,
            "split": self.split,
            "geometry_policy": {
                "policy_version": self.geometry_policy.policy_version,
                "position_tolerance": self.geometry_policy.position_tolerance,
                "orientation_tolerance": self.geometry_policy.orientation_tolerance,
                "spacing_tolerance": self.geometry_policy.spacing_tolerance,
                "minimum_usable_slices": self.geometry_policy.minimum_usable_slices,
                "require_consistent_dimensions": self.geometry_policy.require_consistent_dimensions,
                "require_consistent_orientation": self.geometry_policy.require_consistent_orientation,
            },
            "archive": {
                "archive_size_bytes": self.archive_identity.archive_size_bytes,
                "zip_entry_count": self.archive_identity.zip_entry_count,
                "central_directory_sha256": self.archive_identity.central_directory_sha256,
                "archive_sha256": self.archive_identity.archive_sha256,
                "fingerprint": self.archive_identity.fingerprint,
            },
            "counts": {
                "source_studies": self.study_count,
                "source_reference_labeled_studies": self.gold_study_count,
                "source_report_only_studies": self.report_only_study_count,
                "source_partial_reference_labeled_studies": len(
                    self.source_manifest.partially_labeled_study_uids
                ),
                "source_series": self.series_count,
                "archive_data_files": self.archive_data_file_count,
                "archive_dicom_files": self.archive_dicom_file_count,
                "readable_headers": self.readable_header_count,
                "unreadable_headers": self.unreadable_header_count,
                "malformed_archive_entries": len(self.malformed_archive_entries),
                "duplicate_archive_entries": len(self.duplicate_archive_entries),
                "extra_series": len(self.extra_series),
                "missing_series": len(self.missing_series),
                "identity_conflicts": self.identity_conflict_count,
                "observed_studies": self.actual_study_count,
            },
            "historical_expectations": {
                "study_count": self.expected_study_count,
                "reference_labeled_study_count": self.expected_gold_count,
                "used_as_expectations_only": True,
            },
            "reconciliation": {
                "source_counts_match_expectations": self.source_counts_match_expectations,
                "full_corpus_reconciled": self.full_corpus_reconciled,
            },
            "transfer_syntax_counts": self.transfer_syntax_counts,
            "frame_count_counts": self.frame_count_counts,
            "orientation_counts": self.orientation_counts,
            "slice_count_bucket_counts": self.slice_count_bucket_counts,
            "dimension_counts": self.dimension_counts,
            "geometry_status_counts": self.geometry_status_counts,
            "descriptor_counts": self.descriptor_counts,
            "diagnostic_status_counts": dict(Counter(item.diagnostic_status for item in self.series)),
        }

    def to_private_dict(self) -> dict[str, object]:
        """Return the identifier-bearing audit record for a private path."""

        return {
            "schema_version": self.schema_version,
            "source_version": self.source_version,
            "split": self.split,
            "geometry_policy": {
                "policy_version": self.geometry_policy.policy_version,
                "position_tolerance": self.geometry_policy.position_tolerance,
                "orientation_tolerance": self.geometry_policy.orientation_tolerance,
                "spacing_tolerance": self.geometry_policy.spacing_tolerance,
                "minimum_usable_slices": self.geometry_policy.minimum_usable_slices,
                "require_consistent_dimensions": self.geometry_policy.require_consistent_dimensions,
                "require_consistent_orientation": self.geometry_policy.require_consistent_orientation,
            },
            "archive": {
                "archive_name": self.archive_identity.archive_name,
                "archive_size_bytes": self.archive_identity.archive_size_bytes,
                "archive_mtime_ns": self.archive_identity.archive_mtime_ns,
                "zip_entry_count": self.archive_identity.zip_entry_count,
                "central_directory_sha256": self.archive_identity.central_directory_sha256,
                "archive_sha256": self.archive_identity.archive_sha256,
                "fingerprint": self.archive_identity.fingerprint,
            },
            "source": {
                "studies": [
                    {
                        "StudyInstanceUID": item.study_instance_uid,
                        "report_present": item.report_present,
                        "reference_labels": list(item.reference_labels),
                    }
                    for item in self.source_manifest.studies
                ],
                "gold_studies": self.gold_study_count,
                "report_only_studies": self.report_only_study_count,
                "partial_reference_labeled_studies": list(
                    self.source_manifest.partially_labeled_study_uids
                ),
                "series": [
                    {
                        "StudyInstanceUID": item.study_instance_uid,
                        "SeriesInstanceUID": item.series_instance_uid,
                        "Fluid_Sensitive": item.fluid_sensitive,
                        "Fat_Suppression": item.fat_suppression,
                        "Anatomical_Plane": item.anatomical_plane,
                    }
                    for item in self.source_manifest.series
                ],
            },
            "inventory": {
                "archive_data_file_count": self.archive_data_file_count,
                "archive_dicom_file_count": self.archive_dicom_file_count,
                "readable_header_count": self.readable_header_count,
                "actual_study_uids": list(self.actual_study_uids),
                "malformed_archive_entries": list(self.malformed_archive_entries),
                "duplicate_archive_entries": list(self.duplicate_archive_entries),
                "extra_series": [list(value) for value in self.extra_series],
                "missing_series": [list(value) for value in self.missing_series],
                "identity_conflict_count": self.identity_conflict_count,
                "unreadable_headers": [
                    {
                        "archive_name": item.archive_name,
                        "expected_study_instance_uid": item.expected_study_instance_uid,
                        "expected_series_instance_uid": item.expected_series_instance_uid,
                        "reason": item.reason,
                        "detail": item.detail,
                    }
                    for item in self.unreadable_headers
                ],
                "series": [_series_to_dict(item) for item in self.series],
            },
            "historical_expectations": {
                "study_count": self.expected_study_count,
                "reference_labeled_study_count": self.expected_gold_count,
                "used_as_expectations_only": True,
            },
            "reconciliation": {
                "source_counts_match_expectations": self.source_counts_match_expectations,
                "full_corpus_reconciled": self.full_corpus_reconciled,
            },
        }


class _SeriesAccumulator:
    def __init__(self, source: SourceSeries, geometry_policy: GeometryPolicy) -> None:
        self.study_instance_uid = source.study_instance_uid
        self.series_instance_uid = source.series_instance_uid
        self.anatomical_plane = source.anatomical_plane
        self.fluid_sensitive = source.fluid_sensitive
        self.fat_suppression = source.fat_suppression
        self.geometry_policy = geometry_policy
        self.entry_names: list[str] = []
        self.compressed_bytes = 0
        self.uncompressed_bytes = 0
        self.readable_header_count = 0
        self.unreadable_header_count = 0
        self.identity_conflict_count = 0
        self.image_type_values: set[str] = set()
        self.orientation_counts: Counter[str] = Counter()
        self.unknown_orientation_count = 0
        self.transfer_syntax_counts: Counter[str] = Counter()
        self.unknown_transfer_syntax_count = 0
        self.frame_count_counts: Counter[int] = Counter()
        self.unknown_frame_count_count = 0
        self.dimension_counts: Counter[str] = Counter()
        self.unknown_dimension_count = 0
        self.photometric_counts: Counter[str] = Counter()
        self.effective_frame_count = 0
        self.geometry_valid_count = 0
        self.geometry_missing_count = 0
        self.geometry_invalid_count = 0
        self.geometry_inconsistent_count = 0
        self.spacing_inconsistent_count = 0
        self.positions: list[float] = []
        self.reference_normal: tuple[float, float, float] | None = None
        self.dicom_series_descriptions: set[str] = set()

    def add_entry(self, name: str, compressed_bytes: int, uncompressed_bytes: int) -> None:
        self.entry_names.append(name)
        self.compressed_bytes += compressed_bytes
        self.uncompressed_bytes += uncompressed_bytes

    def add_header(self, header: HeaderInspection) -> None:
        self.readable_header_count += 1
        if header.identity_conflicts:
            self.identity_conflict_count += 1
        self.image_type_values.update(header.image_type_values)
        if header.orientation_status != "present" or header.image_orientation is None:
            self.unknown_orientation_count += 1
        else:
            orientation_key = _orientation_key(
                header.image_orientation,
                self.geometry_policy.orientation_tolerance,
            )
            if orientation_key is None:
                self.unknown_orientation_count += 1
            else:
                self.orientation_counts[orientation_key] += 1
        if header.transfer_syntax_uid is None:
            self.unknown_transfer_syntax_count += 1
        else:
            self.transfer_syntax_counts[header.transfer_syntax_uid] += 1
        if header.frame_count is None:
            self.unknown_frame_count_count += 1
        else:
            self.frame_count_counts[header.frame_count] += 1
            self.effective_frame_count += header.frame_count
        if header.rows is None or header.columns is None:
            self.unknown_dimension_count += 1
        else:
            self.dimension_counts[f"{header.rows}x{header.columns}"] += 1
        if header.photometric_interpretation is not None:
            self.photometric_counts[header.photometric_interpretation] += 1
        if header.position_status == "missing" or header.orientation_status == "missing":
            self.geometry_missing_count += 1
        elif header.position_status == "invalid" or header.orientation_status == "invalid":
            self.geometry_invalid_count += 1
        else:
            position = header.image_position
            orientation = header.image_orientation
            assert position is not None and orientation is not None
            normal = _normal_from_orientation(orientation, self.geometry_policy.orientation_tolerance)
            if normal is None:
                self.geometry_invalid_count += 1
            elif self.reference_normal is None:
                self.reference_normal = normal
                self.positions.append(_project(position, normal))
                self.geometry_valid_count += 1
            elif _dot(self.reference_normal, normal) < 1.0 - self.geometry_policy.orientation_tolerance:
                self.geometry_inconsistent_count += 1
            else:
                self.positions.append(_project(position, self.reference_normal))
                self.geometry_valid_count += 1
        description = header.dicom_series_description
        if description:
            self.dicom_series_descriptions.add(description)

    def _diagnostic_status(self) -> str:
        metadata = tuple(self.dicom_series_descriptions) + tuple(self.image_type_values)
        if not metadata:
            return "unknown"
        if _has_non_diagnostic_token(metadata):
            return "excluded_non_diagnostic"
        return "diagnostic"

    def add_failure(self) -> None:
        self.unreadable_header_count += 1

    def finish(self) -> SeriesInventory:
        duplicate_position_count = 0
        if len(self.positions) > 1:
            ordered = sorted(self.positions)
            duplicate_position_count = sum(
                1
                for previous, current in zip(ordered, ordered[1:])
                if current - previous <= self.geometry_policy.position_tolerance
            )
            spacings = [current - previous for previous, current in zip(ordered, ordered[1:])]
            reference_spacing = spacings[0]
            self.spacing_inconsistent_count = sum(
                1
                for spacing in spacings[1:]
                if not math.isclose(
                    spacing,
                    reference_spacing,
                    rel_tol=self.geometry_policy.spacing_tolerance,
                    abs_tol=self.geometry_policy.spacing_tolerance,
                )
            )
        if not self.readable_header_count:
            geometry_status = "unreadable"
        elif self.geometry_invalid_count:
            geometry_status = "invalid"
        elif self.geometry_missing_count:
            geometry_status = "missing"
        elif self.geometry_inconsistent_count:
            geometry_status = "inconsistent"
        elif self.spacing_inconsistent_count:
            geometry_status = "inconsistent"
        elif duplicate_position_count:
            geometry_status = "duplicate"
        elif self.effective_frame_count < self.geometry_policy.minimum_usable_slices:
            geometry_status = "insufficient_slices"
        else:
            geometry_status = "complete"
        return SeriesInventory(
            study_instance_uid=self.study_instance_uid,
            series_instance_uid=self.series_instance_uid,
            anatomical_plane=self.anatomical_plane,
            fluid_sensitive=self.fluid_sensitive,
            fat_suppression=self.fat_suppression,
            entry_names=tuple(sorted(self.entry_names)),
            compressed_bytes=self.compressed_bytes,
            uncompressed_bytes=self.uncompressed_bytes,
            entry_count=len(self.entry_names),
            readable_header_count=self.readable_header_count,
            unreadable_header_count=self.unreadable_header_count,
            identity_conflict_count=self.identity_conflict_count,
            diagnostic_status=self._diagnostic_status(),
            image_type_values=tuple(sorted(self.image_type_values)),
            orientation_counts=dict(sorted(self.orientation_counts.items())),
            unknown_orientation_count=self.unknown_orientation_count,
            transfer_syntax_counts=dict(sorted(self.transfer_syntax_counts.items())),
            unknown_transfer_syntax_count=self.unknown_transfer_syntax_count,
            frame_count_counts=dict(sorted(self.frame_count_counts.items())),
            unknown_frame_count_count=self.unknown_frame_count_count,
            dimension_counts=dict(sorted(self.dimension_counts.items())),
            unknown_dimension_count=self.unknown_dimension_count,
            photometric_counts=dict(sorted(self.photometric_counts.items())),
            effective_frame_count=self.effective_frame_count,
            geometry_valid_count=self.geometry_valid_count,
            geometry_missing_count=self.geometry_missing_count,
            geometry_invalid_count=self.geometry_invalid_count,
            geometry_inconsistent_count=self.geometry_inconsistent_count,
            spacing_inconsistent_count=self.spacing_inconsistent_count,
            duplicate_position_count=duplicate_position_count,
            geometry_status=geometry_status,
            dicom_series_descriptions=tuple(sorted(self.dicom_series_descriptions)),
        )


def _series_to_dict(item: SeriesInventory) -> dict[str, object]:
    return {
        "StudyInstanceUID": item.study_instance_uid,
        "SeriesInstanceUID": item.series_instance_uid,
        "Fluid_Sensitive": item.fluid_sensitive,
        "Fat_Suppression": item.fat_suppression,
        "Anatomical_Plane": item.anatomical_plane,
        "entry_names": list(item.entry_names),
        "compressed_bytes": item.compressed_bytes,
        "uncompressed_bytes": item.uncompressed_bytes,
        "entry_count": item.entry_count,
        "readable_header_count": item.readable_header_count,
        "unreadable_header_count": item.unreadable_header_count,
        "identity_conflict_count": item.identity_conflict_count,
        "diagnostic_status": item.diagnostic_status,
        "image_type_values": list(item.image_type_values),
        "orientation_counts": item.orientation_counts,
        "unknown_orientation_count": item.unknown_orientation_count,
        "transfer_syntax_counts": item.transfer_syntax_counts,
        "unknown_transfer_syntax_count": item.unknown_transfer_syntax_count,
        "frame_count_counts": {str(key): value for key, value in item.frame_count_counts.items()},
        "unknown_frame_count_count": item.unknown_frame_count_count,
        "dimension_counts": item.dimension_counts,
        "unknown_dimension_count": item.unknown_dimension_count,
        "photometric_counts": item.photometric_counts,
        "effective_frame_count": item.effective_frame_count,
        "geometry_valid_count": item.geometry_valid_count,
        "geometry_missing_count": item.geometry_missing_count,
        "geometry_invalid_count": item.geometry_invalid_count,
        "geometry_inconsistent_count": item.geometry_inconsistent_count,
        "spacing_inconsistent_count": item.spacing_inconsistent_count,
        "duplicate_position_count": item.duplicate_position_count,
        "geometry_status": item.geometry_status,
        "diagnostic_status": item.diagnostic_status,
        "dicom_series_descriptions": list(item.dicom_series_descriptions),
        "complete": item.is_complete,
    }


def _stratum_to_dict(item: InventoryStratum) -> dict[str, object]:
    return {
        "transfer_syntaxes": list(item.transfer_syntaxes),
        "dimensions": list(item.dimensions),
        "frame_counts": list(item.frame_counts),
        "orientations": list(item.orientations),
        "geometry_status": item.geometry_status,
        "anatomical_plane": item.anatomical_plane,
        "fluid_sensitive": item.fluid_sensitive,
        "fat_suppression": item.fat_suppression,
        "slice_count_bucket": item.slice_count_bucket,
        "diagnostic_status": item.diagnostic_status,
        "unknown_transfer_syntax": item.unknown_transfer_syntax,
        "unknown_dimension": item.unknown_dimension,
        "unknown_frame_count": item.unknown_frame_count,
        "unknown_orientation": item.unknown_orientation,
    }


def _storage_to_dict(estimate: StorageEstimate) -> dict[str, object]:
    return {
        "current_project_bytes": estimate.current_project_bytes,
        "source_archive_bytes": estimate.source_archive_bytes,
        "extracted_bytes": estimate.extracted_bytes,
        "pixel_cache_bytes": estimate.pixel_cache_bytes,
        "temporary_bytes": estimate.temporary_bytes,
        "projected_peak_bytes": estimate.projected_peak_bytes,
        "soft_budget_bytes": estimate.soft_budget_bytes,
        "over_budget_bytes": estimate.over_budget_bytes,
        "within_soft_budget": estimate.within_soft_budget,
    }


def write_inventory_reports(
    report: InventoryReport,
    *,
    private_path: str | Path,
    public_path: str | Path,
) -> tuple[Path, Path]:
    """Atomically publish private diagnostics and an identifier-free summary."""

    private_destination = Path(private_path)
    public_destination = Path(public_path)
    if private_destination.resolve() == public_destination.resolve():
        raise InventoryError("private and public inventory reports must use different paths")
    atomic_json_write(private_destination, report.to_private_dict())
    atomic_json_write(public_destination, report.to_public_dict())
    return private_destination, public_destination


def write_selection_reports(
    selection: ExtractionSelection,
    *,
    private_path: str | Path,
    public_path: str | Path,
) -> tuple[Path, Path]:
    """Atomically publish private selection identifiers and a public summary."""

    private_destination = Path(private_path)
    public_destination = Path(public_path)
    if private_destination.resolve() == public_destination.resolve():
        raise InventoryError("private and public selection reports must use different paths")
    atomic_json_write(private_destination, selection.to_private_dict())
    atomic_json_write(public_destination, selection.to_public_dict())
    return private_destination, public_destination


def _check_execution_context(*, require_slurm: bool, allow_local_smoke: bool) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if require_slurm and (not job_id or partition != "cluster1-cpu"):
        raise RuntimeError("ticket-03 inventory must run in a cluster1-cpu Slurm allocation")
    if not job_id and not allow_local_smoke:
        raise RuntimeError(
            "refusing to scan competition data outside Slurm; "
            "use --allow-local-smoke only with a tiny fixture archive"
        )


def _inventory_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--report-only-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--expected-studies", type=int, default=4407)
    parser.add_argument("--expected-gold", type=int, default=58)
    parser.add_argument("--supported-transfer-syntax", action="append", default=None)
    parser.add_argument("--include-archive-sha256", action="store_true")
    parser.add_argument("--extract-to", type=Path, default=None)
    parser.add_argument("--current-project-bytes", type=int, default=0)
    parser.add_argument("--approve-over-budget", action="store_true")
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--allow-local-smoke", action="store_true")
    return parser


def run_inventory(
    archive_path: Path,
    output_dir: Path,
    *,
    source_version: str,
    report_only_count: int = 100,
    seed: int = 20260911,
    expected_study_count: int | None = 4407,
    expected_gold_count: int | None = 58,
    supported_transfer_syntaxes: Iterable[str] | None = None,
    include_archive_sha256: bool = False,
    extraction_path: Path | None = None,
    current_project_bytes: int = 0,
    budget_approved: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Run the header audit and optionally admit one selected private batch."""

    started = time.perf_counter()
    report = inventory_archive(
        archive_path,
        source_version=source_version,
        expected_study_count=expected_study_count,
        expected_gold_count=expected_gold_count,
        include_archive_sha256=include_archive_sha256,
        progress_callback=progress_callback,
    )
    selection = select_representative_batch(
        report,
        report_only_count=report_only_count,
        seed=seed,
        supported_transfer_syntaxes=supported_transfer_syntaxes,
        current_project_bytes=current_project_bytes,
    )
    private_dir = output_dir / "private"
    public_dir = output_dir / "public"
    write_inventory_reports(
        report,
        private_path=private_dir / "inventory.json",
        public_path=public_dir / "inventory.json",
    )
    write_selection_reports(
        selection,
        private_path=private_dir / "selection.json",
        public_path=public_dir / "selection.json",
    )
    extraction_summary: dict[str, object] | None = None
    if extraction_path is not None:
        try:
            if selection.blocked_series_uids:
                raise SelectionError(
                    "selected studies include blocked series; repair or quarantine them before extraction"
                )
            if selection.unsupported_transfer_syntaxes:
                raise SelectionError(
                    "selected classes include transfer syntaxes without explicit support: "
                    + ", ".join(selection.unsupported_transfer_syntaxes)
                )
            extraction = extract_batch(
                archive_path,
                selection.entry_names,
                extraction_path,
                storage_estimate=selection.storage_estimate,
                expected_archive_identity=report.archive_identity,
                budget_approved=budget_approved,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            failed = failed_extraction_report(
                archive_path,
                extraction_path,
                len(selection.entry_names),
                reason="extraction_precondition",
                detail=str(exc),
            )
            write_extraction_report(failed, private_dir / "extraction.json")
            atomic_json_write(
                public_dir / "extraction.json",
                {
                    "expected_file_count": failed.expected_file_count,
                    "extracted_file_count": 0,
                    "failure_count": failed.failure_count,
                    "complete": False,
                    "verified": False,
                },
            )
            raise
        write_extraction_report(extraction, private_dir / "extraction.json")
        extraction_summary = {
            "expected_file_count": extraction.expected_file_count,
            "extracted_file_count": len(extraction.extracted_files),
            "failure_count": extraction.failure_count,
            "complete": extraction.complete,
            "verified": extraction.verified,
        }
        atomic_json_write(public_dir / "extraction.json", extraction_summary)
    summary: dict[str, object] = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "ticket": "03",
        "status": "verified_header_inventory" if report.full_corpus_reconciled else "partial_header_inventory",
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "inventory": report.to_public_dict(),
        "selection": selection.to_public_dict(),
    }
    if extraction_summary is not None:
        summary["extraction"] = extraction_summary
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _inventory_parser().parse_args(argv)
    try:
        _check_execution_context(
            require_slurm=args.require_slurm,
            allow_local_smoke=args.allow_local_smoke,
        )
        summary = run_inventory(
            args.archive,
            args.output_dir,
            source_version=args.source_version,
            report_only_count=args.report_only_count,
            seed=args.seed,
            expected_study_count=args.expected_studies,
            expected_gold_count=args.expected_gold,
            supported_transfer_syntaxes=args.supported_transfer_syntax,
            include_archive_sha256=args.include_archive_sha256,
            extraction_path=args.extract_to,
            current_project_bytes=args.current_project_bytes,
            budget_approved=args.approve_over_budget,
            progress_callback=lambda current, total: print(
                f"ticket-03 header inventory: {current}/{total} archive members",
                file=sys.stderr,
                flush=True,
            ),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ticket-03 inventory failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    extraction = summary.get("extraction")
    if isinstance(extraction, dict) and extraction.get("verified") is not True:
        return 1
    return 0


def _normal_from_orientation(
    orientation: tuple[float, float, float, float, float, float],
    tolerance: float,
) -> tuple[float, float, float] | None:
    row = orientation[:3]
    column = orientation[3:]
    row_norm = math.sqrt(sum(value * value for value in row))
    column_norm = math.sqrt(sum(value * value for value in column))
    if not math.isclose(row_norm, 1.0, abs_tol=tolerance) or not math.isclose(column_norm, 1.0, abs_tol=tolerance):
        return None
    if not math.isclose(sum(left * right for left, right in zip(row, column)), 0.0, abs_tol=tolerance):
        return None
    normal = (
        row[1] * column[2] - row[2] * column[1],
        row[2] * column[0] - row[0] * column[2],
        row[0] * column[1] - row[1] * column[0],
    )
    norm = math.sqrt(sum(value * value for value in normal))
    if norm <= tolerance:
        return None
    return (normal[0] / norm, normal[1] / norm, normal[2] / norm)


def _orientation_key(
    orientation: tuple[float, float, float, float, float, float],
    tolerance: float,
) -> str | None:
    if _normal_from_orientation(orientation, tolerance) is None:
        return None
    return ",".join(f"{value:.6f}" for value in orientation)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(first * second for first, second in zip(left, right))


def _project(position: Sequence[float], normal: Sequence[float]) -> float:
    return _dot(position, normal)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_identity(
    source: Path,
    archive: zipfile.ZipFile,
    *,
    source_version: str,
    include_sha256: bool,
) -> ArchiveIdentity:
    try:
        stat = source.stat()
    except OSError as exc:
        raise InventoryError(f"could not stat source archive {source}") from exc
    try:
        infos = archive.infolist()
        zip_entry_count = len(infos)
    except (OSError, zipfile.BadZipFile) as exc:
        raise InventoryError(f"could not read archive directory for {source}") from exc
    directory_digest = hashlib.sha256()
    for info in infos:
        directory_digest.update(
            "\0".join(
                (
                    info.filename,
                    str(info.compress_size),
                    str(info.file_size),
                    str(info.CRC),
                    str(info.header_offset),
                )
            ).encode("utf-8")
        )
        directory_digest.update(b"\n")
    central_directory_sha256 = directory_digest.hexdigest()
    archive_sha256 = _sha256_file(source) if include_sha256 else None
    fingerprint_payload = "|".join(
        (
            source_version,
            source.name,
            str(stat.st_size),
            str(stat.st_mtime_ns),
            str(zip_entry_count),
            central_directory_sha256,
            archive_sha256 or "",
        )
    ).encode("utf-8")
    return ArchiveIdentity(
        source_version=source_version,
        archive_name=source.name,
        archive_size_bytes=stat.st_size,
        archive_mtime_ns=stat.st_mtime_ns,
        zip_entry_count=zip_entry_count,
        central_directory_sha256=central_directory_sha256,
        archive_sha256=archive_sha256,
        fingerprint=hashlib.sha256(fingerprint_payload).hexdigest(),
    )


def identify_archive(
    archive_path: str | Path,
    *,
    source_version: str,
    include_sha256: bool = False,
) -> ArchiveIdentity:
    """Record archive identity without reading DICOM pixels or extracting files."""

    source_version = source_version.strip()
    if not source_version:
        raise InventoryError("source_version must be non-empty")
    source = Path(archive_path)
    try:
        archive = zipfile.ZipFile(source, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise InventoryError(f"could not open source archive {source}") from exc
    with archive:
        return _archive_identity(
            source,
            archive,
            source_version=source_version,
            include_sha256=include_sha256,
        )


def _optional_int(
    dataset: Any,
    name: str,
    *,
    default: int | None = None,
    minimum: int = 1,
) -> int | None:
    value = getattr(dataset, name, None)
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= minimum else None


def _optional_float_tuple(dataset: Any, name: str, length: int) -> tuple[tuple[float, ...] | None, str]:
    value = getattr(dataset, name, None)
    if value is None:
        return None, "missing"
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None, "invalid"
    if len(values) != length or not all(math.isfinite(item) for item in values):
        return None, "invalid"
    return values, "present"


def _read_header(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    expected_study_uid: str,
    expected_series_uid: str,
) -> HeaderInspection:
    with archive.open(info, "r") as stream:
        dataset = pydicom.dcmread(stream, stop_before_pixels=True, force=False)
    file_meta = getattr(dataset, "file_meta", None)
    transfer_syntax_value = getattr(file_meta, "TransferSyntaxUID", None)
    transfer_syntax_uid = str(transfer_syntax_value).strip() if transfer_syntax_value is not None else None
    transfer_syntax_uid = transfer_syntax_uid or None
    study_uid_value = getattr(dataset, "StudyInstanceUID", None)
    series_uid_value = getattr(dataset, "SeriesInstanceUID", None)
    sop_uid_value = getattr(dataset, "SOPInstanceUID", None)
    study_uid = str(study_uid_value).strip() if study_uid_value is not None else None
    series_uid = str(series_uid_value).strip() if series_uid_value is not None else None
    sop_uid = str(sop_uid_value).strip() if sop_uid_value is not None else None
    stem = Path(info.filename).stem
    identity_conflicts: list[str] = []
    if study_uid != expected_study_uid:
        identity_conflicts.append("study_instance_uid")
    if series_uid != expected_series_uid:
        identity_conflicts.append("series_instance_uid")
    if sop_uid is None:
        identity_conflicts.append("sop_instance_uid_missing")
    elif sop_uid != stem:
        identity_conflicts.append("sop_instance_uid_filename")
    position_value, position_status = _optional_float_tuple(dataset, "ImagePositionPatient", 3)
    orientation_value, orientation_status = _optional_float_tuple(dataset, "ImageOrientationPatient", 6)
    position = cast(tuple[float, float, float] | None, position_value)
    orientation = cast(tuple[float, float, float, float, float, float] | None, orientation_value)
    frame_count = _optional_int(dataset, "NumberOfFrames", default=1)
    rows = _optional_int(dataset, "Rows")
    columns = _optional_int(dataset, "Columns")
    photometric_value = getattr(dataset, "PhotometricInterpretation", None)
    photometric = str(photometric_value).strip().upper() if photometric_value is not None else None
    image_type_value = getattr(dataset, "ImageType", None)
    if image_type_value is None:
        image_type_values: tuple[str, ...] = ()
    elif isinstance(image_type_value, str):
        normalized_image_type = image_type_value.strip().upper()
        image_type_values = (normalized_image_type,) if normalized_image_type else ()
    else:
        image_type_values = tuple(
            normalized
            for value in image_type_value
            if (normalized := str(value).strip().upper())
        )
    description_value = getattr(dataset, "SeriesDescription", None)
    description = str(description_value).strip() if description_value is not None else ""
    return HeaderInspection(
        archive_name=info.filename,
        expected_study_instance_uid=expected_study_uid,
        expected_series_instance_uid=expected_series_uid,
        study_instance_uid=study_uid,
        series_instance_uid=series_uid,
        sop_instance_uid=sop_uid,
        transfer_syntax_uid=transfer_syntax_uid,
        frame_count=frame_count,
        rows=rows,
        columns=columns,
        image_position=position,
        image_orientation=orientation,
        position_status=position_status,
        orientation_status=orientation_status,
        image_type_values=image_type_values,
        photometric_interpretation=photometric,
        bits_allocated=_optional_int(dataset, "BitsAllocated"),
        bits_stored=_optional_int(dataset, "BitsStored"),
        high_bit=_optional_int(dataset, "HighBit"),
        pixel_representation=_optional_int(dataset, "PixelRepresentation", default=0, minimum=0),
        samples_per_pixel=_optional_int(dataset, "SamplesPerPixel", default=1),
        dicom_series_description=description,
        identity_conflicts=tuple(identity_conflicts),
    )


def inventory_archive(
    archive_path: str | Path,
    *,
    source_version: str,
    split: str = "train",
    expected_study_count: int | None = None,
    expected_gold_count: int | None = None,
    include_archive_sha256: bool = False,
    geometry_policy: GeometryPolicy = GeometryPolicy(),
    progress_callback: Callable[[int, int], None] | None = None,
) -> InventoryReport:
    """Inventory every split member's header while retaining all failures.

    The function never accesses ``pixel_array``.  It accounts for malformed
    paths, unreadable headers and source/header identity conflicts explicitly.
    """

    source_version = source_version.strip()
    if not source_version:
        raise InventoryError("source_version must be non-empty")
    source = Path(archive_path)
    source_manifest = load_source_manifest(source, split=split)
    try:
        archive = zipfile.ZipFile(source, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise InventoryError(f"could not open source archive {source}") from exc
    with archive:
        identity = _archive_identity(
            source,
            archive,
            source_version=source_version,
            include_sha256=include_archive_sha256,
        )
        accumulators = {
            item.series_instance_uid: _SeriesAccumulator(item, geometry_policy) for item in source_manifest.series
        }
        archive_data_file_count = 0
        archive_dicom_file_count = 0
        readable_header_count = 0
        unreadable: list[HeaderFailure] = []
        malformed_entries: list[str] = []
        duplicate_entries: list[str] = []
        extra_series: set[tuple[str, str]] = set()
        actual_study_uids: set[str] = set()
        seen_names: set[str] = set()
        prefix = f"{split}_series/"
        infos = archive.infolist()
        total_members = len(infos)
        for member_index, info in enumerate(infos, start=1):
            if progress_callback is not None and member_index % 10_000 == 0:
                progress_callback(member_index, total_members)
            name = info.filename
            if info.is_dir() or not name.startswith(prefix):
                continue
            archive_data_file_count += 1
            if name in seen_names:
                duplicate_entries.append(name)
                continue
            seen_names.add(name)
            try:
                safe_member_path(name)
            except ArchivePathError:
                malformed_entries.append(name)
                continue
            parts = name.split("/")
            if len(parts) != 4 or parts[0] != f"{split}_series" or not parts[3].lower().endswith(".dcm"):
                malformed_entries.append(name)
                continue
            archive_dicom_file_count += 1
            study_uid, series_uid = parts[1], parts[2]
            actual_study_uids.add(study_uid)
            accumulator = accumulators.get(series_uid)
            if accumulator is None or accumulator.study_instance_uid != study_uid:
                extra_series.add((study_uid, series_uid))
                continue
            accumulator.add_entry(name, info.compress_size, info.file_size)
            try:
                header = _read_header(
                    archive,
                    info,
                    expected_study_uid=study_uid,
                    expected_series_uid=series_uid,
                )
            except (
                EOFError,
                InvalidDicomError,
                KeyError,
                OSError,
                TypeError,
                ValueError,
                IndexError,
                AttributeError,
                NotImplementedError,
                OverflowError,
                UnicodeError,
                zipfile.BadZipFile,
            ) as exc:
                accumulator.add_failure()
                unreadable.append(
                    HeaderFailure(
                        archive_name=name,
                        expected_study_instance_uid=study_uid,
                        expected_series_instance_uid=series_uid,
                        reason=(
                            "invalid_dicom"
                            if isinstance(exc, (InvalidDicomError, EOFError, UnicodeError))
                            else "archive_integrity"
                            if isinstance(exc, zipfile.BadZipFile)
                            else "header_error"
                        ),
                        detail=str(exc),
                    )
                )
                continue
            readable_header_count += 1
            accumulator.add_header(header)
        if progress_callback is not None:
            progress_callback(total_members, total_members)
        series = tuple(accumulators[uid].finish() for uid in sorted(accumulators))
    expected_series_keys = {
        (item.study_instance_uid, item.series_instance_uid) for item in source_manifest.series
    }
    present_series_keys = {
        (item.study_instance_uid, item.series_instance_uid)
        for item in series
        if item.entry_count
    }
    missing_series = tuple(sorted(expected_series_keys - present_series_keys))
    return InventoryReport(
        schema_version=INVENTORY_SCHEMA_VERSION,
        source_version=source_version,
        split=split,
        archive_identity=identity,
        geometry_policy=geometry_policy,
        source_manifest=source_manifest,
        archive_data_file_count=archive_data_file_count,
        archive_dicom_file_count=archive_dicom_file_count,
        readable_header_count=readable_header_count,
        unreadable_headers=tuple(unreadable),
        malformed_archive_entries=tuple(sorted(malformed_entries)),
        duplicate_archive_entries=tuple(sorted(duplicate_entries)),
        extra_series=tuple(sorted(extra_series)),
        missing_series=missing_series,
        identity_conflict_count=sum(item.identity_conflict_count for item in series),
        actual_study_uids=tuple(sorted(actual_study_uids)),
        series=series,
        expected_study_count=expected_study_count,
        expected_gold_count=expected_gold_count,
    )


def _coverage_tokens(stratum: InventoryStratum) -> set[object]:
    tokens: set[object] = {("class", stratum)}
    tokens.update(("transfer_syntax", value) for value in stratum.transfer_syntaxes)
    tokens.update(("dimension", value) for value in stratum.dimensions)
    tokens.update(("frame_count", value) for value in stratum.frame_counts)
    tokens.update(("orientation", value) for value in stratum.orientations)
    tokens.add(("geometry", stratum.geometry_status))
    tokens.add(("anatomical_plane", stratum.anatomical_plane))
    tokens.add(("fluid_sensitive", stratum.fluid_sensitive))
    tokens.add(("fat_suppression", stratum.fat_suppression))
    tokens.add(("slice_count", stratum.slice_count_bucket))
    tokens.add(("diagnostic_status", stratum.diagnostic_status))
    if stratum.unknown_transfer_syntax:
        tokens.add(("unknown_transfer_syntax", True))
    if stratum.unknown_dimension:
        tokens.add(("unknown_dimension", True))
    if stratum.unknown_frame_count:
        tokens.add(("unknown_frame_count", True))
    if stratum.unknown_orientation:
        tokens.add(("unknown_orientation", True))
    return tokens


def _stratum_sort_key(stratum: InventoryStratum) -> tuple[object, ...]:
    return (
        stratum.transfer_syntaxes,
        stratum.dimensions,
        stratum.frame_counts,
        stratum.orientations,
        stratum.geometry_status,
        stratum.anatomical_plane or "",
        str(stratum.fluid_sensitive),
        str(stratum.fat_suppression),
        stratum.slice_count_bucket,
        stratum.diagnostic_status,
        str(stratum.unknown_transfer_syntax),
        str(stratum.unknown_dimension),
        str(stratum.unknown_frame_count),
        str(stratum.unknown_orientation),
    )


def estimate_storage(
    inventory: InventoryReport,
    study_uids: Iterable[str],
    *,
    image_size: tuple[int, int] = (128, 128),
    bytes_per_pixel: int = 4,
    current_project_bytes: int = 0,
    soft_budget_bytes: int = 2_000_000_000_000,
) -> StorageEstimate:
    """Estimate retained archive, extracted source and float32 cache peak use."""

    if len(image_size) != 2 or any(size <= 0 for size in image_size):
        raise ValueError("image_size must contain two positive dimensions")
    if bytes_per_pixel <= 0 or current_project_bytes < 0 or soft_budget_bytes <= 0:
        raise ValueError("storage sizes must be positive, with non-negative current usage")
    selected_studies = set(study_uids)
    unknown_studies = selected_studies - set(inventory.source_manifest.studies_by_uid)
    if unknown_studies:
        raise ValueError(f"storage estimate references unknown studies: {sorted(unknown_studies)[:3]}")
    selected_series = [item for item in inventory.series if item.study_instance_uid in selected_studies]
    extracted_bytes = sum(item.uncompressed_bytes for item in selected_series if item.is_complete)
    cached_frames = sum(item.effective_frame_count for item in selected_series if item.is_complete)
    pixels_per_frame = image_size[0] * image_size[1] * bytes_per_pixel
    pixel_cache_bytes = cached_frames * pixels_per_frame
    temporary_bytes = extracted_bytes + pixel_cache_bytes
    projected_peak_bytes = (
        current_project_bytes
        + inventory.archive_identity.archive_size_bytes
        + extracted_bytes
        + pixel_cache_bytes
        + temporary_bytes
    )
    return StorageEstimate(
        current_project_bytes=current_project_bytes,
        source_archive_bytes=inventory.archive_identity.archive_size_bytes,
        extracted_bytes=extracted_bytes,
        pixel_cache_bytes=pixel_cache_bytes,
        temporary_bytes=temporary_bytes,
        projected_peak_bytes=projected_peak_bytes,
        soft_budget_bytes=soft_budget_bytes,
    )


def enforce_storage_budget(estimate: StorageEstimate, *, approved: bool = False) -> None:
    """Refuse an over-budget batch unless an explicit approval is supplied."""

    if not estimate.within_soft_budget and not approved:
        raise StorageBudgetExceeded(
            "projected extraction peak exceeds the 2 TB soft budget by "
            f"{estimate.over_budget_bytes} bytes"
        )


def select_representative_batch(
    inventory: InventoryReport,
    *,
    report_only_count: int = 100,
    seed: int = 20260911,
    supported_transfer_syntaxes: Iterable[str] | None = None,
    current_project_bytes: int = 0,
    soft_budget_bytes: int = 2_000_000_000_000,
) -> ExtractionSelection:
    """Select all gold studies and a deterministic, coverage-seeking sample.

    Only complete series enter ``entry_names``.  A selected study or series
    with an unreadable header remains in ``blocked_series_uids`` and cannot be
    admitted by the extraction adapter.
    """

    if not 50 <= report_only_count <= 200:
        raise SelectionError("report_only_count must be between 50 and 200")
    if supported_transfer_syntaxes is None:
        supported = None
    else:
        normalized_supported = tuple(value.strip() for value in supported_transfer_syntaxes)
        if any(not value for value in normalized_supported):
            raise SelectionError("supported transfer syntaxes must be non-empty strings")
        supported = frozenset(normalized_supported)
    source = inventory.source_manifest
    if source.partially_labeled_study_uids:
        raise SelectionError(
            "source metadata contains partially reference-labeled studies; resolve them before selection"
        )
    source_by_study = source.series_by_study
    series_by_uid = inventory.series_by_uid
    gold_studies = source.gold_study_uids
    report_only_studies = source.report_only_study_uids
    complete_series = [item for item in inventory.series if item.is_complete]
    complete_by_study: dict[str, list[SeriesInventory]] = {}
    for item in complete_series:
        complete_by_study.setdefault(item.study_instance_uid, []).append(item)
    available_report_only = [uid for uid in report_only_studies if complete_by_study.get(uid)]
    if len(available_report_only) < report_only_count:
        raise SelectionError(
            f"only {len(available_report_only)} report-only studies have complete series; "
            f"cannot select {report_only_count}"
        )

    def study_tokens(study_uid: str) -> set[object]:
        tokens: set[object] = set()
        for item in complete_by_study.get(study_uid, ()):
            tokens.update(_coverage_tokens(item.stratum))
        return tokens

    all_strata = {item.stratum for item in inventory.series if item.entry_count}
    covered_tokens: set[object] = set()
    for study_uid in gold_studies:
        covered_tokens.update(study_tokens(study_uid))
    remaining = set(available_report_only)
    chosen: list[str] = []

    def stable_rank(study_uid: str) -> str:
        return hashlib.sha256(f"{seed}:{study_uid}".encode("utf-8")).hexdigest()

    while len(chosen) < report_only_count:
        ranked = sorted(
            remaining,
            key=lambda study_uid: (
                -len(study_tokens(study_uid) - covered_tokens),
                stable_rank(study_uid),
                study_uid,
            ),
        )
        selected_uid = ranked[0]
        chosen.append(selected_uid)
        remaining.remove(selected_uid)
        covered_tokens.update(study_tokens(selected_uid))

    selected_studies = gold_studies + tuple(chosen)
    selected_series: list[SeriesInventory] = []
    blocked_series: list[str] = []
    for study_uid in selected_studies:
        for source_series in source_by_study.get(study_uid, ()):
            item = series_by_uid[source_series.series_instance_uid]
            if item.is_complete:
                selected_series.append(item)
            else:
                blocked_series.append(item.series_instance_uid)
    selected_series.sort(key=lambda item: (item.study_instance_uid, item.series_instance_uid))
    selected_strata = {item.stratum for item in selected_series}
    selected_syntaxes = {
        syntax
        for item in selected_series
        for syntax in item.transfer_syntax_counts
        if syntax != "unknown"
    }
    unsupported = tuple(sorted(selected_syntaxes if supported is None else selected_syntaxes - supported))
    selected_series_uids = tuple(item.series_instance_uid for item in selected_series)
    selected_entry_names = tuple(sorted(name for item in selected_series for name in item.entry_names))
    storage_estimate = estimate_storage(
        inventory,
        selected_studies,
        current_project_bytes=current_project_bytes,
        soft_budget_bytes=soft_budget_bytes,
    )
    return ExtractionSelection(
        seed=seed,
        source_version=inventory.source_version,
        archive_fingerprint=inventory.archive_identity.fingerprint,
        requested_report_only_count=report_only_count,
        gold_study_uids=gold_studies,
        report_only_study_uids=tuple(chosen),
        selected_series_uids=selected_series_uids,
        blocked_series_uids=tuple(sorted(blocked_series)),
        entry_names=selected_entry_names,
        covered_strata=tuple(sorted(selected_strata, key=_stratum_sort_key)),
        unrepresented_strata=tuple(sorted(all_strata - selected_strata, key=_stratum_sort_key)),
        unsupported_transfer_syntaxes=unsupported,
        storage_estimate=storage_estimate,
    )


def _read_csv_rows(
    archive: zipfile.ZipFile,
    name: str,
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        stream = archive.open(name, "r")
    except KeyError as exc:
        raise SourceManifestError(f"archive is missing required metadata file {name!r}") from exc
    with stream, io.TextIOWrapper(stream, encoding="utf-8", newline="") as text:
        reader = csv.DictReader(text, strict=True)
        if reader.fieldnames is None:
            raise SourceManifestError(f"metadata file {name!r} has no header")
        fields = tuple(field for field in reader.fieldnames if field is not None)
        invalid_header = (
            len(fields) != len(reader.fieldnames)
            or len(set(fields)) != len(fields)
            or any(not field.strip() for field in fields)
        )
        if invalid_header:
            raise SourceManifestError(f"metadata file {name!r} has duplicate or empty column names")
        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise SourceManifestError(f"metadata file {name!r} row {row_number} has the wrong number of columns")
            rows.append({key: value for key, value in row.items() if key is not None})
        return fields, rows


def _required_text(row: Mapping[str, str], field: str, row_number: int, filename: str) -> str:
    value = row.get(field)
    if value is None or not value.strip():
        raise SourceManifestError(f"{filename} row {row_number} has no {field}")
    return value.strip()


def _optional_binary(row: Mapping[str, str], field: str, row_number: int, filename: str) -> bool | None:
    value = row.get(field)
    if value is None or not value.strip():
        return None
    normalized = value.strip()
    if normalized not in {"0", "1"}:
        raise SourceManifestError(f"{filename} row {row_number} has invalid {field}={value!r}")
    return normalized == "1"


def _optional_label(row: Mapping[str, str], field: str, row_number: int, filename: str) -> int | None:
    value = row.get(field)
    if value is None or not value.strip():
        return None
    normalized = value.strip()
    if normalized not in {"0", "1"}:
        raise SourceManifestError(f"{filename} row {row_number} has invalid {field}={value!r}")
    return int(normalized)


def load_source_manifest(archive_path: str | Path, *, split: str = "train") -> SourceManifest:
    """Load and reconcile the split CSVs without reading any DICOM pixels."""

    if not split or "/" in split or "\\" in split:
        raise SourceManifestError("split must be a simple non-empty name")
    source = Path(archive_path)
    try:
        archive = zipfile.ZipFile(source, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise SourceManifestError(f"could not open source archive {source}") from exc
    with archive:
        study_filename = f"{split}.csv"
        series_filename = f"{split}_series.csv"
        study_fields, study_rows = _read_csv_rows(archive, study_filename)
        series_fields, series_rows = _read_csv_rows(archive, series_filename)
    required_study_fields = {"StudyInstanceUID"}
    if split == "train":
        required_study_fields.update(("Report", *TARGET_COLUMNS))
    missing_study_fields = required_study_fields - set(study_fields)
    if missing_study_fields:
        raise SourceManifestError(
            f"{study_filename} is missing required columns: {sorted(missing_study_fields)}"
        )
    required_series_fields = {
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Fluid_Sensitive",
        "Fat_Suppression",
        "Anatomical_Plane",
    }
    missing_series_fields = required_series_fields - set(series_fields)
    if missing_series_fields:
        raise SourceManifestError(
            f"{series_filename} is missing required columns: {sorted(missing_series_fields)}"
        )

    studies: list[SourceStudy] = []
    for row_number, row in enumerate(study_rows, start=2):
        study_uid = _required_text(row, "StudyInstanceUID", row_number, study_filename)
        labels = tuple(
            _optional_label(row, target, row_number, study_filename) for target in TARGET_COLUMNS
        )
        studies.append(
            SourceStudy(
                study_instance_uid=study_uid,
                report_present=bool(row.get("Report", "").strip()),
                reference_labels=labels,
            )
        )

    series: list[SourceSeries] = []
    for row_number, row in enumerate(series_rows, start=2):
        series.append(
            SourceSeries(
                study_instance_uid=_required_text(row, "StudyInstanceUID", row_number, series_filename),
                series_instance_uid=_required_text(row, "SeriesInstanceUID", row_number, series_filename),
                fluid_sensitive=_optional_binary(row, "Fluid_Sensitive", row_number, series_filename),
                fat_suppression=_optional_binary(row, "Fat_Suppression", row_number, series_filename),
                anatomical_plane=(row.get("Anatomical_Plane") or "").strip() or None,
            )
        )
    return SourceManifest(split=split, studies=tuple(studies), series=tuple(series))


__all__ = [
    "INVENTORY_SCHEMA_VERSION",
    "InventoryError",
    "SourceManifestError",
    "SelectionError",
    "StorageBudgetExceeded",
    "SourceStudy",
    "SourceSeries",
    "SourceManifest",
    "ArchiveIdentity",
    "HeaderInspection",
    "HeaderFailure",
    "SeriesInventory",
    "InventoryStratum",
    "StorageEstimate",
    "ExtractionSelection",
    "InventoryReport",
    "load_source_manifest",
    "identify_archive",
    "inventory_archive",
    "estimate_storage",
    "enforce_storage_budget",
    "select_representative_batch",
    "write_inventory_reports",
    "write_selection_reports",
    "run_inventory",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
