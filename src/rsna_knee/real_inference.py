"""Private real-DICOM integration and inference orchestration.

This module is the boundary between a private ticket-03 selection and the
shared decoder, geometry, preprocessing, hierarchical model, and submission
interfaces.  Production verification consumes the original private ZIP; an
unverified directory is accepted only when an explicit local fixture-smoke
flag is used.  No report or training module is imported here.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import time
import zipfile
import zlib
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator, Mapping, Sequence, cast

import numpy as np
import torch

from .archive import atomic_json_write, safe_member_path
from .dicom import (
    DamagedDicomError,
    DecoderInfrastructureError,
    DecoderPolicy,
    DicomDecodeError,
    DicomValidationError,
    UnsupportedTransferSyntaxError,
    decode_stream,
)
from .domain import DicomSlice, SeriesRecord, StudyRecord
from .folds import (
    FOLD_SCHEMA_VERSION,
    load_fold_manifest,
    load_inventory_source,
    source_manifest_checksum,
)
from .geometry import GeometryPolicy
from .inference import InferenceEngine, StudyPrediction
from .model import ModelConfig, StudyModel, build_study_model
from .preprocessing import (
    PREPROCESS_SCHEMA_VERSION,
    PackingConfig,
    PreprocessConfig,
    SeriesFailure,
)
from .recovery import (
    ReferenceFallbackPrior,
    compute_laplace_smoothed_reference_priors,
    default_reference_fallback_prior,
    validate_reference_label_vector,
)
from .submission import write_submission
from .series_policy import has_non_diagnostic_token
from .series_cache import SeriesCache
from .targets import NUM_TARGETS


INTEGRATION_MANIFEST_SCHEMA_VERSION = "rsna-knee-real-integration-v1"
INTEGRATION_EVIDENCE_SCHEMA_VERSION = "rsna-knee-real-integration-evidence-v1"
INVENTORY_SCHEMA_VERSION = "rsna-knee-inventory-v1"
SELECTION_SCHEMA_VERSION = "rsna-knee-extraction-selection-v1"
MAX_STORAGE_BUDGET_BYTES = 2_000_000_000_000
LOCAL_SMOKE_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_REAL_GEOMETRY_POLICY = GeometryPolicy(minimum_usable_slices=3)
DEFAULT_REAL_DECODER_POLICY = DecoderPolicy(require_modality=True)
DEFAULT_REAL_PREPROCESS_CONFIG = PreprocessConfig(
    geometry_policy=DEFAULT_REAL_GEOMETRY_POLICY,
)


class RealIntegrationError(ValueError):
    """Raised when a private manifest or real-input admission is unsafe."""


class RealInferenceError(RuntimeError):
    """Raised for a global integration or model failure."""


@dataclass(frozen=True)
class KaggleCompetitionSource:
    """Attested official Kaggle competition input mounted read-only for a private run."""

    root: Path
    competition_slug: str
    kernel_id: str
    request_checksum: str

    def __post_init__(self) -> None:
        if self.competition_slug != "rsna-knee-abnormality-detection":
            raise RealIntegrationError("Kaggle source is not the approved RSNA competition")
        if not self.kernel_id.strip():
            raise RealIntegrationError("Kaggle source attestation requires a kernel identity")
        if re.fullmatch(r"[0-9a-f]{64}", self.request_checksum) is None:
            raise RealIntegrationError("Kaggle source attestation has an invalid request checksum")


class UnsupportedSourceEntryError(DicomDecodeError):
    """Raised when the archive cannot provide a selected member to the decoder."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _nonempty(value: object, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise RealIntegrationError(f"{field} must be non-empty")
    return text


def _tuple_of_strings(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise RealIntegrationError(f"{field} must be an array")
    values = tuple(str(item).strip() for item in value)
    if any(not item for item in values):
        raise RealIntegrationError(f"{field} must not contain empty values")
    if len(set(values)) != len(values):
        raise RealIntegrationError(f"{field} must not contain duplicates")
    return values


@dataclass(frozen=True)
class IntegrationSeries:
    """One private series-to-source-entry mapping from the integration manifest."""

    study_instance_uid: str
    series_instance_uid: str
    entries: tuple[str, ...]
    anatomical_plane: str | None = None
    fluid_sensitive: bool | None = None
    fat_suppression: bool | None = None
    diagnostic_status: str = "unknown"
    expected_transfer_syntaxes: tuple[str, ...] = ()
    expected_dimensions: tuple[str, ...] = ()
    expected_frame_counts: tuple[int, ...] = ()
    expected_entry_count: int | None = None

    def __post_init__(self) -> None:
        if not self.study_instance_uid.strip() or not self.series_instance_uid.strip():
            raise RealIntegrationError("integration series requires study and series identifiers")
        if not self.entries:
            raise RealIntegrationError(f"series {self.series_instance_uid} has no source entries")
        if len(set(self.entries)) != len(self.entries):
            raise RealIntegrationError(f"series {self.series_instance_uid} has duplicate source entries")
        for entry in self.entries:
            safe_member_path(entry)
        if self.diagnostic_status not in {"unknown", "diagnostic", "excluded_non_diagnostic"}:
            raise RealIntegrationError(f"unknown diagnostic status {self.diagnostic_status!r}")
        if self.expected_entry_count is not None and self.expected_entry_count != len(self.entries):
            raise RealIntegrationError(
                f"series {self.series_instance_uid} expected entry count does not match its entries"
            )
        if any(count <= 0 for count in self.expected_frame_counts):
            raise RealIntegrationError("expected frame counts must be positive")


@dataclass(frozen=True)
class IntegrationStudy:
    """One private study and its selected series."""

    study_instance_uid: str
    series: tuple[IntegrationSeries, ...]

    def __post_init__(self) -> None:
        if not self.study_instance_uid.strip():
            raise RealIntegrationError("integration study requires a StudyInstanceUID")
        if len({item.series_instance_uid for item in self.series}) != len(self.series):
            raise RealIntegrationError(f"study {self.study_instance_uid} has duplicate series")
        if any(item.study_instance_uid != self.study_instance_uid for item in self.series):
            raise RealIntegrationError(f"study {self.study_instance_uid} contains a foreign series")


@dataclass(frozen=True)
class IntegrationManifest:
    """Checksum-covered private input manifest with frozen processing policies."""

    source_version: str
    archive_fingerprint: str
    geometry_policy: GeometryPolicy
    decoder_policy: DecoderPolicy
    preprocess_config: PreprocessConfig
    studies: tuple[IntegrationStudy, ...]
    source_split: str = "train"
    packing_config: PackingConfig = PackingConfig()
    source_manifest_checksum: str | None = None
    schema_version: str = INTEGRATION_MANIFEST_SCHEMA_VERSION
    manifest_checksum: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != INTEGRATION_MANIFEST_SCHEMA_VERSION:
            raise RealIntegrationError(f"unexpected integration manifest schema {self.schema_version!r}")
        if not self.source_version.strip() or not self.archive_fingerprint.strip():
            raise RealIntegrationError("integration manifest requires source identity")
        if self.source_manifest_checksum is not None and not self.source_manifest_checksum.strip():
            raise RealIntegrationError("source manifest checksum must be non-empty when present")
        if self.source_split != "train":
            raise RealIntegrationError("real integration currently accepts only the train split")
        if self.preprocess_config.geometry_policy != self.geometry_policy:
            raise RealIntegrationError("manifest geometry and preprocessing geometry policies differ")
        study_uids = [item.study_instance_uid for item in self.studies]
        if not self.studies or len(set(study_uids)) != len(study_uids):
            raise RealIntegrationError("integration manifest studies must be non-empty and unique")
        all_series = [series for study in self.studies for series in study.series]
        series_uids = [series.series_instance_uid for series in all_series]
        if len(set(series_uids)) != len(series_uids):
            raise RealIntegrationError("integration manifest series identifiers must be globally unique")
        all_entries = [entry for series in all_series for entry in series.entries]
        if len(set(all_entries)) != len(all_entries):
            raise RealIntegrationError("integration manifest source entries must be globally unique")

    @property
    def study_uids(self) -> tuple[str, ...]:
        return tuple(item.study_instance_uid for item in self.studies)

    @property
    def series_count(self) -> int:
        return sum(len(item.series) for item in self.studies)


@dataclass(frozen=True)
class SeriesLoadFailure:
    """Private per-file or per-series failure retained by a real load."""

    series_instance_uid: str
    source_entry_name: str
    reason: str
    detail: str
    decoder: str = ""
    attempts: tuple[str, ...] = ()
    recovery: str = "skip_unusable_input"
    affected_source_entry_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoadedSeries:
    """Result of loading one manifest series before model preprocessing."""

    manifest_series: IntegrationSeries
    series: SeriesRecord | None
    status: str
    expected_slice_count: int
    decoded_slice_count: int
    failures: tuple[SeriesLoadFailure, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.status == "accepted" and self.decoded_slice_count == self.expected_slice_count


@dataclass(frozen=True)
class LoadedStudy:
    """Study record plus all private loading diagnostics."""

    study: StudyRecord
    series_results: tuple[LoadedSeries, ...]
    warnings: tuple[str, ...] = ()

    @property
    def failures(self) -> tuple[SeriesLoadFailure, ...]:
        return tuple(failure for item in self.series_results for failure in item.failures)


def _series_to_dict(item: IntegrationSeries) -> dict[str, object]:
    return {
        "StudyInstanceUID": item.study_instance_uid,
        "SeriesInstanceUID": item.series_instance_uid,
        "entries": list(item.entries),
        "Anatomical_Plane": item.anatomical_plane,
        "Fluid_Sensitive": item.fluid_sensitive,
        "Fat_Suppression": item.fat_suppression,
        "diagnostic_status": item.diagnostic_status,
        "expected_transfer_syntaxes": list(item.expected_transfer_syntaxes),
        "expected_dimensions": list(item.expected_dimensions),
        "expected_frame_counts": list(item.expected_frame_counts),
        "expected_entry_count": item.expected_entry_count,
    }


def _study_to_dict(item: IntegrationStudy) -> dict[str, object]:
    return {
        "StudyInstanceUID": item.study_instance_uid,
        "series": [_series_to_dict(series) for series in item.series],
    }


def _manifest_payload(manifest: IntegrationManifest) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "source_version": manifest.source_version,
        "source_split": manifest.source_split,
        "archive_fingerprint": manifest.archive_fingerprint,
        "source_manifest_checksum": manifest.source_manifest_checksum,
        "policies": {
            "geometry": manifest.geometry_policy.to_dict(),
            "decoder": manifest.decoder_policy.to_dict(),
            "preprocessing": manifest.preprocess_config.cache_config(),
            "packing": manifest.packing_config.to_dict(),
        },
        "studies": [_study_to_dict(item) for item in manifest.studies],
    }


def write_integration_manifest(manifest: IntegrationManifest, output_path: str | Path) -> Path:
    """Atomically publish a private checksum-covered integration manifest."""

    payload = _manifest_payload(manifest)
    payload["manifest_checksum"] = _json_sha256(payload)
    return atomic_json_write(output_path, payload)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RealIntegrationError(f"{field} must be an object")
    return value


def _optional_bool(value: object, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise RealIntegrationError(f"{field} must be boolean or null")
    return value


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, field)


def _optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RealIntegrationError(f"{field} must be an integer or null")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RealIntegrationError(f"{field} must be an integer or null") from exc
    if parsed < 0:
        raise RealIntegrationError(f"{field} must be non-negative")
    return parsed


def _required_nonnegative_int(value: object, field: str) -> int:
    parsed = _optional_int(value, field)
    if parsed is None:
        raise RealIntegrationError(f"{field} is required")
    return parsed


def _preprocess_from_cache_config(value: object, geometry_policy: GeometryPolicy) -> PreprocessConfig:
    payload = _mapping(value, "preprocessing policy")
    if payload.get("schema_version") != PREPROCESS_SCHEMA_VERSION:
        raise RealIntegrationError("unsupported preprocessing policy schema")
    image_size_value = payload.get("image_size")
    if not isinstance(image_size_value, (list, tuple)) or len(image_size_value) != 2:
        raise RealIntegrationError("preprocessing image_size must contain two values")
    percentiles_value = payload.get("normalization_percentiles", (0.0, 100.0))
    if not isinstance(percentiles_value, (list, tuple)) or len(percentiles_value) != 2:
        raise RealIntegrationError("preprocessing percentiles must contain two values")
    return PreprocessConfig(
        image_size=(int(str(image_size_value[0])), int(str(image_size_value[1]))),
        geometry_policy=geometry_policy,
        normalization_domain=str(payload.get("normalization_domain", "valid_slice")),
        normalization_percentiles=(float(str(percentiles_value[0])), float(str(percentiles_value[1]))),
        use_voi_lut=bool(payload.get("use_voi_lut", False)),
        interpolation=str(payload.get("interpolation", "nearest")),
        padding_fill_value=float(str(payload.get("padding_fill_value", 0.0))),
        output_dtype=str(payload.get("output_dtype", "float32")),
        representation_id=(
            str(payload["representation_id"])
            if payload.get("representation_id") is not None
            else None
        ),
    )


def _integration_series_from_dict(value: object) -> IntegrationSeries:
    payload = _mapping(value, "integration series")
    entries = _tuple_of_strings(payload.get("entries", ()), "integration series entries")
    expected_frames_value = payload.get("expected_frame_counts", ())
    if not isinstance(expected_frames_value, (list, tuple)):
        raise RealIntegrationError("expected_frame_counts must be an array")
    expected_frames = tuple(int(str(item)) for item in expected_frames_value)
    return IntegrationSeries(
        study_instance_uid=_nonempty(payload.get("StudyInstanceUID", ""), "StudyInstanceUID"),
        series_instance_uid=_nonempty(payload.get("SeriesInstanceUID", ""), "SeriesInstanceUID"),
        entries=entries,
        anatomical_plane=(str(payload["Anatomical_Plane"]).strip() if payload.get("Anatomical_Plane") else None),
        fluid_sensitive=_optional_bool(payload.get("Fluid_Sensitive"), "Fluid_Sensitive"),
        fat_suppression=_optional_bool(payload.get("Fat_Suppression"), "Fat_Suppression"),
        diagnostic_status=str(payload.get("diagnostic_status", "unknown")),
        expected_transfer_syntaxes=_tuple_of_strings(
            payload.get("expected_transfer_syntaxes", ()),
            "expected_transfer_syntaxes",
        ),
        expected_dimensions=_tuple_of_strings(payload.get("expected_dimensions", ()), "expected_dimensions"),
        expected_frame_counts=expected_frames,
        expected_entry_count=_optional_int(payload.get("expected_entry_count"), "expected_entry_count"),
    )


def _integration_study_from_dict(value: object) -> IntegrationStudy:
    payload = _mapping(value, "integration study")
    series_value = payload.get("series")
    if not isinstance(series_value, (list, tuple)):
        raise RealIntegrationError("integration study series must be an array")
    return IntegrationStudy(
        study_instance_uid=_nonempty(payload.get("StudyInstanceUID", ""), "StudyInstanceUID"),
        series=tuple(_integration_series_from_dict(item) for item in series_value),
    )


def load_integration_manifest(path: str | Path) -> IntegrationManifest:
    """Load and verify a private integration manifest before reading pixels."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RealIntegrationError(f"could not read integration manifest {source}") from exc
    root = _mapping(payload, "integration manifest")
    if root.get("schema_version") != INTEGRATION_MANIFEST_SCHEMA_VERSION:
        raise RealIntegrationError("unexpected integration manifest schema")
    stored_checksum = _nonempty(root.get("manifest_checksum", ""), "manifest_checksum")
    without_checksum = dict(root)
    without_checksum.pop("manifest_checksum", None)
    if _json_sha256(without_checksum) != stored_checksum:
        raise RealIntegrationError("integration manifest checksum does not match its contents")
    policies = _mapping(root.get("policies"), "policies")
    geometry_policy = GeometryPolicy.from_dict(policies.get("geometry"))
    decoder_policy = DecoderPolicy.from_dict(policies.get("decoder"))
    preprocess_config = _preprocess_from_cache_config(policies.get("preprocessing"), geometry_policy)
    packing_payload = _mapping(policies.get("packing"), "packing policy")
    packing_config = PackingConfig(
        max_slices=int(str(packing_payload.get("max_slices", 8))),
        max_series=int(str(packing_payload.get("max_series", 16))),
    )
    studies_value = root.get("studies")
    if not isinstance(studies_value, (list, tuple)):
        raise RealIntegrationError("integration manifest studies must be an array")
    manifest = IntegrationManifest(
        source_version=_nonempty(root.get("source_version", ""), "source_version"),
        archive_fingerprint=_nonempty(root.get("archive_fingerprint", ""), "archive_fingerprint"),
        geometry_policy=geometry_policy,
        decoder_policy=decoder_policy,
        preprocess_config=preprocess_config,
        studies=tuple(_integration_study_from_dict(item) for item in studies_value),
        source_split=_nonempty(root.get("source_split", ""), "source_split"),
        packing_config=packing_config,
        source_manifest_checksum=_optional_text(
            root.get("source_manifest_checksum"), "source_manifest_checksum"
        ),
        schema_version=str(root.get("schema_version", "")),
        manifest_checksum=stored_checksum,
    )
    return manifest


def _load_json_object(path: str | Path, description: str) -> Mapping[str, object]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RealIntegrationError(f"could not read {description} {source}") from exc
    return _mapping(value, description)


def _required_array(payload: Mapping[str, object], field: str) -> tuple[object, ...]:
    value = payload.get(field)
    if not isinstance(value, (list, tuple)):
        raise RealIntegrationError(f"{field} must be an array")
    return tuple(value)


def _mapping_by_uid(items: tuple[object, ...], field: str) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for value in items:
        item = _mapping(value, field)
        uid = _nonempty(item.get("SeriesInstanceUID", ""), f"{field}.SeriesInstanceUID")
        if uid in result:
            raise RealIntegrationError(f"{field} contains duplicate SeriesInstanceUID {uid}")
        result[uid] = item
    return result


def _expected_frame_counts(value: object, field: str) -> tuple[int, ...]:
    payload = _mapping(value, field)
    values: list[int] = []
    for key in payload:
        try:
            count = int(key)
        except (TypeError, ValueError) as exc:
            raise RealIntegrationError(f"{field} contains a non-integer frame count") from exc
        if count <= 0:
            raise RealIntegrationError(f"{field} contains a non-positive frame count")
        values.append(count)
    return tuple(sorted(values))


def _expected_keys(value: object, field: str) -> tuple[str, ...]:
    payload = _mapping(value, field)
    return tuple(sorted(_nonempty(key, field) for key in payload))


def _validate_storage_budget(
    inventory: Mapping[str, object],
    selection: Mapping[str, object],
    inventory_series: Mapping[str, Mapping[str, object]],
    selected_series_uids: Sequence[str],
    image_size: tuple[int, int],
) -> None:
    """Recompute the selected batch's conservative 2 TB peak before admission."""

    archive = _mapping(inventory.get("archive"), "inventory archive")
    storage = _mapping(selection.get("storage_estimate"), "selection storage_estimate")
    current_project_bytes = _required_nonnegative_int(
        storage.get("current_project_bytes"), "storage current_project_bytes"
    )
    soft_budget_bytes = _required_nonnegative_int(storage.get("soft_budget_bytes"), "storage soft_budget_bytes")
    if soft_budget_bytes > MAX_STORAGE_BUDGET_BYTES:
        raise RealIntegrationError(
            f"selection storage budget cannot exceed {MAX_STORAGE_BUDGET_BYTES} bytes"
        )
    archive_bytes = _required_nonnegative_int(archive.get("archive_size_bytes"), "archive archive_size_bytes")
    selected_items = [inventory_series[uid] for uid in selected_series_uids]
    extracted_bytes = sum(
        _required_nonnegative_int(item.get("uncompressed_bytes"), f"series {uid}.uncompressed_bytes")
        for uid, item in zip(selected_series_uids, selected_items)
    )
    cached_frames = sum(
        _required_nonnegative_int(item.get("effective_frame_count"), f"series {uid}.effective_frame_count")
        for uid, item in zip(selected_series_uids, selected_items)
    )
    pixel_cache_bytes = cached_frames * image_size[0] * image_size[1] * 4
    temporary_bytes = extracted_bytes + pixel_cache_bytes
    projected_peak_bytes = (
        current_project_bytes
        + archive_bytes
        + extracted_bytes
        + pixel_cache_bytes
        + temporary_bytes
    )
    expected = {
        "source_archive_bytes": archive_bytes,
        "extracted_bytes": extracted_bytes,
        "pixel_cache_bytes": pixel_cache_bytes,
        "temporary_bytes": temporary_bytes,
        "projected_peak_bytes": projected_peak_bytes,
    }
    for field, value in expected.items():
        if _required_nonnegative_int(storage.get(field), f"storage {field}") != value:
            raise RealIntegrationError(f"selection storage estimate is stale for {field}")
    if projected_peak_bytes > soft_budget_bytes or storage.get("within_soft_budget") is not True:
        raise RealIntegrationError(
            f"selected integration batch exceeds the {soft_budget_bytes}-byte storage budget"
        )


def build_integration_manifest(
    inventory_path: str | Path,
    selection_path: str | Path,
    output_path: str | Path,
    *,
    geometry_policy: GeometryPolicy = DEFAULT_REAL_GEOMETRY_POLICY,
    decoder_policy: DecoderPolicy = DEFAULT_REAL_DECODER_POLICY,
    preprocess_config: PreprocessConfig | None = None,
    packing_config: PackingConfig | None = None,
) -> IntegrationManifest:
    """Build a private pixel-verification manifest from ticket-03 artifacts.

    The ticket-03 inventory remains the source of truth for source membership,
    header counts, descriptors, and archive identity.  Decoder admission is
    recomputed against the explicitly supplied policy because an old selection
    may have been created before decoder support was recorded.
    """

    inventory = _load_json_object(inventory_path, "inventory")
    selection = _load_json_object(selection_path, "selection")
    if inventory.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        raise RealIntegrationError("inventory has an unexpected schema")
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise RealIntegrationError("selection has an unexpected schema")
    reconciliation = _mapping(inventory.get("reconciliation"), "inventory reconciliation")
    if reconciliation.get("full_corpus_reconciled") is not True:
        raise RealIntegrationError("ticket-03 inventory is not a complete corpus reconciliation")
    source_version = _nonempty(inventory.get("source_version", ""), "inventory source_version")
    if _nonempty(selection.get("source_version", ""), "selection source_version") != source_version:
        raise RealIntegrationError("inventory and selection source versions differ")
    archive = _mapping(inventory.get("archive"), "inventory archive")
    archive_fingerprint = _nonempty(archive.get("fingerprint", ""), "archive fingerprint")
    if _nonempty(selection.get("archive_fingerprint", ""), "selection archive_fingerprint") != archive_fingerprint:
        raise RealIntegrationError("inventory and selection archive identities differ")
    try:
        inventory_source = load_inventory_source(inventory_path)
    except (OSError, ValueError) as exc:
        raise RealIntegrationError("could not recover the inventory source identity") from exc
    if inventory_source.source_version != source_version:
        raise RealIntegrationError("inventory source identity differs from its source manifest")
    if inventory_source.source_fingerprint != archive_fingerprint:
        raise RealIntegrationError("inventory archive identity differs from its source manifest")
    source_checksum = source_manifest_checksum(inventory_source.source_manifest)

    blocked = _tuple_of_strings(selection.get("blocked_series_uids", ()), "blocked_series_uids")
    if blocked:
        raise RealIntegrationError("selection contains blocked series and cannot be admitted")
    gold_studies = _tuple_of_strings(selection.get("gold_study_uids", ()), "gold_study_uids")
    report_only_studies = _tuple_of_strings(
        selection.get("report_only_study_uids", ()),
        "report_only_study_uids",
    )
    selected_studies = gold_studies + report_only_studies
    if len(set(selected_studies)) != len(selected_studies):
        raise RealIntegrationError("selection repeats a study across cohorts")
    selected_series_uids = _tuple_of_strings(
        selection.get("selected_series_uids", ()),
        "selected_series_uids",
    )
    selected_entries = _tuple_of_strings(selection.get("entry_names", ()), "entry_names")
    inventory_payload = _mapping(inventory.get("inventory"), "inventory")
    inventory_series = _mapping_by_uid(_required_array(inventory_payload, "series"), "inventory series")
    actual_entries: list[str] = []
    series_by_study: dict[str, list[IntegrationSeries]] = {study: [] for study in selected_studies}
    selected_uids = set(selected_series_uids)
    if len(selected_uids) != len(selected_series_uids):
        raise RealIntegrationError("selection contains duplicate series")
    for series_uid in selected_series_uids:
        item = inventory_series.get(series_uid)
        if item is None:
            raise RealIntegrationError(f"selection references unknown series {series_uid}")
        if item.get("complete") is not True:
            raise RealIntegrationError(f"selected series {series_uid} is not complete in the inventory")
        study_uid = _nonempty(item.get("StudyInstanceUID", ""), "inventory StudyInstanceUID")
        if study_uid not in series_by_study:
            raise RealIntegrationError(f"selected series {series_uid} belongs to an unselected study")
        entries = _tuple_of_strings(item.get("entry_names", ()), f"series {series_uid}.entry_names")
        if not entries:
            raise RealIntegrationError(f"selected series {series_uid} has no source entries")
        actual_entries.extend(entries)
        expected_syntaxes = _expected_keys(item.get("transfer_syntax_counts", {}), "transfer_syntax_counts")
        unsupported = set(expected_syntaxes) - set(decoder_policy.supported_transfer_syntaxes)
        if unsupported:
            raise RealIntegrationError(
                f"decoder policy does not admit selected series {series_uid} transfer syntaxes: "
                + ", ".join(sorted(unsupported))
            )
        expected_dimensions = _expected_keys(item.get("dimension_counts", {}), "dimension_counts")
        expected_frames = _expected_frame_counts(item.get("frame_count_counts", {}), "frame_count_counts")
        if any(frame_count != 1 for frame_count in expected_frames):
            raise RealIntegrationError(
                f"selected series {series_uid} contains unsupported multiframe entries"
            )
        effective_frame_count = _optional_int(item.get("effective_frame_count"), "effective_frame_count")
        if effective_frame_count is None:
            effective_frame_count = len(entries)
        if effective_frame_count < geometry_policy.minimum_usable_slices:
            raise RealIntegrationError(
                f"selected series {series_uid} has a declared frame count below the "
                f"minimum usable slice count {geometry_policy.minimum_usable_slices}"
            )
        series = IntegrationSeries(
            study_instance_uid=study_uid,
            series_instance_uid=series_uid,
            entries=entries,
            anatomical_plane=(str(item["Anatomical_Plane"]).strip() if item.get("Anatomical_Plane") else None),
            fluid_sensitive=_optional_bool(item.get("Fluid_Sensitive"), "Fluid_Sensitive"),
            fat_suppression=_optional_bool(item.get("Fat_Suppression"), "Fat_Suppression"),
            diagnostic_status=str(item.get("diagnostic_status") or "unknown"),
            expected_transfer_syntaxes=expected_syntaxes,
            expected_dimensions=expected_dimensions,
            expected_frame_counts=expected_frames,
            expected_entry_count=int(str(item.get("entry_count", len(entries)))),
        )
        series_by_study[study_uid].append(series)
    if set(actual_entries) != set(selected_entries):
        raise RealIntegrationError("selection entry_names do not match selected series entries")
    if len(actual_entries) != len(set(actual_entries)):
        raise RealIntegrationError("selected series contain duplicate source entries")
    if set(series_by_study) != set(selected_studies):
        raise RealIntegrationError("selection contains a study without a selected series")

    materialization = preprocess_config or PreprocessConfig(geometry_policy=geometry_policy)
    if materialization.geometry_policy != geometry_policy:
        raise RealIntegrationError("preprocess_config geometry policy differs from manifest policy")
    _validate_storage_budget(
        inventory,
        selection,
        inventory_series,
        selected_series_uids,
        materialization.image_size,
    )
    manifest = IntegrationManifest(
        source_version=source_version,
        archive_fingerprint=archive_fingerprint,
        geometry_policy=geometry_policy,
        decoder_policy=decoder_policy,
        preprocess_config=materialization,
        studies=tuple(
            IntegrationStudy(study_uid, tuple(sorted(series_by_study[study_uid], key=lambda item: item.series_instance_uid)))
            for study_uid in sorted(selected_studies)
        ),
        source_split=str(inventory.get("split", "train")),
        packing_config=packing_config or PackingConfig(),
        source_manifest_checksum=source_checksum,
    )
    write_integration_manifest(manifest, output_path)
    return load_integration_manifest(output_path)


class _SourceReader:
    """Read entries from a verified archive, local fixture, or attested Kaggle input."""

    def __init__(
        self,
        source: Path,
        *,
        allow_unverified_directory: bool = False,
        kaggle_source: KaggleCompetitionSource | None = None,
    ) -> None:
        self.source = source
        self.allow_unverified_directory = allow_unverified_directory
        self.kaggle_source = kaggle_source
        self.archive: zipfile.ZipFile | None = None
        self.root: Path | None = None

    def __enter__(self) -> "_SourceReader":
        if self.kaggle_source is not None and not self.source.is_dir():
            raise RealIntegrationError("Kaggle source attestation requires the mounted source directory")
        if self.source.is_dir():
            if self.kaggle_source is not None:
                _validate_kaggle_source_root(self.source, self.kaggle_source)
                self.root = self.source.resolve()
                return self
            if not self.allow_unverified_directory:
                raise RealIntegrationError(
                    "real integration requires a source ZIP with archive identity; "
                    "directories require an explicit local fixture or Kaggle source attestation"
                )
            self.root = self.source.resolve()
            return self
        try:
            self.archive = zipfile.ZipFile(self.source, "r")
        except (OSError, zipfile.BadZipFile) as exc:
            raise RealIntegrationError(f"could not open real-DICOM source {self.source}") from exc
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self.archive is not None:
            self.archive.close()

    @staticmethod
    def _hash_stream(stream: BinaryIO) -> str:
        digest = hashlib.sha256()
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()

    def decode(self, entry_name: str, policy: DecoderPolicy) -> DicomSlice:
        safe = safe_member_path(entry_name)
        if self.archive is not None:
            try:
                hash_stream = self.archive.open(safe.as_posix(), "r")
            except (KeyError, OSError, EOFError, zipfile.BadZipFile) as exc:
                raise DamagedDicomError(f"source entry is missing: {entry_name}") from exc
            except NotImplementedError as exc:
                raise UnsupportedSourceEntryError(
                    f"archive compression is unsupported for {entry_name}"
                ) from exc
            try:
                with hash_stream:
                    source_content_sha256 = self._hash_stream(cast(BinaryIO, hash_stream))
            except (EOFError, OSError, zipfile.BadZipFile, zlib.error) as exc:
                raise DamagedDicomError(f"archive integrity failed for {entry_name}") from exc
            try:
                stream = self.archive.open(safe.as_posix(), "r")
            except (KeyError, OSError, EOFError, zipfile.BadZipFile) as exc:
                raise DamagedDicomError(f"source entry is missing: {entry_name}") from exc
            except NotImplementedError as exc:
                raise UnsupportedSourceEntryError(
                    f"archive compression is unsupported for {entry_name}"
                ) from exc
            try:
                with stream:
                    decoded = decode_stream(cast(BinaryIO, stream), filename=entry_name, policy=policy)
                    return replace(decoded, source_content_sha256=source_content_sha256)
            except (EOFError, OSError, zipfile.BadZipFile, zlib.error) as exc:
                raise DamagedDicomError(f"archive integrity failed for {entry_name}") from exc
        if self.root is None:
            raise RealIntegrationError("source reader was used outside its context")
        candidate = self.root / Path(*safe.parts)
        if candidate.is_symlink():
            raise DamagedDicomError(f"source entry is not a regular file: {entry_name}")
        target = candidate.resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise RealIntegrationError(f"source entry escapes the source root: {entry_name}") from exc
        if target.is_symlink() or not target.is_file():
            raise DamagedDicomError(f"source entry is not a regular file: {entry_name}")
        try:
            with target.open("rb") as hash_stream:
                source_content_sha256 = self._hash_stream(hash_stream)
            with target.open("rb") as stream:
                decoded = decode_stream(stream, filename=entry_name, policy=policy)
                return replace(decoded, source_content_sha256=source_content_sha256)
        except (FileNotFoundError, OSError) as exc:
            raise DamagedDicomError(f"source entry is missing: {entry_name}") from exc


def _failure_reason(error: DicomDecodeError) -> str:
    if isinstance(error, UnsupportedSourceEntryError):
        return "unsupported_source_encoding"
    if isinstance(error, UnsupportedTransferSyntaxError):
        return "unsupported_transfer_syntax"
    if isinstance(error, DecoderInfrastructureError):
        return "decoder_infrastructure"
    if isinstance(error, DamagedDicomError):
        return "damaged_input"
    if isinstance(error, DicomValidationError):
        return "dicom_validation"
    return "decode_error"


def _has_non_diagnostic_token(values: Sequence[str]) -> bool:
    return has_non_diagnostic_token(values)


def _validate_slice_against_manifest(
    item: Any,
    manifest_series: IntegrationSeries,
    entry_name: str,
) -> None:
    if item.study_instance_uid != manifest_series.study_instance_uid:
        raise DicomValidationError(f"study identity disagrees with integration manifest for {entry_name}")
    if item.series_instance_uid != manifest_series.series_instance_uid:
        raise DicomValidationError(f"series identity disagrees with integration manifest for {entry_name}")
    if item.sop_instance_uid != PurePosixPath(entry_name).stem:
        raise DicomValidationError(f"SOP identity disagrees with source filename for {entry_name}")
    if (
        manifest_series.expected_transfer_syntaxes
        and item.transfer_syntax_uid not in manifest_series.expected_transfer_syntaxes
    ):
        raise DicomValidationError(f"transfer syntax disagrees with inventory for {entry_name}")
    if (
        manifest_series.expected_dimensions
        and f"{item.rows}x{item.columns}" not in manifest_series.expected_dimensions
    ):
        raise DicomValidationError(f"pixel dimensions disagree with inventory for {entry_name}")
    if manifest_series.expected_frame_counts and item.number_of_frames not in manifest_series.expected_frame_counts:
        raise DicomValidationError(f"frame count disagrees with inventory for {entry_name}")


def _load_series(
    reader: _SourceReader,
    manifest_series: IntegrationSeries,
    decoder_policy: DecoderPolicy,
    geometry_policy: GeometryPolicy,
    *,
    require_complete_series: bool,
) -> LoadedSeries:
    expected_count = len(manifest_series.entries)
    if manifest_series.diagnostic_status == "excluded_non_diagnostic":
        return LoadedSeries(
            manifest_series=manifest_series,
            series=None,
            status="excluded_non_diagnostic",
            expected_slice_count=expected_count,
            decoded_slice_count=0,
            warnings=("excluded_non_diagnostic",),
        )

    decoded: list[Any] = []
    decoded_entries: list[str] = []
    failures: list[SeriesLoadFailure] = []
    for entry_name in manifest_series.entries:
        try:
            item = reader.decode(entry_name, decoder_policy)
            _validate_slice_against_manifest(item, manifest_series, entry_name)
        except DecoderInfrastructureError as exc:
            raise RealInferenceError(
                f"decoder infrastructure failure for {manifest_series.series_instance_uid}/"
                f"{entry_name}: {exc}"
            ) from exc
        except DicomDecodeError as exc:
            failures.append(
                SeriesLoadFailure(
                    series_instance_uid=manifest_series.series_instance_uid,
                    source_entry_name=entry_name,
                    reason=_failure_reason(exc),
                    detail=str(exc),
                    decoder=decoder_policy.attempt_order[-1],
                    attempts=decoder_policy.attempt_order,
                    recovery=(
                        "reject_series_on_any_failure"
                        if require_complete_series
                        else "skip_damaged_input_continue_with_survivors"
                    ),
                    affected_source_entry_names=(entry_name,),
                )
            )
            continue
        decoded.append(item)
        decoded_entries.append(entry_name)

    metadata = [
        value
        for item in decoded
        for value in (item.series_description, *item.image_type_values)
        if value
    ]
    if _has_non_diagnostic_token(metadata):
        return LoadedSeries(
            manifest_series=manifest_series,
            series=None,
            status="excluded_non_diagnostic",
            expected_slice_count=expected_count,
            decoded_slice_count=len(decoded),
            failures=tuple(failures),
            warnings=("excluded_non_diagnostic",),
        )

    warnings: list[str] = []
    if not metadata:
        warnings.append("diagnostic_metadata_unknown")
    if len(decoded) < geometry_policy.minimum_usable_slices:
        failures.append(
            SeriesLoadFailure(
                series_instance_uid=manifest_series.series_instance_uid,
                source_entry_name="",
                reason="insufficient_slices",
                detail=(
                    f"{len(decoded)} decoded slices are below minimum "
                    f"{geometry_policy.minimum_usable_slices}"
                ),
                decoder="geometry",
                recovery="reject_series_below_minimum_slices",
                affected_source_entry_names=manifest_series.entries,
            )
        )
    if failures and require_complete_series:
        return LoadedSeries(
            manifest_series=manifest_series,
            series=None,
            status="rejected",
            expected_slice_count=expected_count,
            decoded_slice_count=len(decoded),
            failures=tuple(failures),
            warnings=tuple(warnings),
        )
    if len(decoded) < geometry_policy.minimum_usable_slices:
        return LoadedSeries(
            manifest_series=manifest_series,
            series=None,
            status="rejected",
            expected_slice_count=expected_count,
            decoded_slice_count=len(decoded),
            failures=tuple(failures),
            warnings=tuple(warnings),
        )

    description_values = tuple(sorted({item.series_description for item in decoded if item.series_description}))
    description = description_values[0] if description_values else ""
    series = SeriesRecord(
        study_instance_uid=manifest_series.study_instance_uid,
        series_instance_uid=manifest_series.series_instance_uid,
        slices=tuple(decoded),
        description=description,
        anatomical_plane=manifest_series.anatomical_plane,
        fluid_sensitive=manifest_series.fluid_sensitive,
        fat_suppression=manifest_series.fat_suppression,
        diagnostic_status="diagnostic",
        source_entry_names=tuple(decoded_entries),
    )
    return LoadedSeries(
        manifest_series=manifest_series,
        series=series,
        status="accepted" if not failures else "accepted_partial",
        expected_slice_count=expected_count,
        decoded_slice_count=len(decoded),
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


def _load_real_study_with_reader(
    manifest: IntegrationManifest,
    study_manifest: IntegrationStudy,
    reader: _SourceReader,
    *,
    require_complete_series: bool,
) -> LoadedStudy:
    results = tuple(
        _load_series(
            reader,
            series,
            manifest.decoder_policy,
            manifest.geometry_policy,
            require_complete_series=require_complete_series,
        )
        for series in study_manifest.series
    )
    accepted = tuple(item.series for item in results if item.series is not None)
    return LoadedStudy(
        study=StudyRecord(study_manifest.study_instance_uid, accepted),
        series_results=results,
        warnings=tuple(warning for item in results for warning in item.warnings),
    )


def load_real_study(
    manifest: IntegrationManifest,
    study_manifest: IntegrationStudy,
    source: str | Path,
    *,
    require_complete_series: bool = True,
    allow_unverified_directory: bool = False,
    kaggle_source: KaggleCompetitionSource | None = None,
) -> LoadedStudy:
    """Load one manifest study from its checksum-bound source."""

    canonical = next(
        (item for item in manifest.studies if item.study_instance_uid == study_manifest.study_instance_uid),
        None,
    )
    if canonical is None:
        raise RealIntegrationError("study is not present in the integration manifest")
    if study_manifest != canonical:
        raise RealIntegrationError("study does not match the checksum-covered integration manifest")
    if allow_unverified_directory:
        _validate_local_smoke_scope(manifest, Path(source))
    with _SourceReader(
        Path(source),
        allow_unverified_directory=allow_unverified_directory,
        kaggle_source=kaggle_source,
    ) as reader:
        return _load_real_study_with_reader(
            manifest,
            canonical,
            reader,
            require_complete_series=require_complete_series,
        )


def iter_real_studies(
    manifest: IntegrationManifest,
    source: str | Path,
    *,
    require_complete_series: bool = True,
    allow_unverified_directory: bool = False,
    study_uids: Sequence[str] | None = None,
    kaggle_source: KaggleCompetitionSource | None = None,
) -> Iterator[LoadedStudy]:
    """Yield canonical manifest studies while keeping a ZIP handle open once.

    ``study_uids`` limits reading to a declared manifest subset. It cannot add
    studies or alter the canonical series mapping and is intended for bounded
    training-only diagnostics.
    """

    if allow_unverified_directory:
        _validate_local_smoke_scope(manifest, Path(source))
    if kaggle_source is not None and allow_unverified_directory:
        raise RealIntegrationError("Kaggle source attestation cannot use local fixture mode")
    selected: set[str] | None = None
    if study_uids is not None:
        selected = set(study_uids)
        if not selected or any(not isinstance(uid, str) or not uid.strip() for uid in study_uids):
            raise RealIntegrationError("selected study identities must be non-empty")
        if len(selected) != len(study_uids):
            raise RealIntegrationError("selected study identities must be unique")
        unknown = selected.difference(manifest.study_uids)
        if unknown:
            raise RealIntegrationError("selected study is absent from the integration manifest")
    with _SourceReader(
        Path(source),
        allow_unverified_directory=allow_unverified_directory,
        kaggle_source=kaggle_source,
    ) as reader:
        for study_manifest in manifest.studies:
            if selected is not None and study_manifest.study_instance_uid not in selected:
                continue
            yield _load_real_study_with_reader(
                manifest,
                study_manifest,
                reader,
                require_complete_series=require_complete_series,
            )


def _validate_kaggle_source_root(
    source: Path,
    attestation: KaggleCompetitionSource,
) -> None:
    """Require the source to be the read-only competition mount in Kaggle."""

    resolved_source = source.resolve()
    resolved_root = attestation.root.resolve()
    kaggle_input = Path("/kaggle/input").resolve()
    if resolved_source != resolved_root:
        raise RealIntegrationError("Kaggle source path differs from its runtime attestation")
    try:
        resolved_root.relative_to(kaggle_input)
    except ValueError as exc:
        raise RealIntegrationError("Kaggle competition inputs must come from /kaggle/input") from exc
    if not resolved_root.is_dir() or source.is_symlink():
        raise RealIntegrationError("Kaggle competition source is not a regular mounted directory")
    if not Path("/kaggle/working").is_dir():
        raise RealIntegrationError("Kaggle working directory is unavailable")


def _check_execution_context(*, require_slurm: bool, allow_local_smoke: bool) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if allow_local_smoke and job_id:
        raise RealInferenceError("local smoke mode cannot run inside a Slurm allocation")
    if require_slurm and (not job_id or partition != "cluster1-cpu"):
        raise RealInferenceError("real integration must run in a cluster1-cpu Slurm allocation")
    if not job_id and not allow_local_smoke:
        raise RealInferenceError(
            "refusing to read real-DICOM inputs outside Slurm; "
            "use --allow-local-smoke only with a tiny fixture"
        )


def _validate_local_smoke_scope(manifest: IntegrationManifest, source: Path) -> None:
    if os.environ.get("SLURM_JOB_ID"):
        raise RealInferenceError("local smoke mode cannot run inside a Slurm allocation")
    if not source.is_dir():
        raise RealInferenceError("local smoke mode requires a small extracted fixture directory")
    if manifest.series_count > 16 or sum(len(item.entries) for study in manifest.studies for item in study.series) > 256:
        raise RealInferenceError("local smoke mode is limited to 16 series and 256 source entries")
    try:
        total_bytes = sum(item.stat().st_size for item in source.rglob("*") if item.is_file())
    except OSError as exc:
        raise RealInferenceError(f"could not inspect local smoke fixture {source}") from exc
    if total_bytes > LOCAL_SMOKE_MAX_BYTES:
        raise RealInferenceError(
            f"local smoke mode is limited to {LOCAL_SMOKE_MAX_BYTES} source bytes"
        )


def _configure_threads() -> int:
    requested = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    try:
        thread_count = max(1, min(16, int(requested)))
    except ValueError:
        thread_count = 1
    torch.set_num_threads(thread_count)
    return thread_count


def _verify_archive_source_identity(source_path: Path, manifest: IntegrationManifest) -> None:
    """Reconcile a ZIP source with the identity used to build the manifest."""

    if not source_path.is_file() or source_path.suffix.casefold() != ".zip":
        return
    from .inventory import identify_archive

    try:
        actual = identify_archive(source_path, source_version=manifest.source_version)
    except (OSError, ValueError) as exc:
        raise RealIntegrationError(f"could not verify source archive {source_path}") from exc
    if actual.fingerprint != manifest.archive_fingerprint:
        raise RealIntegrationError(
            "source archive identity does not match the private integration manifest"
        )


def _load_reference_labels_json(
    path: str | Path,
) -> tuple[dict[str, tuple[int | None, ...]], dict[str, int] | None]:
    """Load trusted reference rows from a private inventory/fold-style JSON."""

    root = _load_json_object(path, "fallback reference labels")
    rows: dict[str, tuple[int | None, ...]] = {}
    folds: dict[str, int] = {}
    labels_value = root.get("labels")
    if isinstance(labels_value, Mapping):
        for raw_uid, raw_labels in labels_value.items():
            uid = _nonempty(raw_uid, "fallback reference label study identifier")
            if uid in rows:
                raise RealIntegrationError(f"fallback reference labels repeat study {uid}")
            rows[uid] = validate_reference_label_vector(raw_labels, field=f"fallback labels for {uid}")
    else:
        memberships = root.get("memberships")
        if isinstance(memberships, list):
            if root.get("schema_version") != FOLD_SCHEMA_VERSION:
                raise RealIntegrationError(
                    "fold-shaped fallback labels must use the canonical fold-manifest schema"
                )
            try:
                load_fold_manifest(path)
            except ValueError as exc:
                raise RealIntegrationError(
                    "fold-shaped fallback labels failed canonical manifest validation"
                ) from exc
            for index, raw_item in enumerate(memberships):
                item = _mapping(raw_item, f"fallback membership {index}")
                if item.get("cohort") != "gold":
                    continue
                uid = _nonempty(item.get("StudyInstanceUID", ""), "fallback membership StudyInstanceUID")
                if uid in rows:
                    raise RealIntegrationError(f"fallback reference labels repeat study {uid}")
                rows[uid] = validate_reference_label_vector(
                    item.get("reference_labels"),
                    field=f"fallback membership labels for {uid}",
                )
                if any(label is None for label in rows[uid]):
                    raise RealIntegrationError(
                        f"fallback gold membership labels must be complete for {uid}"
                    )
                raw_fold = item.get("fold")
                if raw_fold is not None:
                    if isinstance(raw_fold, bool):
                        raise RealIntegrationError(f"fallback membership fold is invalid for {uid}")
                    try:
                        fold = int(str(raw_fold))
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise RealIntegrationError(
                            f"fallback membership fold is invalid for {uid}"
                        ) from exc
                    if fold < 0:
                        raise RealIntegrationError(f"fallback membership fold is invalid for {uid}")
                    folds[uid] = fold
        else:
            source = root.get("source")
            source_mapping = source if isinstance(source, Mapping) else root
            studies = source_mapping.get("studies") if isinstance(source_mapping, Mapping) else None
            if isinstance(studies, list):
                if root.get("schema_version") != INVENTORY_SCHEMA_VERSION:
                    raise RealIntegrationError(
                        "inventory-shaped fallback labels must use the canonical inventory schema"
                    )
                reconciliation = root.get("reconciliation")
                if (
                    not isinstance(reconciliation, Mapping)
                    or reconciliation.get("full_corpus_reconciled") is not True
                ):
                    raise RealIntegrationError(
                        "inventory-shaped fallback labels must be from a reconciled inventory"
                    )
                for index, raw_item in enumerate(studies):
                    item = _mapping(raw_item, f"fallback source study {index}")
                    labels = validate_reference_label_vector(
                        item.get("reference_labels"),
                        field=f"fallback source labels {index}",
                    )
                    if all(label is not None for label in labels):
                        uid = _nonempty(
                            item.get("StudyInstanceUID", ""),
                            "fallback source StudyInstanceUID",
                        )
                        if uid in rows:
                            raise RealIntegrationError(f"fallback reference labels repeat study {uid}")
                        rows[uid] = labels
    if not rows:
        raise RealIntegrationError(
            "fallback reference labels contain no complete gold/reference rows"
        )
    if len(rows) != len(set(rows)):
        raise RealIntegrationError("fallback reference labels contain duplicate study identifiers")
    return rows, (folds if folds else None)


def _load_study_uid_list_json(path: str | Path) -> tuple[str, ...]:
    source = Path(path)
    try:
        root_value: object = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RealIntegrationError(f"could not read study identifier list {source}") from exc
    if isinstance(root_value, list):
        values = root_value
    elif isinstance(root_value, Mapping):
        values = root_value.get("study_uids", root_value.get("eligible_study_uids"))
        if not isinstance(values, list):
            raise RealIntegrationError(
                f"study identifier list {source} must be an array or contain study_uids"
            )
    else:
        raise RealIntegrationError(f"study identifier list {source} must contain an array")
    return _tuple_of_strings(values, "study identifier list")


def _private_series_rejections(
    loaded: LoadedStudy,
    prediction: StudyPrediction,
) -> list[dict[str, object]]:
    """Expand non-loader rejections with the files and admission stage involved."""

    result: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for failure in prediction.rejected_series:
        manifest_series = next(
            (
                item
                for item in loaded.series_results
                if item.manifest_series.series_instance_uid == failure.series_instance_uid
            ),
            None,
        )
        if manifest_series is None:
            continue
        loader_reasons = {
            f"{item.reason}: {item.detail}" for item in manifest_series.failures
        }
        if failure.reason in loader_reasons:
            continue
        key = (failure.series_instance_uid, failure.reason)
        if key in seen:
            continue
        seen.add(key)
        is_preprocessing_rejection = manifest_series.series is not None
        result.append(
            {
                "series_instance_uid": failure.series_instance_uid,
                "affected_files": list(manifest_series.manifest_series.entries),
                "surviving_files": (
                    list(manifest_series.series.source_entry_names)
                    if manifest_series.series is not None
                    else []
                ),
                "decoder": "preprocessing_or_geometry" if is_preprocessing_rejection else "admission_policy",
                "attempts": [],
                "error": failure.reason,
                "recovery": (
                    "study_fallback_reference_prior"
                    if prediction.used_fallback
                    else (
                        "skip_unusable_series_continue_with_survivors"
                        if is_preprocessing_rejection
                        else "exclude_non_diagnostic_series"
                    )
                ),
            }
        )
    return result


def _private_study_evidence(
    loaded: LoadedStudy,
    prediction: StudyPrediction,
) -> dict[str, object]:
    return {
        "StudyInstanceUID": loaded.study.study_instance_uid,
        "series": [
            {
                "SeriesInstanceUID": item.manifest_series.series_instance_uid,
                "status": item.status,
                "expected_slice_count": item.expected_slice_count,
                "decoded_slice_count": item.decoded_slice_count,
                "warnings": list(item.warnings),
                "failures": [
                    {
                        "source_entry_name": failure.source_entry_name,
                        "affected_files": list(
                            failure.affected_source_entry_names
                            or ((failure.source_entry_name,) if failure.source_entry_name else ())
                        ),
                        "reason": failure.reason,
                        "detail": failure.detail,
                        "decoder": failure.decoder,
                        "attempts": list(failure.attempts),
                        "recovery": failure.recovery,
                    }
                    for failure in item.failures
                ],
            }
            for item in loaded.series_results
        ],
        "prediction": {
            "used_fallback": prediction.used_fallback,
            "valid_series_count": prediction.valid_series_count,
            "rejected_series": [
                {"series_instance_uid": item.series_instance_uid, "reason": item.reason}
                for item in prediction.rejected_series
            ],
            "warnings": list(prediction.warnings),
            "sampled_source_indices": [list(indices) for indices in prediction.sampled_source_indices],
        },
        "recovery": {
            "manifest_series_count": len(loaded.series_results),
            "remaining_usable_series": prediction.valid_series_count,
            "used_fallback": prediction.used_fallback,
            "action": (
                "study_fallback_reference_prior"
                if prediction.used_fallback
                else (
                    "continue_with_surviving_series"
                    if loaded.failures or prediction.rejected_series
                    else "normal_model_inference"
                )
            ),
            "full_study_fallback": prediction.used_fallback,
            "slice_loss": {
                "expected": sum(item.expected_slice_count for item in loaded.series_results),
                "decoded": sum(item.decoded_slice_count for item in loaded.series_results),
                "lost": sum(
                    item.expected_slice_count - item.decoded_slice_count
                    for item in loaded.series_results
                ),
            },
            "series_loss": {
                "expected": len(loaded.series_results),
                "admitted_by_loader": sum(item.series is not None for item in loaded.series_results),
                "usable": prediction.valid_series_count,
                "lost": len(loaded.series_results) - prediction.valid_series_count,
            },
            "series_rejections": _private_series_rejections(loaded, prediction),
            "fallback_investigation": {
                "required": prediction.used_fallback,
                "status": "pending_operator_review" if prediction.used_fallback else "not_required",
                "investigated": not prediction.used_fallback,
                "unavoidable": None,
                "promotion_eligible": not prediction.used_fallback,
            },
        },
    }


def run_real_inference(
    manifest_path: str | Path,
    source_path: str | Path,
    output_path: str | Path,
    *,
    evidence_path: str | Path | None = None,
    private_evidence_path: str | Path | None = None,
    model: StudyModel | None = None,
    model_seed: int = 20260911,
    device: str | torch.device = "cpu",
    cache_root: str | Path | None = None,
    fallback_prior: ReferenceFallbackPrior | None = None,
    fallback_reference_labels: Mapping[str, Sequence[int | None]] | None = None,
    fallback_eligible_study_uids: Sequence[str] | None = None,
    fallback_held_out_study_uids: Sequence[str] = (),
    fallback_alpha: float = 1.0,
    evidence_ticket: str = "05",
    recover_damaged_inputs: bool | None = None,
    require_slurm: bool = False,
    allow_local_smoke: bool = False,
) -> dict[str, object]:
    """Run all manifest studies through the shared model and submission path."""

    _check_execution_context(require_slurm=require_slurm, allow_local_smoke=allow_local_smoke)
    if not evidence_ticket.strip():
        raise RealIntegrationError("evidence_ticket must be non-empty")
    if recover_damaged_inputs is None:
        recover_damaged_inputs = evidence_ticket == "07"
    thread_count = _configure_threads()
    manifest = load_integration_manifest(manifest_path)
    source = Path(source_path)
    if allow_local_smoke:
        _validate_local_smoke_scope(manifest, source)
    _verify_archive_source_identity(source, manifest)
    if fallback_prior is not None and fallback_reference_labels is not None:
        raise RealIntegrationError(
            "fallback_prior and fallback_reference_labels are mutually exclusive"
        )
    if fallback_reference_labels is not None:
        prior = compute_laplace_smoothed_reference_priors(
            fallback_reference_labels,
            eligible_study_uids=fallback_eligible_study_uids,
            held_out_study_uids=fallback_held_out_study_uids,
            alpha=fallback_alpha,
        )
    else:
        prior = fallback_prior or default_reference_fallback_prior()
    if model is None:
        torch.manual_seed(model_seed)
        model = build_study_model(ModelConfig(image_size=manifest.preprocess_config.image_size))
    series_cache = SeriesCache(cache_root) if cache_root is not None else None
    engine = InferenceEngine(
        model,
        manifest.preprocess_config,
        device=device,
        fallback_prior=prior,
        packing_config=manifest.packing_config,
        series_cache=series_cache,
    )

    started = time.perf_counter()
    predictions: list[StudyPrediction] = []
    private_study_evidence: list[dict[str, object]] = []
    manifest_series_count = 0
    accepted_series_count = 0
    usable_series_count = 0
    complete_series_count = 0
    rejected_series_count = 0
    excluded_series_count = 0
    series_with_decode_failures = 0
    expected_slice_count = 0
    decoded_slice_count = 0
    prediction_rejected_series_count = 0
    warning_count = 0
    failure_reason_counts: dict[str, int] = {}
    for loaded in iter_real_studies(
        manifest,
        source,
        require_complete_series=not recover_damaged_inputs,
        allow_unverified_directory=allow_local_smoke,
    ):
        manifest_series_count += len(loaded.series_results)
        accepted_series_count += sum(
            item.status in {"accepted", "accepted_partial"} for item in loaded.series_results
        )
        complete_series_count += sum(item.complete for item in loaded.series_results)
        rejected_series_count += sum(item.status == "rejected" for item in loaded.series_results)
        excluded_series_count += sum(
            item.status == "excluded_non_diagnostic" for item in loaded.series_results
        )
        series_with_decode_failures += sum(bool(item.failures) for item in loaded.series_results)
        expected_slice_count += sum(item.expected_slice_count for item in loaded.series_results)
        decoded_slice_count += sum(item.decoded_slice_count for item in loaded.series_results)
        loader_failures = tuple(
            SeriesFailure(
                item.manifest_series.series_instance_uid,
                f"{failure.reason}: {failure.detail}",
            )
            for item in loaded.series_results
            for failure in item.failures
        )
        excluded_failures = tuple(
            SeriesFailure(item.manifest_series.series_instance_uid, "excluded_non_diagnostic")
            for item in loaded.series_results
            if item.status == "excluded_non_diagnostic"
        )
        prediction = engine.predict_study(
            loaded.study,
            input_failures=loader_failures + excluded_failures,
            input_warnings=loaded.warnings,
        )
        private_study_evidence.append(_private_study_evidence(loaded, prediction))
        usable_series_count += prediction.valid_series_count
        for failure in loaded.failures:
            failure_reason_counts[failure.reason] = failure_reason_counts.get(failure.reason, 0) + 1
        loader_failure_reasons = {
            (
                item.manifest_series.series_instance_uid,
                f"{failure.reason}: {failure.detail}",
            )
            for item in loaded.series_results
            for failure in item.failures
        }
        preprocessing_rejection_count = sum(
            (
                (failure.series_instance_uid, failure.reason) not in loader_failure_reasons
                and any(
                    item.manifest_series.series_instance_uid == failure.series_instance_uid
                    and item.series is not None
                    for item in loaded.series_results
                )
            )
            for failure in prediction.rejected_series
        )
        if preprocessing_rejection_count:
            failure_reason_counts["preprocessing_or_geometry"] = (
                failure_reason_counts.get("preprocessing_or_geometry", 0)
                + preprocessing_rejection_count
            )
        prediction_rejected_series_count += len(prediction.rejected_series)
        warning_count += len(prediction.warnings)
        predictions.append(prediction)

    if not predictions:
        raise RealInferenceError("integration manifest produced no study predictions")
    study_uids = tuple(item.study_instance_uid for item in manifest.studies)
    result_uids = tuple(item.study_instance_uid for item in predictions)
    if result_uids != study_uids:
        raise RealInferenceError("real loader changed the manifest study order or membership")
    elapsed_seconds = time.perf_counter() - started

    fallback_numerator = sum(item.used_fallback for item in predictions)
    fallback_denominator = len(predictions)
    fallback_rate = fallback_numerator / fallback_denominator
    if fallback_numerator == 0:
        fallback_status = "zero_preferred_achieved"
    elif fallback_rate < 0.005:
        fallback_status = "investigation_required_before_promotion"
    elif fallback_rate <= 0.01:
        fallback_status = "normally_blocks_promotion"
    else:
        fallback_status = "blocks_promotion"
    lost_series_count = manifest_series_count - usable_series_count
    lost_slice_count = expected_slice_count - decoded_slice_count

    scores = np.concatenate([item.scores for item in predictions])
    public_evidence: dict[str, object] = {
        "schema_version": INTEGRATION_EVIDENCE_SCHEMA_VERSION,
        "ticket": evidence_ticket,
        "status": "completed_real_integration",
        "scope": "private representative real DICOM series; no identifiers or paths in this record",
        "source": {
            "source_version": manifest.source_version,
            "archive_fingerprint": manifest.archive_fingerprint,
            "manifest_checksum": manifest.manifest_checksum,
        },
        "policies": {
            "geometry": manifest.geometry_policy.to_dict(),
            "decoder": manifest.decoder_policy.to_dict(),
            "preprocessing": manifest.preprocess_config.cache_config(),
            "packing": manifest.packing_config.to_dict(),
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_isolated_venv": sys.prefix != sys.base_prefix,
            "platform": platform.system(),
            "execution": "slurm_cluster1_cpu" if os.environ.get("SLURM_JOB_ID") else "local_smoke",
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "pydicom_version": manifest.decoder_policy.pydicom_version,
            "torch_num_threads": thread_count,
        },
        "counts": {
            "input_studies": len(predictions),
            "manifest_series": manifest_series_count,
            "accepted_series": accepted_series_count,
            "usable_series": usable_series_count,
            "complete_series": complete_series_count,
            "rejected_series": rejected_series_count,
            "excluded_non_diagnostic_series": excluded_series_count,
            "series_with_decode_failures": series_with_decode_failures,
            "expected_slices": expected_slice_count,
            "decoded_slices": decoded_slice_count,
            "fallback_studies": sum(item.used_fallback for item in predictions),
            "prediction_rejected_series": prediction_rejected_series_count,
            "diagnostic_warning_count": warning_count,
            "scores_finite": bool(np.isfinite(scores).all()),
            "scores_bounded": bool(((scores >= 0.0) & (scores <= 1.0)).all()),
            "output_rows": len(predictions),
            "output_columns": NUM_TARGETS + 1,
            "full_study_fallback_numerator": fallback_numerator,
            "full_study_fallback_denominator": fallback_denominator,
            "full_study_fallback_rate": fallback_rate,
            "lost_series": lost_series_count,
            "lost_slices": lost_slice_count,
        },
        "fallback_prior": prior.to_public_dict(),
        "recovery": {
            "full_study_fallback": {
                "numerator": fallback_numerator,
                "denominator": fallback_denominator,
                "rate": fallback_rate,
                "status": fallback_status,
            },
            "slice_loss": {
                "expected": expected_slice_count,
                "decoded": decoded_slice_count,
                "lost": lost_slice_count,
            },
            "series_loss": {
                "expected": manifest_series_count,
                "admitted_by_loader": accepted_series_count,
                "usable": usable_series_count,
                "lost": lost_series_count,
                "rejected": rejected_series_count,
                "excluded_non_diagnostic": excluded_series_count,
            },
            "failure_reasons": dict(sorted(failure_reason_counts.items())),
            "global_failures_are_fatal": True,
            "fallback_investigation": {
                "required": fallback_numerator > 0,
                "fallback_cases": fallback_numerator,
                "investigated_cases": 0,
                "unavoidable_cases": 0,
                "status": "pending_operator_review" if fallback_numerator else "not_required",
                "complete": fallback_numerator == 0,
                "promotion_blocked_until_complete": fallback_numerator > 0,
            },
        },
        "timing": {"elapsed_seconds": elapsed_seconds},
        "cache_boundary": {
            "cache_configuration": manifest.preprocess_config.cache_config(),
            "sampling_is_post_cache": True,
            "model_architecture_is_not_in_cache_identity": True,
            "series_cache_enabled": series_cache is not None,
        },
    }
    destinations = [Path(output_path)]
    if evidence_path is not None:
        destinations.append(Path(evidence_path))
    if private_evidence_path is not None:
        destinations.append(Path(private_evidence_path))
    if len(set(destinations)) != len(destinations):
        raise RealInferenceError("submission and evidence destinations must be distinct")
    for destination in destinations:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite protected integration artifact {destination}")
    if evidence_path is not None:
        atomic_json_write(evidence_path, public_evidence)
    if private_evidence_path is not None:
        atomic_json_write(
            private_evidence_path,
            {
                "schema_version": INTEGRATION_EVIDENCE_SCHEMA_VERSION,
                "ticket": evidence_ticket,
                "manifest_checksum": manifest.manifest_checksum,
                "source_version": manifest.source_version,
                "fallback_prior": prior.to_private_dict(),
                "studies": private_study_evidence,
            },
        )
    write_submission(
        output_path,
        expected_study_uids=study_uids,
        predictions={item.study_instance_uid: item.scores for item in predictions},
    )
    return public_evidence


def _parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=None)
    parser.add_argument("--private-evidence", type=Path, default=None)
    parser.add_argument("--inventory", type=Path, default=None)
    parser.add_argument("--selection", type=Path, default=None)
    parser.add_argument("--model-seed", type=int, default=20260911)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument(
        "--fallback-reference-labels",
        type=Path,
        default=None,
        help="private JSON object containing trusted reference-label rows",
    )
    parser.add_argument(
        "--fallback-eligible-study-uids",
        type=Path,
        default=None,
        help="private JSON array/object naming the reference rows allowed in the prior",
    )
    parser.add_argument(
        "--fallback-held-out-study-uids",
        type=Path,
        default=None,
        help="private JSON array/object naming rows excluded from the prior",
    )
    parser.add_argument(
        "--fallback-held-out-fold",
        type=int,
        default=None,
        help="derive eligible and held-out gold rows from this fold in a private fold manifest",
    )
    parser.add_argument(
        "--fallback-final-refit",
        action="store_true",
        help="explicitly use all trusted reference rows for final-refit fallback inference",
    )
    parser.add_argument("--fallback-alpha", type=float, default=1.0)
    parser.add_argument("--evidence-ticket", default="05")
    parser.add_argument(
        "--recover-damaged-inputs",
        action="store_true",
        default=None,
        help="continue with surviving slices and series instead of strict series admission",
    )
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--allow-local-smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if (args.inventory is None) != (args.selection is None):
            raise RealIntegrationError("--inventory and --selection must be supplied together")
        if args.inventory is not None and args.selection is not None:
            build_integration_manifest(args.inventory, args.selection, args.manifest)
        fallback_reference_labels = None
        fallback_label_folds = None
        fallback_eligible_study_uids = None
        fallback_held_out_study_uids: tuple[str, ...] | None = None
        if args.fallback_reference_labels is not None:
            fallback_reference_labels, fallback_label_folds = _load_reference_labels_json(
                args.fallback_reference_labels
            )
        if args.fallback_eligible_study_uids is not None:
            fallback_eligible_study_uids = _load_study_uid_list_json(args.fallback_eligible_study_uids)
        if args.fallback_held_out_study_uids is not None:
            fallback_held_out_study_uids = _load_study_uid_list_json(args.fallback_held_out_study_uids)
        if args.fallback_final_refit:
            if fallback_reference_labels is None:
                raise RealIntegrationError(
                    "--fallback-final-refit requires a reference-label file"
                )
            if args.fallback_held_out_fold is not None:
                raise RealIntegrationError(
                    "--fallback-final-refit and --fallback-held-out-fold are mutually exclusive"
                )
            if fallback_eligible_study_uids is not None or fallback_held_out_study_uids is not None:
                raise RealIntegrationError(
                    "--fallback-final-refit cannot be combined with explicit study partitions"
                )
            fallback_eligible_study_uids = tuple(sorted(fallback_reference_labels))
            fallback_held_out_study_uids = ()
        elif args.fallback_held_out_fold is not None:
            if fallback_reference_labels is None or fallback_label_folds is None:
                raise RealIntegrationError(
                    "--fallback-held-out-fold requires a fold-shaped reference-label file"
                )
            if set(fallback_label_folds) != set(fallback_reference_labels):
                raise RealIntegrationError(
                    "fold-shaped reference labels must declare a fold for every gold row"
                )
            derived_held_out = tuple(
                sorted(
                    uid
                    for uid, fold in fallback_label_folds.items()
                    if fold == args.fallback_held_out_fold
                )
            )
            derived_eligible = tuple(
                sorted(
                    uid
                    for uid, fold in fallback_label_folds.items()
                    if fold != args.fallback_held_out_fold
                )
            )
            if not derived_held_out:
                raise RealIntegrationError(
                    f"selected held-out fold {args.fallback_held_out_fold} has no gold rows"
                )
            if (
                fallback_eligible_study_uids is not None
                and set(fallback_eligible_study_uids) != set(derived_eligible)
            ):
                raise RealIntegrationError(
                    "explicit eligible studies disagree with the selected held-out fold"
                )
            if (
                fallback_held_out_study_uids is not None
                and set(fallback_held_out_study_uids) != set(derived_held_out)
            ):
                raise RealIntegrationError(
                    "explicit held-out studies disagree with the selected held-out fold"
                )
            fallback_eligible_study_uids = derived_eligible
            fallback_held_out_study_uids = derived_held_out
        elif fallback_reference_labels is not None:
            if fallback_label_folds is None and fallback_eligible_study_uids is None:
                raise RealIntegrationError(
                    "--fallback-eligible-study-uids is required for reference labels "
                    "without fold assignments"
                )
            if (
                fallback_label_folds is not None
                and (
                    fallback_eligible_study_uids is None
                    or fallback_held_out_study_uids is None
                )
            ):
                raise RealIntegrationError(
                    "fold-shaped fallback labels require a held-out fold, explicit "
                    "eligible and held-out lists, or --fallback-final-refit"
                )
        evidence = run_real_inference(
            args.manifest,
            args.source,
            args.output,
            evidence_path=args.evidence,
            private_evidence_path=args.private_evidence,
            model_seed=args.model_seed,
            cache_root=args.cache_root,
            fallback_reference_labels=fallback_reference_labels,
            fallback_eligible_study_uids=fallback_eligible_study_uids,
            fallback_held_out_study_uids=fallback_held_out_study_uids or (),
            fallback_alpha=args.fallback_alpha,
            evidence_ticket=args.evidence_ticket,
            recover_damaged_inputs=args.recover_damaged_inputs,
            require_slurm=args.require_slurm,
            allow_local_smoke=args.allow_local_smoke,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"real inference failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


__all__ = [
    "INTEGRATION_MANIFEST_SCHEMA_VERSION",
    "INTEGRATION_EVIDENCE_SCHEMA_VERSION",
    "DEFAULT_REAL_GEOMETRY_POLICY",
    "DEFAULT_REAL_DECODER_POLICY",
    "DEFAULT_REAL_PREPROCESS_CONFIG",
    "RealIntegrationError",
    "RealInferenceError",
    "KaggleCompetitionSource",
    "IntegrationSeries",
    "IntegrationStudy",
    "IntegrationManifest",
    "SeriesLoadFailure",
    "LoadedSeries",
    "LoadedStudy",
    "write_integration_manifest",
    "load_integration_manifest",
    "build_integration_manifest",
    "load_real_study",
    "iter_real_studies",
    "run_real_inference",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
