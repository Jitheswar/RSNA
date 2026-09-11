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
import sys
import time
import zipfile
import zlib
from dataclasses import dataclass
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
from .geometry import GeometryPolicy
from .inference import InferenceEngine, StudyPrediction
from .model import HierarchicalStudyModel, ModelConfig
from .preprocessing import (
    PREPROCESS_SCHEMA_VERSION,
    PackingConfig,
    PreprocessConfig,
    SeriesFailure,
)
from .submission import write_submission
from .series_policy import has_non_diagnostic_token
from .targets import NUM_TARGETS


INTEGRATION_MANIFEST_SCHEMA_VERSION = "rsna-knee-real-integration-v1"
INTEGRATION_EVIDENCE_SCHEMA_VERSION = "rsna-knee-real-integration-evidence-v1"
INVENTORY_SCHEMA_VERSION = "rsna-knee-inventory-v1"
SELECTION_SCHEMA_VERSION = "rsna-knee-extraction-selection-v1"
DEFAULT_REAL_GEOMETRY_POLICY = GeometryPolicy(minimum_usable_slices=3)
DEFAULT_REAL_DECODER_POLICY = DecoderPolicy(require_modality=True)
DEFAULT_REAL_PREPROCESS_CONFIG = PreprocessConfig(
    geometry_policy=DEFAULT_REAL_GEOMETRY_POLICY,
)
class RealIntegrationError(ValueError):
    """Raised when a private manifest or real-input admission is unsafe."""


class RealInferenceError(RuntimeError):
    """Raised for a global integration or model failure."""


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
    schema_version: str = INTEGRATION_MANIFEST_SCHEMA_VERSION
    manifest_checksum: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != INTEGRATION_MANIFEST_SCHEMA_VERSION:
            raise RealIntegrationError(f"unexpected integration manifest schema {self.schema_version!r}")
        if not self.source_version.strip() or not self.archive_fingerprint.strip():
            raise RealIntegrationError("integration manifest requires source identity")
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
    )
    write_integration_manifest(manifest, output_path)
    return load_integration_manifest(output_path)


class _SourceReader:
    """Read manifest entries from one verified ZIP or an explicit local fixture."""

    def __init__(self, source: Path, *, allow_unverified_directory: bool = False) -> None:
        self.source = source
        self.allow_unverified_directory = allow_unverified_directory
        self.archive: zipfile.ZipFile | None = None
        self.root: Path | None = None

    def __enter__(self) -> "_SourceReader":
        if self.source.is_dir():
            if not self.allow_unverified_directory:
                raise RealIntegrationError(
                    "real integration requires a source ZIP with archive identity; "
                    "unverified directories are allowed only for local fixture smoke tests"
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

    def decode(self, entry_name: str, policy: DecoderPolicy) -> DicomSlice:
        safe = safe_member_path(entry_name)
        if self.archive is not None:
            try:
                stream = self.archive.open(safe.as_posix(), "r")
            except (KeyError, OSError, zipfile.BadZipFile) as exc:
                raise DamagedDicomError(f"source entry is missing: {entry_name}") from exc
            try:
                with stream:
                    decoded = decode_stream(cast(BinaryIO, stream), filename=entry_name, policy=policy)
                    while stream.read(1024 * 1024):
                        pass
                    return decoded
            except (OSError, RuntimeError, zipfile.BadZipFile, zlib.error) as exc:
                raise DamagedDicomError(f"archive integrity failed for {entry_name}") from exc
        if self.root is None:
            raise RealIntegrationError("source reader was used outside its context")
        target = (self.root / Path(*safe.parts)).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise RealIntegrationError(f"source entry escapes the source root: {entry_name}") from exc
        try:
            with target.open("rb") as stream:
                return decode_stream(stream, filename=entry_name, policy=policy)
        except (FileNotFoundError, OSError) as exc:
            raise DamagedDicomError(f"source entry is missing: {entry_name}") from exc


def _failure_reason(error: DicomDecodeError) -> str:
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
                )
            )
            continue
        decoded.append(item)

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
        source_entry_names=manifest_series.entries,
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
) -> LoadedStudy:
    """Load one manifest study; directories are reserved for local fixtures."""

    canonical = next(
        (item for item in manifest.studies if item.study_instance_uid == study_manifest.study_instance_uid),
        None,
    )
    if canonical is None:
        raise RealIntegrationError("study is not present in the integration manifest")
    if study_manifest != canonical:
        raise RealIntegrationError("study does not match the checksum-covered integration manifest")
    with _SourceReader(Path(source), allow_unverified_directory=allow_unverified_directory) as reader:
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
) -> Iterator[LoadedStudy]:
    """Yield canonical manifest studies while keeping a ZIP handle open once."""

    with _SourceReader(Path(source), allow_unverified_directory=allow_unverified_directory) as reader:
        for study_manifest in manifest.studies:
            yield _load_real_study_with_reader(
                manifest,
                study_manifest,
                reader,
                require_complete_series=require_complete_series,
            )


def _check_execution_context(*, require_slurm: bool, allow_local_smoke: bool) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if require_slurm and (not job_id or partition != "cluster1-cpu"):
        raise RealInferenceError("real integration must run in a cluster1-cpu Slurm allocation")
    if not job_id and not allow_local_smoke:
        raise RealInferenceError(
            "refusing to read real-DICOM inputs outside Slurm; "
            "use --allow-local-smoke only with a tiny fixture"
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
                        "reason": failure.reason,
                        "detail": failure.detail,
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
    }


def run_real_inference(
    manifest_path: str | Path,
    source_path: str | Path,
    output_path: str | Path,
    *,
    evidence_path: str | Path | None = None,
    private_evidence_path: str | Path | None = None,
    model: HierarchicalStudyModel | None = None,
    model_seed: int = 20260911,
    device: str | torch.device = "cpu",
    require_slurm: bool = False,
    allow_local_smoke: bool = False,
) -> dict[str, object]:
    """Run all manifest studies through the shared model and submission path."""

    _check_execution_context(require_slurm=require_slurm, allow_local_smoke=allow_local_smoke)
    thread_count = _configure_threads()
    manifest = load_integration_manifest(manifest_path)
    _verify_archive_source_identity(Path(source_path), manifest)
    if model is None:
        torch.manual_seed(model_seed)
        model = HierarchicalStudyModel(ModelConfig(image_size=manifest.preprocess_config.image_size))
    engine = InferenceEngine(
        model,
        manifest.preprocess_config,
        device=device,
        packing_config=manifest.packing_config,
    )

    started = time.perf_counter()
    predictions: list[StudyPrediction] = []
    private_study_evidence: list[dict[str, object]] = []
    manifest_series_count = 0
    accepted_series_count = 0
    complete_series_count = 0
    rejected_series_count = 0
    excluded_series_count = 0
    series_with_decode_failures = 0
    expected_slice_count = 0
    decoded_slice_count = 0
    prediction_rejected_series_count = 0
    warning_count = 0
    for loaded in iter_real_studies(
        manifest,
        source_path,
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

    scores = np.concatenate([item.scores for item in predictions])
    public_evidence: dict[str, object] = {
        "schema_version": INTEGRATION_EVIDENCE_SCHEMA_VERSION,
        "ticket": "05",
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
        },
        "timing": {"elapsed_seconds": elapsed_seconds},
        "cache_boundary": {
            "cache_configuration": manifest.preprocess_config.cache_config(),
            "sampling_is_post_cache": True,
            "model_architecture_is_not_in_cache_identity": True,
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
                "ticket": "05",
                "manifest_checksum": manifest.manifest_checksum,
                "source_version": manifest.source_version,
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
        evidence = run_real_inference(
            args.manifest,
            args.source,
            args.output,
            evidence_path=args.evidence,
            private_evidence_path=args.private_evidence,
            model_seed=args.model_seed,
            require_slurm=args.require_slurm,
            allow_local_smoke=args.allow_local_smoke,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ticket-05 real inference failed: {exc}", file=sys.stderr)
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
