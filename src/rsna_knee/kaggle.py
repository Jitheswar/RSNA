"""Prepare and record the approval-gated Kaggle delivery seam.

This module deliberately does not upload notebooks, models, or submissions.
It validates one immutable local candidate, creates an exact execution request,
and records externally supplied Kaggle evidence in a new immutable artifact.
The candidate itself is never edited when verification or promotion evidence is
written.
"""

from __future__ import annotations

import ast
import csv
import gc
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import io
import json
import math
import os
import platform
import re
import resource
import shutil
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.parser import Parser
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, cast

import numpy as np
import pydicom
import torch
from pydicom.dataset import Dataset

from .archive import atomic_json_write
from .candidate import CandidateBundle, CandidateError, _load_candidate_models, load_candidate
from .dicom import (
    DecoderInfrastructureError,
    DecoderPolicy,
    DicomDecodeError,
    decode_stream,
)
from .domain import SeriesRecord, StudyRecord
from .geometry import GeometryPolicy
from .inference import InferenceEngine, StudyPrediction
from .preprocessing import PackingConfig, PreprocessConfig, SeriesFailure, pack_study, prepare_study
from .submission import SubmissionError, validate_submission_csv, write_submission
from .targets import NUM_TARGETS, SUBMISSION_COLUMNS


DEFAULT_KAGGLE_ADAPTER_PATH = Path("src/rsna_knee/kaggle.py")


KAGGLE_DELIVERY_SCHEMA_VERSION = "rsna-knee-kaggle-delivery-v1"
KAGGLE_APPROVAL_SCHEMA_VERSION = "rsna-knee-kaggle-owner-approval-v1"
KAGGLE_EXECUTION_SCHEMA_VERSION = "rsna-knee-kaggle-execution-v1"
KAGGLE_VERIFICATION_SCHEMA_VERSION = "rsna-knee-kaggle-verification-v1"
KAGGLE_PROMOTION_SCHEMA_VERSION = "rsna-knee-kaggle-promotion-v1"
KAGGLE_ENGINEERING_TARGET_SECONDS = 25_920
KAGGLE_HARD_LIMIT_SECONDS = 32_400
KAGGLE_TARGET_STUDIES = 1_300
KAGGLE_COMPETITION_SLUG = "rsna-knee-abnormality-detection"
KAGGLE_RUNTIME_STAGES_V1 = (
    "startup_setup",
    "imports",
    "model_metadata_loading",
    "dicom_read_decode",
    "preprocessing",
    "inference_ensembling",
    "csv_creation",
)
KAGGLE_RUNTIME_STAGES = (*KAGGLE_RUNTIME_STAGES_V1, "study_iteration_overhead")
KAGGLE_PROFILE_SCHEMA_VERSION = "rsna-knee-kaggle-profile-v2"
KAGGLE_RUNTIME_STAGES_BY_PROFILE = {
    "rsna-knee-kaggle-profile-v1": KAGGLE_RUNTIME_STAGES_V1,
    KAGGLE_PROFILE_SCHEMA_VERSION: KAGGLE_RUNTIME_STAGES,
}
KAGGLE_PREDICTION_CHECKPOINT_SCHEMA_VERSION = "rsna-knee-kaggle-prediction-checkpoint-v1"
OWNER_APPROVAL_SCOPE = "ticket-12-exact-bundle-upload-and-execution"
TICKET_24_APPROVAL_SCOPE = "ticket-24-hidden-test-delivery"
OWNER_APPROVAL_SCOPES = frozenset({OWNER_APPROVAL_SCOPE, TICKET_24_APPROVAL_SCOPE})

_NETWORK_TEXT_TOKENS = (
    "http://",
    "https://",
    "requests.",
    "urllib.",
    "pip install",
    "conda install",
    "git clone",
    "wget ",
    "curl ",
)
_NETWORK_IMPORTS = frozenset({"requests", "urllib", "urllib3", "httpx", "wget", "curl", "socket"})
_PRIVATE_METADATA_KEYS = frozenset(
    {
        "studyinstanceuid",
        "studyinstanceuids",
        "study_instance_uid",
        "study_instance_uids",
        "seriesinstanceuid",
        "seriesinstanceuids",
        "series_instance_uid",
        "series_instance_uids",
        "sopinstanceuid",
        "sopinstanceuids",
        "sop_instance_uid",
        "sop_instance_uids",
        "sourceentryname",
        "sourceentrynames",
        "source_entry_name",
        "source_entry_names",
        "heldoutstudyuid",
        "heldoutstudyuids",
        "held_out_study_uid",
        "held_out_study_uids",
    }
)


class KaggleDeliveryError(ValueError):
    """Raised when a Kaggle request, execution, or promotion is unsafe."""


@dataclass(frozen=True)
class KaggleDeliveryRequest:
    """A checksum-verified owner-approval request for one exact bundle."""

    root: Path
    request_checksum: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class _ValidatedKaggleInputs:
    """The one-process integrity proof produced by :func:`discover_kaggle_inputs`."""

    request: KaggleDeliveryRequest
    candidate_bundle: CandidateBundle
    candidate_reference: dict[str, object]
    approval_sha256: str
    delivery_notebook_sha256: str
    delivery_adapter_sha256: str
    runtime_wheel_audits: tuple[Mapping[str, object], ...]
    candidate_load_hash_verification_seconds: float


@dataclass(frozen=True)
class KaggleVerification:
    """A checksum-verified record of one accepted Kaggle execution."""

    root: Path
    verification_checksum: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class KagglePromotion:
    """A checksum-verified promotion pointer, separate from the candidate."""

    root: Path
    promotion_checksum: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class KaggleInferenceResult:
    """The offline notebook adapter's output and identifier-free evidence."""

    output_path: Path
    evidence: Mapping[str, object]
    private_evidence_path: Path | None = None


@dataclass(frozen=True)
class KaggleInputLayout:
    """Validated paths for one namespaced Kaggle notebook input mount."""

    input_root: Path
    delivery_root: Path
    candidate_root: Path
    candidate_code_root: Path
    code_root: Path
    request_path: Path
    approval_path: Path
    competition_root: Path
    delivery_notebook_path: Path
    delivery_adapter_path: Path
    runtime_wheel_path: Path | None
    runtime_wheel_sha256: str | None
    request_checksum: str
    candidate_manifest_checksum: str
    runtime_wheel_paths: tuple[Path, ...] = ()
    runtime_wheel_sha256s: tuple[str, ...] = ()
    workload_kind: str = "hidden_test"
    input_manifest_sha256: str | None = None
    validation: _ValidatedKaggleInputs | None = None


@dataclass
class _KaggleDicomProfile:
    """Wall-clock and access counters for one directory-backed Kaggle run."""

    archive_index_discovery_seconds: float = 0.0
    dicom_read_decode_seconds: float = 0.0
    pixel_decode_seconds: float = 0.0
    index_scan_count: int = 0
    indexed_file_count: int = 0
    dicom_file_open_count: int = 0
    dicom_file_reopen_count: int = 0
    decode_call_count: int = 0
    expected_slice_count: int = 0
    decoded_slice_count: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)
    rss_after_study_decode_bytes: list[int] = field(default_factory=list)
    rss_after_study_prediction_bytes: list[int] = field(default_factory=list)
    rss_after_study_release_bytes: list[int] = field(default_factory=list)
    stage_rss_bytes: dict[str, int | None] = field(default_factory=dict)
    max_active_studies: int = 1

    @property
    def dicom_read_parse_seconds(self) -> float:
        return max(0.0, self.dicom_read_decode_seconds - self.pixel_decode_seconds)

    def public_dict(self) -> dict[str, object]:
        return {
            "archive_index_discovery_seconds": self.archive_index_discovery_seconds,
            "dicom_read_decode_seconds": self.dicom_read_decode_seconds,
            "dicom_read_parse_seconds": self.dicom_read_parse_seconds,
            "pixel_decode_seconds": self.pixel_decode_seconds,
            "index_scan_count": self.index_scan_count,
            "indexed_file_count": self.indexed_file_count,
            "dicom_file_open_count": self.dicom_file_open_count,
            "dicom_file_reopen_count": self.dicom_file_reopen_count,
            "decode_call_count": self.decode_call_count,
            "expected_slice_count": self.expected_slice_count,
            "decoded_slice_count": self.decoded_slice_count,
            "failure_reasons": dict(sorted(self.failure_reasons.items())),
            "max_active_studies": self.max_active_studies,
            "source_kind": "directory",
            "archive_access_mode": "not_applicable_directory_input",
            "dicom_access_mode": "serial_one_open_per_slice",
        }


@contextmanager
def _timed_pixel_decoding(profile: _KaggleDicomProfile) -> Any:
    """Time the frozen decoder's lazy pydicom pixel materialization."""

    original_property = cast(property, getattr(Dataset, "pixel_array"))
    getter = cast(Callable[[Dataset], Any] | None, original_property.fget)
    if getter is None:
        yield
        return

    def timed_pixel_array(dataset: Dataset) -> Any:
        started = time.perf_counter()
        try:
            return getter(dataset)
        finally:
            profile.pixel_decode_seconds += time.perf_counter() - started

    setattr(Dataset, "pixel_array", property(
        timed_pixel_array,
        original_property.fset,
        original_property.fdel,
        original_property.__doc__,
    ))
    try:
        yield
    finally:
        setattr(Dataset, "pixel_array", original_property)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    """Return stable summary statistics for one measured workload dimension."""

    if not values:
        raise KaggleDeliveryError("cannot summarize an empty workload dimension")
    ordered = sorted(int(value) for value in values)
    return {
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "average": sum(ordered) / len(ordered),
        "median": ordered[len(ordered) // 2],
    }


def _percentile_threshold(values: Sequence[int], percentile: float) -> int:
    """Use the nearest-rank threshold for a deterministic long-tail rule."""

    if not values or not 0.0 < percentile <= 1.0:
        raise KaggleDeliveryError("invalid workload percentile threshold")
    ordered = sorted(int(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _workload_profile(series_per_study: Sequence[int], slices_per_study: Sequence[int]) -> dict[str, object]:
    """Describe study/series/slice distribution without exposing identifiers."""

    if len(series_per_study) != len(slices_per_study) or not series_per_study:
        raise KaggleDeliveryError("workload dimensions do not cover the same studies")
    series_threshold = _percentile_threshold(series_per_study, 0.90)
    slice_threshold = _percentile_threshold(slices_per_study, 0.90)
    long_tail_count = sum(
        series >= series_threshold or slices >= slice_threshold
        for series, slices in zip(series_per_study, slices_per_study)
    )
    return {
        "series_per_study": _distribution(series_per_study),
        "slices_per_study": _distribution(slices_per_study),
        "long_tail": {
            "definition": "series or expected slices at or above the nearest-rank 90th percentile",
            "percentile": 0.90,
            "series_threshold": series_threshold,
            "slice_threshold": slice_threshold,
            "study_count": long_tail_count,
        },
    }


def _directory_size_bytes(root: Path) -> int:
    """Measure a mounted delivery tree without reading its file contents."""

    if not root.is_dir() or root.is_symlink():
        raise KaggleDeliveryError(f"cannot measure a non-directory delivery root: {root}")
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise KaggleDeliveryError(f"delivery root contains a symlink: {path}")
        if path.is_file():
            total += path.stat().st_size
    return total


def _peak_rss_bytes() -> int:
    """Return the process peak resident set size in bytes on supported hosts."""

    usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage if platform.system() == "Darwin" else usage * 1024


def _fixed_plus_scaling_projection(
    *,
    whole_seconds: float,
    study_count: int,
    target_studies: int,
    stages: Mapping[str, float],
) -> dict[str, object]:
    """Project fixed setup plus measured workload-dependent stage cost."""

    if whole_seconds <= 0.0 or study_count <= 0 or target_studies <= 0:
        raise KaggleDeliveryError("runtime projection inputs must be positive")
    fixed_stage_names = (
        "startup_setup",
        "imports",
        "input_discovery",
        "adapter_preflight",
        "model_metadata_loading",
    )
    scaling_stage_names = (
        "dicom_read_decode",
        "preprocessing",
        "inference_ensembling",
        "study_iteration_overhead",
        "csv_creation",
    )
    fixed_observed = sum(max(0.0, float(stages.get(name, 0.0))) for name in fixed_stage_names)
    scaling_observed = sum(max(0.0, float(stages.get(name, 0.0))) for name in scaling_stage_names)
    unattributed_observed = max(0.0, whole_seconds - fixed_observed - scaling_observed)
    fixed_total = fixed_observed + unattributed_observed
    projected_scaling = scaling_observed * target_studies / study_count
    projected = fixed_total + projected_scaling
    return {
        "target_studies": target_studies,
        "projected_seconds_at_target": projected,
        "method": "fixed_plus_measured_scaling",
        "fixed_seconds_observed": fixed_total,
        "fixed_stage_seconds_observed": fixed_observed,
        "unattributed_seconds_counted_as_fixed": unattributed_observed,
        "scaling_seconds_observed": scaling_observed,
        "scaling_seconds_per_study_observed": scaling_observed / study_count,
        "projected_scaling_seconds_at_target": projected_scaling,
        "engineering_target_seconds": KAGGLE_ENGINEERING_TARGET_SECONDS,
        "hard_limit_seconds": KAGGLE_HARD_LIMIT_SECONDS,
        "engineering_margin_seconds": KAGGLE_ENGINEERING_TARGET_SECONDS - projected,
        "hard_limit_margin_seconds": KAGGLE_HARD_LIMIT_SECONDS - projected,
    }


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise KaggleDeliveryError(f"artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise KaggleDeliveryError(f"could not read artifact {path}") from exc
    return digest.hexdigest()


def _normalize_package_name(value: object, field: str) -> str:
    return re.sub(r"[-_.]+", "-", _nonempty(value, field).casefold())


def audit_kaggle_runtime_wheel(
    path: str | Path,
    *,
    expected_package: str | None = "pydicom",
    expected_version: str | None = "3.0.1",
) -> dict[str, object]:
    """Validate one local wheel before an offline Kaggle installation."""

    wheel_path = Path(path)
    if not wheel_path.name.endswith(".whl"):
        raise KaggleDeliveryError(f"offline runtime artifact is not a wheel: {wheel_path}")
    try:
        wheel_bytes = wheel_path.read_bytes()
        with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as archive:
            members = tuple(archive.namelist())
            metadata_members = tuple(
                name for name in members if name.endswith(".dist-info/METADATA") and not name.startswith("/")
            )
            wheel_members = tuple(
                name for name in members if name.endswith(".dist-info/WHEEL") and not name.startswith("/")
            )
            if len(metadata_members) != 1 or len(wheel_members) != 1:
                raise KaggleDeliveryError("offline runtime wheel must contain one dist-info METADATA and WHEEL")
            metadata = Parser().parsestr(archive.read(metadata_members[0]).decode("utf-8"))
            wheel_metadata = archive.read(wheel_members[0]).decode("utf-8")
        wheel_sha256 = hashlib.sha256(wheel_bytes).hexdigest()
    except (OSError, UnicodeError, zipfile.BadZipFile, KeyError) as exc:
        raise KaggleDeliveryError(f"could not inspect offline runtime wheel: {wheel_path}") from exc
    package = _nonempty(metadata.get("Name"), "offline wheel package")
    version = _nonempty(metadata.get("Version"), "offline wheel version")
    if expected_package is not None:
        expected_package_name = _normalize_package_name(expected_package, "expected wheel package")
        if _normalize_package_name(package, "offline wheel package") != expected_package_name:
            raise KaggleDeliveryError(
                f"offline runtime wheel package {package!r} does not match {expected_package_name!r}"
            )
    if expected_version is not None:
        expected_version_text = _nonempty(expected_version, "expected wheel version")
        if version != expected_version_text:
            raise KaggleDeliveryError(
                f"offline runtime wheel version {version!r} does not match {expected_version_text!r}"
            )
    tags = tuple(
        line.split(":", 1)[1].strip()
        for line in wheel_metadata.splitlines()
        if line.startswith("Tag:") and ":" in line
    )
    if not tags:
        raise KaggleDeliveryError("offline runtime wheel does not declare a compatibility tag")
    return {
        "filename": wheel_path.name,
        "sha256": wheel_sha256,
        "package": package,
        "version": version,
        "tags": list(tags),
    }


def _runtime_wheel_references(value: object) -> tuple[Mapping[str, object], ...]:
    """Normalize legacy and current request wheelhouse references."""

    runtime = _mapping(value, "request delivery runtime")
    if "offline_wheels" in runtime:
        raw_wheels = runtime.get("offline_wheels")
        if not isinstance(raw_wheels, list) or not raw_wheels:
            raise KaggleDeliveryError("request offline runtime wheels must be a non-empty list")
        return tuple(
            _mapping(item, f"request offline runtime wheel {index}")
            for index, item in enumerate(raw_wheels)
        )
    if "offline_wheel" in runtime:
        return (_mapping(runtime.get("offline_wheel"), "request offline runtime wheel"),)
    raise KaggleDeliveryError("request delivery runtime has no offline wheel references")


def _runtime_wheel_paths(
    singular: str | Path | None,
    plural: Sequence[str | Path] | None,
) -> tuple[Path, ...]:
    """Normalize the versioned API's singular and wheelhouse arguments."""

    if singular is not None and plural is not None:
        raise KaggleDeliveryError("supply either one runtime wheel or a runtime wheelhouse, not both")
    if plural is not None:
        if isinstance(plural, (str, bytes, Path)):
            raise KaggleDeliveryError("runtime wheelhouse must be a sequence of wheel paths")
        paths = tuple(Path(item) for item in plural)
    elif singular is not None:
        paths = (Path(singular),)
    else:
        paths = ()
    if len({str(path) for path in paths}) != len(paths):
        raise KaggleDeliveryError("runtime wheelhouse contains duplicate paths")
    return paths


def _audit_runtime_wheels_for_pins(
    paths: Sequence[Path],
    runtime_pins: Mapping[str, object],
    *,
    require_all_pins: bool,
) -> tuple[dict[str, object], ...]:
    """Audit a local wheel set and bind each wheel to one frozen package pin."""

    expected_pins = {
        _normalize_package_name(package, "runtime package"): _nonempty(version, f"runtime pin {package}")
        for package, version in runtime_pins.items()
    }
    if not expected_pins:
        raise KaggleDeliveryError("candidate runtime pins are empty")
    audited: list[dict[str, object]] = []
    seen_packages: set[str] = set()
    for path in paths:
        wheel = audit_kaggle_runtime_wheel(path, expected_package=None, expected_version=None)
        package = _normalize_package_name(wheel["package"], "offline wheel package")
        if package not in expected_pins:
            raise KaggleDeliveryError(f"offline runtime wheel package {package!r} is not frozen by the candidate")
        if package in seen_packages:
            raise KaggleDeliveryError(f"runtime wheelhouse contains duplicate package {package!r}")
        expected_version = expected_pins[package]
        if wheel["version"] != expected_version:
            raise KaggleDeliveryError(
                f"offline runtime wheel {package!r} is {wheel['version']!r}, expected frozen {expected_version!r}"
            )
        seen_packages.add(package)
        audited.append(wheel)
    if require_all_pins and seen_packages != set(expected_pins):
        missing = sorted(set(expected_pins) - seen_packages)
        raise KaggleDeliveryError("runtime wheelhouse does not cover every frozen pin: " + ", ".join(missing))
    return tuple(sorted(audited, key=lambda item: (str(item["package"]).casefold(), str(item["filename"]))))


def validate_kaggle_runtime_pins(
    expected_pins: Mapping[str, object],
    *,
    expected_python_version: str | None = None,
) -> dict[str, object]:
    """Check every locked package and the Python runtime before DICOM work."""

    normalized_expected = {
        _normalize_package_name(package, "runtime package"): _nonempty(version, f"runtime pin {package}")
        for package, version in expected_pins.items()
    }
    if not normalized_expected:
        raise KaggleDeliveryError("candidate runtime pins are empty")
    metadata_versions: dict[str, str] = {}
    imported_versions: dict[str, str] = {}
    importable_modules = {"numpy": "numpy", "pydicom": "pydicom", "torch": "torch"}
    importlib.invalidate_caches()
    for package, expected_version in sorted(normalized_expected.items()):
        try:
            actual_metadata_version = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError as exc:
            raise KaggleDeliveryError(f"required runtime package is not installed: {package}") from exc
        metadata_versions[package] = actual_metadata_version
        if actual_metadata_version != expected_version:
            raise KaggleDeliveryError(
                f"runtime package {package!r} is {actual_metadata_version!r}, expected pinned {expected_version!r}"
            )
        module_name = importable_modules.get(package)
        if module_name is not None:
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                raise KaggleDeliveryError(f"required runtime module is not importable: {package}") from exc
            actual_module_version = _nonempty(
                getattr(module, "__version__", None),
                f"runtime module {package} version",
            )
            imported_versions[package] = actual_module_version
            if actual_module_version != expected_version:
                raise KaggleDeliveryError(
                    f"imported runtime module {package!r} is {actual_module_version!r}, "
                    f"expected pinned {expected_version!r}"
                )
    python_version = platform.python_version()
    if expected_python_version is not None and python_version != expected_python_version:
        raise KaggleDeliveryError(
            f"runtime Python {python_version!r} does not match pinned {expected_python_version!r}"
        )
    return {
        "python_version": python_version,
        "python_pin": expected_python_version,
        "packages": metadata_versions,
        "imported_packages": imported_versions,
    }


def _read_json(path: Path, description: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise KaggleDeliveryError(f"could not read {description} {path}") from exc
    if not isinstance(value, Mapping):
        raise KaggleDeliveryError(f"{description} must be a JSON object")
    return cast(Mapping[str, object], value)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise KaggleDeliveryError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _nonempty(value: object, field: str) -> str:
    if value is None:
        raise KaggleDeliveryError(f"{field} must be non-empty")
    text = str(value).strip()
    if not text:
        raise KaggleDeliveryError(f"{field} must be non-empty")
    return text


def _safe_identifier(value: object, field: str) -> str:
    text = _nonempty(value, field)
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise KaggleDeliveryError(f"{field} must be path-safe")
    return text


def _required_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise KaggleDeliveryError(f"{field} must be boolean")
    return value


def _required_int(value: object, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise KaggleDeliveryError(f"{field} must be an integer")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise KaggleDeliveryError(f"{field} must be an integer") from exc
    if parsed < (1 if positive else 0):
        qualifier = "positive " if positive else "non-negative "
        raise KaggleDeliveryError(f"{field} must be a {qualifier}integer")
    return parsed


def _required_float(value: object, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise KaggleDeliveryError(f"{field} must be a number")
    try:
        parsed = float(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise KaggleDeliveryError(f"{field} must be a number") from exc
    if not math.isfinite(parsed) or (positive and parsed <= 0.0) or (not positive and parsed < 0.0):
        raise KaggleDeliveryError(f"{field} must be a finite {'positive ' if positive else 'non-negative '}number")
    return parsed


def _optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _required_int(value, field)


def _private_metadata_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9_]", "", str(key).casefold())
            if normalized in _PRIVATE_METADATA_KEYS:
                found.add(str(key))
            found.update(_private_metadata_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_private_metadata_keys(child))
    return found


def _set_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            if path.is_dir():
                path.chmod(0o555)
            else:
                path.chmod(0o444)
        except OSError as exc:
            raise KaggleDeliveryError(f"could not make artifact immutable: {path}") from exc
    root.chmod(0o555)


def _publish_directory(output_root: Path, writer: Callable[[Path], None]) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite protected Kaggle artifact {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.partial-", dir=output_root.parent))
    try:
        writer(staging)
        _set_read_only(staging)
        try:
            os.rename(staging, output_root)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite protected Kaggle artifact {output_root}") from exc
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _ensure_outside(path: Path, protected: Sequence[Path], field: str) -> None:
    resolved = path.resolve()
    for item in protected:
        protected_resolved = item.resolve()
        try:
            resolved.relative_to(protected_resolved)
        except ValueError:
            try:
                protected_resolved.relative_to(resolved)
            except ValueError:
                continue
        raise KaggleDeliveryError(f"{field} must remain separate from protected artifact {item}")


def _notebook_audit(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise KaggleDeliveryError(f"delivery notebook is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise KaggleDeliveryError(f"delivery notebook is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping) or payload.get("nbformat") != 4:
        raise KaggleDeliveryError("delivery notebook must use nbformat 4")
    text = path.read_text(encoding="utf-8").casefold()
    if "candidate" not in text or "submission.csv" not in text:
        raise KaggleDeliveryError("delivery notebook must name the candidate and submission.csv")
    found = tuple(token for token in _NETWORK_TEXT_TOKENS if token in text)
    if found:
        raise KaggleDeliveryError("delivery notebook contains runtime network instructions")
    private_keys = _private_metadata_keys(payload)
    if private_keys:
        raise KaggleDeliveryError(
            "delivery notebook contains identifier-bearing fields: " + ", ".join(sorted(private_keys))
        )
    return {
        "filename": path.name,
        "sha256": _sha256_file(path),
        "nbformat": 4,
        "network_instructions": False,
        "private_metadata_keys": [],
    }


def _adapter_audit(path: Path) -> dict[str, object]:
    """Audit the path/orchestration adapter shipped beside the candidate."""

    if not path.is_file() or path.is_symlink():
        raise KaggleDeliveryError(f"delivery adapter is not a regular file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise KaggleDeliveryError(f"could not read delivery adapter {path}") from exc
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        raise KaggleDeliveryError("delivery adapter is not parseable Python") from exc
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    forbidden = sorted(imports & _NETWORK_IMPORTS)
    if forbidden:
        raise KaggleDeliveryError(f"delivery adapter contains network imports: {forbidden}")
    if "run_kaggle_baseline" not in text or "submission.csv" not in text:
        raise KaggleDeliveryError("delivery adapter does not expose the Kaggle inference path")
    return {
        "filename": path.name,
        "sha256": _sha256_file(path),
        "network_instructions": False,
    }


def _artifact_records(bundle: CandidateBundle) -> dict[str, Mapping[str, object]]:
    records_value = bundle.manifest.get("artifacts")
    if not isinstance(records_value, list):
        raise KaggleDeliveryError("candidate artifact records are missing")
    records: dict[str, Mapping[str, object]] = {}
    for item_value in records_value:
        item = _mapping(item_value, "candidate artifact")
        path = _nonempty(item.get("path"), "candidate artifact path")
        records[path] = item
    return records


def _candidate_reference(bundle: CandidateBundle) -> dict[str, object]:
    records = _artifact_records(bundle)
    members_value = _mapping(bundle.manifest.get("model"), "candidate model").get("members")
    if not isinstance(members_value, list) or not members_value:
        raise KaggleDeliveryError("candidate has no model checkpoints")
    model_checkpoints: list[dict[str, object]] = []
    model_size_bytes = 0
    for item_value in members_value:
        item = _mapping(item_value, "candidate model member")
        path = _nonempty(item.get("path"), "candidate model path")
        artifact = records.get(path)
        if artifact is None or artifact.get("kind") != "model_checkpoint":
            raise KaggleDeliveryError(f"candidate model is not checksum-covered: {path}")
        actual_sha = _sha256_file(bundle.root / path)
        expected_sha = _nonempty(artifact.get("sha256"), f"candidate model checksum {path}")
        if actual_sha != expected_sha or str(item.get("sha256", expected_sha)) != expected_sha:
            raise KaggleDeliveryError(f"candidate model checksum is inconsistent: {path}")
        model_size_bytes += _required_int(artifact.get("size_bytes"), f"candidate model size {path}", positive=True)
        model_checkpoints.append(
            {
                "fold": _required_int(item.get("fold"), f"candidate model fold {path}"),
                "path": path,
                "sha256": expected_sha,
                "checkpoint_kind": _nonempty(item.get("checkpoint_kind"), f"candidate checkpoint kind {path}"),
            }
        )
    promotion = _mapping(bundle.manifest.get("promotion"), "candidate promotion")
    if promotion.get("status") != "locally_validated_only" or promotion.get("kaggle_verified") is not False:
        raise KaggleDeliveryError("Kaggle preparation requires a locally validated, not Kaggle-verified candidate")
    provenance = _mapping(bundle.manifest.get("provenance"), "candidate provenance")
    environment = _mapping(bundle.manifest.get("environment"), "candidate environment")
    cluster_environment = _mapping(environment.get("cluster"), "candidate cluster environment")
    offline_environment = _mapping(environment.get("offline"), "candidate offline environment")
    if environment.get("runtime_network_download") is not False:
        raise KaggleDeliveryError("candidate environment permits runtime downloads")
    cluster_pins = _mapping(cluster_environment.get("runtime_pins"), "candidate cluster runtime pins")
    offline_pins = _mapping(offline_environment.get("runtime_pins"), "candidate offline runtime pins")
    if dict(cluster_pins) != dict(offline_pins):
        raise KaggleDeliveryError("candidate cluster and offline runtime pins differ")
    code_revision = _nonempty(provenance.get("code_revision"), "candidate code revision")
    code_digest = _nonempty(environment.get("shared_code_digest"), "candidate shared code digest")
    return {
        "candidate_id": bundle.candidate_id,
        "manifest_checksum": bundle.manifest_checksum,
        "code_revision": code_revision,
        "shared_code_digest": code_digest,
        "inference_configuration_sha256": _json_sha256(bundle.inference_config),
        "model_checkpoints": model_checkpoints,
        "model_size_bytes": model_size_bytes,
        "candidate_bundle_size_bytes": sum(
            _required_int(item.get("size_bytes"), f"candidate artifact size {path}", positive=True)
            for path, item in records.items()
        ),
        "environment": {
            "cluster_requirements_sha256": _nonempty(
                cluster_environment.get("sha256"), "candidate cluster requirements sha256"
            ),
            "offline_requirements_sha256": _nonempty(
                offline_environment.get("sha256"), "candidate offline requirements sha256"
            ),
            "runtime_pins": dict(offline_pins),
            "runtime_network_download": False,
        },
    }


def _local_submission_audit(path: Path, expected_study_uids: Sequence[str] | None) -> dict[str, object]:
    if expected_study_uids is not None:
        try:
            validation = validate_submission_csv(path, expected_study_uids)
        except (OSError, SubmissionError) as exc:
            raise KaggleDeliveryError(f"local submission failed exact validation: {path}") from exc
        return {**validation, "sha256": _sha256_file(path), "available": True}
    if not path.is_file() or path.is_symlink():
        raise KaggleDeliveryError(f"local submission is not a regular file: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise KaggleDeliveryError(f"could not read local submission {path}") from exc
    if not rows or rows[0] != list(SUBMISSION_COLUMNS):
        raise KaggleDeliveryError("local submission header does not match the exact target order")
    observed_uids: list[str] = []
    for row in rows[1:]:
        if len(row) != NUM_TARGETS + 1 or not row[0]:
            raise KaggleDeliveryError("local submission contains an invalid row")
        observed_uids.append(row[0])
        try:
            scores = [float(value) for value in row[1:]]
        except (TypeError, ValueError, OverflowError) as exc:
            raise KaggleDeliveryError("local submission contains a non-numeric score") from exc
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in scores):
            raise KaggleDeliveryError("local submission contains a non-finite or unbounded score")
    if len(set(observed_uids)) != len(observed_uids):
        raise KaggleDeliveryError("local submission contains duplicate study rows")
    return {
        "row_count": len(rows) - 1,
        "column_count": len(rows[0]),
        "exact_header": True,
        "exact_uid_order": False,
        "scores_finite": True,
        "scores_bounded": True,
        "sha256": _sha256_file(path),
        "available": True,
    }


def _local_evidence_audit(
    path: Path,
    candidate: Mapping[str, object],
    *,
    local_submission_path: Path | None,
    expected_study_uids: Sequence[str] | None,
) -> dict[str, object]:
    evidence = _read_json(path, "local candidate evidence")
    schema = str(evidence.get("schema_version", ""))
    if not schema.startswith("rsna-knee-real-integration-evidence-"):
        raise KaggleDeliveryError("local evidence has an unexpected schema")
    if evidence.get("status") != "completed_real_integration":
        raise KaggleDeliveryError("local evidence is not a completed real integration")
    if _private_metadata_keys(evidence):
        raise KaggleDeliveryError("local evidence contains identifier-bearing fields")
    evidence_candidate = _mapping(evidence.get("candidate"), "local evidence candidate")
    if evidence_candidate.get("candidate_manifest_checksum") != candidate.get("manifest_checksum"):
        raise KaggleDeliveryError("local evidence candidate manifest checksum does not match the candidate")

    counts = _mapping(evidence.get("counts"), "local evidence counts")
    input_studies = _required_int(counts.get("input_studies"), "local evidence input studies", positive=True)
    output_rows = _required_int(counts.get("output_rows"), "local evidence output rows", positive=True)
    if output_rows != input_studies:
        raise KaggleDeliveryError("local evidence output rows do not cover every measured study")
    if _required_int(counts.get("output_columns"), "local evidence output columns") != NUM_TARGETS + 1:
        raise KaggleDeliveryError("local evidence output does not have the thirteen-column contract")
    if counts.get("scores_finite") is not True or counts.get("scores_bounded") is not True:
        raise KaggleDeliveryError("local evidence does not prove finite bounded scores")
    submission_validation = _mapping(evidence.get("submission_validation"), "local submission validation")
    for field in ("exact_header", "exact_uid_order", "scores_finite", "scores_bounded"):
        if submission_validation.get(field) is not True:
            raise KaggleDeliveryError(f"local evidence does not prove submission {field}")
    if _required_int(submission_validation.get("row_count"), "local submission validation row count", positive=True) != input_studies:
        raise KaggleDeliveryError("local submission validation row count differs from local workload")
    if _required_int(submission_validation.get("column_count"), "local submission validation column count") != NUM_TARGETS + 1:
        raise KaggleDeliveryError("local submission validation column count is not thirteen")

    fallback_numerator = _required_int(
        counts.get("full_study_fallback_numerator"),
        "local fallback numerator",
    )
    fallback_denominator = _required_int(
        counts.get("full_study_fallback_denominator"),
        "local fallback denominator",
        positive=True,
    )
    if fallback_denominator != input_studies or fallback_numerator > fallback_denominator:
        raise KaggleDeliveryError("local fallback denominator does not cover the measured workload")
    fallback_rate = _required_float(counts.get("full_study_fallback_rate"), "local fallback rate")
    if not math.isclose(fallback_rate, fallback_numerator / fallback_denominator, rel_tol=1e-9, abs_tol=1e-12):
        raise KaggleDeliveryError("local fallback rate is inconsistent with its numerator and denominator")
    recovery = _mapping(evidence.get("recovery"), "local recovery evidence")
    recovery_fallback = _mapping(recovery.get("full_study_fallback"), "local recovery fallback")
    if (
        _required_int(recovery_fallback.get("numerator"), "recovery fallback numerator") != fallback_numerator
        or _required_int(recovery_fallback.get("denominator"), "recovery fallback denominator", positive=True)
        != fallback_denominator
    ):
        raise KaggleDeliveryError("local recovery fallback counts disagree with execution counts")
    if fallback_numerator:
        investigation = _mapping(recovery.get("fallback_investigation"), "local fallback investigation")
        if investigation.get("complete") is not True:
            raise KaggleDeliveryError("every local fallback case must be investigated before delivery")

    benchmark = _mapping(evidence.get("runtime_benchmark"), "local runtime benchmark")
    observed = _required_float(
        benchmark.get("whole_pipeline_seconds_observed"),
        "local observed whole-pipeline runtime",
        positive=True,
    )
    projected = _required_float(
        benchmark.get("projected_seconds_at_target"),
        "local projected runtime",
        positive=True,
    )
    if benchmark.get("engineering_target_seconds") != KAGGLE_ENGINEERING_TARGET_SECONDS:
        raise KaggleDeliveryError("local benchmark uses a different engineering target")
    if benchmark.get("hard_limit_seconds") != KAGGLE_HARD_LIMIT_SECONDS:
        raise KaggleDeliveryError("local benchmark uses a different hard runtime limit")
    if benchmark.get("within_engineering_target") is not (projected <= KAGGLE_ENGINEERING_TARGET_SECONDS):
        raise KaggleDeliveryError("local engineering-target result is inconsistent with its projection")
    if benchmark.get("within_hard_limit_projection") is not (projected <= KAGGLE_HARD_LIMIT_SECONDS):
        raise KaggleDeliveryError("local hard-limit result is inconsistent with its projection")
    if benchmark.get("hidden_execution_unknown") is not True:
        raise KaggleDeliveryError("local evidence must keep hidden execution explicitly unknown")
    if benchmark.get("full_scale_compliance_claim") is not False:
        raise KaggleDeliveryError("local evidence must not claim hidden-scale compliance")
    execution = _mapping(evidence.get("candidate_execution"), "local candidate execution")
    if execution.get("runtime_network_download") is not False or execution.get("reports_required") is not False:
        raise KaggleDeliveryError("local candidate execution is not offline and report-free")

    model_and_bundle_bytes = _optional_int(
        benchmark.get("model_and_bundle_bytes"),
        "local model and bundle size",
    )
    peak_vram_bytes = _optional_int(benchmark.get("peak_vram_bytes"), "local peak VRAM")

    local_submission: dict[str, object] = {"available": False}
    if local_submission_path is not None:
        local_submission = _local_submission_audit(local_submission_path, expected_study_uids)
        if local_submission["row_count"] != input_studies:
            raise KaggleDeliveryError("local submission row count differs from local evidence")
    return {
        "evidence_schema_version": schema,
        "evidence_sha256": _sha256_file(path),
        "measured_studies": input_studies,
        "measured_runtime_seconds": observed,
        "projected_runtime_seconds_at_1300": projected,
        "fallback": {
            "status": "measured",
            "numerator": fallback_numerator,
            "denominator": fallback_denominator,
            "rate": fallback_rate,
            "hidden_failure_rate_unknown": True,
        },
        "submission": {
            "row_count": input_studies,
            "column_count": NUM_TARGETS + 1,
            "exact_header": True,
            "exact_uid_order": True,
            "scores_finite": True,
            "scores_bounded": True,
            "local_artifact": local_submission,
        },
        "runtime": {
            "engineering_target_seconds": KAGGLE_ENGINEERING_TARGET_SECONDS,
            "hard_limit_seconds": KAGGLE_HARD_LIMIT_SECONDS,
            "within_engineering_target_projection": projected <= KAGGLE_ENGINEERING_TARGET_SECONDS,
            "within_hard_limit_projection": projected <= KAGGLE_HARD_LIMIT_SECONDS,
            "hidden_execution_unknown": True,
            "model_and_bundle_bytes": model_and_bundle_bytes,
            "peak_vram_bytes": peak_vram_bytes,
        },
        "recovery": {
            "save_reload_verified": recovery.get("save_reload_verified") is True,
            "interruption_recovery_verified": recovery.get("interruption_recovery_verified") is True,
            "rollback_seconds": _optional_int(recovery.get("rollback_seconds"), "local recovery rollback seconds"),
            "evidence_sha256": _sha256_file(path),
        },
    }


def _request_payload(root: Path) -> Mapping[str, object]:
    if not root.is_dir() or root.is_symlink():
        raise KaggleDeliveryError(f"Kaggle request root is not a directory: {root}")
    payload = _read_json(root / "request.json", "Kaggle delivery request")
    checksum = _nonempty(payload.get("request_checksum"), "request_checksum")
    without_checksum = dict(payload)
    without_checksum.pop("request_checksum", None)
    if _json_sha256(without_checksum) != checksum:
        raise KaggleDeliveryError("Kaggle delivery request checksum does not match its contents")
    if payload.get("schema_version") != KAGGLE_DELIVERY_SCHEMA_VERSION:
        raise KaggleDeliveryError("unsupported Kaggle delivery request schema")
    if payload.get("status") != "owner_approval_required":
        raise KaggleDeliveryError("Kaggle delivery request is not approval-gated")
    _request_approval_scope(payload)
    return payload


def _request_approval_scope(payload: Mapping[str, object]) -> str:
    approval = _mapping(payload.get("approval"), "request approval")
    scope = _nonempty(approval.get("scope"), "request approval scope")
    if scope not in OWNER_APPROVAL_SCOPES:
        raise KaggleDeliveryError("Kaggle request has an unsupported owner approval scope")
    return scope


def load_kaggle_delivery(
    root: str | Path,
    *,
    candidate_root: str | Path | None = None,
    delivery_notebook_path: str | Path | None = None,
    delivery_adapter_path: str | Path | None = None,
    delivery_runtime_wheel_path: str | Path | None = None,
    delivery_runtime_wheel_paths: Sequence[str | Path] | None = None,
) -> KaggleDeliveryRequest:
    """Load a request and optionally recheck the referenced local artifacts."""

    request_root = Path(root)
    payload = _request_payload(request_root)
    checksum = _nonempty(payload.get("request_checksum"), "request_checksum")
    if candidate_root is not None:
        bundle = load_candidate(candidate_root)
        candidate = _mapping(payload.get("candidate"), "request candidate")
        if candidate.get("candidate_id") != bundle.candidate_id or candidate.get("manifest_checksum") != bundle.manifest_checksum:
            raise KaggleDeliveryError("request does not reference the supplied candidate")
        if _candidate_reference(bundle) != dict(candidate):
            raise KaggleDeliveryError("request candidate reference differs from the supplied candidate")
    if delivery_notebook_path is not None:
        notebook = _notebook_audit(Path(delivery_notebook_path))
        expected = _mapping(payload.get("delivery_notebook"), "request delivery notebook")
        if notebook["sha256"] != expected.get("sha256"):
            raise KaggleDeliveryError("request notebook checksum differs from the supplied notebook")
    if delivery_adapter_path is not None:
        adapter = _adapter_audit(Path(delivery_adapter_path))
        expected = _mapping(payload.get("delivery_adapter"), "request delivery adapter")
        if adapter["sha256"] != expected.get("sha256"):
            raise KaggleDeliveryError("request adapter checksum differs from the supplied adapter")
    supplied_runtime_wheels = _runtime_wheel_paths(delivery_runtime_wheel_path, delivery_runtime_wheel_paths)
    if supplied_runtime_wheels:
        runtime_value = payload.get("delivery_runtime")
        if runtime_value is None:
            raise KaggleDeliveryError("supplied runtime wheels are not referenced by the request")
        expected_wheels = _runtime_wheel_references(runtime_value)
        if len(expected_wheels) != len(supplied_runtime_wheels):
            raise KaggleDeliveryError("supplied runtime wheel count differs from the request")
        for path, expected_wheel in zip(supplied_runtime_wheels, expected_wheels):
            expected_package = _nonempty(expected_wheel.get("package"), "request runtime wheel package")
            expected_version = _nonempty(expected_wheel.get("version"), "request runtime wheel version")
            wheel = audit_kaggle_runtime_wheel(
                path,
                expected_package=expected_package,
                expected_version=expected_version,
            )
            filename = _safe_identifier(expected_wheel.get("filename"), "request runtime wheel filename")
            relative_path = _nonempty(expected_wheel.get("relative_path"), "request runtime wheel relative path")
            if relative_path != f"runtime/{filename}":
                raise KaggleDeliveryError("request runtime wheel path is not the canonical runtime layout")
            if wheel["filename"] != filename or wheel["sha256"] != expected_wheel.get("sha256"):
                raise KaggleDeliveryError("request runtime wheel checksum differs from the supplied wheel")
    return KaggleDeliveryRequest(request_root.resolve(), checksum, payload)


def prepare_kaggle_delivery(
    candidate_root: str | Path,
    local_evidence_path: str | Path,
    output_root: str | Path,
    *,
    delivery_notebook_path: str | Path | None = None,
    delivery_adapter_path: str | Path = DEFAULT_KAGGLE_ADAPTER_PATH,
    delivery_runtime_wheel_path: str | Path | None = None,
    delivery_runtime_wheel_paths: Sequence[str | Path] | None = None,
    local_submission_path: str | Path | None = None,
    expected_study_uids: Sequence[str] | None = None,
    execution_input: Mapping[str, object] | None = None,
    approval_scope: str = OWNER_APPROVAL_SCOPE,
    request_id: str | None = None,
) -> KaggleDeliveryRequest:
    """Create one immutable, approval-required Kaggle execution request.

    Preparation is intentionally a local read/validate/write operation.  It
    never calls Kaggle and never changes the candidate or the local evidence.
    """

    bundle = load_candidate(candidate_root)
    candidate = _candidate_reference(bundle)
    candidate_root_path = Path(candidate_root).resolve()
    evidence_path = Path(local_evidence_path)
    notebook_path = (
        Path(delivery_notebook_path)
        if delivery_notebook_path is not None
        else candidate_root_path / "notebook" / "ticket11_offline_candidate.ipynb"
    )
    notebook = _notebook_audit(notebook_path)
    adapter_path = Path(delivery_adapter_path)
    adapter = _adapter_audit(adapter_path)
    candidate_environment = _mapping(candidate.get("environment"), "candidate environment")
    runtime_pins = _mapping(candidate_environment.get("runtime_pins"), "candidate runtime pins")
    supplied_runtime_wheels = _runtime_wheel_paths(delivery_runtime_wheel_path, delivery_runtime_wheel_paths)
    runtime_wheels = _audit_runtime_wheels_for_pins(
        supplied_runtime_wheels,
        runtime_pins,
        require_all_pins=delivery_runtime_wheel_paths is not None,
    ) if supplied_runtime_wheels else ()
    evidence = _local_evidence_audit(
        evidence_path,
        candidate,
        local_submission_path=(Path(local_submission_path) if local_submission_path is not None else None),
        expected_study_uids=expected_study_uids,
    )
    request_name = request_id or f"ticket-12-{candidate['candidate_id']}"
    request_name = _safe_identifier(request_name, "request_id")
    request_approval_scope = _nonempty(approval_scope, "owner approval scope")
    if request_approval_scope not in OWNER_APPROVAL_SCOPES:
        raise KaggleDeliveryError("owner approval scope is not supported")
    output = Path(output_root)
    _ensure_outside(output, (candidate_root_path,), "Kaggle request")
    payload: dict[str, object] = {
        "schema_version": KAGGLE_DELIVERY_SCHEMA_VERSION,
        "request_id": request_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "owner_approval_required",
        "candidate": candidate,
        "delivery_notebook": {
            "filename": notebook["filename"],
            "sha256": notebook["sha256"],
            "source": "explicit_delivery_adapter" if delivery_notebook_path is not None else "frozen_candidate",
            "network_instructions": False,
        },
        "delivery_adapter": {
            "filename": adapter["filename"],
            "sha256": adapter["sha256"],
            "source": "explicit_delivery_adapter",
            "network_instructions": False,
        },
        "delivery_layout": {
            "request": {"relative_path": "request.json"},
            "candidate": {
                "relative_path": "candidate/",
                "manifest_checksum": candidate["manifest_checksum"],
            },
            "notebook": {
                "relative_path": str(notebook["filename"]),
                "sha256": notebook["sha256"],
            },
            "adapter": {
                "relative_path": f"code/rsna_knee/{adapter['filename']}",
                "sha256": adapter["sha256"],
            },
            "offline_requirements": {
                "relative_path": "candidate/environment/offline-requirements.lock",
                "sha256": _mapping(candidate["environment"], "candidate environment")[
                    "offline_requirements_sha256"
                ],
                "runtime_pins": _mapping(candidate["environment"], "candidate environment")["runtime_pins"],
            },
        },
        "local_validation": evidence,
        "approval": {
            "required": True,
            "approved": False,
            "status": "pending_owner_confirmation",
            "scope": request_approval_scope,
            "rules_and_eligibility_must_be_rechecked": True,
        },
        "upload": {
            "allowed": False,
            "submission_allowed": False,
            "actions_not_performed": ["notebook_upload", "model_upload", "competition_submission"],
        },
        "claims": {
            "kaggle_verified": False,
            "hidden_scale_runtime_verified": False,
            "hidden_fallback_rate_known": False,
            "owner_selected_final": False,
        },
    }
    if runtime_wheels:
        if delivery_runtime_wheel_paths is None:
            runtime_wheel = runtime_wheels[0]
            payload["delivery_runtime"] = {
                "offline_wheel": {
                    **runtime_wheel,
                    "relative_path": f"runtime/{runtime_wheel['filename']}",
                    "source": "owner-authorized-local-wheel",
                    "no_index": True,
                    "no_deps": True,
                },
                "no_index": True,
                "no_deps": True,
                "local_wheel_only": True,
            }
        else:
            payload["delivery_runtime"] = {
                "offline_wheels": [
                    {
                        **runtime_wheel,
                        "relative_path": f"runtime/{runtime_wheel['filename']}",
                        "source": "owner-authorized-local-wheel",
                        "no_index": True,
                        "no_deps": True,
                    }
                    for runtime_wheel in runtime_wheels
                ],
                "no_index": True,
                "no_deps": True,
                "local_wheel_only": True,
            }
    payload["request_checksum"] = _json_sha256(payload)

    if execution_input is not None:
        input_reference = dict(execution_input)
        if _private_metadata_keys(input_reference):
            raise KaggleDeliveryError("execution input reference contains identifier-bearing fields")
        _nonempty(input_reference.get("mount_slug"), "execution input mount_slug")
        _nonempty(input_reference.get("workload_kind"), "execution input workload_kind")
        _safe_identifier(input_reference["mount_slug"], "execution input mount slug")
        if input_reference["workload_kind"] == "representative":
            for field in ("manifest_sha256", "test_csv_sha256", "test_series_csv_sha256"):
                _nonempty(input_reference.get(field), f"execution input {field}")
            _safe_identifier(input_reference["manifest_filename"], "execution input manifest filename")
        elif input_reference["workload_kind"] == "hidden_test":
            if input_reference["mount_slug"] != KAGGLE_COMPETITION_SLUG:
                raise KaggleDeliveryError("hidden-test input must use the official competition mount")
            if any(
                field in input_reference
                for field in ("manifest_filename", "manifest_sha256", "test_csv_sha256", "test_series_csv_sha256")
            ):
                raise KaggleDeliveryError("hidden-test input cannot pin replaceable competition files")
        else:
            raise KaggleDeliveryError("execution input workload kind must be hidden_test or representative")
        payload["execution_input"] = input_reference
        payload.pop("request_checksum", None)
        payload["request_checksum"] = _json_sha256(payload)

    def writer(staging: Path) -> None:
        atomic_json_write(staging / "request.json", payload)

    _publish_directory(output, writer)
    return load_kaggle_delivery(output)


def _approval_payload(
    path: Path,
    request_checksum: str,
    *,
    expected_scope: str = OWNER_APPROVAL_SCOPE,
) -> Mapping[str, object]:
    if not path.is_file() or path.is_symlink():
        raise KaggleDeliveryError(f"owner approval is not a regular file: {path}")
    payload = _read_json(path, "owner approval")
    if payload.get("schema_version") != KAGGLE_APPROVAL_SCHEMA_VERSION:
        raise KaggleDeliveryError("owner approval has an unexpected schema")
    if payload.get("request_checksum") != request_checksum:
        raise KaggleDeliveryError("owner approval does not name this request")
    if expected_scope not in OWNER_APPROVAL_SCOPES or payload.get("approval_scope") != expected_scope:
        raise KaggleDeliveryError("owner approval scope does not match the exact delivery request")
    if payload.get("approved") is not True:
        raise KaggleDeliveryError("owner approval is not granted")
    for field in (
        "rules_rechecked",
        "rules_accepted_by_owner",
        "account_eligible",
        "upload_allowed",
        "competition_submission_allowed",
    ):
        if payload.get(field) is not True:
            raise KaggleDeliveryError(f"owner approval does not confirm {field}")
    rules_evidence = _mapping(payload.get("rules_evidence"), "owner rules evidence")
    checked_at = _nonempty(rules_evidence.get("checked_at_utc"), "owner rules evidence timestamp")
    try:
        datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise KaggleDeliveryError("owner rules evidence timestamp is not ISO-8601") from exc
    sources = rules_evidence.get("sources")
    if not isinstance(sources, list) or not sources or not all(str(source).strip() for source in sources):
        raise KaggleDeliveryError("owner rules evidence must list its official sources")
    if not isinstance(rules_evidence.get("material_changes_reconciled"), bool):
        raise KaggleDeliveryError("owner rules evidence must state whether material changes were reconciled")
    _nonempty(payload.get("approver"), "owner approval approver")
    timestamp = _nonempty(payload.get("approved_at_utc"), "owner approval timestamp")
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise KaggleDeliveryError("owner approval timestamp is not ISO-8601") from exc
    return payload


def _execution_input_reference(
    payload: Mapping[str, object],
    *,
    default_mount_slug: str,
) -> Mapping[str, object]:
    """Load and validate the request-bound input workload reference."""

    value = payload.get("execution_input")
    if value is None:
        return {"mount_slug": default_mount_slug, "workload_kind": "hidden_test"}
    reference = _mapping(value, "execution input reference")
    mount_slug = _safe_identifier(reference.get("mount_slug"), "execution input mount slug")
    workload_kind = _nonempty(reference.get("workload_kind"), "execution input workload kind")
    if workload_kind not in {"hidden_test", "representative"}:
        raise KaggleDeliveryError("execution input workload kind must be hidden_test or representative")
    if workload_kind == "hidden_test":
        if mount_slug != KAGGLE_COMPETITION_SLUG:
            raise KaggleDeliveryError("hidden-test input must use the official competition mount")
        if any(
            field in reference
            for field in ("manifest_filename", "manifest_sha256", "test_csv_sha256", "test_series_csv_sha256")
        ):
            raise KaggleDeliveryError("hidden-test input cannot pin replaceable competition files")
        return {"mount_slug": mount_slug, "workload_kind": workload_kind}
    manifest_filename = _safe_identifier(reference.get("manifest_filename"), "execution input manifest filename")
    for field in ("manifest_sha256", "test_csv_sha256", "test_series_csv_sha256"):
        _nonempty(reference.get(field), f"execution input {field}")
    return {
        **dict(reference),
        "mount_slug": mount_slug,
        "workload_kind": workload_kind,
        "manifest_filename": manifest_filename,
    }


def _input_mount_files(root: Path, filename: str, *, max_depth: int = 5) -> tuple[Path, ...]:
    """Find a shallowly mounted input file without walking DICOM payloads."""

    if not root.is_dir() or root.is_symlink():
        raise KaggleDeliveryError(f"Kaggle input root is not a directory: {root}")
    found: list[Path] = []
    frontier: list[tuple[Path, int]] = [(root, 0)]
    # These directories contain the competition payload or immutable bundle
    # contents, never a delivery manifest or owner approval file.  Pruning
    # them keeps startup discovery independent of the hidden DICOM count.
    payload_directory_names = {"candidate", "code", "test_series", "train", "test"}
    while frontier:
        directory, depth = frontier.pop(0)
        try:
            children = sorted(directory.iterdir(), key=lambda path: str(path))
        except OSError as exc:
            raise KaggleDeliveryError(f"could not inspect Kaggle input directory: {directory}") from exc
        found_in_directory = False
        for child in children:
            if child.is_symlink():
                continue
            if child.is_file() and child.name == filename:
                found.append(child)
                found_in_directory = True
        if found_in_directory or depth >= max_depth:
            continue
        for child in children:
            if child.is_symlink() or not child.is_dir() or child.name in payload_directory_names:
                continue
            frontier.append((child, depth + 1))
    return tuple(found)


def _input_mount_directories(root: Path, name: str, *, max_depth: int = 4) -> tuple[Path, ...]:
    """Find shallowly mounted source directories without traversing their contents."""

    if not root.is_dir() or root.is_symlink():
        raise KaggleDeliveryError(f"Kaggle input root is not a directory: {root}")
    found: list[Path] = []
    frontier: list[tuple[Path, int]] = [(root, 0)]
    payload_directory_names = {"candidate", "code", "test_series", "train", "test"}
    while frontier:
        directory, depth = frontier.pop(0)
        try:
            children = sorted(directory.iterdir(), key=lambda path: str(path))
        except OSError as exc:
            raise KaggleDeliveryError(f"could not inspect Kaggle input directory: {directory}") from exc
        for child in children:
            if child.is_symlink() or not child.is_dir():
                continue
            if child.name == name:
                found.append(child)
                continue
            if depth >= max_depth or child.name in payload_directory_names:
                continue
            frontier.append((child, depth + 1))
    return tuple(found)


def _input_namespace_root(root: Path, namespace: str) -> Path:
    """Use Kaggle's dedicated mount namespace when it is present."""

    candidate = root / namespace
    if candidate.is_dir() and not candidate.is_symlink():
        return candidate
    return root


def _immutable_file_map(root: Path, description: str) -> dict[Path, str]:
    """Return checksums for a regular, symlink-free input tree."""

    if not root.is_dir() or root.is_symlink():
        raise KaggleDeliveryError(f"{description} is not a regular directory: {root}")
    files: dict[Path, str] = {}
    try:
        paths = sorted(root.rglob("*"), key=lambda path: str(path))
    except OSError as exc:
        raise KaggleDeliveryError(f"could not inspect {description}: {root}") from exc
    for path in paths:
        if path.is_symlink():
            raise KaggleDeliveryError(f"{description} contains a symlink: {path}")
        if path.is_file():
            files[path.relative_to(root)] = _sha256_file(path)
    return files


def _verify_delivery_code_mirror(candidate_root: Path, code_root: Path, adapter_path: Path) -> None:
    """Prove the imported package is the immutable candidate code plus adapter."""

    candidate_code = _immutable_file_map(candidate_root / "code", "candidate code")
    delivery_code = _immutable_file_map(code_root, "delivery code")
    adapter_relative = adapter_path.relative_to(code_root)
    expected_delivery_paths = set(candidate_code) | {adapter_relative}
    if set(delivery_code) != expected_delivery_paths:
        missing = sorted(str(path) for path in expected_delivery_paths - set(delivery_code))
        unexpected = sorted(str(path) for path in set(delivery_code) - expected_delivery_paths)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise KaggleDeliveryError("delivery code does not mirror immutable candidate code (" + "; ".join(details) + ")")
    for relative_path, expected_sha256 in candidate_code.items():
        # The adapter is the one deliberately versioned file shipped beside
        # the frozen candidate.  Every other imported module must be byte
        # identical to the candidate snapshot.
        if relative_path == adapter_relative:
            continue
        if delivery_code[relative_path] != expected_sha256:
            raise KaggleDeliveryError(f"delivery code differs from immutable candidate code: {relative_path}")


def discover_kaggle_inputs(
    input_root: str | Path = "/kaggle/input",
    *,
    expected_request_checksum: str,
    expected_candidate_manifest_checksum: str,
    competition_slug: str = KAGGLE_COMPETITION_SLUG,
    delivery_root_override: str | Path | None = None,
) -> KaggleInputLayout:
    """Resolve and validate Kaggle's namespaced input mounts.

    Kaggle may mount a dataset at ``/kaggle/input/datasets/<owner>/<slug>``
    and a competition at ``/kaggle/input/competitions/<slug>``. Search these
    namespaces separately so metadata lookup never walks the competition's
    image payload while looking for delivery or approval files. Inputs are
    still discovered by their checksummed contents and every selected path is
    verified before inference. A notebook that safely stages a verified
    archive-backed delivery bundle may provide its extracted root while
    competition and approval inputs continue to resolve from ``input_root``.
    """

    root = Path(input_root)
    request_checksum = _nonempty(expected_request_checksum, "expected request checksum")
    candidate_manifest_checksum = _nonempty(
        expected_candidate_manifest_checksum,
        "expected candidate manifest checksum",
    )
    competition_name = _safe_identifier(competition_slug, "competition slug")
    dataset_mount_root = _input_namespace_root(root, "datasets")
    competition_mount_root = _input_namespace_root(root, "competitions")

    request_matches: list[tuple[Path, Mapping[str, object]]] = []
    if delivery_root_override is None:
        for path in _input_mount_files(dataset_mount_root, "request.json"):
            try:
                payload = _request_payload(path.parent)
            except KaggleDeliveryError:
                continue
            if payload.get("request_checksum") == request_checksum:
                request_matches.append((path, payload))
    else:
        extracted_delivery_root = Path(delivery_root_override)
        payload = _request_payload(extracted_delivery_root)
        if payload.get("request_checksum") != request_checksum:
            raise KaggleDeliveryError("extracted delivery request checksum differs from the expected request")
        request_matches.append((extracted_delivery_root / "request.json", payload))
    if len(request_matches) != 1:
        raise KaggleDeliveryError(
            f"expected exactly one mounted request with checksum {request_checksum}; found {len(request_matches)}"
        )
    request_path, request_payload = request_matches[0]
    delivery_root = request_path.parent
    request = KaggleDeliveryRequest(
        delivery_root.resolve(),
        _nonempty(request_payload.get("request_checksum"), "request_checksum"),
        request_payload,
    )
    request_candidate = _mapping(request.payload.get("candidate"), "request candidate")
    if request_candidate.get("manifest_checksum") != candidate_manifest_checksum:
        raise KaggleDeliveryError("mounted request does not reference the expected candidate manifest")
    input_reference = _execution_input_reference(
        request.payload,
        default_mount_slug=competition_name,
    )
    if input_reference["mount_slug"] != competition_name:
        raise KaggleDeliveryError("the requested input mount slug differs from the discovery slug")

    candidate_started = time.perf_counter()
    candidate_root = delivery_root / "candidate"
    candidate_bundle = load_candidate(candidate_root)
    candidate_reference = _candidate_reference(candidate_bundle)
    candidate_load_hash_verification_seconds = time.perf_counter() - candidate_started
    if candidate_bundle.manifest_checksum != candidate_manifest_checksum:
        raise KaggleDeliveryError("mounted candidate manifest checksum differs from the expected candidate")
    if request_candidate.get("candidate_id") != candidate_bundle.candidate_id:
        raise KaggleDeliveryError("mounted request does not reference the mounted candidate")
    if candidate_reference != dict(request_candidate):
        raise KaggleDeliveryError("mounted request candidate reference differs from the mounted candidate")

    notebook_reference = _mapping(request.payload.get("delivery_notebook"), "request delivery notebook")
    notebook_name = _safe_identifier(notebook_reference.get("filename"), "delivery notebook filename")
    delivery_notebook_path = delivery_root / notebook_name
    adapter_reference = _mapping(request.payload.get("delivery_adapter"), "request delivery adapter")
    adapter_name = _safe_identifier(adapter_reference.get("filename"), "delivery adapter filename")
    code_root = delivery_root / "code"
    delivery_adapter_path = code_root / "rsna_knee" / adapter_name
    runtime_wheel_paths: tuple[Path, ...] = ()
    runtime_wheel_sha256s: tuple[str, ...] = ()
    runtime_wheel_audits: list[Mapping[str, object]] = []
    runtime_value = request.payload.get("delivery_runtime")
    if runtime_value is not None:
        for index, wheel_reference in enumerate(_runtime_wheel_references(runtime_value)):
            wheel_name = _safe_identifier(
                wheel_reference.get("filename"),
                f"request runtime wheel {index} filename",
            )
            wheel_relative_path = _nonempty(
                wheel_reference.get("relative_path"),
                f"request runtime wheel {index} relative path",
            )
            if wheel_relative_path != f"runtime/{wheel_name}":
                raise KaggleDeliveryError("request runtime wheel path is not the canonical runtime layout")
            wheel_path = delivery_root / wheel_relative_path
            wheel_audit = audit_kaggle_runtime_wheel(
                wheel_path,
                expected_package=_nonempty(
                    wheel_reference.get("package"),
                    f"request runtime wheel {index} package",
                ),
                expected_version=_nonempty(
                    wheel_reference.get("version"),
                    f"request runtime wheel {index} version",
                ),
            )
            if wheel_audit["filename"] != wheel_name or wheel_audit["sha256"] != wheel_reference.get("sha256"):
                raise KaggleDeliveryError("mounted runtime wheel checksum differs from the request")
            runtime_wheel_paths += (wheel_path,)
            runtime_wheel_sha256s += (str(wheel_audit["sha256"]),)
            runtime_wheel_audits.append(wheel_audit)
    notebook_audit = _notebook_audit(delivery_notebook_path)
    if notebook_audit["filename"] != notebook_name:
        raise KaggleDeliveryError("request notebook filename differs from the mounted notebook")
    if notebook_audit["sha256"] != notebook_reference.get("sha256"):
        raise KaggleDeliveryError("request notebook checksum differs from the mounted notebook")
    adapter_audit = _adapter_audit(delivery_adapter_path)
    if adapter_audit["filename"] != adapter_name:
        raise KaggleDeliveryError("request adapter filename differs from the mounted adapter")
    if adapter_audit["sha256"] != adapter_reference.get("sha256"):
        raise KaggleDeliveryError("request adapter checksum differs from the mounted adapter")
    _verify_delivery_code_mirror(candidate_root, code_root, delivery_adapter_path)

    approval_matches: list[Path] = []
    approval_scope = _request_approval_scope(request.payload)
    for path in _input_mount_files(dataset_mount_root, "owner-approval.json"):
        try:
            _approval_payload(path, request_checksum, expected_scope=approval_scope)
        except KaggleDeliveryError:
            continue
        approval_matches.append(path)
    if len(approval_matches) != 1:
        raise KaggleDeliveryError(
            f"expected exactly one owner approval for request {request_checksum}; found {len(approval_matches)}"
    )
    approval_path = approval_matches[0]
    approval_sha256 = _sha256_file(approval_path)

    competition_matches: list[Path] = []
    for path in _input_mount_directories(competition_mount_root, competition_name):
        required_files = (path / "test.csv", path / "test_series.csv")
        test_series_root = path / "test_series"
        if (
            all(item.is_file() and not item.is_symlink() for item in required_files)
            and test_series_root.is_dir()
            and not test_series_root.is_symlink()
        ):
            competition_matches.append(path)
    if len(competition_matches) != 1:
        raise KaggleDeliveryError(
            f"expected exactly one mounted input {competition_name}; found {len(competition_matches)}"
        )
    input_manifest_sha256: str | None = None
    if input_reference["workload_kind"] == "representative":
        input_root_path = competition_matches[0]
        manifest_path = input_root_path / str(input_reference["manifest_filename"])
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise KaggleDeliveryError(f"mounted execution input is missing its manifest: {manifest_path}")
        input_manifest_sha256 = _sha256_file(manifest_path)
        if input_manifest_sha256 != input_reference["manifest_sha256"]:
            raise KaggleDeliveryError("mounted execution input manifest checksum differs from the request")
        if _sha256_file(input_root_path / "test.csv") != input_reference["test_csv_sha256"]:
            raise KaggleDeliveryError("mounted execution input test.csv checksum differs from the request")
        if _sha256_file(input_root_path / "test_series.csv") != input_reference["test_series_csv_sha256"]:
            raise KaggleDeliveryError("mounted execution input test_series.csv checksum differs from the request")

    validation = _ValidatedKaggleInputs(
        request=request,
        candidate_bundle=candidate_bundle,
        candidate_reference=candidate_reference,
        approval_sha256=approval_sha256,
        delivery_notebook_sha256=str(notebook_audit["sha256"]),
        delivery_adapter_sha256=str(adapter_audit["sha256"]),
        runtime_wheel_audits=tuple(runtime_wheel_audits),
        candidate_load_hash_verification_seconds=candidate_load_hash_verification_seconds,
    )

    return KaggleInputLayout(
        input_root=root,
        delivery_root=delivery_root,
        candidate_root=candidate_root,
        candidate_code_root=candidate_root / "code",
        code_root=code_root,
        request_path=request_path,
        approval_path=approval_path,
        competition_root=competition_matches[0],
        delivery_notebook_path=delivery_notebook_path,
        delivery_adapter_path=delivery_adapter_path,
        runtime_wheel_path=runtime_wheel_paths[0] if len(runtime_wheel_paths) == 1 else None,
        runtime_wheel_sha256=runtime_wheel_sha256s[0] if len(runtime_wheel_sha256s) == 1 else None,
        request_checksum=request_checksum,
        candidate_manifest_checksum=candidate_manifest_checksum,
        runtime_wheel_paths=runtime_wheel_paths,
        runtime_wheel_sha256s=runtime_wheel_sha256s,
        workload_kind=str(input_reference["workload_kind"]),
        input_manifest_sha256=input_manifest_sha256,
        validation=validation,
    )


def _reuse_discovered_kaggle_inputs(
    layout: KaggleInputLayout,
    *,
    candidate_root: str | Path,
    input_root: str | Path,
    delivery_request_root: str | Path,
    owner_approval_path: str | Path,
    delivery_notebook_path: str | Path,
    delivery_adapter_path: str | Path,
    delivery_request_checksum: str | None,
    delivery_notebook_sha256: str | None,
    delivery_adapter_sha256: str | None,
    delivery_runtime_wheel_path: str | Path | None,
    delivery_runtime_wheel_paths: Sequence[str | Path] | None,
) -> _ValidatedKaggleInputs:
    """Reuse discovery's proof only when every run binding is unchanged.

    This is intentionally an in-process handoff, not a serialized trust token.
    Kaggle input mounts are immutable after discovery; callers that do not have
    a layout returned by ``discover_kaggle_inputs`` retain full revalidation.
    """

    validation = layout.validation
    if validation is None:
        raise KaggleDeliveryError("run validation reuse requires a discovered Kaggle input layout")

    candidate_path = Path(candidate_root).resolve()
    input_path = Path(input_root).resolve()
    request_root = Path(delivery_request_root).resolve()
    approval_path = Path(owner_approval_path).resolve()
    notebook_path = Path(delivery_notebook_path).resolve()
    adapter_path = Path(delivery_adapter_path).resolve()

    bindings = (
        (layout.candidate_root.resolve(), candidate_path, "candidate root"),
        (layout.competition_root.resolve(), input_path, "competition input root"),
        (layout.delivery_root.resolve(), request_root, "delivery request root"),
        (layout.approval_path.resolve(), approval_path, "owner approval path"),
        (layout.delivery_notebook_path.resolve(), notebook_path, "delivery notebook path"),
        (layout.delivery_adapter_path.resolve(), adapter_path, "delivery adapter path"),
    )
    for discovered, supplied, field in bindings:
        if discovered != supplied:
            raise KaggleDeliveryError(f"validated Kaggle layout does not match supplied {field}")

    if layout.request_path.resolve() != request_root / "request.json":
        raise KaggleDeliveryError("validated Kaggle layout has an unexpected request path")
    if layout.candidate_code_root.resolve() != candidate_path / "code":
        raise KaggleDeliveryError("validated Kaggle layout has an unexpected candidate code path")
    if layout.code_root.resolve() != request_root / "code":
        raise KaggleDeliveryError("validated Kaggle layout has an unexpected delivery code path")
    if validation.request.root != request_root or validation.candidate_bundle.root != candidate_path:
        raise KaggleDeliveryError("validated Kaggle proof is detached from its input layout")
    if layout.request_checksum != validation.request.request_checksum:
        raise KaggleDeliveryError("validated Kaggle layout request checksum is inconsistent")
    if layout.candidate_manifest_checksum != validation.candidate_bundle.manifest_checksum:
        raise KaggleDeliveryError("validated Kaggle layout candidate checksum is inconsistent")
    if validation.candidate_reference.get("candidate_id") != validation.candidate_bundle.candidate_id:
        raise KaggleDeliveryError("validated Kaggle candidate reference is inconsistent")
    if validation.candidate_reference.get("manifest_checksum") != validation.candidate_bundle.manifest_checksum:
        raise KaggleDeliveryError("validated Kaggle candidate reference checksum is inconsistent")

    if delivery_request_checksum is not None and delivery_request_checksum != validation.request.request_checksum:
        raise KaggleDeliveryError("supplied delivery request checksum differs from discovered request.json")
    if delivery_notebook_sha256 is not None and delivery_notebook_sha256 != validation.delivery_notebook_sha256:
        raise KaggleDeliveryError("supplied delivery notebook checksum differs from discovered notebook")
    if delivery_adapter_sha256 is not None and delivery_adapter_sha256 != validation.delivery_adapter_sha256:
        raise KaggleDeliveryError("supplied delivery adapter checksum differs from discovered adapter")

    supplied_wheels = _runtime_wheel_paths(delivery_runtime_wheel_path, delivery_runtime_wheel_paths)
    discovered_wheels = tuple(path.resolve() for path in layout.runtime_wheel_paths)
    if tuple(path.resolve() for path in supplied_wheels) != discovered_wheels:
        raise KaggleDeliveryError("supplied runtime wheelhouse differs from discovered runtime wheelhouse")
    runtime_value = validation.request.payload.get("delivery_runtime")
    expected_wheel_references = _runtime_wheel_references(runtime_value) if runtime_value is not None else ()
    if len(expected_wheel_references) != len(supplied_wheels):
        raise KaggleDeliveryError("supplied runtime wheelhouse does not match discovered request.json")
    if tuple(layout.runtime_wheel_sha256s) != tuple(
        str(audit.get("sha256")) for audit in validation.runtime_wheel_audits
    ):
        raise KaggleDeliveryError("validated runtime wheel checksums are inconsistent")
    for audit, expected in zip(validation.runtime_wheel_audits, expected_wheel_references):
        if (
            audit.get("filename") != expected.get("filename")
            or audit.get("sha256") != expected.get("sha256")
            or audit.get("package") != expected.get("package")
            or audit.get("version") != expected.get("version")
        ):
            raise KaggleDeliveryError("validated runtime wheel audit differs from request.json")

    return validation


def _runtime_install_audit(
    value: Mapping[str, object] | None,
    request: KaggleDeliveryRequest,
    runtime_audit: Mapping[str, object],
) -> dict[str, object] | None:
    """Validate the recorded before/after offline wheel installation."""

    runtime_value = request.payload.get("delivery_runtime")
    if runtime_value is None:
        if value is not None:
            raise KaggleDeliveryError("runtime installation evidence is not allowed without a delivery wheel")
        return None
    if value is None:
        raise KaggleDeliveryError("offline runtime installation evidence is required for this delivery")
    evidence = dict(value)
    if _private_metadata_keys(evidence):
        raise KaggleDeliveryError("runtime installation evidence contains identifier-bearing fields")
    if evidence.get("status") != "installed_offline":
        raise KaggleDeliveryError("runtime installation was not recorded as offline")
    installation = _mapping(evidence.get("installation"), "runtime installation evidence")
    for field in ("no_index", "no_deps", "local_wheel_only"):
        if installation.get(field) is not True:
            raise KaggleDeliveryError(f"runtime installation does not confirm {field}")
    if installation.get("internet_enabled") is not False:
        raise KaggleDeliveryError("runtime installation does not confirm internet is disabled")
    runtime = _mapping(runtime_value, "request delivery runtime")
    expected_wheels = _runtime_wheel_references(runtime)
    if "wheels" in evidence:
        raw_observed_wheels = evidence.get("wheels")
        if not isinstance(raw_observed_wheels, list):
            raise KaggleDeliveryError("runtime installation wheels must be a list")
        observed_wheels = tuple(
            _mapping(item, f"runtime installation wheel {index}")
            for index, item in enumerate(raw_observed_wheels)
        )
    elif len(expected_wheels) == 1 and "wheel" in evidence:
        observed_wheels = (_mapping(evidence.get("wheel"), "runtime installation wheel"),)
    else:
        raise KaggleDeliveryError("runtime installation evidence does not list every offline wheel")
    if len(observed_wheels) != len(expected_wheels):
        raise KaggleDeliveryError("runtime installation wheel count differs from the request")
    for index, (observed_wheel, expected_wheel) in enumerate(zip(observed_wheels, expected_wheels)):
        for field in ("filename", "sha256", "package", "version"):
            if observed_wheel.get(field) != expected_wheel.get(field):
                raise KaggleDeliveryError(f"runtime installation wheel {index} {field} differs from the request")
        if "tags" in expected_wheel and observed_wheel.get("tags") != expected_wheel.get("tags"):
            raise KaggleDeliveryError(f"runtime installation wheel {index} tags differ from the request")

    candidate = _mapping(request.payload.get("candidate"), "request candidate")
    environment = _mapping(candidate.get("environment"), "request candidate environment")
    expected_pins = {
        _normalize_package_name(package, "runtime package"): _nonempty(version, f"runtime pin {package}")
        for package, version in _mapping(environment.get("runtime_pins"), "request runtime pins").items()
    }
    before = _mapping(evidence.get("environment_before"), "runtime environment before installation")
    after = _mapping(evidence.get("environment_after"), "runtime environment after installation")
    _nonempty(before.get("python_version"), "runtime Python before installation")
    if _nonempty(after.get("python_version"), "runtime Python after installation") != runtime_audit.get("python_version"):
        raise KaggleDeliveryError("runtime installation evidence has a different Python version")
    before_packages = _mapping(before.get("packages"), "runtime packages before installation")
    after_packages = _mapping(after.get("packages"), "runtime packages after installation")
    if set(str(key) for key in after_packages) != set(expected_pins):
        raise KaggleDeliveryError("runtime installation evidence does not cover every locked package")
    if {str(key): str(value) for key, value in after_packages.items()} != expected_pins:
        raise KaggleDeliveryError("runtime packages after installation differ from the frozen pins")
    if dict(after_packages) != dict(_mapping(runtime_audit.get("packages"), "runtime package audit")):
        raise KaggleDeliveryError("runtime package evidence differs from the imported runtime audit")
    if set(str(key) for key in before_packages) != set(expected_pins):
        raise KaggleDeliveryError("runtime packages before installation do not cover every locked package")
    for package in expected_pins:
        _nonempty(before_packages.get(package), f"runtime package before installation {package}")
    imported_after = _mapping(after.get("imported_packages"), "imported runtime packages after installation")
    imported_audit = _mapping(runtime_audit.get("imported_packages"), "imported runtime package audit")
    if dict(imported_after) != dict(imported_audit):
        raise KaggleDeliveryError("imported runtime package evidence differs from the runtime audit")
    installation_wheel_sha256s = tuple(str(item.get("sha256")) for item in observed_wheels)
    if len(set(installation_wheel_sha256s)) != len(installation_wheel_sha256s):
        raise KaggleDeliveryError("runtime installation evidence repeats a wheel checksum")
    return evidence


def _private_execution_audit(
    path: Path,
    candidate_manifest_checksum: str,
    expected_study_count: int,
) -> tuple[str, int]:
    if not path.is_file() or path.is_symlink():
        raise KaggleDeliveryError(f"private Kaggle execution evidence is not a regular file: {path}")
    payload = _read_json(path, "private Kaggle execution evidence")
    if payload.get("schema_version") != KAGGLE_EXECUTION_SCHEMA_VERSION:
        raise KaggleDeliveryError("private Kaggle execution evidence has an unexpected schema")
    if payload.get("candidate_manifest_checksum") != candidate_manifest_checksum:
        raise KaggleDeliveryError("private Kaggle execution evidence uses a different candidate")
    studies = payload.get("studies")
    if not isinstance(studies, list) or len(studies) != expected_study_count:
        raise KaggleDeliveryError("private Kaggle execution evidence does not cover every study")
    seen_uids: set[str] = set()
    fallback_count = 0
    for item in studies:
        study = _mapping(item, "private Kaggle study evidence")
        study_uid = _nonempty(study.get("StudyInstanceUID"), "private study identifier")
        if study_uid in seen_uids:
            raise KaggleDeliveryError("private Kaggle execution evidence repeats a study")
        seen_uids.add(study_uid)
        prediction = _mapping(study.get("prediction"), "private study prediction evidence")
        if not isinstance(prediction.get("used_full_study_fallback"), bool):
            raise KaggleDeliveryError("private study prediction evidence must state fallback usage")
        if prediction.get("used_full_study_fallback") is True:
            fallback_count += 1
        if not isinstance(prediction.get("valid_series_count"), int):
            raise KaggleDeliveryError("private study prediction evidence must state valid series")
        if not isinstance(prediction.get("rejected_series"), list):
            raise KaggleDeliveryError("private study prediction evidence must list rejected series")
        if not isinstance(prediction.get("warnings"), list):
            raise KaggleDeliveryError("private study prediction evidence must list warnings")
    return _sha256_file(path), fallback_count


def _execution_payload(
    path: Path,
    request: KaggleDeliveryRequest,
    *,
    submission_path: Path | None,
    private_evidence_path: Path | None,
    expected_study_uids: Sequence[str] | None,
    engineering_margin_approved: bool,
) -> dict[str, object]:
    raw = _read_json(path, "Kaggle execution evidence")
    if raw.get("schema_version") != KAGGLE_EXECUTION_SCHEMA_VERSION:
        raise KaggleDeliveryError("Kaggle execution evidence has an unexpected schema")
    if _private_metadata_keys(raw):
        raise KaggleDeliveryError("Kaggle execution evidence contains identifier-bearing fields")
    if raw.get("status") != "accepted_by_kaggle":
        raise KaggleDeliveryError("Kaggle execution was not accepted by Kaggle")
    bundle = _mapping(raw.get("bundle"), "Kaggle execution bundle")
    request_candidate = _mapping(request.payload.get("candidate"), "request candidate")
    if bundle.get("request_checksum") != request.request_checksum:
        raise KaggleDeliveryError("Kaggle execution used a different delivery request")
    if bundle.get("candidate_manifest_checksum") != request_candidate.get("manifest_checksum"):
        raise KaggleDeliveryError("Kaggle execution used a different candidate manifest")
    if bundle.get("notebook_sha256") != _mapping(request.payload.get("delivery_notebook"), "request notebook").get("sha256"):
        raise KaggleDeliveryError("Kaggle execution used a different delivery notebook")
    if bundle.get("adapter_sha256") != _mapping(request.payload.get("delivery_adapter"), "request delivery adapter").get("sha256"):
        raise KaggleDeliveryError("Kaggle execution used a different delivery adapter")
    execution_models = bundle.get("model_checkpoints")
    if execution_models != request_candidate.get("model_checkpoints"):
        raise KaggleDeliveryError("Kaggle execution used a different checkpoint set")
    if bundle.get("model_size_bytes") != request_candidate.get("model_size_bytes"):
        raise KaggleDeliveryError("Kaggle execution used a different model size")
    if bundle.get("candidate_bundle_size_bytes") != request_candidate.get("candidate_bundle_size_bytes"):
        raise KaggleDeliveryError("Kaggle execution used a different candidate package size")
    if bundle.get("environment") != request_candidate.get("environment"):
        raise KaggleDeliveryError("Kaggle execution used a different pinned environment")

    environment = _mapping(raw.get("environment"), "Kaggle execution environment")
    if environment.get("runtime") != "kaggle":
        raise KaggleDeliveryError("execution environment is not Kaggle")
    if environment.get("internet_enabled") is not False or environment.get("runtime_network_download") is not False:
        raise KaggleDeliveryError("Kaggle execution was not demonstrably offline")
    if environment.get("reports_required") is not False:
        raise KaggleDeliveryError("Kaggle execution requires reports at inference")
    for field in ("python_version", "torch_version", "accelerator"):
        _nonempty(environment.get(field), f"Kaggle environment {field}")
    requested_environment = _mapping(request_candidate.get("environment"), "request candidate environment")
    requested_runtime_pins = _mapping(requested_environment.get("runtime_pins"), "request runtime pins")
    installed_runtime_pins_payload = _mapping(
        environment.get("installed_runtime_pins"),
        "Kaggle installed runtime pins",
    )
    installed_runtime_pins = {
        package: _nonempty(installed_runtime_pins_payload.get(package), f"Kaggle installed {package} pin")
        for package in ("numpy", "pydicom", "torch")
    }
    if installed_runtime_pins != {str(key): str(value) for key, value in requested_runtime_pins.items()}:
        raise KaggleDeliveryError("Kaggle installed runtime pins differ from the pinned offline environment")

    resources = _mapping(raw.get("resources"), "Kaggle execution resources")
    model_size_bytes = _required_int(resources.get("model_size_bytes"), "Kaggle model size", positive=True)
    candidate_bundle_size_bytes = _required_int(
        resources.get("candidate_bundle_size_bytes"),
        "Kaggle candidate package size",
        positive=True,
    )
    peak_vram_bytes = _required_int(resources.get("peak_vram_bytes"), "Kaggle peak VRAM")
    ensemble_model_count = _required_int(resources.get("ensemble_model_count"), "Kaggle ensemble model count", positive=True)
    if model_size_bytes != request_candidate.get("model_size_bytes"):
        raise KaggleDeliveryError("Kaggle resource evidence has a different model size")
    if candidate_bundle_size_bytes != request_candidate.get("candidate_bundle_size_bytes"):
        raise KaggleDeliveryError("Kaggle resource evidence has a different candidate package size")
    expected_checkpoints = request_candidate.get("model_checkpoints")
    if not isinstance(expected_checkpoints, list):
        raise KaggleDeliveryError("request candidate checkpoint set is invalid")
    if ensemble_model_count != len(expected_checkpoints):
        raise KaggleDeliveryError("Kaggle resource evidence has a different ensemble size")

    workload = _mapping(raw.get("workload"), "Kaggle execution workload")
    study_count = _required_int(workload.get("study_count"), "Kaggle workload study count", positive=True)
    _required_int(workload.get("series_count"), "Kaggle workload series count")
    _required_int(workload.get("slice_count"), "Kaggle workload slice count")
    if workload.get("actual_workload_recorded") is not True:
        raise KaggleDeliveryError("Kaggle execution does not record its actual workload")
    workload_kind = str(workload.get("workload_kind", "hidden_test")).strip()
    if workload_kind not in {"hidden_test", "representative"}:
        raise KaggleDeliveryError("Kaggle execution has an unsupported workload kind")
    if workload.get("hidden_test") is not (workload_kind == "hidden_test") or workload.get("example_only") is not False:
        raise KaggleDeliveryError("example-only execution cannot establish Kaggle verification")
    private_evidence_result = (
        _private_execution_audit(
            Path(private_evidence_path),
            str(request_candidate.get("manifest_checksum")),
            study_count,
        )
        if private_evidence_path is not None
        else None
    )
    private_evidence_sha256 = private_evidence_result[0] if private_evidence_result is not None else None
    private_fallback_count = private_evidence_result[1] if private_evidence_result is not None else None

    timing = _mapping(raw.get("timing"), "Kaggle execution timing")
    whole_seconds = _required_float(timing.get("whole_notebook_seconds"), "whole-notebook runtime", positive=True)
    if whole_seconds > KAGGLE_HARD_LIMIT_SECONDS:
        raise KaggleDeliveryError("Kaggle execution exceeded the nine-hour hard limit")
    stages = _mapping(timing.get("stages"), "Kaggle runtime stages")
    detailed_timing = _mapping(timing.get("detailed"), "Kaggle detailed timing")
    profile_schema = _nonempty(detailed_timing.get("schema_version"), "Kaggle profile schema version")
    profile_stages = KAGGLE_RUNTIME_STAGES_BY_PROFILE.get(profile_schema)
    if profile_stages is None:
        raise KaggleDeliveryError("Kaggle execution has an unsupported profile schema")
    normalized_stages: dict[str, float] = {}
    for stage in profile_stages:
        normalized_stages[stage] = _required_float(stages.get(stage), f"Kaggle runtime stage {stage}")
    if timing.get("stage_timing_source") != "notebook_wall_clock":
        raise KaggleDeliveryError("Kaggle runtime stages must be measured by the notebook wall clock")
    projection = _mapping(timing.get("projection"), "Kaggle runtime projection")
    if _required_int(projection.get("target_studies"), "Kaggle projection target studies", positive=True) != KAGGLE_TARGET_STUDIES:
        raise KaggleDeliveryError("Kaggle runtime projection must target approximately 1,300 studies")
    projected_seconds = _required_float(
        projection.get("projected_seconds_at_target"),
        "Kaggle projected runtime",
        positive=True,
    )
    _nonempty(projection.get("method"), "Kaggle runtime projection method")
    observed_within_target = whole_seconds <= KAGGLE_ENGINEERING_TARGET_SECONDS
    within_target = projected_seconds <= KAGGLE_ENGINEERING_TARGET_SECONDS
    if not within_target and not engineering_margin_approved:
        v16_status = "blocked_engineering_margin_not_approved"
    elif within_target:
        v16_status = "passed"
    else:
        v16_status = "passed_with_owner_margin_approval"

    output = _mapping(raw.get("output"), "Kaggle execution output")
    if output.get("filename") != "submission.csv":
        raise KaggleDeliveryError("Kaggle output is not named submission.csv")
    if output.get("accepted_by_kaggle") is not True:
        raise KaggleDeliveryError("Kaggle did not accept submission.csv")
    if _required_int(output.get("row_count"), "Kaggle output row count", positive=True) != study_count:
        raise KaggleDeliveryError("Kaggle output rows do not cover the recorded workload")
    if _required_int(output.get("column_count"), "Kaggle output column count") != NUM_TARGETS + 1:
        raise KaggleDeliveryError("Kaggle output does not have thirteen columns")
    for field in ("exact_header", "exact_uid_order", "scores_finite", "scores_bounded"):
        if output.get(field) is not True:
            raise KaggleDeliveryError(f"Kaggle output does not prove {field}")
    output_sha = str(output.get("sha256", "")).strip() or None
    if submission_path is not None:
        local_output = _local_submission_audit(Path(submission_path), expected_study_uids)
        if local_output["row_count"] != study_count:
            raise KaggleDeliveryError("accepted Kaggle output row count differs from the supplied CSV")
        if output_sha is not None and output_sha != local_output["sha256"]:
            raise KaggleDeliveryError("accepted Kaggle output checksum differs from the supplied CSV")
        output_sha = cast(str, local_output["sha256"])
    if output_sha is None:
        raise KaggleDeliveryError("accepted Kaggle output needs a checksum or a supplied submission.csv")

    fallback = _mapping(raw.get("fallback"), "Kaggle fallback evidence")
    fallback_status = str(fallback.get("status", "")).strip()
    denominator = _required_int(fallback.get("denominator"), "Kaggle fallback denominator", positive=True)
    if denominator != study_count:
        raise KaggleDeliveryError("Kaggle fallback denominator differs from the actual workload")
    measured_fallback_numerator: int | None = None
    if fallback_status == "unknown":
        if fallback.get("numerator") is not None or fallback.get("rate") is not None:
            raise KaggleDeliveryError("unknown fallback evidence must not contain invented zero values")
        if fallback.get("unknown_hidden_failure_rate") is not True:
            raise KaggleDeliveryError("unknown fallback evidence must explicitly remain unknown")
        fallback_summary: dict[str, object] = {
            "status": "unknown",
            "numerator": None,
            "denominator": denominator,
            "rate": None,
            "hidden_failure_rate_unknown": True,
        }
        v17_status = "blocked_hidden_fallback_rate_unknown"
        fallback_eligible = False
    elif fallback_status == "measured":
        numerator = _required_int(fallback.get("numerator"), "Kaggle fallback numerator")
        if numerator > denominator:
            raise KaggleDeliveryError("Kaggle fallback numerator exceeds its denominator")
        measured_fallback_numerator = numerator
        rate = _required_float(fallback.get("rate"), "Kaggle fallback rate")
        if not math.isclose(rate, numerator / denominator, rel_tol=1e-9, abs_tol=1e-12):
            raise KaggleDeliveryError("Kaggle fallback rate is inconsistent with its counts")
        if fallback.get("unknown_hidden_failure_rate") is not False:
            raise KaggleDeliveryError("measured fallback evidence must say that its rate is known")
        investigation = _mapping(fallback.get("investigation", {}), "Kaggle fallback investigation")
        if numerator > 0 and investigation.get("complete") is not True:
            raise KaggleDeliveryError("nonzero Kaggle fallback requires complete case investigation")
        fallback_summary = {
            "status": "measured",
            "numerator": numerator,
            "denominator": denominator,
            "rate": rate,
            "hidden_failure_rate_unknown": False,
            "investigation": {
                "complete": investigation.get("complete") is True,
                "investigated_cases": investigation.get("investigated_cases"),
                "unavoidable_cases": investigation.get("unavoidable_cases"),
            },
        }
        v17_status = "passed" if numerator == 0 else "blocked_nonzero_fallback"
        fallback_eligible = numerator == 0
    else:
        raise KaggleDeliveryError("Kaggle fallback status must be measured or unknown")

    if fallback_status == "measured" and private_fallback_count != measured_fallback_numerator:
        raise KaggleDeliveryError("private fallback report count differs from accepted fallback evidence")

    recovery = _mapping(raw.get("recovery"), "Kaggle recovery evidence")
    for field in ("save_reload_verified", "interruption_recovery_verified"):
        if not isinstance(recovery.get(field), bool):
            raise KaggleDeliveryError(f"Kaggle recovery evidence {field} must be boolean")
    rollback_seconds = _optional_int(recovery.get("rollback_seconds"), "Kaggle recovery rollback seconds")
    recovery_eligible = (
        recovery.get("save_reload_verified") is True
        and recovery.get("interruption_recovery_verified") is True
    )
    if fallback_eligible and recovery_eligible is False:
        v17_status = "blocked_recovery_not_verified"
    fallback_eligible = fallback_eligible and recovery_eligible
    accepted_output_preserved = submission_path is not None
    if fallback_eligible and not accepted_output_preserved:
        v17_status = "blocked_accepted_output_not_preserved"
    private_evidence_preserved = private_evidence_sha256 is not None
    if fallback_eligible and accepted_output_preserved and not private_evidence_preserved:
        v17_status = "blocked_private_failure_report_not_preserved"

    normalized = {
        "execution_evidence_sha256": _sha256_file(path),
        "environment": {
            "runtime": "kaggle",
            "internet_enabled": False,
            "runtime_network_download": False,
            "reports_required": False,
            "python_version": _nonempty(environment.get("python_version"), "Kaggle environment python_version"),
            "torch_version": _nonempty(environment.get("torch_version"), "Kaggle environment torch_version"),
            "accelerator": _nonempty(environment.get("accelerator"), "Kaggle environment accelerator"),
            "installed_runtime_pins": installed_runtime_pins,
        },
        "resources": {
            "model_size_bytes": model_size_bytes,
            "candidate_bundle_size_bytes": candidate_bundle_size_bytes,
            "peak_vram_bytes": peak_vram_bytes,
            "ensemble_model_count": ensemble_model_count,
        },
        "workload": {
            "study_count": study_count,
            "series_count": _required_int(workload.get("series_count"), "Kaggle workload series count"),
            "slice_count": _required_int(workload.get("slice_count"), "Kaggle workload slice count"),
            "actual_workload_recorded": True,
            "workload_kind": workload_kind,
            "hidden_test": workload_kind == "hidden_test",
            "example_only": False,
            "series_per_study": workload.get("series_per_study"),
            "slices_per_study": workload.get("slices_per_study"),
            "long_tail": workload.get("long_tail"),
        },
        "timing": {
            "whole_notebook_seconds": whole_seconds,
            "stages": normalized_stages,
            "engineering_target_seconds": KAGGLE_ENGINEERING_TARGET_SECONDS,
            "hard_limit_seconds": KAGGLE_HARD_LIMIT_SECONDS,
            "within_engineering_target": within_target,
            "observed_within_engineering_target": observed_within_target,
            "within_hard_limit": True,
            "stage_timing_source": "notebook_wall_clock",
            "projection": {
                "target_studies": KAGGLE_TARGET_STUDIES,
                "projected_seconds_at_target": projected_seconds,
                "method": str(projection.get("method")),
            },
        },
        "output": {
            "filename": "submission.csv",
            "accepted_by_kaggle": True,
            "row_count": study_count,
            "column_count": NUM_TARGETS + 1,
            "exact_header": True,
            "exact_uid_order": True,
            "scores_finite": True,
            "scores_bounded": True,
            "sha256": output_sha,
            "accepted_output_preserved": accepted_output_preserved,
        },
        "fallback": fallback_summary,
        "private_evidence": {
            "sha256": private_evidence_sha256,
            "preserved": private_evidence_preserved,
        },
        "recovery": {
            "save_reload_verified": recovery.get("save_reload_verified"),
            "interruption_recovery_verified": recovery.get("interruption_recovery_verified"),
            "rollback_seconds": rollback_seconds,
        },
        "gates": {
            "v08_exact_offline_bundle_and_output": "passed",
            "v12_owner_approved_exact_request": "passed",
            "v16_whole_notebook_runtime": v16_status,
            "v17_complete_rows_and_fallback": v17_status,
        },
        "promotion_eligible": bool(
            fallback_eligible
            and accepted_output_preserved
            and private_evidence_preserved
            and (within_target or engineering_margin_approved)
        ),
    }
    return normalized


def record_kaggle_verification(
    delivery_root: str | Path,
    candidate_root: str | Path,
    approval_path: str | Path,
    execution_evidence_path: str | Path,
    output_root: str | Path,
    *,
    submission_path: str | Path | None = None,
    private_evidence_path: str | Path | None = None,
    expected_study_uids: Sequence[str] | None = None,
) -> KaggleVerification:
    """Record owner-approved external Kaggle evidence without mutating inputs."""

    request = load_kaggle_delivery(delivery_root, candidate_root=candidate_root)
    approval_file = Path(approval_path)
    approval = _approval_payload(
        approval_file,
        request.request_checksum,
        expected_scope=_request_approval_scope(request.payload),
    )
    output = Path(output_root)
    candidate = Path(candidate_root)
    _ensure_outside(output, (request.root, candidate.resolve()), "Kaggle verification")
    execution_path = Path(execution_evidence_path)
    engineering_margin_approved = approval.get("engineering_margin_approved") is True
    execution = _execution_payload(
        execution_path,
        request,
        submission_path=(Path(submission_path) if submission_path is not None else None),
        private_evidence_path=(Path(private_evidence_path) if private_evidence_path is not None else None),
        expected_study_uids=expected_study_uids,
        engineering_margin_approved=engineering_margin_approved,
    )
    request_candidate = _mapping(request.payload.get("candidate"), "request candidate")
    payload: dict[str, object] = {
        "schema_version": KAGGLE_VERIFICATION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "kaggle_execution_accepted",
        "request_checksum": request.request_checksum,
        "candidate": dict(request_candidate),
        "approval": {
            "sha256": _sha256_file(approval_file),
            "approver": _nonempty(approval.get("approver"), "approval approver"),
            "approved_at_utc": _nonempty(approval.get("approved_at_utc"), "approval timestamp"),
            "engineering_margin_approved": engineering_margin_approved,
        },
        "execution": execution,
        "environment": execution["environment"],
        "workload": execution["workload"],
        "timing": execution["timing"],
        "output": execution["output"],
        "fallback": execution["fallback"],
        "gates": execution["gates"],
        "promotion": {
            "eligible": execution["promotion_eligible"],
            "level": "kaggle_verified" if execution["promotion_eligible"] else "kaggle_execution_recorded_pending_gates",
            "final_candidate": False,
            "owner_selection_required": True,
            "candidate_mutated": False,
            "export_allowed": False,
        },
        "preservation": {
            "request_unchanged": True,
            "candidate_unchanged": True,
            "accepted_output_preserved": submission_path is not None,
            "private_failure_report_preserved": private_evidence_path is not None,
        },
    }
    payload["verification_checksum"] = _json_sha256(payload)

    def writer(staging: Path) -> None:
        atomic_json_write(staging / "verification.json", payload)
        if submission_path is not None:
            private = staging / "private"
            private.mkdir(parents=True, exist_ok=True)
            source = Path(submission_path)
            if not source.is_file() or source.is_symlink():
                raise KaggleDeliveryError(f"accepted submission is not a regular file: {source}")
            shutil.copyfile(source, private / "submission.csv")
        if private_evidence_path is not None:
            private = staging / "private"
            private.mkdir(parents=True, exist_ok=True)
            source = Path(private_evidence_path)
            if not source.is_file() or source.is_symlink():
                raise KaggleDeliveryError(f"private execution evidence is not a regular file: {source}")
            shutil.copyfile(source, private / "execution.json")

    _publish_directory(output, writer)
    return load_kaggle_verification(output)


def _verification_payload(root: Path) -> Mapping[str, object]:
    if not root.is_dir() or root.is_symlink():
        raise KaggleDeliveryError(f"Kaggle verification root is not a directory: {root}")
    payload = _read_json(root / "verification.json", "Kaggle verification")
    checksum = _nonempty(payload.get("verification_checksum"), "verification_checksum")
    without_checksum = dict(payload)
    without_checksum.pop("verification_checksum", None)
    if _json_sha256(without_checksum) != checksum:
        raise KaggleDeliveryError("Kaggle verification checksum does not match its contents")
    if payload.get("schema_version") != KAGGLE_VERIFICATION_SCHEMA_VERSION:
        raise KaggleDeliveryError("unsupported Kaggle verification schema")
    if payload.get("status") != "kaggle_execution_accepted":
        raise KaggleDeliveryError("Kaggle verification is not an accepted execution")
    preservation = _mapping(payload.get("preservation"), "Kaggle verification preservation")
    if preservation.get("accepted_output_preserved") is True:
        execution = _mapping(payload.get("execution"), "Kaggle verification execution")
        output = _mapping(execution.get("output"), "Kaggle verification output")
        private_output = root / "private" / "submission.csv"
        if not private_output.is_file() or private_output.is_symlink():
            raise KaggleDeliveryError("Kaggle verification is missing its preserved submission.csv")
        if _sha256_file(private_output) != _nonempty(output.get("sha256"), "Kaggle output sha256"):
            raise KaggleDeliveryError("preserved Kaggle submission checksum does not match evidence")
    if preservation.get("private_failure_report_preserved") is True:
        execution = _mapping(payload.get("execution"), "Kaggle verification execution")
        private_evidence = _mapping(execution.get("private_evidence"), "Kaggle private evidence")
        private_output = root / "private" / "execution.json"
        if not private_output.is_file() or private_output.is_symlink():
            raise KaggleDeliveryError("Kaggle verification is missing its private execution evidence")
        if _sha256_file(private_output) != _nonempty(private_evidence.get("sha256"), "private evidence sha256"):
            raise KaggleDeliveryError("preserved private execution evidence checksum does not match evidence")
    return payload


def load_kaggle_verification(root: str | Path) -> KaggleVerification:
    """Load one immutable external-execution evidence record."""

    verification_root = Path(root)
    payload = _verification_payload(verification_root)
    return KaggleVerification(
        verification_root.resolve(),
        _nonempty(payload.get("verification_checksum"), "verification_checksum"),
        payload,
    )


def promote_kaggle_candidate(
    delivery_root: str | Path,
    candidate_root: str | Path,
    verification_root: str | Path,
    output_root: str | Path,
) -> KagglePromotion:
    """Append a Kaggle-verified pointer while keeping the candidate immutable."""

    request = load_kaggle_delivery(delivery_root, candidate_root=candidate_root)
    verification = load_kaggle_verification(verification_root)
    if verification.payload.get("request_checksum") != request.request_checksum:
        raise KaggleDeliveryError("verification does not belong to this delivery request")
    candidate = _mapping(request.payload.get("candidate"), "request candidate")
    if verification.payload.get("candidate") != candidate:
        raise KaggleDeliveryError("verification candidate reference differs from the request")
    promotion = _mapping(verification.payload.get("promotion"), "verification promotion")
    if promotion.get("eligible") is not True:
        raise KaggleDeliveryError("Kaggle verification is not eligible for promotion")
    output = Path(output_root)
    _ensure_outside(output, (request.root, Path(candidate_root).resolve(), Path(verification_root).resolve()), "Kaggle promotion")
    payload: dict[str, object] = {
        "schema_version": KAGGLE_PROMOTION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "kaggle_verified_only",
        "candidate": dict(candidate),
        "request_checksum": request.request_checksum,
        "verification_checksum": verification.verification_checksum,
        "promotion": {
            "level": "kaggle_verified",
            "final_candidate": False,
            "owner_selection_required": True,
            "candidate_mutated": False,
            "export_allowed": False,
        },
        "gates": _mapping(_mapping(verification.payload.get("execution"), "verification execution").get("gates"), "verification gates"),
        "claims": {
            "actual_kaggle_workload_recorded": True,
            "hidden_scale_runtime_verified": True,
            "hidden_fallback_rate_known": True,
            "owner_selected_final": False,
        },
    }
    payload["promotion_checksum"] = _json_sha256(payload)

    def writer(staging: Path) -> None:
        atomic_json_write(staging / "promotion.json", payload)

    _publish_directory(output, writer)
    return load_kaggle_promotion(output)


def _promotion_payload(root: Path) -> Mapping[str, object]:
    if not root.is_dir() or root.is_symlink():
        raise KaggleDeliveryError(f"Kaggle promotion root is not a directory: {root}")
    payload = _read_json(root / "promotion.json", "Kaggle promotion")
    checksum = _nonempty(payload.get("promotion_checksum"), "promotion_checksum")
    without_checksum = dict(payload)
    without_checksum.pop("promotion_checksum", None)
    if _json_sha256(without_checksum) != checksum:
        raise KaggleDeliveryError("Kaggle promotion checksum does not match its contents")
    if payload.get("schema_version") != KAGGLE_PROMOTION_SCHEMA_VERSION:
        raise KaggleDeliveryError("unsupported Kaggle promotion schema")
    if payload.get("status") != "kaggle_verified_only":
        raise KaggleDeliveryError("Kaggle promotion is not a verified-only pointer")
    promotion = _mapping(payload.get("promotion"), "Kaggle promotion state")
    if promotion.get("final_candidate") is not False or promotion.get("owner_selection_required") is not True:
        raise KaggleDeliveryError("Kaggle promotion cannot silently become final")
    return payload


def load_kaggle_promotion(root: str | Path) -> KagglePromotion:
    """Load one immutable Kaggle promotion pointer."""

    promotion_root = Path(root)
    payload = _promotion_payload(promotion_root)
    return KagglePromotion(
        promotion_root.resolve(),
        _nonempty(payload.get("promotion_checksum"), "promotion_checksum"),
        payload,
    )


def _candidate_runtime(bundle: CandidateBundle) -> tuple[DecoderPolicy, PreprocessConfig, PackingConfig]:
    inference = bundle.inference_config
    packing_payload = _mapping(inference.get("packing"), "candidate packing")
    packing = PackingConfig(
        max_slices=_required_int(packing_payload.get("max_slices"), "candidate max_slices", positive=True),
        max_series=_required_int(packing_payload.get("max_series"), "candidate max_series", positive=True),
    )
    preprocessing = _mapping(inference.get("preprocessing"), "candidate preprocessing")
    try:
        geometry = GeometryPolicy.from_dict(preprocessing.get("geometry_policy"))
    except (TypeError, ValueError) as exc:
        raise KaggleDeliveryError("candidate geometry policy is invalid") from exc
    image_size = preprocessing.get("image_size")
    percentiles = preprocessing.get("normalization_percentiles")
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise KaggleDeliveryError("candidate preprocessing image_size is invalid")
    if not isinstance(percentiles, (list, tuple)) or len(percentiles) != 2:
        raise KaggleDeliveryError("candidate preprocessing percentiles are invalid")
    preprocess = PreprocessConfig(
        image_size=(int(str(image_size[0])), int(str(image_size[1]))),
        max_slices=packing.max_slices,
        max_series=packing.max_series,
        geometry_policy=geometry,
        normalization_domain=str(preprocessing.get("normalization_domain", "valid_slice")),
        normalization_percentiles=(float(str(percentiles[0])), float(str(percentiles[1]))),
        use_voi_lut=bool(preprocessing.get("use_voi_lut", False)),
        interpolation=str(preprocessing.get("interpolation", "nearest")),
        padding_fill_value=float(str(preprocessing.get("padding_fill_value", 0.0))),
        output_dtype=str(preprocessing.get("output_dtype", "float32")),
    )
    if preprocess.image_size != bundle.model_config.image_size:
        raise KaggleDeliveryError("candidate model and preprocessing image sizes differ")
    try:
        decoder = DecoderPolicy.from_dict(inference.get("decoder"))
        decoder.validate_runtime()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise KaggleDeliveryError("candidate decoder policy cannot run in this environment") from exc
    return decoder, preprocess, packing


def _load_kaggle_ensemble(bundle: CandidateBundle, device: str | torch.device) -> Any:
    """Load the candidate's own tested model/aggregation implementation."""

    try:
        return _load_candidate_models(bundle, device)
    except CandidateError as exc:
        raise KaggleDeliveryError("candidate model ensemble could not be restored") from exc


def _read_metadata_rows(path: Path, required_fields: Sequence[str], description: str) -> list[dict[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise KaggleDeliveryError(f"{description} is not a regular file: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            fields = tuple(reader.fieldnames or ())
            if not fields or len(set(fields)) != len(fields) or any(not field.strip() for field in fields):
                raise KaggleDeliveryError(f"{description} has an invalid header")
            missing = set(required_fields) - set(fields)
            if missing:
                raise KaggleDeliveryError(f"{description} is missing fields: {sorted(missing)}")
            rows: list[dict[str, str]] = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise KaggleDeliveryError(f"{description} contains a malformed row")
                rows.append({key: value for key, value in row.items() if key is not None})
    except (OSError, UnicodeError, csv.Error) as exc:
        raise KaggleDeliveryError(f"could not read {description}: {path}") from exc
    return rows


def _row_text(row: Mapping[str, str], field: str, description: str) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise KaggleDeliveryError(f"{description} has an empty {field}")
    return value


def _row_optional_bool(row: Mapping[str, str], field: str, description: str) -> bool | None:
    value = str(row.get(field, "")).strip()
    if not value:
        return None
    if value not in {"0", "1"}:
        raise KaggleDeliveryError(f"{description} has an invalid {field}={value!r}")
    return value == "1"


def _safe_input_component(value: str, field: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise KaggleDeliveryError(f"{field} is not a safe input path component")
    return value


def _series_input_files(series_dir: Path, input_root: Path) -> tuple[Path, ...]:
    if not series_dir.is_dir() or series_dir.is_symlink():
        return ()
    files: list[Path] = []
    for path in series_dir.rglob("*"):
        if path.is_symlink():
            raise KaggleDeliveryError(f"Kaggle series input contains a symlink: {path}")
        if path.is_file():
            files.append(path)
    return tuple(sorted(files, key=lambda path: path.relative_to(input_root).as_posix()))



def _read_kaggle_study_metadata(
    input_root: Path,
) -> tuple[tuple[str, ...], dict[str, list[Mapping[str, str]]]]:
    if not input_root.is_dir() or input_root.is_symlink():
        raise KaggleDeliveryError(f"Kaggle input root is not a directory: {input_root}")
    test_csv = input_root / "test.csv"
    test_series_csv = input_root / "test_series.csv"
    series_root = input_root / "test_series"
    if not series_root.is_dir() or series_root.is_symlink():
        raise KaggleDeliveryError(f"Kaggle input is missing test_series/: {series_root}")
    study_rows = _read_metadata_rows(test_csv, ("StudyInstanceUID",), "test.csv")
    series_rows = _read_metadata_rows(
        test_series_csv,
        ("StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"),
        "test_series.csv",
    )
    study_uids = tuple(_row_text(row, "StudyInstanceUID", "test.csv") for row in study_rows)
    if not study_uids:
        raise KaggleDeliveryError("test.csv contains no studies")
    if len(set(study_uids)) != len(study_uids):
        raise KaggleDeliveryError("test.csv repeats a StudyInstanceUID")
    study_set = set(study_uids)
    by_study: dict[str, list[Mapping[str, str]]] = {uid: [] for uid in study_uids}
    seen_series: set[str] = set()
    for row in series_rows:
        study_uid = _row_text(row, "StudyInstanceUID", "test_series.csv")
        series_uid = _row_text(row, "SeriesInstanceUID", "test_series.csv")
        if study_uid not in study_set:
            raise KaggleDeliveryError(f"test_series.csv references a study absent from test.csv: {study_uid}")
        if series_uid in seen_series:
            raise KaggleDeliveryError(f"test_series.csv repeats a SeriesInstanceUID: {series_uid}")
        seen_series.add(series_uid)
        by_study[study_uid].append(row)
    return study_uids, by_study


def _load_kaggle_study(
    input_root: Path,
    decoder: DecoderPolicy,
    study_uid: str,
    series_rows: Sequence[Mapping[str, str]],
    timing_profile: _KaggleDicomProfile,
) -> tuple[StudyRecord, dict[str, object]]:
    series_records: list[SeriesRecord] = []
    study_private: dict[str, object] = {"StudyInstanceUID": study_uid, "series": []}
    private_series = cast(list[dict[str, object]], study_private["series"])
    study_dir = input_root / "test_series" / _safe_input_component(study_uid, "StudyInstanceUID")
    for series_row in series_rows:
        series_uid = _row_text(series_row, "SeriesInstanceUID", "test_series.csv")
        series_dir = study_dir / _safe_input_component(series_uid, "SeriesInstanceUID")
        if series_dir.exists() and series_dir.is_symlink():
            raise KaggleDeliveryError(f"Kaggle series directory is a symlink: {series_dir}")
        index_started = time.perf_counter()
        files = _series_input_files(series_dir, input_root)
        index_seconds = time.perf_counter() - index_started
        timing_profile.archive_index_discovery_seconds += index_seconds
        timing_profile.index_scan_count += 1
        timing_profile.indexed_file_count += len(files)
        slices = []
        failures: list[dict[str, object]] = []
        timing_profile.expected_slice_count += len(files)
        for source_path in files:
            relative = source_path.relative_to(input_root).as_posix()
            timing_profile.decode_call_count += 1
            timing_profile.dicom_file_open_count += 1
            file_decode_started = time.perf_counter()
            try:
                with source_path.open("rb") as handle:
                    with _timed_pixel_decoding(timing_profile):
                        decoded = decode_stream(handle, filename=relative, policy=decoder)
                if decoded.study_instance_uid not in (None, study_uid):
                    raise DicomDecodeError("decoded StudyInstanceUID disagrees with test.csv")
                if decoded.series_instance_uid not in (None, series_uid):
                    raise DicomDecodeError("decoded SeriesInstanceUID disagrees with test_series.csv")
                slices.append(decoded)
                timing_profile.decoded_slice_count += 1
            except DecoderInfrastructureError:
                raise
            except (DicomDecodeError, OSError) as exc:
                reason = type(exc).__name__
                timing_profile.failure_reasons[reason] = timing_profile.failure_reasons.get(reason, 0) + 1
                failures.append({"source_entry_name": relative, "reason": reason, "detail": str(exc)})
            finally:
                timing_profile.dicom_read_decode_seconds += time.perf_counter() - file_decode_started
        if not files:
            timing_profile.failure_reasons["missing_series_input"] = (
                timing_profile.failure_reasons.get("missing_series_input", 0) + 1
            )
            failures.append(
                {"source_entry_name": "", "reason": "missing_series_input", "detail": str(series_dir)}
            )
        series_records.append(
            SeriesRecord(
                study_instance_uid=study_uid,
                series_instance_uid=series_uid,
                slices=tuple(slices),
                anatomical_plane=str(series_row.get("Anatomical_Plane", "")).strip() or None,
                fluid_sensitive=_row_optional_bool(series_row, "Fluid_Sensitive", "test_series.csv"),
                fat_suppression=_row_optional_bool(series_row, "Fat_Suppression", "test_series.csv"),
                source_entry_names=tuple(
                    source_path.relative_to(input_root).as_posix() for source_path in files
                ),
            )
        )
        private_series.append(
            {
                "SeriesInstanceUID": series_uid,
                "expected_slice_count": len(files),
                "decoded_slice_count": len(slices),
                "failures": failures,
            }
        )
    return StudyRecord(study_uid, tuple(series_records)), study_private


def _iter_kaggle_studies(
    input_root: Path,
    decoder: DecoderPolicy,
    *,
    study_uids: Sequence[str] | None = None,
    by_study: Mapping[str, Sequence[Mapping[str, str]]] | None = None,
    skip_studies: int = 0,
    profile: _KaggleDicomProfile | None = None,
) -> Iterator[tuple[int, StudyRecord, dict[str, object]]]:
    """Decode one study per yield and drop it before beginning the next study."""

    if study_uids is None or by_study is None:
        study_uids, by_study = _read_kaggle_study_metadata(input_root)
    if skip_studies < 0 or skip_studies > len(study_uids):
        raise KaggleDeliveryError("prediction checkpoint is outside the study order")
    decoder.validate_runtime()
    timing_profile = profile or _KaggleDicomProfile()
    for study_index in range(skip_studies, len(study_uids)):
        study_uid = study_uids[study_index]
        study, study_private = _load_kaggle_study(
            input_root,
            decoder,
            study_uid,
            by_study[study_uid],
            timing_profile,
        )
        yield study_index, study, study_private
        study = None  # type: ignore[assignment]
        study_private = {}
        gc.collect()
        rss_after_release = _current_rss_bytes()
        if rss_after_release is not None:
            timing_profile.rss_after_study_release_bytes.append(rss_after_release)


def _prediction_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.progress.jsonl")


class _KagglePredictionCheckpoint:
    """Private, checksum-bound study results used for safe restart recovery."""

    def __init__(
        self,
        path: Path,
        *,
        identity: Mapping[str, object],
        expected_uids: Sequence[str],
    ) -> None:
        self.path = path
        self.identity = dict(identity)
        self.expected_uids = tuple(expected_uids)
        self.uid_order_sha256 = _json_sha256(list(self.expected_uids))
        self.predictions: dict[str, np.ndarray] = {}
        self.private_records: list[dict[str, object]] = []
        self.resumed_row_count = 0
        self._header: dict[str, object] = {
            "kind": "header",
            "schema_version": KAGGLE_PREDICTION_CHECKPOINT_SCHEMA_VERSION,
            "identity": self.identity,
            "expected_study_count": len(self.expected_uids),
            "uid_order_sha256": self.uid_order_sha256,
        }
        self._open_or_load()

    @property
    def completed_count(self) -> int:
        return len(self.private_records)

    def _open_or_load(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise KaggleDeliveryError("prediction checkpoint cannot be a symlink")
        if not self.path.exists():
            try:
                with self.path.open("xb") as handle:
                    handle.write((_canonical_json(self._header) + "\n").encode("utf-8"))
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError as exc:
                raise KaggleDeliveryError("prediction checkpoint appeared concurrently") from exc
            return
        if not self.path.is_file():
            raise KaggleDeliveryError("prediction checkpoint is not a regular file")
        try:
            data = self.path.read_bytes()
        except OSError as exc:
            raise KaggleDeliveryError("prediction checkpoint could not be read") from exc
        lines = data.splitlines(keepends=True)
        if not lines or not lines[0].endswith(b"\n"):
            raise KaggleDeliveryError("prediction checkpoint header is incomplete")
        try:
            header = json.loads(lines[0].decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise KaggleDeliveryError("prediction checkpoint header is invalid") from exc
        if header != self._header:
            raise KaggleDeliveryError("prediction checkpoint belongs to a different candidate or input")
        valid_bytes = len(lines[0])
        for line in lines[1:]:
            if not line.endswith(b"\n"):
                break
            try:
                row = json.loads(line.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise KaggleDeliveryError("prediction checkpoint contains a malformed completed row") from exc
            index = len(self.private_records)
            if not isinstance(row, dict):
                raise KaggleDeliveryError("prediction checkpoint contains a malformed completed row")
            row_checksum = row.get("row_sha256")
            unsigned_row = dict(row)
            unsigned_row.pop("row_sha256", None)
            if row_checksum != _json_sha256(unsigned_row):
                raise KaggleDeliveryError("prediction checkpoint row checksum is invalid")
            if not isinstance(row, dict) or row.get("kind") != "study" or row.get("index") != index:
                raise KaggleDeliveryError("prediction checkpoint study order is invalid")
            if index >= len(self.expected_uids) or row.get("study_uid") != self.expected_uids[index]:
                raise KaggleDeliveryError("prediction checkpoint UID order differs from the current input")
            raw_scores = row.get("scores")
            private_record = row.get("private_record")
            if not isinstance(raw_scores, list) or len(raw_scores) != NUM_TARGETS or not isinstance(private_record, dict):
                raise KaggleDeliveryError("prediction checkpoint row has an invalid payload")
            if private_record.get("StudyInstanceUID") != self.expected_uids[index]:
                raise KaggleDeliveryError("prediction checkpoint diagnostics belong to a different study")
            try:
                scores = np.asarray([float(value) for value in raw_scores], dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise KaggleDeliveryError("prediction checkpoint contains invalid scores") from exc
            if not np.isfinite(scores).all() or (scores < 0.0).any() or (scores > 1.0).any():
                raise KaggleDeliveryError("prediction checkpoint scores are outside [0, 1]")
            self.predictions[self.expected_uids[index]] = scores
            self.private_records.append(private_record)
            valid_bytes += len(line)
        if valid_bytes != len(data):
            try:
                with self.path.open("r+b") as handle:
                    handle.truncate(valid_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise KaggleDeliveryError("incomplete prediction checkpoint tail could not be removed") from exc
        self.resumed_row_count = self.completed_count

    def append(
        self,
        index: int,
        study_uid: str,
        scores: np.ndarray,
        private_record: dict[str, object],
    ) -> None:
        if index != self.completed_count or index >= len(self.expected_uids):
            raise KaggleDeliveryError("prediction checkpoint rows must be committed in UID order")
        if study_uid != self.expected_uids[index]:
            raise KaggleDeliveryError("prediction checkpoint row does not match the expected UID")
        score_values = np.asarray(scores, dtype=np.float64)
        if score_values.shape != (NUM_TARGETS,) or not np.isfinite(score_values).all():
            raise KaggleDeliveryError("prediction checkpoint received invalid scores")
        if (score_values < 0.0).any() or (score_values > 1.0).any():
            raise KaggleDeliveryError("prediction checkpoint scores are outside [0, 1]")
        row: dict[str, object] = {
            "kind": "study",
            "index": index,
            "study_uid": study_uid,
            "scores": [float(value) for value in score_values],
            "private_record": private_record,
        }
        row["row_sha256"] = _json_sha256(row)
        try:
            with self.path.open("ab") as handle:
                handle.write((_canonical_json(row) + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise KaggleDeliveryError("prediction checkpoint row could not be committed") from exc
        self.predictions[study_uid] = score_values.copy()
        self.private_records.append(private_record)


def _checkpoint_identity(
    *,
    candidate_manifest_checksum: str,
    delivery_request_checksum: str | None,
    test_csv_sha256: str,
    test_series_csv_sha256: str,
    input_manifest_sha256: str | None,
) -> dict[str, object]:
    return {
        "candidate_manifest_checksum": candidate_manifest_checksum,
        "delivery_request_checksum": delivery_request_checksum,
        "test_csv_sha256": test_csv_sha256,
        "test_series_csv_sha256": test_series_csv_sha256,
        "input_manifest_sha256": input_manifest_sha256,
    }


def _execution_input_summary(
    expected_uids: Sequence[str],
    total_series_count: int,
    private_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(private_records) != len(expected_uids):
        raise KaggleDeliveryError("completed prediction checkpoint does not cover every study")
    series_per_study: list[int] = []
    slices_per_study: list[int] = []
    decoded_slices = 0
    failure_reasons: dict[str, int] = {}
    for record in private_records:
        series = cast(list[Mapping[str, object]], record.get("series", []))
        series_per_study.append(len(series))
        study_expected = 0
        for series_record in series:
            study_expected += int(str(series_record.get("expected_slice_count", 0)))
            decoded_slices += int(str(series_record.get("decoded_slice_count", 0)))
            for failure in cast(list[Mapping[str, object]], series_record.get("failures", [])):
                reason = str(failure.get("reason", "unknown"))
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        slices_per_study.append(study_expected)
    workload_profile = _workload_profile(series_per_study, slices_per_study)
    long_tail = cast(Mapping[str, object], workload_profile["long_tail"])
    series_threshold = int(str(long_tail["series_threshold"]))
    slice_threshold = int(str(long_tail["slice_threshold"]))
    for record, study_series_count, study_slice_count in zip(
        private_records,
        series_per_study,
        slices_per_study,
    ):
        if isinstance(record, dict):
            record["workload"] = {
                "series_count": study_series_count,
                "expected_slice_count": study_slice_count,
                "long_tail": (
                    study_series_count >= series_threshold
                    or study_slice_count >= slice_threshold
                ),
            }
    return {
        "study_count": len(expected_uids),
        "series_count": total_series_count,
        "slice_count": sum(slices_per_study),
        "decoded_slice_count": decoded_slices,
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "decode_failure_count": sum(failure_reasons.values()),
        **workload_profile,
    }


def _current_rss_bytes() -> int | None:
    """Return current RSS where Linux procfs exposes it."""

    try:
        with Path("/proc/self/statm").open(encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _system_memory_snapshot_bytes() -> dict[str, int | None]:
    """Read exposed system and cgroup memory values without inferring a limit."""

    memory: dict[str, int | None] = {
        "system_total_bytes": None,
        "system_available_bytes": None,
        "cgroup_limit_bytes": None,
        "cgroup_current_bytes": None,
    }
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, _, raw_value = line.partition(":")
            if key not in {"MemTotal", "MemAvailable"}:
                continue
            value_kib = int(raw_value.strip().split()[0])
            field_name = "system_total_bytes" if key == "MemTotal" else "system_available_bytes"
            memory[field_name] = value_kib * 1024
    except (OSError, ValueError, IndexError):
        pass
    for limit_path, current_path in (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    ):
        try:
            raw_limit = limit_path.read_text(encoding="ascii").strip()
            if raw_limit != "max":
                memory["cgroup_limit_bytes"] = int(raw_limit)
            raw_current = current_path.read_text(encoding="ascii").strip()
            memory["cgroup_current_bytes"] = int(raw_current)
            break
        except (OSError, ValueError):
            continue
    return memory


def _rss_sample_summary(values: Sequence[int]) -> dict[str, object]:
    if not values:
        return {
            "sample_count": 0,
            "minimum_bytes": None,
            "maximum_bytes": None,
            "first_bytes": None,
            "last_bytes": None,
        }
    return {
        "sample_count": len(values),
        "minimum_bytes": min(values),
        "maximum_bytes": max(values),
        "first_bytes": values[0],
        "last_bytes": values[-1],
    }


def _predict_kaggle_study(
    engine: InferenceEngine,
    study: StudyRecord,
    input_failures: Sequence[SeriesFailure],
    input_warnings: Sequence[str] = (),
) -> tuple[StudyPrediction, dict[str, float]]:
    preprocessing_started = time.perf_counter()
    prepared = prepare_study(study, engine.preprocess_config)
    preprocessing_resize_seconds = time.perf_counter() - preprocessing_started
    packing_started = time.perf_counter()
    packed = pack_study(
        study,
        prepared.series,
        engine.preprocess_config,
        packing_config=engine.packing_config,
    )
    packing_seconds = time.perf_counter() - packing_started
    preprocessing_seconds = preprocessing_resize_seconds + packing_seconds
    valid_series_count = int(packed.series_mask.sum())
    rejected_series = tuple(input_failures) + prepared.failures
    warnings = tuple(input_warnings) + prepared.warnings
    if valid_series_count == 0:
        return (
            StudyPrediction(
                study_instance_uid=study.study_instance_uid,
                scores=engine.fallback_scores.copy(),
                used_fallback=True,
                valid_series_count=0,
                rejected_series=rejected_series,
                sampled_source_indices=packed.sampled_source_indices,
                warnings=warnings,
            ),
            {
                "preprocessing_seconds": preprocessing_seconds,
                "preprocessing_resize_seconds": preprocessing_resize_seconds,
                "packing_seconds": packing_seconds,
                "inference_ensembling_seconds": 0.0,
            },
        )
    inference_started = time.perf_counter()
    images = torch.from_numpy(packed.images).unsqueeze(0).to(engine.device)
    slice_mask = torch.from_numpy(packed.slice_mask).unsqueeze(0).to(engine.device)
    series_mask = torch.from_numpy(packed.series_mask).unsqueeze(0).to(engine.device)
    with torch.inference_mode():
        scores_tensor = cast(Any, engine.model).predict_proba(images, slice_mask, series_mask)
    scores = np.asarray(scores_tensor.squeeze(0).detach().cpu().numpy(), dtype=np.float64)
    inference_seconds = time.perf_counter() - inference_started
    if scores.shape != (NUM_TARGETS,) or not np.isfinite(scores).all() or (scores < 0.0).any() or (scores > 1.0).any():
        raise KaggleDeliveryError(f"model returned invalid scores for study {study.study_instance_uid}")
    return (
        StudyPrediction(
            study_instance_uid=study.study_instance_uid,
            scores=scores,
            used_fallback=False,
            valid_series_count=valid_series_count,
            rejected_series=rejected_series,
            sampled_source_indices=packed.sampled_source_indices,
            warnings=warnings,
        ),
        {
            "preprocessing_seconds": preprocessing_seconds,
            "preprocessing_resize_seconds": preprocessing_resize_seconds,
            "packing_seconds": packing_seconds,
            "inference_ensembling_seconds": inference_seconds,
        },
    )


def run_kaggle_baseline(
    candidate_root: str | Path,
    input_root: str | Path,
    output_path: str | Path,
    *,
    evidence_path: str | Path | None = None,
    private_evidence_path: str | Path | None = None,
    delivery_request_checksum: str | None = None,
    delivery_notebook_sha256: str | None = None,
    delivery_adapter_sha256: str | None = None,
    delivery_request_root: str | Path | None = None,
    owner_approval_path: str | Path | None = None,
    delivery_notebook_path: str | Path | None = None,
    delivery_adapter_path: str | Path | None = None,
    delivery_runtime_wheel_path: str | Path | None = None,
    delivery_runtime_wheel_paths: Sequence[str | Path] | None = None,
    runtime_install_evidence: Mapping[str, object] | None = None,
    notebook_stage_timings: Mapping[str, float] | None = None,
    notebook_timing_details: Mapping[str, object] | None = None,
    validated_layout: KaggleInputLayout | None = None,
    device: str | torch.device = "cpu",
    allow_local_smoke: bool = False,
    notebook_started_at: float | None = None,
) -> KaggleInferenceResult:
    """Run the frozen model on Kaggle's test CSV/DICOM layout.

    The adapter reads ``test.csv``, ``test_series.csv``, and ``test_series/``
    from an already-mounted Kaggle input.  It reuses the candidate's decoder,
    preprocessing, hierarchical model and exact submission writer; it does
    not read reports or perform network operations.  Local execution requires
    ``allow_local_smoke=True`` and is not Kaggle verification.
    """

    if not allow_local_smoke and not (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE")
    ):
        raise KaggleDeliveryError("Kaggle baseline must run in a Kaggle notebook; use local smoke only explicitly")
    adapter_preflight_started = time.perf_counter()
    process_wall_started = adapter_preflight_started
    process_cpu_started = os.times()
    timing_details_input = dict(notebook_timing_details or {})
    delivery_request_validation_seconds = 0.0
    candidate_load_hash_verification_seconds = 0.0
    runtime_pin_gate_seconds = 0.0
    approval_sha256: str | None = None
    delivery_validation_reused = False
    candidate_hashes_reused = False
    request: KaggleDeliveryRequest | None = None
    bundle: CandidateBundle | None = None
    candidate: dict[str, object] | None = None
    workload_kind = "local_smoke"
    input_manifest_sha256: str | None = None
    if not allow_local_smoke:
        required_paths = (
            (delivery_request_root, "delivery_request_root"),
            (owner_approval_path, "owner_approval_path"),
            (delivery_notebook_path, "delivery_notebook_path"),
            (delivery_adapter_path, "delivery_adapter_path"),
        )
        missing = [field for value, field in required_paths if value is None]
        if missing:
            raise KaggleDeliveryError("actual Kaggle execution requires: " + ", ".join(missing))
        if notebook_started_at is None:
            raise KaggleDeliveryError("actual Kaggle execution requires a notebook-start timer")
        if notebook_stage_timings is None:
            raise KaggleDeliveryError("actual Kaggle execution requires notebook startup/import timings")
        if validated_layout is not None:
            validation = _reuse_discovered_kaggle_inputs(
                validated_layout,
                candidate_root=candidate_root,
                input_root=input_root,
                delivery_request_root=cast(str | Path, delivery_request_root),
                owner_approval_path=cast(str | Path, owner_approval_path),
                delivery_notebook_path=cast(str | Path, delivery_notebook_path),
                delivery_adapter_path=cast(str | Path, delivery_adapter_path),
                delivery_request_checksum=delivery_request_checksum,
                delivery_notebook_sha256=delivery_notebook_sha256,
                delivery_adapter_sha256=delivery_adapter_sha256,
                delivery_runtime_wheel_path=delivery_runtime_wheel_path,
                delivery_runtime_wheel_paths=delivery_runtime_wheel_paths,
            )
            request = validation.request
            bundle = validation.candidate_bundle
            candidate = dict(validation.candidate_reference)
            delivery_request_checksum = request.request_checksum
            delivery_notebook_sha256 = validation.delivery_notebook_sha256
            delivery_adapter_sha256 = validation.delivery_adapter_sha256
            approval_sha256 = validation.approval_sha256
            candidate_load_hash_verification_seconds = validation.candidate_load_hash_verification_seconds
            delivery_validation_reused = True
            candidate_hashes_reused = True
            workload_kind = validated_layout.workload_kind
            input_manifest_sha256 = validated_layout.input_manifest_sha256
        else:
            request_validation_started = time.perf_counter()
            request = load_kaggle_delivery(
                cast(str | Path, delivery_request_root),
                candidate_root=candidate_root,
                delivery_notebook_path=cast(str | Path, delivery_notebook_path),
                delivery_adapter_path=cast(str | Path, delivery_adapter_path),
                delivery_runtime_wheel_path=delivery_runtime_wheel_path,
                delivery_runtime_wheel_paths=delivery_runtime_wheel_paths,
            )
            if request.payload.get("delivery_runtime") is not None and not _runtime_wheel_paths(
                delivery_runtime_wheel_path,
                delivery_runtime_wheel_paths,
            ):
                raise KaggleDeliveryError("actual execution requires the request's offline runtime wheelhouse")
            if delivery_request_checksum is not None and delivery_request_checksum != request.request_checksum:
                raise KaggleDeliveryError("supplied delivery request checksum differs from request.json")
            delivery_request_checksum = request.request_checksum
            request_notebook = _mapping(request.payload.get("delivery_notebook"), "request delivery notebook")
            request_adapter = _mapping(request.payload.get("delivery_adapter"), "request delivery adapter")
            request_notebook_sha256 = _nonempty(request_notebook.get("sha256"), "delivery notebook sha256")
            request_adapter_sha256 = _nonempty(request_adapter.get("sha256"), "delivery adapter sha256")
            if delivery_notebook_sha256 is not None and delivery_notebook_sha256 != request_notebook_sha256:
                raise KaggleDeliveryError("supplied delivery notebook checksum differs from request.json")
            if delivery_adapter_sha256 is not None and delivery_adapter_sha256 != request_adapter_sha256:
                raise KaggleDeliveryError("supplied delivery adapter checksum differs from request.json")
            delivery_notebook_sha256 = request_notebook_sha256
            delivery_adapter_sha256 = request_adapter_sha256
            approval_path = Path(cast(str | Path, owner_approval_path))
            _approval_payload(
                approval_path,
                request.request_checksum,
                expected_scope=_request_approval_scope(request.payload),
            )
            approval_sha256 = _sha256_file(approval_path)
            delivery_request_validation_seconds = time.perf_counter() - request_validation_started
            input_reference = _execution_input_reference(
                request.payload,
                default_mount_slug=KAGGLE_COMPETITION_SLUG,
            )
            workload_kind = str(input_reference["workload_kind"])
            if input_reference["workload_kind"] == "representative":
                manifest_path = Path(input_root) / str(input_reference["manifest_filename"])
                if manifest_path.is_file() and not manifest_path.is_symlink():
                    input_manifest_sha256 = _sha256_file(manifest_path)
        _nonempty(delivery_request_checksum, "delivery_request_checksum")
        for stage in ("startup_setup", "imports"):
            _required_float(notebook_stage_timings.get(stage), f"notebook {stage} seconds")
    else:
        notebook_stage_timings = {"startup_setup": 0.0, "imports": 0.0}
        representative_manifest = Path(input_root) / "input-manifest.json"
        if representative_manifest.is_file() and not representative_manifest.is_symlink():
            input_manifest_sha256 = _sha256_file(representative_manifest)
            workload_kind = "representative"
    if bundle is None or candidate is None:
        candidate_started = time.perf_counter()
        bundle = load_candidate(candidate_root)
        candidate = _candidate_reference(bundle)
        candidate_load_hash_verification_seconds = time.perf_counter() - candidate_started
    candidate_checkpoints = candidate.get("model_checkpoints")
    if not isinstance(candidate_checkpoints, list):
        raise KaggleDeliveryError("candidate checkpoint reference is invalid")
    runtime_environment = _mapping(candidate.get("environment"), "candidate environment")
    runtime_pin_started = time.perf_counter()
    runtime_audit = validate_kaggle_runtime_pins(
        _mapping(runtime_environment.get("runtime_pins"), "candidate runtime pins")
    )
    runtime_pin_gate_seconds = time.perf_counter() - runtime_pin_started
    audited_runtime_install: dict[str, object] | None = None
    if not allow_local_smoke:
        if request is None:
            raise KaggleDeliveryError("actual Kaggle execution did not load a delivery request")
        audited_runtime_install = _runtime_install_audit(runtime_install_evidence, request, runtime_audit)
    elif runtime_install_evidence is not None:
        raise KaggleDeliveryError("local smoke cannot claim an offline Kaggle runtime installation")
    delivery_size_bytes = _directory_size_bytes(request.root) if request is not None else None
    checkpoint_sizes_bytes = {
        str(item["path"]): (bundle.root / str(item["path"])).stat().st_size
        for item in candidate_checkpoints
    }
    runtime_package_sizes_bytes = {
        str(audit.get("package")): int(path.stat().st_size)
        for path, audit in zip(
            _runtime_wheel_paths(delivery_runtime_wheel_path, delivery_runtime_wheel_paths),
            _runtime_wheel_references(request.payload.get("delivery_runtime"))
            if request is not None and request.payload.get("delivery_runtime") is not None
            else (),
        )
    }
    output = Path(output_path)
    evidence_destination = Path(evidence_path) if evidence_path is not None else None
    private_destination = Path(private_evidence_path) if private_evidence_path is not None else None
    destinations = [output] + ([evidence_destination] if evidence_destination is not None else [])
    destinations += [private_destination] if private_destination is not None else []
    if len(set(destinations)) != len(destinations):
        raise KaggleDeliveryError("Kaggle output and evidence destinations must be distinct")
    for destination in destinations:
        if destination is None:
            continue
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite protected Kaggle output {destination}")
    _ensure_outside(output, (Path(candidate_root).resolve(),), "Kaggle output")
    if evidence_destination is not None:
        _ensure_outside(evidence_destination, (Path(candidate_root).resolve(),), "Kaggle evidence")
    if private_destination is not None:
        _ensure_outside(private_destination, (Path(candidate_root).resolve(),), "Kaggle private evidence")


    adapter_preflight_seconds = time.perf_counter() - adapter_preflight_started
    started = notebook_started_at if notebook_started_at is not None else time.perf_counter()
    input_path = Path(input_root)
    metadata_paths = (input_path / "test.csv", input_path / "test_series.csv")
    test_csv_sha256 = _sha256_file(metadata_paths[0])
    test_series_csv_sha256 = _sha256_file(metadata_paths[1])
    decoder, preprocess, packing = _candidate_runtime(bundle)
    metadata_started = time.perf_counter()
    expected_uids, by_study = _read_kaggle_study_metadata(input_path)
    metadata_seconds = time.perf_counter() - metadata_started
    series_count = sum(len(rows) for rows in by_study.values())
    checkpoint_identity = _checkpoint_identity(
        candidate_manifest_checksum=str(candidate["manifest_checksum"]),
        delivery_request_checksum=delivery_request_checksum,
        test_csv_sha256=test_csv_sha256,
        test_series_csv_sha256=test_series_csv_sha256,
        input_manifest_sha256=input_manifest_sha256,
    )
    checkpoint_path = _prediction_checkpoint_path(output)
    checkpoint = _KagglePredictionCheckpoint(
        checkpoint_path,
        identity=checkpoint_identity,
        expected_uids=expected_uids,
    )
    resumed_row_count = checkpoint.resumed_row_count
    dicom_profile = _KaggleDicomProfile()
    dicom_profile.stage_rss_bytes["adapter_start"] = _current_rss_bytes()
    dicom_profile.stage_rss_bytes["after_metadata"] = _current_rss_bytes()
    system_memory_start = _system_memory_snapshot_bytes()

    device_object = torch.device(device)
    if device_object.type == "cuda":
        if not torch.cuda.is_available():
            raise KaggleDeliveryError("CUDA was requested but is unavailable")
        torch.cuda.reset_peak_memory_stats(device_object)
    model_seconds = 0.0
    model = None
    engine: InferenceEngine | None = None
    if checkpoint.completed_count < len(expected_uids):
        model_started = time.perf_counter()
        model = _load_kaggle_ensemble(bundle, device)
        model_seconds = time.perf_counter() - model_started
        engine = InferenceEngine(
            cast(Any, model),
            preprocess,
            device=device,
            fallback_scores=tuple([0.5] * NUM_TARGETS),
            packing_config=packing,
        )
    dicom_profile.stage_rss_bytes["after_model_load"] = _current_rss_bytes()

    preprocessing_seconds = 0.0
    preprocessing_resize_seconds = 0.0
    packing_seconds = 0.0
    inference_seconds = 0.0
    decode_started = time.perf_counter()
    for study_index, study, study_private in _iter_kaggle_studies(
        input_path,
        decoder,
        study_uids=expected_uids,
        by_study=by_study,
        skip_studies=checkpoint.completed_count,
        profile=dicom_profile,
    ):
        if engine is None:
            raise KaggleDeliveryError("prediction checkpoint requested inference without a loaded model")
        rss_after_decode = _current_rss_bytes()
        if rss_after_decode is not None:
            dicom_profile.rss_after_study_decode_bytes.append(rss_after_decode)
        input_failures = tuple(
            SeriesFailure(
                str(series["SeriesInstanceUID"]),
                str(failure["reason"]),
            )
            for series in cast(list[Mapping[str, object]], study_private["series"])
            for failure in cast(list[Mapping[str, object]], series["failures"])
        )
        prediction, timing = _predict_kaggle_study(engine, study, input_failures)
        preprocessing_seconds += timing["preprocessing_seconds"]
        preprocessing_resize_seconds += timing.get(
            "preprocessing_resize_seconds",
            timing["preprocessing_seconds"],
        )
        packing_seconds += timing.get("packing_seconds", 0.0)
        inference_seconds += timing["inference_ensembling_seconds"]
        study_private["prediction"] = {
            "used_full_study_fallback": prediction.used_fallback,
            "valid_series_count": prediction.valid_series_count,
            "rejected_series": [
                {
                    "series_instance_uid": failure.series_instance_uid,
                    "reason": failure.reason,
                }
                for failure in prediction.rejected_series
            ],
        }
        checkpoint.append(study_index, study.study_instance_uid, prediction.scores, study_private)
        rss_after_prediction = _current_rss_bytes()
        if rss_after_prediction is not None:
            dicom_profile.rss_after_study_prediction_bytes.append(rss_after_prediction)
        del prediction, timing, input_failures, study, study_private
    decode_seconds = time.perf_counter() - decode_started
    study_iteration_overhead_seconds = max(
        0.0,
        decode_seconds
        - dicom_profile.dicom_read_decode_seconds
        - preprocessing_seconds
        - inference_seconds,
    )
    if checkpoint.completed_count != len(expected_uids):
        raise KaggleDeliveryError("study stream ended before every expected UID was checkpointed")
    # Reopen and validate the durable prefix before publishing the CSV.
    reloaded_checkpoint = _KagglePredictionCheckpoint(
        checkpoint_path,
        identity=checkpoint_identity,
        expected_uids=expected_uids,
    )
    if reloaded_checkpoint.predictions.keys() != checkpoint.predictions.keys() or any(
        not np.array_equal(reloaded_checkpoint.predictions[uid], checkpoint.predictions[uid])
        for uid in checkpoint.predictions
    ):
        raise KaggleDeliveryError("prediction checkpoint save/load changed one or more scores")
    if reloaded_checkpoint.private_records != checkpoint.private_records:
        raise KaggleDeliveryError("prediction checkpoint save/load changed study diagnostics")
    checkpoint = reloaded_checkpoint
    input_summary = _execution_input_summary(
        expected_uids,
        series_count,
        checkpoint.private_records,
    )
    del engine, model
    gc.collect()
    dicom_profile.stage_rss_bytes["after_inference_release"] = _current_rss_bytes()
    system_memory_after_inference = _system_memory_snapshot_bytes()

    output_started = time.perf_counter()
    write_submission(
        output,
        expected_study_uids=expected_uids,
        predictions=checkpoint.predictions,
    )
    submission_validation = validate_submission_csv(output, expected_uids)
    csv_seconds = time.perf_counter() - output_started
    dicom_profile.stage_rss_bytes["after_csv_validation"] = _current_rss_bytes()
    system_memory_after_csv = _system_memory_snapshot_bytes()
    current_rss_after_study_decode = _rss_sample_summary(dicom_profile.rss_after_study_decode_bytes)
    current_rss_after_study_prediction = _rss_sample_summary(
        dicom_profile.rss_after_study_prediction_bytes
    )
    current_rss_after_study_release = _rss_sample_summary(dicom_profile.rss_after_study_release_bytes)
    memory_profile = {
        "sampling_source": "/proc/self/statm (current RSS)" if dicom_profile.stage_rss_bytes["adapter_start"] is not None else "unavailable",
        "stage_rss_bytes": dict(dicom_profile.stage_rss_bytes),
        "after_each_study_decode": current_rss_after_study_decode,
        "after_each_study_prediction": current_rss_after_study_prediction,
        "after_each_study_release": current_rss_after_study_release,
        "active_study_limit": 1,
        "decoded_pixel_arrays_retained_after_prediction": False,
        "release_action": "drop_study_references_and_collect_garbage",
        "exposed_system_memory": {
            "start": system_memory_start,
            "after_inference": system_memory_after_inference,
            "after_csv_validation": system_memory_after_csv,
        },
    }
    stage_timings = {
        "startup_setup": float(notebook_stage_timings["startup_setup"]),
        "imports": float(notebook_stage_timings["imports"]),
        "model_metadata_loading": metadata_seconds + model_seconds,
        "dicom_read_decode": dicom_profile.dicom_read_decode_seconds,
        "preprocessing": preprocessing_seconds,
        "inference_ensembling": inference_seconds,
        "study_iteration_overhead": study_iteration_overhead_seconds,
        "csv_creation": csv_seconds,
    }
    fallback_numerator = sum(
        bool(
            cast(
                Mapping[str, object],
                record.get("prediction", {}),
            ).get("used_full_study_fallback", False)
        )
        for record in checkpoint.private_records
    )
    fallback_denominator = len(expected_uids)
    fallback_rate = fallback_numerator / fallback_denominator
    if device_object.type == "cuda":
        peak_vram_bytes = int(torch.cuda.max_memory_allocated(device_object))
    else:
        peak_vram_bytes = 0

    def timing_detail_seconds(name: str, fallback: float = 0.0) -> float:
        return _required_float(timing_details_input.get(name, fallback), f"Kaggle timing detail {name}")

    whole_seconds = time.perf_counter() - started
    process_wall_seconds = time.perf_counter() - process_wall_started
    process_cpu_finished = os.times()
    process_user_seconds = process_cpu_finished.user - process_cpu_started.user
    process_system_seconds = process_cpu_finished.system - process_cpu_started.system
    process_cpu_seconds = process_user_seconds + process_system_seconds
    try:
        affinity_cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_cpu_count = None
    try:
        interop_thread_count: int | None = torch.get_num_interop_threads()
    except RuntimeError:
        interop_thread_count = None
    projection = _fixed_plus_scaling_projection(
        whole_seconds=whole_seconds,
        study_count=len(expected_uids),
        target_studies=KAGGLE_TARGET_STUDIES,
        stages={
            **stage_timings,
            "input_discovery": timing_detail_seconds("input_discovery_seconds"),
            "adapter_preflight": adapter_preflight_seconds,
        },
    )

    notebook_startup_preflight_seconds = timing_detail_seconds(
        "notebook_startup_preflight_seconds",
        stage_timings["startup_setup"],
    )
    offline_wheel_verification_seconds = timing_detail_seconds("offline_wheel_verification_seconds")
    offline_wheel_install_seconds = timing_detail_seconds("offline_wheel_install_seconds")
    offline_runtime_probe_seconds = timing_detail_seconds("offline_runtime_probe_seconds")
    input_discovery_seconds = timing_detail_seconds("input_discovery_seconds")
    accounted_seconds = (
        notebook_startup_preflight_seconds
        + stage_timings["imports"]
        + input_discovery_seconds
        + adapter_preflight_seconds
        + stage_timings["model_metadata_loading"]
        + stage_timings["dicom_read_decode"]
        + stage_timings["preprocessing"]
        + stage_timings["inference_ensembling"]
        + stage_timings["study_iteration_overhead"]
        + stage_timings["csv_creation"]
    )
    detailed_timing = {
        "schema_version": KAGGLE_PROFILE_SCHEMA_VERSION,
        "notebook_startup_preflight_seconds": notebook_startup_preflight_seconds,
        "offline_wheel_verification_seconds": offline_wheel_verification_seconds,
        "offline_wheel_install_seconds": offline_wheel_install_seconds,
        "offline_runtime_probe_seconds": offline_runtime_probe_seconds,
        "input_discovery_seconds": input_discovery_seconds,
        "adapter_preflight_seconds": adapter_preflight_seconds,
        "delivery_validation_mode": (
            "discovered_layout_reused" if delivery_validation_reused else "full_request_revalidation"
        ),
        "delivery_validation_reused": delivery_validation_reused,
        "candidate_hashes_reused": candidate_hashes_reused,
        "runtime_wheel_hashes_reused": delivery_validation_reused,
        "approval_hash_reused": delivery_validation_reused,
        "delivery_request_validation_seconds": delivery_request_validation_seconds,
        "candidate_load_hash_verification_seconds": candidate_load_hash_verification_seconds,
        "runtime_pin_gate_seconds": runtime_pin_gate_seconds,
        "dicom_archive_index_discovery_seconds": dicom_profile.archive_index_discovery_seconds,
        "dicom_read_decode_seconds": dicom_profile.dicom_read_decode_seconds,
        "study_iteration_loop_seconds": decode_seconds,
        "study_iteration_overhead_seconds": study_iteration_overhead_seconds,
        "dicom_read_parse_seconds": dicom_profile.dicom_read_parse_seconds,
        "pixel_decode_seconds": dicom_profile.pixel_decode_seconds,
        "preprocessing_resize_seconds": preprocessing_resize_seconds,
        "preprocessing_packing_seconds": packing_seconds,
        "model_metadata_loading_seconds": max(0.0, metadata_seconds),
        "model_load_seconds": model_seconds,
        "model_inference_seconds": inference_seconds,
        "csv_generation_validation_seconds": csv_seconds,
        "total_runtime_seconds": whole_seconds,
        "accounted_runtime_seconds": accounted_seconds,
        "unattributed_runtime_seconds": max(0.0, whole_seconds - accounted_seconds),
        "dicom_file_open_count": dicom_profile.dicom_file_open_count,
        "dicom_file_reopen_count": dicom_profile.dicom_file_reopen_count,
        "dicom_decode_call_count": dicom_profile.decode_call_count,
        "dicom_index_scan_count": dicom_profile.index_scan_count,
        "dicom_indexed_file_count": dicom_profile.indexed_file_count,
        "dicom_access": {
            "source_kind": "directory",
            "archive_access_mode": "not_applicable_directory_input",
            "dicom_access_mode": "serial_one_open_per_slice",
            "files_reopened_or_rescanned": False,
        },
        "cpu": {
            "logical_cpu_count": os.cpu_count(),
            "affinity_cpu_count": affinity_cpu_count,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": interop_thread_count,
            "process_user_seconds": process_user_seconds,
            "process_system_seconds": process_system_seconds,
            "process_cpu_seconds": process_cpu_seconds,
            "process_wall_seconds": process_wall_seconds,
            "process_cpu_utilization_ratio": (
                process_cpu_seconds / process_wall_seconds if process_wall_seconds > 0.0 else 0.0
            ),
        },
        "peak_ram_bytes": _peak_rss_bytes(),
        "dicom_profile": dicom_profile.public_dict(),
        "memory_profile": memory_profile,
    }
    evidence: dict[str, object] = {
        "schema_version": KAGGLE_EXECUTION_SCHEMA_VERSION,
        "ticket": "24",
        "status": "completed_kaggle_notebook",
        "bundle": {
            "request_checksum": delivery_request_checksum,
            "candidate_manifest_checksum": candidate["manifest_checksum"],
            "notebook_sha256": delivery_notebook_sha256,
            "adapter_sha256": delivery_adapter_sha256,
            "model_checkpoints": candidate_checkpoints,
            "model_size_bytes": candidate["model_size_bytes"],
            "candidate_bundle_size_bytes": candidate["candidate_bundle_size_bytes"],
            "environment": candidate["environment"],
        },
        "environment": {
            "runtime": "kaggle",
            "internet_enabled": False,
            "runtime_network_download": False,
            "reports_required": False,
            "python_version": runtime_audit["python_version"],
            "torch_version": _mapping(runtime_audit.get("imported_packages"), "runtime imported packages")["torch"],
            "accelerator": str(device),
            "cpu_count": os.cpu_count(),
            "gpu_name": (
                torch.cuda.get_device_name(device_object)
                if device_object.type == "cuda" and torch.cuda.is_available()
                else None
            ),
            "installed_runtime_pins": dict(_mapping(runtime_audit.get("packages"), "runtime package audit")),
            "imported_runtime_pins": dict(
                _mapping(runtime_audit.get("imported_packages"), "runtime imported package audit")
            ),
        },
        "resources": {
            "model_size_bytes": candidate["model_size_bytes"],
            "candidate_bundle_size_bytes": candidate["candidate_bundle_size_bytes"],
            "delivery_size_bytes": delivery_size_bytes,
            "checkpoint_sizes_bytes": checkpoint_sizes_bytes,
            "runtime_package_sizes_bytes": runtime_package_sizes_bytes,
            "peak_ram_bytes": detailed_timing["peak_ram_bytes"],
            "peak_vram_bytes": peak_vram_bytes,
            "ensemble_model_count": len(candidate_checkpoints),
        },
        "input": {
            "test_csv_sha256": test_csv_sha256,
            "test_series_csv_sha256": test_series_csv_sha256,
            "input_layout": "test.csv + test_series.csv + test_series/",
            "source": "competition_hidden_test" if workload_kind == "hidden_test" else "private_representative_dataset",
            "workload_kind": workload_kind,
            "manifest_sha256": input_manifest_sha256,
        },
        "workload": {
            "study_count": input_summary["study_count"],
            "series_count": input_summary["series_count"],
            "slice_count": input_summary["slice_count"],
            "decoded_slice_count": input_summary["decoded_slice_count"],
            "actual_workload_recorded": True,
            "workload_kind": workload_kind,
            "hidden_test": workload_kind == "hidden_test",
            "example_only": allow_local_smoke,
            "series_per_study": input_summary["series_per_study"],
            "slices_per_study": input_summary["slices_per_study"],
            "long_tail": input_summary["long_tail"],
        },
        "timing": {
            "whole_notebook_seconds": whole_seconds,
            "stages": stage_timings,
            "detailed": detailed_timing,
            "stage_timing_source": "notebook_wall_clock" if not allow_local_smoke else "local_smoke_adapter",
            "projection": projection,
        },
        "output": {
            "filename": "submission.csv",
            "accepted_by_kaggle": False,
            "accepted_by_local_contract": True,
            **submission_validation,
            "expected_row_count": len(expected_uids),
            "missing_rows": 0,
            "duplicate_rows": 0,
            "decode_failure_count": input_summary["decode_failure_count"],
            "sha256": _sha256_file(output),
        },
        "fallback": {
            "status": "measured",
            "numerator": fallback_numerator,
            "denominator": fallback_denominator,
            "rate": fallback_rate,
            "unknown_hidden_failure_rate": allow_local_smoke,
            "investigation": {
                "complete": fallback_numerator == 0,
                "investigated_cases": 0,
                "unavoidable_cases": 0,
            },
        },
        "recovery": {
            "checkpoint_schema_version": KAGGLE_PREDICTION_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_rows_written_this_run": checkpoint.completed_count - resumed_row_count,
            "checkpoint_rows_resumed": resumed_row_count,
            "save_reload_verified": True,
            "interruption_recovery_verified": resumed_row_count > 0,
            "rollback_seconds": None,
            "status": "resumed_after_interruption" if resumed_row_count else "study_checkpoint_saved_reloaded",
        },
        "diagnostics": {
            "failure_reasons": input_summary["failure_reasons"],
            "fallback_cases_are_private": True,
        },
    }
    if approval_sha256 is not None:
        evidence["approval"] = {"sha256": approval_sha256}
    if audited_runtime_install is not None:
        evidence["runtime_install"] = audited_runtime_install
    if evidence_destination is not None:
        atomic_json_write(evidence_destination, evidence)
    if private_destination is not None:
        atomic_json_write(
            private_destination,
            {
                "schema_version": KAGGLE_EXECUTION_SCHEMA_VERSION,
                "ticket": "24",
                "candidate_manifest_checksum": candidate["manifest_checksum"],
                "studies": checkpoint.private_records,
            },
        )
    try:
        checkpoint_path.unlink()
    except FileNotFoundError:
        pass
    return KaggleInferenceResult(output, evidence, private_destination)


__all__ = [
    "KAGGLE_APPROVAL_SCHEMA_VERSION",
    "KAGGLE_COMPETITION_SLUG",
    "DEFAULT_KAGGLE_ADAPTER_PATH",
    "KAGGLE_DELIVERY_SCHEMA_VERSION",
    "KAGGLE_ENGINEERING_TARGET_SECONDS",
    "KAGGLE_EXECUTION_SCHEMA_VERSION",
    "KAGGLE_HARD_LIMIT_SECONDS",
    "KAGGLE_PROMOTION_SCHEMA_VERSION",
    "KAGGLE_PROFILE_SCHEMA_VERSION",
    "KAGGLE_RUNTIME_STAGES",
    "KAGGLE_TARGET_STUDIES",
    "KAGGLE_VERIFICATION_SCHEMA_VERSION",
    "KaggleDeliveryError",
    "KaggleDeliveryRequest",
    "KaggleInferenceResult",
    "KaggleInputLayout",
    "KagglePromotion",
    "KaggleVerification",
    "OWNER_APPROVAL_SCOPE",
    "audit_kaggle_runtime_wheel",
    "discover_kaggle_inputs",
    "load_kaggle_delivery",
    "load_kaggle_promotion",
    "load_kaggle_verification",
    "prepare_kaggle_delivery",
    "promote_kaggle_candidate",
    "record_kaggle_verification",
    "run_kaggle_baseline",
    "validate_kaggle_runtime_pins",
]
