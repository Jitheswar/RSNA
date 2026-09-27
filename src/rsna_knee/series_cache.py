"""Atomic, source-aware storage for deterministic prepared MRI series.

The cache stores materialized slices, not model inputs.  Sampling controls and
model architecture therefore do not participate in a cache key.  A published
entry is a directory containing one NumPy array and checksum-covered metadata;
the directory is renamed into place only after both files are complete.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .domain import DicomSlice, SeriesRecord, StudyRecord
from .geometry import GeometryError
from .preprocessing import (
    PREPROCESS_SCHEMA_VERSION,
    PreparedSeries,
    PreparedStudy,
    PreprocessConfig,
    PreprocessingError,
    SeriesFailure,
    prepare_series,
)


SERIES_CACHE_SCHEMA_VERSION = "rsna-knee-series-cache-v1"
_CACHE_KEY_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PARTIAL_PATTERN = re.compile(r"^\.([0-9a-f]{64})\.partial-")
_INVALID_PATTERN = re.compile(r"^\.([0-9a-f]{64})\.invalid-")
_ARRAY_FILENAME = "images.npy"
_METADATA_FILENAME = "metadata.json"
_ROOT_MARKER_FILENAME = ".rsna-series-cache.json"
_ROOT_MARKER_SCHEMA_VERSION = "rsna-knee-disposable-cache-v1"


class SeriesCacheError(RuntimeError):
    """Base class for cache configuration, publication, or integrity errors."""


class SeriesCacheIntegrityError(SeriesCacheError):
    """Raised when a complete-looking cache entry fails validation."""


class SeriesCacheLockError(SeriesCacheError):
    """Raised when a cache-key lock cannot be acquired safely."""


@dataclass(frozen=True)
class SeriesCacheResult:
    """The prepared series and the cache event that produced it."""

    prepared: PreparedSeries
    cache_key: str
    hit: bool
    reason: str
    source_fingerprint: str
    bytes_read: int = 0
    bytes_written: int = 0
    temporary_bytes: int = 0


@dataclass(frozen=True)
class CacheCleanupReport:
    """What an explicitly requested disposable-cache cleanup removed."""

    removed_keys: tuple[str, ...]
    removed_partial_paths: tuple[str, ...] = ()
    reclaimed_bytes: int = 0


@dataclass(frozen=True)
class CacheBenchmark:
    """Mechanics measurements for one bounded cache exercise."""

    series_count: int
    worker_count: int
    elapsed_seconds: float
    cache_hits: int
    cache_misses: int
    bytes_read: int
    bytes_written: int
    peak_temporary_bytes: int


class _CacheMiss(Exception):
    pass


class _CacheAlreadyPublished(Exception):
    def __init__(self, prepared: PreparedSeries, bytes_read: int) -> None:
        super().__init__("another writer published a valid cache entry")
        self.prepared = prepared
        self.bytes_read = bytes_read


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _validate_cache_key(value: str) -> str:
    if not isinstance(value, str) or _CACHE_KEY_PATTERN.fullmatch(value) is None:
        raise SeriesCacheError(f"invalid series cache key {value!r}")
    return value


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _decoded_slice_content_hash(item: DicomSlice) -> str:
    """Hash the decoded source record when a raw source hash is unavailable."""

    pixels = np.asarray(item.pixel_array)
    contiguous = np.ascontiguousarray(pixels)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    metadata = {
        "sop_instance_uid": item.sop_instance_uid,
        "study_instance_uid": item.study_instance_uid,
        "series_instance_uid": item.series_instance_uid,
        "image_position": item.image_position,
        "image_orientation": item.image_orientation,
        "rescale_slope": item.rescale_slope,
        "rescale_intercept": item.rescale_intercept,
        "photometric_interpretation": item.photometric_interpretation,
        "padding_value": _optional_float(item.padding_value),
        "padding_range_limit": _optional_float(item.padding_range_limit),
        "transfer_syntax_uid": item.transfer_syntax_uid,
        "modality": item.modality,
        "rows": item.rows,
        "columns": item.columns,
        "number_of_frames": item.number_of_frames,
        "samples_per_pixel": item.samples_per_pixel,
        "bits_allocated": item.bits_allocated,
        "bits_stored": item.bits_stored,
        "high_bit": item.high_bit,
        "pixel_representation": item.pixel_representation,
        "image_type_values": item.image_type_values,
        "series_description": item.series_description,
    }
    digest.update(_canonical_json(metadata).encode("utf-8"))
    return digest.hexdigest()


def _source_slice_payload(
    series: SeriesRecord,
    source_content_hashes: Sequence[str] | None,
) -> tuple[dict[str, object], ...]:
    if source_content_hashes is not None and len(source_content_hashes) != len(series.slices):
        raise SeriesCacheError("source_content_hashes must align with series.slices")
    if series.source_entry_names and len(series.source_entry_names) != len(series.slices):
        raise SeriesCacheError("source_entry_names must align with series.slices")

    payload: list[dict[str, object]] = []
    seen_uids: set[str] = set()
    for index, item in enumerate(series.slices):
        if not item.sop_instance_uid.strip() or item.sop_instance_uid in seen_uids:
            raise SeriesCacheError("series source slices must have unique non-empty SOP identifiers")
        seen_uids.add(item.sop_instance_uid)
        source_identity = item.filename.strip()
        if not source_identity and series.source_entry_names:
            source_identity = series.source_entry_names[index].strip()
        if not source_identity:
            source_identity = item.sop_instance_uid
        if not source_identity:
            raise SeriesCacheError(f"slice {item.sop_instance_uid} has no source identity")
        content_hash = (
            str(source_content_hashes[index]).strip()
            if source_content_hashes is not None
            else (item.source_content_sha256 or _decoded_slice_content_hash(item)).strip()
        )
        if not content_hash:
            raise SeriesCacheError(f"slice {item.sop_instance_uid} has no content hash")
        payload.append(
            {
                "sop_instance_uid": item.sop_instance_uid,
                "source_identity": source_identity,
                "content_sha256": content_hash,
            }
        )
    return tuple(sorted(payload, key=lambda item: str(item["sop_instance_uid"])))


def _source_payload(series: SeriesRecord, source_content_hashes: Sequence[str] | None) -> dict[str, object]:
    return {
        "study_instance_uid": series.study_instance_uid,
        "series_instance_uid": series.series_instance_uid,
        "slices": list(_source_slice_payload(series, source_content_hashes)),
    }


def _prepared_array(prepared: PreparedSeries) -> np.ndarray:
    images = np.asarray(prepared.images)
    if images.ndim != 4 or images.shape[1] != 1 or images.dtype != np.dtype("float32"):
        raise SeriesCacheIntegrityError(
            f"prepared series {prepared.series_instance_uid} must be float32 [slice, 1, height, width]"
        )
    if images.shape[0] != len(prepared.source_slice_uids):
        raise SeriesCacheIntegrityError("prepared image count does not match source slice order")
    if len(set(prepared.source_slice_uids)) != len(prepared.source_slice_uids):
        raise SeriesCacheIntegrityError("prepared source slice order contains duplicates")
    if not np.isfinite(images).all():
        raise SeriesCacheIntegrityError("prepared series contains non-finite pixels")
    return np.ascontiguousarray(images)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


class SeriesCache:
    """Persist and recover deterministic per-series materializations.

    ``root`` is the only location this class reads or removes.  It should be a
    disposable cache directory, never a raw-data, report, checkpoint, fold, or
    candidate directory.
    """

    def __init__(self, root: str | Path, *, read_only: bool = False) -> None:
        self.root = Path(root)
        self.entries_root = self.root / "entries"
        self.locks_root = self.root / "locks"
        self.read_only = read_only
        if self.root.is_symlink():
            raise SeriesCacheError("series cache root must not be a symlink")
        root_preexisted = self.root.exists()
        if root_preexisted and not self.root.is_dir():
            raise SeriesCacheError("series cache root must be a directory")
        if self.read_only:
            if not root_preexisted:
                raise SeriesCacheError("read-only series cache root does not exist")
            self._require_private_directory(self.root)
            self._ensure_disposable_root(create=False)
        else:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            if root_preexisted:
                self._require_private_directory(self.root)
            self._ensure_disposable_root()
        for control_path in (self.entries_root, self.locks_root):
            if control_path.exists() and control_path.is_symlink():
                raise SeriesCacheError("series cache control directories must not be symlinks")
            if self.read_only:
                if not control_path.is_dir():
                    raise SeriesCacheError("read-only series cache control directory is missing")
            else:
                control_path.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._require_private_directory(control_path)

    def cache_key(
        self,
        series: SeriesRecord,
        config: PreprocessConfig = PreprocessConfig(),
        *,
        source_content_hashes: Sequence[str] | None = None,
    ) -> str:
        """Return a sampling-independent key for one source series."""

        return self._cache_key_for(
            _source_payload(series, source_content_hashes),
            config,
        )

    def entry_path(self, cache_key: str) -> Path:
        """Return the on-disk path for a validated-looking cache key."""

        key = _validate_cache_key(cache_key)
        return self.entries_root / key[:2] / key

    def get_or_prepare(
        self,
        series: SeriesRecord,
        config: PreprocessConfig = PreprocessConfig(),
        *,
        source_content_hashes: Sequence[str] | None = None,
    ) -> SeriesCacheResult:
        """Load a validated entry or prepare and atomically publish one."""

        source = _source_payload(series, source_content_hashes)
        cache_key = self._cache_key_for(source, config)
        source_fingerprint = _json_sha256(source)
        if self.read_only:
            try:
                prepared, bytes_read = self._load(series, config, source, cache_key)
            except (_CacheMiss, SeriesCacheIntegrityError) as exc:
                raise SeriesCacheError(
                    "read-only series cache has no exact valid entry for this source series"
                ) from exc
            return SeriesCacheResult(
                prepared=prepared,
                cache_key=cache_key,
                hit=True,
                reason="hit",
                source_fingerprint=source_fingerprint,
                bytes_read=bytes_read,
            )
        with self._key_lock(cache_key):
            invalid_entry = False
            try:
                prepared, bytes_read = self._load(
                    series,
                    config,
                    source,
                    cache_key,
                )
            except (_CacheMiss, SeriesCacheIntegrityError):
                entry = self.entry_path(cache_key)
                invalid_entry = entry.exists() or entry.is_symlink()
                prepared = prepare_series(series, config)
                try:
                    bytes_written, temporary_bytes = self._publish(
                        prepared,
                        series,
                        config,
                        source,
                        cache_key,
                    )
                except _CacheAlreadyPublished as published:
                    return SeriesCacheResult(
                        prepared=published.prepared,
                        cache_key=cache_key,
                        hit=True,
                        reason="hit",
                        source_fingerprint=source_fingerprint,
                        bytes_read=published.bytes_read,
                    )
                return SeriesCacheResult(
                    prepared=prepared,
                    cache_key=cache_key,
                    hit=False,
                    reason="rebuild" if invalid_entry else "miss",
                    source_fingerprint=source_fingerprint,
                    bytes_written=bytes_written,
                    temporary_bytes=temporary_bytes,
                )
            return SeriesCacheResult(
                prepared=prepared,
                cache_key=cache_key,
                hit=True,
                reason="hit",
                source_fingerprint=source_fingerprint,
                bytes_read=bytes_read,
            )

    def load_validated_entry(
        self,
        cache_key: str,
        *,
        study_instance_uid: str,
        series_instance_uid: str,
        expected_source_entry_names: Sequence[str],
        config: PreprocessConfig = PreprocessConfig(),
    ) -> PreparedSeries:
        """Read one immutable cache entry against its frozen source membership.

        This boundary supports feature extraction from an already materialized
        series without reopening the source DICOM pixels. It never rebuilds a
        missing or stale entry.
        """

        key = _validate_cache_key(cache_key)
        expected_entries = tuple(expected_source_entry_names)
        if (
            not study_instance_uid.strip()
            or not series_instance_uid.strip()
            or not expected_entries
            or len(set(expected_entries)) != len(expected_entries)
            or any(not isinstance(item, str) or not item.strip() for item in expected_entries)
        ):
            raise SeriesCacheError("cached series reads require exact study, series and entry identities")
        entry = self.entry_path(key)
        metadata = self._read_metadata(entry)
        source = metadata.get("source")
        if not isinstance(source, dict):
            raise SeriesCacheIntegrityError("cache entry source identity is malformed")
        slices_value = source.get("slices")
        if not isinstance(slices_value, list) or not slices_value:
            raise SeriesCacheIntegrityError("cache entry source slices are missing")
        if (
            source.get("study_instance_uid") != study_instance_uid
            or source.get("series_instance_uid") != series_instance_uid
            or metadata.get("preprocessing") != config.cache_config()
        ):
            raise SeriesCacheIntegrityError("cache entry identity differs from the frozen request")
        try:
            ordered_slices = sorted(
                (
                    {
                        "sop_instance_uid": str(item["sop_instance_uid"]),
                        "source_identity": str(item["source_identity"]),
                        "content_sha256": str(item["content_sha256"]),
                    }
                    for item in slices_value
                    if isinstance(item, dict)
                ),
                key=lambda item: item["sop_instance_uid"],
            )
        except (KeyError, TypeError) as exc:
            raise SeriesCacheIntegrityError("cache entry source slice identity is malformed") from exc
        if (
            len(ordered_slices) != len(slices_value)
            or tuple(item["source_identity"] for item in ordered_slices)
            != tuple(sorted(expected_entries))
            or len({item["sop_instance_uid"] for item in ordered_slices}) != len(ordered_slices)
            or any(
                len(item["content_sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in item["content_sha256"])
                for item in ordered_slices
            )
        ):
            raise SeriesCacheIntegrityError("cache entry source entries differ from the frozen manifest")
        series = SeriesRecord(
            study_instance_uid=study_instance_uid,
            series_instance_uid=series_instance_uid,
            slices=tuple(
                DicomSlice(
                    sop_instance_uid=item["sop_instance_uid"],
                    image_position=None,
                    image_orientation=None,
                    pixel_array=np.empty((0, 0), dtype=np.uint8),
                    study_instance_uid=study_instance_uid,
                    series_instance_uid=series_instance_uid,
                    filename=item["source_identity"],
                    source_content_sha256=item["content_sha256"],
                )
                for item in ordered_slices
            ),
            source_entry_names=tuple(item["source_identity"] for item in ordered_slices),
        )
        source_payload = _source_payload(
            series,
            tuple(item["content_sha256"] for item in ordered_slices),
        )
        if self._cache_key_for(source_payload, config) != key:
            raise SeriesCacheIntegrityError("cache entry key is inconsistent with its source identity")
        prepared, _ = self._load(series, config, source_payload, key)
        return prepared

    def prepare_study(
        self,
        study: StudyRecord,
        config: PreprocessConfig = PreprocessConfig(),
    ) -> PreparedStudy:
        """Prepare every series through this cache while retaining failures."""

        prepared: list[PreparedSeries] = []
        failures: list[SeriesFailure] = []
        for series in sorted(study.series, key=lambda item: item.series_instance_uid):
            try:
                prepared.append(self.get_or_prepare(series, config).prepared)
            except (GeometryError, PreprocessingError) as exc:
                failures.append(SeriesFailure(series.series_instance_uid, str(exc)))
        return PreparedStudy(
            study_instance_uid=study.study_instance_uid,
            series=tuple(prepared),
            failures=tuple(failures),
            warnings=tuple(warning for item in prepared for warning in item.warnings),
        )

    def list_complete_keys(self) -> tuple[str, ...]:
        """List only entries with a complete, checksum-valid metadata file."""

        keys: list[str] = []
        if not self.entries_root.exists():
            return ()
        for prefix in sorted(self.entries_root.iterdir()):
            if not prefix.is_dir() or prefix.is_symlink():
                continue
            for entry in sorted(prefix.iterdir()):
                if not entry.is_dir() or entry.is_symlink() or _CACHE_KEY_PATTERN.fullmatch(entry.name) is None:
                    continue
                try:
                    metadata = self._read_metadata(entry)
                except SeriesCacheIntegrityError:
                    continue
                if metadata.get("state") != "complete":
                    continue
                array_payload = metadata.get("array")
                array_path = entry / _ARRAY_FILENAME
                if (
                    not isinstance(array_payload, dict)
                    or array_payload.get("filename") != _ARRAY_FILENAME
                    or not array_path.is_file()
                    or array_path.is_symlink()
                ):
                    continue
                try:
                    array_hash, array_bytes = _file_sha256(array_path)
                except OSError:
                    continue
                if (
                    array_hash != array_payload.get("sha256")
                    or array_bytes != array_payload.get("bytes")
                ):
                    continue
                keys.append(entry.name)
        return tuple(keys)

    def cleanup_disposable(
        self,
        *,
        referenced_keys: set[str],
        remove_partial: bool = False,
    ) -> CacheCleanupReport:
        """Remove only unreferenced material inside this disposable cache root.

        Partial staging directories are retained by default so an interrupted
        writer remains diagnosable.  Removing them requires the explicit
        ``remove_partial=True`` opt-in.
        """

        references = {_validate_cache_key(key) for key in referenced_keys}
        removed: list[str] = []
        removed_partials: list[str] = []
        reclaimed = 0
        for prefix in sorted(self.entries_root.iterdir()) if self.entries_root.exists() else ():
            if not prefix.is_dir() or prefix.is_symlink():
                continue
            for entry in sorted(prefix.iterdir()):
                if entry.is_symlink():
                    continue
                if entry.name.startswith(".") and ".partial-" in entry.name:
                    if remove_partial:
                        partial_match = _PARTIAL_PATTERN.match(entry.name)
                        if partial_match is None:
                            continue
                        with self._key_lock(partial_match.group(1)):
                            if entry.exists() and not entry.is_symlink():
                                reclaimed += self._path_size(entry)
                                self._remove_cache_material(entry)
                                removed_partials.append(str(entry.relative_to(self.root)))
                    continue
                if entry.name.startswith(".") and ".invalid-" in entry.name:
                    if remove_partial:
                        invalid_match = _INVALID_PATTERN.match(entry.name)
                        if invalid_match is None:
                            continue
                        with self._key_lock(invalid_match.group(1)):
                            if entry.exists() and not entry.is_symlink():
                                reclaimed += self._path_size(entry)
                                self._remove_cache_material(entry)
                                removed_partials.append(str(entry.relative_to(self.root)))
                    continue
                if not entry.is_dir() or _CACHE_KEY_PATTERN.fullmatch(entry.name) is None:
                    continue
                if entry.name in references:
                    continue
                with self._key_lock(entry.name):
                    if entry.exists() and not entry.is_symlink():
                        reclaimed += self._path_size(entry)
                        self._remove_cache_material(entry)
                        removed.append(entry.name)
        return CacheCleanupReport(
            removed_keys=tuple(sorted(removed)),
            removed_partial_paths=tuple(sorted(removed_partials)),
            reclaimed_bytes=reclaimed,
        )

    def benchmark(
        self,
        series: Sequence[SeriesRecord],
        config: PreprocessConfig = PreprocessConfig(),
        *,
        workers: int = 1,
    ) -> CacheBenchmark:
        """Exercise bounded cache reads/writes with declared worker concurrency."""

        if workers <= 0:
            raise ValueError("workers must be positive")
        items = tuple(series)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = tuple(executor.map(lambda item: self.get_or_prepare(item, config), items))
        return CacheBenchmark(
            series_count=len(items),
            worker_count=workers,
            elapsed_seconds=time.perf_counter() - started,
            cache_hits=sum(item.hit for item in results),
            cache_misses=sum(not item.hit for item in results),
            bytes_read=sum(item.bytes_read for item in results),
            bytes_written=sum(item.bytes_written for item in results),
            peak_temporary_bytes=sum(
                sorted(
                    (item.temporary_bytes for item in results if item.temporary_bytes > 0),
                    reverse=True,
                )[:workers]
            ),
        )

    @contextmanager
    def _key_lock(self, cache_key: str) -> Iterator[None]:
        key = _validate_cache_key(cache_key)
        lock_path = self.locks_root / f"{key}.lock"
        try:
            lock_flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(lock_path, lock_flags, 0o600)
        except OSError as exc:
            raise SeriesCacheLockError(f"could not open cache lock {lock_path}") from exc
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            except OSError as exc:
                raise SeriesCacheLockError(f"could not lock cache key {key}") from exc
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _cache_key_for(self, source: dict[str, object], config: PreprocessConfig) -> str:
        return _json_sha256(
            {
                "schema_version": SERIES_CACHE_SCHEMA_VERSION,
                "source": source,
                "preprocessing": config.cache_config(),
            }
        )

    def _ensure_disposable_root(self, *, create: bool = True) -> None:
        marker = self.root / _ROOT_MARKER_FILENAME
        if marker.exists() or marker.is_symlink():
            if marker.is_symlink() or not marker.is_file():
                raise SeriesCacheError("series cache root marker is not a regular file")
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise SeriesCacheError("series cache root marker cannot be read") from exc
            if payload != {
                "schema_version": _ROOT_MARKER_SCHEMA_VERSION,
                "role": "disposable-series-cache",
            }:
                raise SeriesCacheError("series cache root marker is invalid")
            return
        if not create:
            raise SeriesCacheError("read-only series cache root marker is missing")
        if any(self.root.iterdir()):
            raise SeriesCacheError(
                "series cache root must be new, empty, or previously marked disposable"
            )
        payload = {
            "schema_version": _ROOT_MARKER_SCHEMA_VERSION,
            "role": "disposable-series-cache",
        }
        try:
            descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return self._ensure_disposable_root()
        except OSError as exc:
            raise SeriesCacheError("could not create series cache root marker") from exc
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            marker.unlink(missing_ok=True)
            raise
        _fsync_directory(self.root)

    @staticmethod
    def _require_private_directory(path: Path) -> None:
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            raise SeriesCacheError(f"could not inspect cache directory permissions {path}") from exc
        if mode & 0o077:
            raise SeriesCacheError(f"series cache directory must be private (mode 0700): {path}")

    def _load(
        self,
        series: SeriesRecord,
        config: PreprocessConfig,
        source: dict[str, object],
        cache_key: str,
    ) -> tuple[PreparedSeries, int]:
        entry = self.entry_path(cache_key)
        if not entry.exists() or not entry.is_dir() or entry.is_symlink():
            raise _CacheMiss()
        metadata = self._read_metadata(entry)
        if metadata.get("state") != "complete":
            raise SeriesCacheIntegrityError("cache entry is not complete")
        if metadata.get("cache_key") != cache_key:
            raise SeriesCacheIntegrityError("cache entry key does not match its path")
        if metadata.get("source") != source:
            raise SeriesCacheIntegrityError("cache entry source identity does not match the request")
        if metadata.get("preprocessing") != config.cache_config():
            raise SeriesCacheIntegrityError("cache entry preprocessing policy does not match the request")
        prepared_payload = metadata.get("prepared")
        array_payload = metadata.get("array")
        if not isinstance(prepared_payload, dict) or not isinstance(array_payload, dict):
            raise SeriesCacheIntegrityError("cache entry is missing prepared or array metadata")
        array_name = array_payload.get("filename")
        if array_name != _ARRAY_FILENAME:
            raise SeriesCacheIntegrityError("cache entry has an unexpected array filename")
        array_path = entry / _ARRAY_FILENAME
        if not array_path.is_file() or array_path.is_symlink():
            raise SeriesCacheIntegrityError("cache entry array is missing")
        expected_hash = array_payload.get("sha256")
        actual_hash, actual_bytes = _file_sha256(array_path)
        if actual_hash != expected_hash:
            raise SeriesCacheIntegrityError("cache entry array checksum does not match")
        if array_payload.get("bytes") != actual_bytes:
            raise SeriesCacheIntegrityError("cache entry array byte count does not match")
        try:
            images = np.load(array_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise SeriesCacheIntegrityError("cache entry array cannot be loaded") from exc
        images = np.asarray(images)
        shape = prepared_payload.get("shape")
        order = prepared_payload.get("order")
        warnings = prepared_payload.get("warnings")
        if (
            not isinstance(shape, list)
            or not isinstance(order, list)
            or not isinstance(warnings, list)
            or any(not isinstance(item, str) for item in order + warnings)
        ):
            raise SeriesCacheIntegrityError("cache entry prepared metadata is malformed")
        if (
            prepared_payload.get("schema_version") != PREPROCESS_SCHEMA_VERSION
            or prepared_payload.get("dtype") != "float32"
            or list(images.shape) != shape
            or images.dtype != np.dtype("float32")
        ):
            raise SeriesCacheIntegrityError("cache entry array shape or dtype does not match metadata")
        if not np.isfinite(images).all():
            raise SeriesCacheIntegrityError("cache entry array contains non-finite values")
        source_slices = source.get("slices")
        if not isinstance(source_slices, list):
            raise SeriesCacheIntegrityError("cache request source metadata is malformed")
        expected_uids = [str(item["sop_instance_uid"]) for item in source_slices]
        if sorted(order) != sorted(expected_uids) or len(order) != images.shape[0]:
            raise SeriesCacheIntegrityError("cache entry source order does not match metadata")
        study_uid = series.study_instance_uid
        series_uid = series.series_instance_uid
        stored_study = prepared_payload.get("study_instance_uid")
        stored_series = prepared_payload.get("series_instance_uid")
        if stored_study != study_uid or stored_series != series_uid:
            raise SeriesCacheIntegrityError("cache entry prepared identifiers do not match request")
        prepared = PreparedSeries(
            study_instance_uid=study_uid,
            series_instance_uid=series_uid,
            images=images,
            source_slice_uids=tuple(order),
            warnings=tuple(warnings),
        )
        metadata_bytes = (entry / _METADATA_FILENAME).stat().st_size
        return prepared, actual_bytes + metadata_bytes

    def _read_metadata(self, entry: Path) -> dict[str, object]:
        metadata_path = entry / _METADATA_FILENAME
        if not metadata_path.is_file() or metadata_path.is_symlink():
            raise SeriesCacheIntegrityError("cache entry metadata is missing")
        try:
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SeriesCacheIntegrityError("cache entry metadata cannot be read") from exc
        if not isinstance(raw, dict):
            raise SeriesCacheIntegrityError("cache entry metadata must be an object")
        checksum = raw.get("metadata_sha256")
        without_checksum = dict(raw)
        without_checksum.pop("metadata_sha256", None)
        if not isinstance(checksum, str) or _json_sha256(without_checksum) != checksum:
            raise SeriesCacheIntegrityError("cache entry metadata checksum does not match")
        if raw.get("schema_version") != SERIES_CACHE_SCHEMA_VERSION:
            raise SeriesCacheIntegrityError("cache entry schema is unsupported")
        return raw

    def _publish(
        self,
        prepared: PreparedSeries,
        series: SeriesRecord,
        config: PreprocessConfig,
        source: dict[str, object],
        cache_key: str,
    ) -> tuple[int, int]:
        images = _prepared_array(prepared)
        source_slices = source.get("slices")
        if not isinstance(source_slices, list):
            raise SeriesCacheIntegrityError("source metadata is malformed before publication")
        expected_uids = [str(item["sop_instance_uid"]) for item in source_slices]
        if sorted(expected_uids) != sorted(prepared.source_slice_uids):
            raise SeriesCacheIntegrityError("prepared order does not agree with source geometry order")
        parent = self.entry_path(cache_key).parent
        if parent.is_symlink():
            raise SeriesCacheError("series cache entry prefix must not be a symlink")
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{cache_key}.partial-", dir=parent))
        try:
            array_path = staging / _ARRAY_FILENAME
            with array_path.open("wb") as handle:
                np.save(handle, images, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            array_hash, array_bytes = _file_sha256(array_path)
            payload: dict[str, object] = {
                "schema_version": SERIES_CACHE_SCHEMA_VERSION,
                "state": "complete",
                "cache_key": cache_key,
                "source": source,
                "preprocessing": config.cache_config(),
                "prepared": {
                    "study_instance_uid": series.study_instance_uid,
                    "series_instance_uid": series.series_instance_uid,
                    "schema_version": PREPROCESS_SCHEMA_VERSION,
                    "shape": list(images.shape),
                    "dtype": str(images.dtype),
                    "order": list(prepared.source_slice_uids),
                    "warnings": list(prepared.warnings),
                },
                "array": {
                    "filename": _ARRAY_FILENAME,
                    "sha256": array_hash,
                    "bytes": array_bytes,
                },
            }
            payload["metadata_sha256"] = _json_sha256(payload)
            metadata_path = staging / _METADATA_FILENAME
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(staging)
            final = self.entry_path(cache_key)
            quarantine: Path | None = None
            if final.exists() or final.is_symlink():
                try:
                    existing, bytes_read = self._load(series, config, source, cache_key)
                except (_CacheMiss, SeriesCacheIntegrityError):
                    quarantine = parent / f".{cache_key}.invalid-{uuid.uuid4().hex}"
                    os.replace(final, quarantine)
                else:
                    raise _CacheAlreadyPublished(existing, bytes_read)
            metadata_bytes = metadata_path.stat().st_size
            try:
                os.replace(staging, final)
                _fsync_directory(parent)
                return array_bytes + metadata_bytes, array_bytes + metadata_bytes
            except BaseException:
                if quarantine is not None and quarantine.exists() and not final.exists():
                    os.replace(quarantine, final)
                raise
            finally:
                if quarantine is not None and quarantine.exists():
                    self._remove_cache_material(quarantine)
        finally:
            if staging.exists():
                self._remove_cache_material(staging)

    @staticmethod
    def _path_size(path: Path) -> int:
        if path.is_file() and not path.is_symlink():
            try:
                return path.stat().st_size
            except OSError:
                return 0
        if not path.is_dir() or path.is_symlink():
            return 0
        total = 0
        for item in path.rglob("*"):
            if item.is_file() and not item.is_symlink():
                try:
                    total += item.stat().st_size
                except OSError:
                    continue
        return total

    @staticmethod
    def _remove_cache_material(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)


__all__ = [
    "SERIES_CACHE_SCHEMA_VERSION",
    "CacheBenchmark",
    "CacheCleanupReport",
    "SeriesCache",
    "SeriesCacheError",
    "SeriesCacheIntegrityError",
    "SeriesCacheLockError",
    "SeriesCacheResult",
]
