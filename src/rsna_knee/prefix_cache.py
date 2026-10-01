"""Immutable unfitted ResNet activations, with explicit packed-slice order."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, cast

import numpy as np

from .preprocessing import SERIES_DESCRIPTOR_COLUMNS


SCHEMA_VERSION = "rsna-resnet-prefix-entry-v1"
ARRAY_KEYS = ("prefix", "slice_mask", "series_mask", "series_descriptors", "valid_flat_indices")


class PrefixCacheError(ValueError):
    """A prefix entry is incomplete, incompatible or changed."""


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identity_digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(dict(value), sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def array_digest(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    result = hashlib.sha256()
    result.update(str(value.dtype).encode("ascii"))
    result.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    result.update(value.tobytes())
    return result.hexdigest()


@dataclass(frozen=True)
class PrefixEntry:
    prefix: np.ndarray
    slice_mask: np.ndarray
    series_mask: np.ndarray
    series_descriptors: np.ndarray
    valid_flat_indices: np.ndarray
    metadata: dict[str, object]


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    prefix, slices, series, descriptors, indices = (arrays[key] for key in ARRAY_KEYS)
    if (slices.shape != (16, 8) or slices.dtype != np.bool_
            or series.shape != (16,) or series.dtype != np.bool_
            or descriptors.shape != (16, len(SERIES_DESCRIPTOR_COLUMNS))
            or descriptors.dtype != np.float32 or not np.isfinite(descriptors).all()):
        raise PrefixCacheError("Packed masks or anatomical descriptors are incompatible")
    expected_indices = np.flatnonzero((slices & series[:, None]).reshape(-1))
    if (not len(expected_indices) or prefix.shape != (len(expected_indices), 256, 14, 14)
            or prefix.dtype != np.float32 or not np.isfinite(prefix).all()
            or indices.dtype != np.int64 or not np.array_equal(indices, expected_indices)):
        raise PrefixCacheError("Prefix dimensions, numerics or valid-slice order are incompatible")


def write_prefix_entry(path: Path, *, prefix: np.ndarray, slice_mask: np.ndarray,
                       series_mask: np.ndarray, series_descriptors: np.ndarray,
                       metadata: Mapping[str, object]) -> str:
    """Publish one complete entry; an existing output is never overwritten."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    if slice_mask.shape != (16, 8) or series_mask.shape != (16,):
        raise PrefixCacheError("Packed masks are incompatible")
    arrays = {"prefix": prefix, "slice_mask": slice_mask, "series_mask": series_mask,
              "series_descriptors": series_descriptors,
              "valid_flat_indices": np.flatnonzero((slice_mask & series_mask[:, None]).reshape(-1))}
    _validate_arrays(arrays)
    if (any(not isinstance(metadata.get(key), str) or not metadata[key]
            for key in ("study_instance_uid", "feature_identity_sha256", "source_identity_sha256",
                        "extraction_plan_sha256", "job_id"))
            or metadata.get("source_kind") not in
            {"frozen-rsna-archive", "ticket35-prepared-series-cache-read-only"}):
        raise PrefixCacheError("Required source and extraction identity is missing")
    record = {**metadata, "schema_version": SCHEMA_VERSION,
              "array_sha256": {key: array_digest(value) for key, value in arrays.items()}}
    descriptor, name = tempfile.mkstemp(prefix=".prefix-", suffix=".npz", dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, allow_pickle=False, prefix=arrays["prefix"], slice_mask=arrays["slice_mask"],
                     series_mask=arrays["series_mask"], series_descriptors=arrays["series_descriptors"],
                     valid_flat_indices=arrays["valid_flat_indices"],
                     metadata=np.asarray(json.dumps(record, sort_keys=True, allow_nan=False)))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        return file_digest(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_prefix_entry(path: Path, *, expected_sha256: str, expected_study_uid: str,
                      expected_identity_sha256: str, expected_source_identity_sha256: str,
                      expected_plan_sha256: str) -> PrefixEntry:
    """Verify bytes, membership, representation, arrays and explicit slice order."""
    if not path.is_file() or path.is_symlink() or file_digest(path) != expected_sha256:
        raise PrefixCacheError("Prefix file is missing, unsafe or changed")
    try:
        with np.load(path, allow_pickle=False) as source:
            if set(source.files) != {*ARRAY_KEYS, "metadata"}:
                raise PrefixCacheError("Prefix payload is incomplete or has extra fields")
            arrays = {key: source[key] for key in ARRAY_KEYS}
            metadata = json.loads(str(source["metadata"].item()))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise PrefixCacheError("Prefix payload could not be decoded") from exc
    _validate_arrays(arrays)
    expected = {"schema_version": SCHEMA_VERSION, "study_instance_uid": expected_study_uid,
                "feature_identity_sha256": expected_identity_sha256,
                "source_identity_sha256": expected_source_identity_sha256,
                "extraction_plan_sha256": expected_plan_sha256,
                "array_sha256": {key: array_digest(value) for key, value in arrays.items()}}
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
        raise PrefixCacheError("Prefix metadata, source or array identity differs")
    return PrefixEntry(prefix=arrays["prefix"], slice_mask=arrays["slice_mask"],
                       series_mask=arrays["series_mask"], series_descriptors=arrays["series_descriptors"],
                       valid_flat_indices=arrays["valid_flat_indices"], metadata=cast(dict[str, object], metadata))
