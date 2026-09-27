"""Content-identity-bound cache for frozen slice encoder features."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .archive import atomic_json_write


FEATURE_CACHE_SCHEMA_VERSION = "rsna-knee-frozen-feature-cache-v1"


class FeatureCacheError(ValueError):
    """Raised when cached features do not satisfy their immutable identity."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenFeatureCache:
    """Store one `.npy` feature tensor per source study under a frozen identity."""

    def __init__(
        self,
        root: str | Path,
        identity: Mapping[str, object],
        *,
        feature_width: int | None = None,
        read_only: bool = False,
    ) -> None:
        if not identity:
            raise FeatureCacheError("feature cache identity must be non-empty")
        if feature_width is not None and feature_width <= 0:
            raise FeatureCacheError("feature_width must be positive when supplied")
        self.identity = dict(identity)
        self.feature_width = feature_width
        self.identity_fingerprint = _json_sha256(
            {"schema_version": FEATURE_CACHE_SCHEMA_VERSION, "identity": self.identity}
        )
        self.root = Path(root) / self.identity_fingerprint
        self.entries = self.root / "entries"
        self.read_only = read_only
        self.hits = 0
        self.misses = 0
        self.fallback_imports = 0
        identity_path = self.root / "identity.json"
        expected = {
            "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "identity": self.identity,
            "identity_fingerprint": self.identity_fingerprint,
        }
        if self.read_only and (
            not self.root.is_dir()
            or self.root.is_symlink()
            or not identity_path.is_file()
            or identity_path.is_symlink()
            or not self.entries.is_dir()
            or self.entries.is_symlink()
        ):
            raise FeatureCacheError("read-only feature cache is missing its frozen identity or entries")
        if not self.read_only:
            self.root.mkdir(parents=True, exist_ok=True)
        if identity_path.exists():
            try:
                actual = json.loads(identity_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise FeatureCacheError("feature cache identity record is unreadable") from exc
            if actual != expected:
                raise FeatureCacheError("feature cache identity record does not match its directory")
        else:
            if self.read_only:
                raise FeatureCacheError("read-only feature cache has no identity record")
            atomic_json_write(identity_path, expected)
        if not self.read_only:
            self.entries.mkdir(exist_ok=True)

    def _paths(self, study_instance_uid: str) -> tuple[Path, Path]:
        if not study_instance_uid.strip():
            raise FeatureCacheError("feature cache entries require a study identity")
        key = hashlib.sha256(study_instance_uid.encode("utf-8")).hexdigest()
        return self.entries / f"{key}.npy", self.entries / f"{key}.json"

    def get_or_compute(
        self,
        study_instance_uid: str,
        packed_images: np.ndarray,
        slice_mask: np.ndarray,
        series_mask: np.ndarray,
        series_descriptors: np.ndarray,
        compute: Callable[[], np.ndarray],
    ) -> tuple[np.ndarray, bool]:
        """Return valid cached features, or atomically rematerialize stale entries."""

        images = np.asarray(packed_images)
        slices = np.asarray(slice_mask, dtype=bool)
        series = np.asarray(series_mask, dtype=bool)
        descriptors = np.asarray(series_descriptors, dtype=np.float32)
        if images.ndim != 5 or slices.shape != images.shape[:2] or series.shape != (images.shape[0],):
            raise FeatureCacheError("packed feature-cache inputs have incompatible shapes")
        if descriptors.ndim != 2 or descriptors.shape[0] != images.shape[0]:
            raise FeatureCacheError("packed series descriptors do not match the study")
        input_fingerprint = _json_sha256(
            {
                "study_instance_uid": study_instance_uid,
                "images": _array_sha256(images),
                "slice_mask": _array_sha256(slices),
                "series_mask": _array_sha256(series),
                "series_descriptors": _array_sha256(descriptors),
            }
        )
        feature_path, metadata_path = self._paths(study_instance_uid)
        if (
            feature_path.is_file()
            and not feature_path.is_symlink()
            and metadata_path.is_file()
            and not metadata_path.is_symlink()
        ):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (
                    metadata.get("schema_version") == FEATURE_CACHE_SCHEMA_VERSION
                    and metadata.get("study_instance_uid") == study_instance_uid
                    and metadata.get("identity_fingerprint") == self.identity_fingerprint
                    and metadata.get("input_fingerprint") == input_fingerprint
                    and metadata.get("feature_file_sha256") == _file_sha256(feature_path)
                ):
                    cached = np.load(feature_path, allow_pickle=False)
                    if (
                        cached.dtype == np.float32
                        and cached.ndim == 3
                        and cached.shape[:2] == images.shape[:2]
                        and (
                            self.feature_width is None
                            or cached.shape[2] == self.feature_width
                        )
                        and np.isfinite(cached).all()
                        and metadata.get("shape") == list(cached.shape)
                        and metadata.get("dtype") == str(cached.dtype)
                        and metadata.get("feature_array_sha256") == _array_sha256(cached)
                    ):
                        self.hits += 1
                        return cached, True
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                pass

        if self.read_only:
            raise FeatureCacheError(
                f"read-only feature cache has no exact valid entry for study {study_instance_uid}"
            )

        computed = np.asarray(compute(), dtype=np.float32)
        if (
            computed.ndim != 3
            or computed.shape[:2] != images.shape[:2]
            or (self.feature_width is not None and computed.shape[2] != self.feature_width)
            or not np.isfinite(computed).all()
        ):
            raise FeatureCacheError("encoder returned invalid study feature dimensions or values")
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{feature_path.name}.partial-", dir=feature_path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.save(handle, computed, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            metadata = {
                "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
                "study_instance_uid": study_instance_uid,
                "identity_fingerprint": self.identity_fingerprint,
                "input_fingerprint": input_fingerprint,
                "feature_array_sha256": _array_sha256(computed),
                "feature_file_sha256": _file_sha256(temporary),
                "shape": list(computed.shape),
                "dtype": str(computed.dtype),
            }
            os.replace(temporary, feature_path)
            atomic_json_write(metadata_path, metadata)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        self.misses += 1
        return computed, False


__all__ = [
    "FEATURE_CACHE_SCHEMA_VERSION",
    "FeatureCacheError",
    "FrozenFeatureCache",
]
