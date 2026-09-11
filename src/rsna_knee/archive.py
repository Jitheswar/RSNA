"""Safe, checksum-verified extraction for selected private archive members."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .inventory import ArchiveIdentity, StorageEstimate


class ArchivePathError(ValueError):
    """Raised when an archive member cannot be safely mapped to a destination."""


class ArchiveIntegrityError(ValueError):
    """Raised when archive structure or a requested member is invalid."""


@dataclass(frozen=True)
class ArchiveMember:
    """Central-directory facts used for bounded extraction planning."""

    name: str
    compressed_bytes: int
    uncompressed_bytes: int
    crc32: int


@dataclass(frozen=True)
class ExtractedFile:
    """Content identity recorded only after a member was read to EOF."""

    archive_name: str
    relative_path: str
    bytes_written: int
    content_sha256: str
    archive_crc32: int


@dataclass(frozen=True)
class ExtractionFailure:
    """Private diagnostic for one member that was not published."""

    archive_name: str
    reason: str
    detail: str


@dataclass(frozen=True)
class ExtractionReport:
    """Completion record distinguishing a verified batch from a partial one."""

    archive_name: str
    destination: str
    expected_file_count: int
    extracted_files: tuple[ExtractedFile, ...]
    failures: tuple[ExtractionFailure, ...]
    complete: bool
    verified: bool

    @property
    def failure_count(self) -> int:
        return len(self.failures)


def safe_member_path(name: str) -> PurePosixPath:
    """Validate a POSIX archive name before joining it to a destination."""

    if not name or "\0" in name or "\\" in name:
        raise ArchivePathError(f"archive member has an invalid name: {name!r}")
    raw_parts = name.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ArchivePathError(f"archive member escapes the extraction root: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute():
        raise ArchivePathError(f"archive member escapes the extraction root: {name!r}")
    return path


def list_archive_members(
    archive_path: str | Path,
    *,
    prefix: str | None = None,
) -> tuple[ArchiveMember, ...]:
    """Return non-directory central-directory entries, optionally by prefix."""

    source = Path(archive_path)
    try:
        archive = zipfile.ZipFile(source, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArchiveIntegrityError(f"could not open archive {source}") from exc
    with archive:
        members: list[ArchiveMember] = []
        seen: set[str] = set()
        for info in archive.infolist():
            if info.is_dir() or (prefix is not None and not info.filename.startswith(prefix)):
                continue
            safe_member_path(info.filename)
            if info.filename in seen:
                raise ArchiveIntegrityError(f"archive contains duplicate member {info.filename!r}")
            seen.add(info.filename)
            members.append(
                ArchiveMember(
                    name=info.filename,
                    compressed_bytes=info.compress_size,
                    uncompressed_bytes=info.file_size,
                    crc32=info.CRC,
                )
            )
    return tuple(members)


def atomic_json_write(path: str | Path, payload: object, *, overwrite: bool = False) -> Path:
    """Publish JSON through a same-directory temporary file and rename."""

    destination = Path(path)
    if not overwrite and (destination.exists() or destination.is_symlink()):
        raise FileExistsError(f"refusing to overwrite protected report {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def _relative_target(root: Path, member_name: str) -> Path:
    relative = safe_member_path(member_name)
    root_resolved = root.resolve()
    target = (root / Path(*relative.parts)).resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise ArchivePathError(f"archive member escapes the extraction root: {member_name!r}") from exc
    return target


def _requested_infos(
    archive: zipfile.ZipFile,
    member_names: Iterable[str],
) -> tuple[tuple[str, zipfile.ZipInfo], ...]:
    names = tuple(member_names)
    if not names:
        raise ArchiveIntegrityError("an extraction batch must contain at least one member")
    if len(set(names)) != len(names):
        raise ArchiveIntegrityError("an extraction batch contains duplicate member names")
    infos: list[tuple[str, zipfile.ZipInfo]] = []
    seen: set[str] = set()
    for info in archive.infolist():
        if info.filename in seen:
            raise ArchiveIntegrityError(f"archive contains duplicate member {info.filename!r}")
        seen.add(info.filename)
    for name in names:
        safe_member_path(name)
        try:
            info = archive.getinfo(name)
        except KeyError as exc:
            raise ArchiveIntegrityError(f"requested archive member is missing: {name!r}") from exc
        if info.is_dir():
            raise ArchiveIntegrityError(f"requested archive member is a directory: {name!r}")
        infos.append((name, info))
    return tuple(infos)


def _copy_verified_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    target: Path,
) -> ExtractedFile:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    bytes_written = 0
    with archive.open(info, "r") as source, target.open("wb") as destination:
        while chunk := source.read(1024 * 1024):
            destination.write(chunk)
            digest.update(chunk)
            bytes_written += len(chunk)
    if bytes_written != info.file_size:
        raise ArchiveIntegrityError(
            f"archive member {info.filename!r} ended at {bytes_written} bytes; expected {info.file_size}"
        )
    return ExtractedFile(
        archive_name=info.filename,
        relative_path=info.filename,
        bytes_written=bytes_written,
        content_sha256=digest.hexdigest(),
        archive_crc32=info.CRC,
    )


def extract_batch(
    archive_path: str | Path,
    member_names: Iterable[str],
    destination: str | Path,
    *,
    storage_estimate: StorageEstimate,
    expected_archive_identity: ArchiveIdentity,
    budget_approved: bool = False,
) -> ExtractionReport:
    """Extract a selected batch into an atomic destination with content hashes.

    The caller must provide the inventory storage projection and the exact
    archive identity used to build the selection.  This keeps the low-level
    adapter from silently bypassing the ticket's admission gates.

    A failed member removes the staging directory and leaves no usable output
    directory.  The returned completion record remains explicit about the
    failed/partial attempt so callers cannot treat it as an admitted batch.
    """

    source = Path(archive_path)
    requested_names = tuple(member_names)
    target_root = Path(destination)
    if target_root.exists() or target_root.is_symlink():
        raise ArchiveIntegrityError(f"extraction destination already exists: {target_root}")
    from .inventory import ArchiveIdentity, identify_archive, enforce_storage_budget, StorageEstimate

    if not isinstance(expected_archive_identity, ArchiveIdentity):
        raise TypeError("expected_archive_identity must be an ArchiveIdentity")
    if not isinstance(storage_estimate, StorageEstimate):
        raise TypeError("storage_estimate must be a StorageEstimate")
    try:
        actual_identity = identify_archive(
            source,
            source_version=expected_archive_identity.source_version,
            include_sha256=expected_archive_identity.archive_sha256 is not None,
        )
    except (OSError, ValueError) as exc:
        raise ArchiveIntegrityError(f"could not verify source archive identity for {source}") from exc
    if actual_identity.fingerprint != expected_archive_identity.fingerprint:
        raise ArchiveIntegrityError(
            "source archive identity does not match the inventory; recompute the selection before extraction"
        )
    try:
        source_size = source.stat().st_size
    except OSError as exc:
        raise ArchiveIntegrityError(f"could not stat source archive {source}") from exc
    if storage_estimate.source_archive_bytes != source_size:
        raise ArchiveIntegrityError(
            "storage estimate does not match the source archive size; recompute it before extraction"
        )
    enforce_storage_budget(storage_estimate, approved=budget_approved)
    target_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        archive = zipfile.ZipFile(source, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArchiveIntegrityError(f"could not open archive {source}") from exc

    staging = Path(tempfile.mkdtemp(prefix=f".{target_root.name}.partial-", dir=target_root.parent))
    extracted: list[ExtractedFile] = []
    failures: list[ExtractionFailure] = []
    try:
        with archive:
            requested = _requested_infos(archive, requested_names)
            for name, info in requested:
                try:
                    extracted.append(_copy_verified_member(archive, info, _relative_target(staging, name)))
                except (EOFError, OSError, RuntimeError, zipfile.BadZipFile, ArchiveIntegrityError) as exc:
                    failures.append(ExtractionFailure(name, "archive_integrity", str(exc)))
                    break
        if failures:
            return ExtractionReport(
                archive_name=source.name,
                destination=str(target_root),
                expected_file_count=len(requested_names),
                extracted_files=tuple(extracted),
                failures=tuple(failures),
                complete=False,
                verified=False,
            )
        expected_names = {name for name, _ in requested}
        actual_names = {
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file()
        }
        if actual_names != expected_names:
            failures.append(
                ExtractionFailure(
                    "",
                    "extraction_reconciliation",
                    f"missing={sorted(expected_names - actual_names)}, extra={sorted(actual_names - expected_names)}",
                )
            )
            return ExtractionReport(
                archive_name=source.name,
                destination=str(target_root),
                expected_file_count=len(requested),
                extracted_files=tuple(extracted),
                failures=tuple(failures),
                complete=False,
                verified=False,
            )
        os.replace(staging, target_root)
        return ExtractionReport(
            archive_name=source.name,
            destination=str(target_root),
            expected_file_count=len(requested),
            extracted_files=tuple(extracted),
            failures=(),
            complete=True,
            verified=True,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def write_extraction_report(report: ExtractionReport, output_path: str | Path) -> Path:
    """Write a private, atomically published extraction completion record."""

    destination = Path(output_path)
    payload = {
        "schema_version": "rsna-knee-extraction-v1",
        "archive_name": report.archive_name,
        "destination": report.destination,
        "expected_file_count": report.expected_file_count,
        "complete": report.complete,
        "verified": report.verified,
        "extracted_files": [
            {
                "archive_name": item.archive_name,
                "relative_path": item.relative_path,
                "bytes_written": item.bytes_written,
                "content_sha256": item.content_sha256,
                "archive_crc32": item.archive_crc32,
            }
            for item in report.extracted_files
        ],
        "failures": [
            {"archive_name": item.archive_name, "reason": item.reason, "detail": item.detail}
            for item in report.failures
        ],
    }
    atomic_json_write(destination, payload)
    return destination


def failed_extraction_report(
    archive_path: str | Path,
    destination: str | Path,
    expected_file_count: int,
    *,
    reason: str,
    detail: str,
) -> ExtractionReport:
    """Build an explicit failed completion record for a rejected attempt."""

    return ExtractionReport(
        archive_name=Path(archive_path).name,
        destination=str(destination),
        expected_file_count=expected_file_count,
        extracted_files=(),
        failures=(ExtractionFailure("", reason, detail),),
        complete=False,
        verified=False,
    )


__all__ = [
    "ArchivePathError",
    "ArchiveIntegrityError",
    "ArchiveMember",
    "ExtractedFile",
    "ExtractionFailure",
    "ExtractionReport",
    "safe_member_path",
    "list_archive_members",
    "extract_batch",
    "atomic_json_write",
    "write_extraction_report",
    "failed_extraction_report",
]
