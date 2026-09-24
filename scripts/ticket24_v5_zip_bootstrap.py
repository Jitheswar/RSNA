"""Stdlib-only bounded extractor embedded in the Ticket 24 v5 notebook."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


_ARCHIVE_LIMITS = {
    "candidate.zip": {
        "max_archive_bytes": 8_000_000,
        "max_entries": 96,
        "max_files": 48,
        "max_total_bytes": 2_000_000,
    },
    "code.zip": {
        "max_archive_bytes": 8_000_000,
        "max_entries": 96,
        "max_files": 48,
        "max_total_bytes": 2_000_000,
    },
    "runtime.zip": {
        "max_archive_bytes": 250_000_000,
        "max_entries": 8,
        "max_files": 4,
        "max_total_bytes": 220_000_000,
    },
}
_MAX_MEMBER_PATH_BYTES = 512
_MAX_MEMBER_DEPTH = 24
_MAX_COMPRESSION_RATIO = 1000
_MAX_STAGED_ARCHIVE_BYTES = 225_000_000
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _json_sha256(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_member(value: str, *, allow_directory: bool = False) -> tuple[PurePosixPath, bool]:
    if not value or len(value.encode("utf-8")) > _MAX_MEMBER_PATH_BYTES:
        raise RuntimeError("archive member path is empty or too long")
    if "\x00" in value or "\\" in value or value.startswith("/"):
        raise RuntimeError(f"archive member path is not relative POSIX: {value!r}")
    is_directory = allow_directory and value.endswith("/")
    normalized = value[:-1] if is_directory else value
    parts = normalized.split("/")
    if (
        not normalized
        or len(parts) > _MAX_MEMBER_DEPTH
        or any(part in {"", ".", ".."} or ":" in part for part in parts)
    ):
        raise RuntimeError(f"archive member path is unsafe: {value!r}")
    path = PurePosixPath(*parts)
    if path.as_posix() != normalized:
        raise RuntimeError(f"archive member path is not canonical: {value!r}")
    return path, is_directory


def _extract_delivery_archive(
    archive_path: Path,
    destination: Path,
    *,
    limits: dict[str, int],
) -> tuple[dict[str, dict[str, int | str]], set[str]]:
    if archive_path.is_symlink() or not archive_path.is_file():
        raise RuntimeError(f"delivery archive is not a regular file: {archive_path.name}")
    if archive_path.stat().st_size > limits["max_archive_bytes"]:
        raise RuntimeError(f"delivery archive exceeds its compressed size bound: {archive_path.name}")
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"delivery extraction destination already exists: {destination.name}")
    destination.mkdir(parents=True, exist_ok=False)

    observed_files: dict[str, dict[str, int | str]] = {}
    observed_directories: set[str] = set()
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            entries = archive.infolist()
            if not entries or len(entries) > limits["max_entries"]:
                raise RuntimeError(f"delivery archive has an invalid entry count: {archive_path.name}")
            names: set[str] = set()
            folded_names: set[str] = set()
            file_sizes: dict[str, int] = {}
            total_bytes = 0
            file_count = 0
            for entry in entries:
                relative, is_directory = _canonical_member(entry.filename, allow_directory=True)
                name = relative.as_posix()
                folded = name.casefold()
                if name in names or folded in folded_names:
                    raise RuntimeError(f"delivery archive has duplicate member paths: {name}")
                names.add(name)
                folded_names.add(folded)
                mode = (entry.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if file_type == stat.S_IFLNK:
                    raise RuntimeError(f"delivery archive contains a symbolic link: {name}")
                if file_type and file_type not in (stat.S_IFREG, stat.S_IFDIR):
                    raise RuntimeError(f"delivery archive contains a special file: {name}")
                if entry.flag_bits & 0x1:
                    raise RuntimeError(f"delivery archive contains an encrypted member: {name}")
                if entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise RuntimeError(f"delivery archive uses an unsupported compression method: {name}")
                if is_directory:
                    if entry.file_size != 0:
                        raise RuntimeError(f"delivery directory member has data: {name}")
                    observed_directories.add(name)
                    continue
                if entry.is_dir():
                    raise RuntimeError(f"delivery archive directory entry is malformed: {name}")
                if entry.file_size < 0 or entry.compress_size < 0:
                    raise RuntimeError(f"delivery archive member has a negative size: {name}")
                if entry.file_size and (
                    not entry.compress_size
                    or entry.file_size > entry.compress_size * _MAX_COMPRESSION_RATIO
                ):
                    raise RuntimeError(f"delivery archive member exceeds the compression-ratio bound: {name}")
                total_bytes += entry.file_size
                file_count += 1
                file_sizes[name] = entry.file_size
            if file_count == 0 or file_count > limits["max_files"]:
                raise RuntimeError(f"delivery archive has an invalid file count: {archive_path.name}")
            if total_bytes > limits["max_total_bytes"]:
                raise RuntimeError(f"delivery archive exceeds its expanded-size bound: {archive_path.name}")
            for name in file_sizes:
                path = PurePosixPath(name)
                if any(parent.as_posix() in file_sizes for parent in path.parents if str(parent) != "."):
                    raise RuntimeError(f"delivery archive file shadows a parent directory: {name}")

            written_bytes = 0
            for entry in sorted(entries, key=lambda item: item.filename):
                relative, is_directory = _canonical_member(entry.filename, allow_directory=True)
                name = relative.as_posix()
                target = destination.joinpath(*relative.parts)
                if is_directory:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                member_bytes = 0
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(target, flags, 0o600)
                try:
                    with archive.open(entry, "r") as source, os.fdopen(descriptor, "wb") as output:
                        while True:
                            block = source.read(1024 * 1024)
                            if not block:
                                break
                            member_bytes += len(block)
                            written_bytes += len(block)
                            if member_bytes > entry.file_size or written_bytes > limits["max_total_bytes"]:
                                raise RuntimeError(f"delivery archive expanded beyond its declared bound: {name}")
                            digest.update(block)
                            output.write(block)
                except BaseException:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                    raise
                if member_bytes != entry.file_size:
                    raise RuntimeError(f"delivery archive member size changed while extracting: {name}")
                observed_files[name] = {"sha256": digest.hexdigest(), "size_bytes": member_bytes}
            if written_bytes != total_bytes:
                raise RuntimeError(f"delivery archive expanded to an unexpected size: {archive_path.name}")
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        shutil.rmtree(destination, ignore_errors=True)
        if isinstance(error, RuntimeError):
            raise
        raise RuntimeError(f"could not safely extract delivery archive {archive_path.name}") from error
    return observed_files, observed_directories


def _manifest_expected_files(manifest: dict[str, Any], expected_checksum: str) -> dict[str, dict[str, int | str]]:
    stored_checksum = manifest.get("manifest_checksum")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_checksum", None)
    if stored_checksum != expected_checksum or _json_sha256(unsigned_manifest) != expected_checksum:
        raise RuntimeError("candidate manifest checksum differs from the exact request")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("candidate manifest has no artifact list")
    expected: dict[str, dict[str, int | str]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise RuntimeError("candidate manifest artifact is malformed")
        raw_path = artifact.get("path")
        if not isinstance(raw_path, str):
            raise RuntimeError("candidate manifest artifact path is missing")
        relative, is_directory = _canonical_member(raw_path)
        if is_directory:
            raise RuntimeError("candidate manifest artifact path is a directory")
        digest = artifact.get("sha256")
        size = artifact.get("size_bytes")
        if not isinstance(digest, str) or _HASH_PATTERN.fullmatch(digest) is None:
            raise RuntimeError(f"candidate manifest artifact hash is malformed: {raw_path}")
        if not isinstance(size, int) or size < 0:
            raise RuntimeError(f"candidate manifest artifact size is malformed: {raw_path}")
        if relative.as_posix() in expected:
            raise RuntimeError(f"candidate manifest repeats an artifact path: {raw_path}")
        expected[relative.as_posix()] = {"sha256": digest, "size_bytes": size}
    return expected


def _verify_archive_files(
    label: str,
    observed: dict[str, dict[str, int | str]],
    observed_directories: set[str],
    expected: dict[str, dict[str, int | str]],
) -> None:
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise RuntimeError(f"{label} archive file set differs from its bindings; missing={missing}, extra={extra}")
    for name, expected_record in expected.items():
        actual = observed[name]
        if actual["sha256"] != expected_record["sha256"]:
            raise RuntimeError(f"{label} archive checksum differs from its binding: {name}")
        expected_size = expected_record.get("size_bytes")
        if expected_size is not None and actual["size_bytes"] != expected_size:
            raise RuntimeError(f"{label} archive size differs from its binding: {name}")
    allowed_directories = {
        parent.as_posix()
        for name in expected
        for parent in PurePosixPath(name).parents
        if str(parent) != "."
    }
    extra_directories = sorted(observed_directories - allowed_directories)
    if extra_directories:
        raise RuntimeError(f"{label} archive has unexpected directories: {extra_directories}")


def stage_delivery_archives(
    source_root: str | Path,
    request_path: str | Path,
    request_payload: dict[str, Any],
    *,
    expected_request_checksum: str,
    expected_candidate_manifest_checksum: str,
    working_root: str | Path,
) -> tuple[Any, Path, dict[str, dict[str, int | str]]]:
    """Safely stage the three Kaggle CLI archives and verify every bound file."""

    source = Path(source_root)
    request_file = Path(request_path)
    work = Path(working_root)
    if source.is_symlink() or not source.is_dir() or request_file.is_symlink() or not request_file.is_file():
        raise RuntimeError("delivery source root or request is not a regular mounted input")
    if request_file.parent.resolve() != source.resolve() or request_file.name != "request.json":
        raise RuntimeError("delivery request is outside its archive mount")
    if work.is_symlink() or not work.is_dir():
        raise RuntimeError("Kaggle working root is not a regular directory")
    unsigned_request = dict(request_payload)
    unsigned_request.pop("request_checksum", None)
    if (
        request_payload.get("request_checksum") != expected_request_checksum
        or _json_sha256(unsigned_request) != expected_request_checksum
    ):
        raise RuntimeError("mounted delivery request checksum differs from its binding")
    request_candidate = request_payload.get("candidate")
    if not isinstance(request_candidate, dict) or request_candidate.get("manifest_checksum") != expected_candidate_manifest_checksum:
        raise RuntimeError("mounted delivery request candidate differs from its binding")
    notebook_reference = request_payload.get("delivery_notebook")
    if not isinstance(notebook_reference, dict):
        raise RuntimeError("delivery request notebook reference is malformed")
    notebook_name = notebook_reference.get("filename")
    notebook_sha256 = notebook_reference.get("sha256")
    if not isinstance(notebook_name, str) or PurePosixPath(notebook_name).name != notebook_name:
        raise RuntimeError("delivery notebook filename is unsafe")
    if not isinstance(notebook_sha256, str) or _HASH_PATTERN.fullmatch(notebook_sha256) is None:
        raise RuntimeError("delivery notebook hash is malformed")
    mounted_notebook = source / notebook_name
    if mounted_notebook.is_symlink() or not mounted_notebook.is_file() or _sha256_path(mounted_notebook) != notebook_sha256:
        raise RuntimeError("mounted delivery notebook checksum differs from the request")
    request_bytes = request_file.read_bytes()
    try:
        parsed_request = json.loads(request_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("mounted delivery request is not valid JSON") from error
    if parsed_request != request_payload:
        raise RuntimeError("mounted delivery request changed after request discovery")

    source_archives: dict[str, Path] = {}
    total_compressed_bytes = 0
    for archive_name, limits in _ARCHIVE_LIMITS.items():
        path = source / archive_name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"mounted delivery archive is missing: {archive_name}")
        archive_size = path.stat().st_size
        if archive_size > limits["max_archive_bytes"]:
            raise RuntimeError(f"mounted delivery archive exceeds its compressed bound: {archive_name}")
        total_compressed_bytes += archive_size
        source_archives[archive_name] = path
    if total_compressed_bytes > _MAX_STAGED_ARCHIVE_BYTES:
        raise RuntimeError("combined mounted delivery archives exceed their staging bound")

    temporary = tempfile.TemporaryDirectory(prefix="ticket24-delivery-", dir=work)
    delivery_root = Path(temporary.name)
    try:
        extracted: dict[str, tuple[dict[str, dict[str, int | str]], set[str]]] = {}
        for archive_name, limits in _ARCHIVE_LIMITS.items():
            label = archive_name.removesuffix(".zip")
            extracted[label] = _extract_delivery_archive(
                source_archives[archive_name],
                delivery_root / label,
                limits=limits,
            )

        candidate_root = delivery_root / "candidate"
        try:
            candidate_manifest = json.loads((candidate_root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError("candidate archive has no readable manifest") from error
        if candidate_manifest.get("candidate_id") != request_candidate.get("candidate_id"):
            raise RuntimeError("candidate archive id differs from the exact request")
        candidate_expected = _manifest_expected_files(candidate_manifest, expected_candidate_manifest_checksum)
        manifest_path = candidate_root / "manifest.json"
        candidate_expected["manifest.json"] = {
            "sha256": _sha256_path(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        }
        _verify_archive_files("candidate", extracted["candidate"][0], extracted["candidate"][1], candidate_expected)

        adapter_reference = request_payload.get("delivery_adapter")
        if not isinstance(adapter_reference, dict):
            raise RuntimeError("delivery adapter reference is malformed")
        adapter_name = adapter_reference.get("filename")
        adapter_sha256 = adapter_reference.get("sha256")
        if not isinstance(adapter_name, str) or PurePosixPath(adapter_name).name != adapter_name:
            raise RuntimeError("delivery adapter filename is unsafe")
        if not isinstance(adapter_sha256, str) or _HASH_PATTERN.fullmatch(adapter_sha256) is None:
            raise RuntimeError("delivery adapter hash is malformed")
        code_expected: dict[str, dict[str, int | str]] = {}
        for record in candidate_manifest["artifacts"]:
            if record.get("kind") != "shared_inference_code":
                continue
            raw_path = record["path"]
            if not isinstance(raw_path, str) or not raw_path.startswith("code/"):
                raise RuntimeError("candidate shared-code path is not canonical")
            relative = raw_path[len("code/"):]
            code_expected[relative] = {"sha256": record["sha256"], "size_bytes": record["size_bytes"]}
        adapter_relative = f"rsna_knee/{adapter_name}"
        if adapter_relative in code_expected:
            raise RuntimeError("delivery adapter path collides with frozen candidate code")
        code_expected[adapter_relative] = {"sha256": adapter_sha256}
        _verify_archive_files("code", extracted["code"][0], extracted["code"][1], code_expected)

        runtime_reference = request_payload.get("delivery_runtime")
        if not isinstance(runtime_reference, dict):
            raise RuntimeError("delivery runtime reference is malformed")
        wheel_references = runtime_reference.get("offline_wheels")
        if not isinstance(wheel_references, list) or not wheel_references:
            raise RuntimeError("delivery request has no offline wheel references")
        runtime_expected: dict[str, dict[str, int | str]] = {}
        for wheel in wheel_references:
            if not isinstance(wheel, dict):
                raise RuntimeError("delivery runtime wheel reference is malformed")
            filename = wheel.get("filename")
            relative_path = wheel.get("relative_path")
            digest = wheel.get("sha256")
            if not isinstance(filename, str) or PurePosixPath(filename).name != filename:
                raise RuntimeError("delivery runtime wheel filename is unsafe")
            if relative_path != f"runtime/{filename}":
                raise RuntimeError("delivery runtime wheel path is not canonical")
            if not isinstance(digest, str) or _HASH_PATTERN.fullmatch(digest) is None:
                raise RuntimeError("delivery runtime wheel hash is malformed")
            if filename in runtime_expected:
                raise RuntimeError(f"delivery runtime repeats a wheel: {filename}")
            runtime_expected[filename] = {"sha256": digest}
        _verify_archive_files("runtime", extracted["runtime"][0], extracted["runtime"][1], runtime_expected)

        shutil.copyfile(request_file, delivery_root / "request.json")
        shutil.copyfile(mounted_notebook, delivery_root / notebook_name)
        audit: dict[str, dict[str, int | str]] = {}
        for archive_name, limits in _ARCHIVE_LIMITS.items():
            label = archive_name.removesuffix(".zip")
            files = extracted[label][0]
            audit[label] = {
                "verified_file_count": len(files),
                "expanded_bytes": sum(int(record["size_bytes"]) for record in files.values()),
                "archive_bytes": source_archives[archive_name].stat().st_size,
                "verified_hash_count": len(files),
            }
        return temporary, delivery_root, audit
    except BaseException:
        temporary.cleanup()
        raise
