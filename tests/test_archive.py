from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rsna_knee.archive import ArchiveIntegrityError, ArchivePathError, extract_batch

from test_inventory import _write_fixture_archive


def test_extract_batch_publishes_only_after_full_member_checksum_verification(tmp_path: Path) -> None:
    from rsna_knee.inventory import estimate_storage, inventory_archive

    archive_path = tmp_path / "source.zip"
    _write_fixture_archive(archive_path)
    inventory = inventory_archive(archive_path, source_version="fixture-2026-09-11")
    destination = tmp_path / "extracted"
    member = "train_series/2.25.100/2.25.101/2.25.400.dcm"

    report = extract_batch(
        archive_path,
        (member,),
        destination,
        storage_estimate=estimate_storage(inventory, ("2.25.100",)),
        expected_archive_identity=inventory.archive_identity,
    )

    assert report.complete is True
    assert report.verified is True
    assert report.failure_count == 0
    extracted_path = destination / member
    assert extracted_path.exists()
    assert report.extracted_files[0].archive_name == member
    assert report.extracted_files[0].content_sha256 == hashlib.sha256(extracted_path.read_bytes()).hexdigest()
    assert not list(tmp_path.glob(".extracted.partial-*"))


def test_extract_batch_rejects_path_traversal_before_creating_output(tmp_path: Path) -> None:
    from rsna_knee.inventory import estimate_storage, inventory_archive

    archive_path = tmp_path / "source.zip"
    _write_fixture_archive(archive_path)
    inventory = inventory_archive(archive_path, source_version="fixture-2026-09-11")

    with pytest.raises(ArchivePathError):
        extract_batch(
            archive_path,
            ("../outside.dcm",),
            tmp_path / "unsafe",
            storage_estimate=estimate_storage(inventory, ("2.25.100",)),
            expected_archive_identity=inventory.archive_identity,
        )

    assert not (tmp_path / "unsafe").exists()


def test_extract_batch_rejects_a_storage_estimate_for_another_archive(tmp_path: Path) -> None:
    from rsna_knee.inventory import estimate_storage, identify_archive, inventory_archive

    archive_path = tmp_path / "source.zip"
    other_archive_path = tmp_path / "other-source.zip"
    _write_fixture_archive(archive_path)
    _write_fixture_archive(other_archive_path, report_only_count=2)
    report = inventory_archive(archive_path, source_version="fixture-2026-09-11")
    estimate = estimate_storage(report, ("2.25.100",))
    other_identity = identify_archive(other_archive_path, source_version="fixture-2026-09-11")

    with pytest.raises(ArchiveIntegrityError, match="does not match"):
        extract_batch(
            other_archive_path,
            ("train_series/2.25.100/2.25.101/2.25.400.dcm",),
            tmp_path / "mismatched",
            storage_estimate=estimate,
            expected_archive_identity=other_identity,
        )


def test_extract_batch_rejects_an_archive_with_a_different_identity(tmp_path: Path) -> None:
    from rsna_knee.inventory import estimate_storage, identify_archive, inventory_archive

    archive_path = tmp_path / "source.zip"
    other_archive_path = tmp_path / "other-source.zip"
    _write_fixture_archive(archive_path)
    _write_fixture_archive(other_archive_path, report_only_count=2)
    expected_identity = identify_archive(other_archive_path, source_version="fixture-2026-09-11")
    inventory = inventory_archive(archive_path, source_version="fixture-2026-09-11")

    with pytest.raises(ArchiveIntegrityError, match="identity"):
        extract_batch(
            archive_path,
            ("train_series/2.25.100/2.25.101/2.25.400.dcm",),
            tmp_path / "wrong-identity",
            storage_estimate=estimate_storage(inventory, ("2.25.100",)),
            expected_archive_identity=expected_identity,
        )
