"""Source integrity checks must reject corruption even with plausible records."""

import hashlib
import importlib.util
from pathlib import Path
import zlib

import pytest


SPEC = importlib.util.spec_from_file_location(
    "ticket59_input_verifier", Path(__file__).parents[1] / "scripts/validate_ticket59_full_input_probe.py"
)
assert SPEC and SPEC.loader
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


@pytest.mark.parametrize("changed_record", ["producer_sha", "archive_crc", "same_size_payload"])
def test_independent_source_check_rejects_corruption(tmp_path, changed_record):
    original = b"original retained MRI payload"
    path = tmp_path / "slice.dcm"
    path.write_bytes(original)
    record = {"bytes": len(original), "actual_sha256": hashlib.sha256(original).hexdigest(),
              "archive_crc32": zlib.crc32(original)}
    assert VERIFIER.verify_payload((path, record)) == len(original)
    if changed_record == "producer_sha":
        record["actual_sha256"] = "0" * 64
    elif changed_record == "archive_crc":
        record["archive_crc32"] ^= 1
    else:
        path.write_bytes(b"X" + original[1:])
    with pytest.raises(ValueError, match="SHA/independent archive CRC differs"):
        VERIFIER.verify_payload((path, record))


def test_payload_symlink_cannot_substitute_for_original_source(tmp_path):
    source = tmp_path / "source.dcm"
    source.write_bytes(b"MRI")
    link = tmp_path / "substitute.dcm"
    link.symlink_to(source)
    record = {"bytes": 3, "actual_sha256": hashlib.sha256(b"MRI").hexdigest(), "archive_crc32": zlib.crc32(b"MRI")}
    with pytest.raises(ValueError, match="symlinked"):
        VERIFIER.verify_payload((link, record))
