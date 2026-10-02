"""Carry an immutable experiment configuration with its resumable checkpoint."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def carry_configuration(source, destination, *, expected_sha256):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file() or destination.is_symlink():
        raise ValueError("Regular immutable configuration files required")
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Original configuration bytes changed")
    value = json.loads(data)
    if not isinstance(value, dict) or any(key not in value for key in
            ("run_id", "config", "scientific_configuration_fingerprint", "semantic_fingerprint", "identity", "fold")):
        raise ValueError("Complete original immutable experiment configuration required")
    if destination.exists():
        if destination.read_bytes() != data:
            raise ValueError("Refusing to replace another immutable configuration")
        return expected_sha256
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="configuration-handoff-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        # Atomic, exclusive publication: readers never see a partial JSON file.
        os.link(temporary, destination)
    finally:
        temporary.unlink()
    return expected_sha256
