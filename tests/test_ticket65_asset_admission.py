"""Reject wrong or unsupported checkpoint assets before any model is built."""

import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / "scripts/ticket65_biomedclip_asset_admission.py"
spec = importlib.util.spec_from_file_location("ticket65_assets", SOURCE)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_complete_hash_and_byte_count_required(tmp_path):
    path = tmp_path / "asset"
    path.write_bytes(b"fixed checkpoint")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    worker.checked_identity(path, path.stat().st_size, checksum)
    with pytest.raises(ValueError):
        worker.checked_identity(path, path.stat().st_size + 1, checksum)
    with pytest.raises(ValueError):
        worker.checked_identity(path, path.stat().st_size, "0" * 64)


def test_observed_flat_and_wrapped_tensor_states():
    tensor = torch.ones(2)
    state, wrapper, prefixed = worker.tensor_state({"visual.proj": tensor})
    assert state["visual.proj"] is tensor and wrapper == "flat" and not prefixed
    state, wrapper, prefixed = worker.tensor_state({"state_dict": {"module.visual.proj": tensor}})
    assert state["visual.proj"] is tensor and wrapper == "state_dict" and prefixed


@pytest.mark.parametrize("value", [
    {}, {"visual.proj": "not a tensor"}, {"text.proj": torch.ones(2)},
    {"module.visual.proj": torch.ones(2), "text.proj": torch.ones(2)},
])
def test_unsupported_states_are_rejected(value):
    with pytest.raises(ValueError):
        worker.tensor_state(value)
