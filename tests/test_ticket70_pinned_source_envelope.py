"""Reject incomplete or newer asset envelopes before any launch admission."""

from datetime import datetime, timezone
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("ticket70_envelope", Path(__file__).resolve().parents[1] / "scripts/prepare_ticket70_pinned_source_envelope.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    metadata = dict(dataset_sources=["owner/one", "owner/two"], model_sources=["model/name/PyTorch/small/1"], enable_internet=False)
    rows = [dict(ref=ref, currentVersionNumber=version, lastUpdated="2026-08-01T00:00:00Z",
                 versions=[dict(versionNumber=version, creationDate="2026-08-01T00:00:00Z", status="Ready")])
            for ref, version in zip(metadata["dataset_sources"], [1, 2])]
    cutoff = datetime(2026, 9, 1, tzinfo=timezone.utc)
    docker = "gcr.io/kaggle-private-byod/python@sha256:" + "a" * 64
    return metadata, rows, cutoff, docker


def test_only_dataset_version_and_saved_container_fields_change():
    metadata, rows, cutoff, docker = fixture()
    actual = module.pinned_metadata(metadata, rows, cutoff, docker)
    assert actual == {**metadata, "dataset_sources":["owner/one/1", "owner/two/2"],
                      "docker_image":docker, "docker_image_pinning_type":"original"}
    assert metadata["dataset_sources"] == ["owner/one", "owner/two"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_ref", "new_update", "new_version", "not_ready", "ambiguous_version", "bool_version", "unpinned_container"])
def test_unsupported_historical_lineage_is_rejected(change):
    metadata, rows, cutoff, docker = fixture()
    if change == "missing": rows.pop()
    if change == "duplicate": rows[1] = rows[0]
    if change == "wrong_ref": rows[1]["ref"] = "other/dataset"
    if change == "new_update": rows[1]["lastUpdated"] = "2026-10-01T00:00:00Z"
    if change == "new_version": rows[1]["versions"][0]["creationDate"] = "2026-10-01T00:00:00Z"
    if change == "not_ready": rows[1]["versions"][0]["status"] = "Pending"
    if change == "ambiguous_version": rows[1]["versions"] *= 2
    if change == "bool_version": rows[1]["currentVersionNumber"] = True
    if change == "unpinned_container": docker = "latest"
    with pytest.raises(ValueError): module.pinned_metadata(metadata, rows, cutoff, docker)
