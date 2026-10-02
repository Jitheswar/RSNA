import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsna_knee.gpu_campaign import (
    begin_gpu_submission_attempt, gpu_campaign_capacity,
    initialize_gpu_campaign_ledger, record_gpu_submission_result,
    reserve_gpu_campaign_request,
)

SPEC = importlib.util.spec_from_file_location("ticket70_lifecycle", Path(__file__).parents[1] / "scripts/ticket70_accelerator_lifecycle.py")
life = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(life)


def quota(gpu_free=20000, reserved=0, paid=False):
    return {"accelerators": {name: {"free_seconds": gpu_free, "time_reserved_seconds": reserved,
                                   "paid_scaling_enabled": paid} for name in ("gpu", "tpu")}}


@pytest.mark.parametrize("value", [quota(gpu_free=14399.99), quota(reserved=.001), quota(paid=True), quota(paid=None)])
def test_full_bound_and_actual_zero_paid_reservations_required(value):
    with pytest.raises(ValueError):
        life.require_free_quota(value)


def test_quota_uses_typed_timedelta_not_malformed_json():
    q = SimpleNamespace(time_used=timedelta(seconds=50821.751), time_reserved=timedelta(0),
                        total_time_allowed=timedelta(seconds=108000), is_pay_to_scale_enabled=False)
    api = SimpleNamespace(quota_view=lambda: SimpleNamespace(gpu_quota=q, tpu_quota=q,
                          quota_refresh_time=datetime(2026, 10, 3)))
    result = life.typed_quota(api)
    assert result["accelerators"]["gpu"]["free_seconds"] == 57178.249
    life.require_free_quota(result)


def terminal_fixture(tmp_path, monkeypatch, status="COMPLETE"):
    root = tmp_path / "run"
    root.mkdir()
    ledger = tmp_path / "ledger.json"
    monkeypatch.setattr(life, "ROOT", root)
    monkeypatch.setattr(life, "LEDGER", ledger)
    monkeypatch.setattr(life, "admission", lambda: None)
    life.write(root / "request.json", {"fixed": "request"})
    checksum = life.digest(root / "request.json")
    handle = "jitheswar/fixed-development"
    life.write(root / "accepted-launch.json", {"kernel_id": handle})
    life.write(root / "kernel-terminal.json", {"status": status, "kernel_id": handle, "kernel_version": 1})
    initialize_gpu_campaign_ledger(ledger, reserved_comparison_hours=0)
    reserve_gpu_campaign_request(ledger, request_checksum=checksum, experiment_id="fixed",
                                 wall_time_minutes=120, retry_allowed=False, remaining_attempts=1, gpu_count=2)
    begin_gpu_submission_attempt(ledger, request_checksum=checksum, attempt_number=1, submitted_epoch=1)
    record_gpu_submission_result(ledger, request_checksum=checksum, attempt_number=1, job_id=handle,
                                 exit_code=0, submission_wait=False)
    return root, ledger


@pytest.mark.parametrize("status", ["COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"])
def test_terminal_charges_entire_bound_exactly_once(tmp_path, monkeypatch, status):
    _, ledger = terminal_fixture(tmp_path, monkeypatch, status)
    life.close_terminal()
    before = ledger.read_bytes()
    life.close_terminal()
    assert ledger.read_bytes() == before
    capacity = gpu_campaign_capacity(ledger)
    assert capacity["spent_seconds"] == 14400
    assert capacity["outstanding_reserved_seconds"] == 0
    entry = json.loads(before)["requests"][0]
    assert entry["actual_allocation_known"] is False
    assert entry["actual_allocated_seconds"] == 0


@pytest.mark.parametrize("update", [{"status": "RUNNING"}, {"kernel_id": "jitheswar/wrong"}, {"kernel_version": 2}])
def test_nonterminal_or_wrong_identity_never_closes_reservation(tmp_path, monkeypatch, update):
    root, ledger = terminal_fixture(tmp_path, monkeypatch)
    value = life.read(root / "kernel-terminal.json")
    value.update(update)
    (root / "kernel-terminal.json").write_text(json.dumps(value))
    before = ledger.read_bytes()
    with pytest.raises(ValueError):
        life.close_terminal()
    assert ledger.read_bytes() == before


def test_consumed_intent_never_calls_api_or_mutates_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(life, "ROOT", tmp_path)
    monkeypatch.setattr(life, "admission", lambda: ({}, {}, {}))
    monkeypatch.setattr(life, "api_client", lambda: pytest.fail("Consumed launch must not call API"))
    (tmp_path / "launch-intent.json").write_text("{}")
    with pytest.raises(FileExistsError):
        life.launch()


def remote_fixture():
    request = {"docker_image": "image@sha256:fixed"}
    accepted = {"kernel_id": "jitheswar/fixed", "kernel_numeric_id": 123,
                "kernel_version": 1, "request_sha256": "a" * 64}
    notebook = {"cells": [{"cell_type": "code", "source": ["print('fixed')\n"]}]}
    metadata = {"dataset_sources": ["owner/asset/2"], "competition_sources": ["competition"],
                "model_sources": ["owner/model/framework/variant/1"], "kernel_sources": []}
    actual = {"ref": accepted["kernel_id"], "id": 123, "currentVersionNumber": 1,
              "isPrivate": True, "enableGpu": True, "enableTpu": False, "enableInternet": False,
              "machineShape": "NvidiaTeslaT4", "dockerImage": request["docker_image"],
              "datasetDataSources": ["owner/asset"], "competitionDataSources": ["competition"],
              "modelDataSources": metadata["model_sources"], "kernelDataSources": []}
    remote = {"metadata": actual, "blob": {"source": json.dumps(notebook)}}
    return remote, request, accepted, notebook, metadata


def test_remote_canonical_refs_preserve_accepted_version_pins():
    args = remote_fixture()
    result = life.validate_remote(*args)
    assert result["status"] == "PASS"
    assert result["numeric_dataset_mount_versions_exposed"] is False
    assert args[-1]["dataset_sources"] == ["owner/asset/2"]


@pytest.mark.parametrize("change", [{"enableInternet": True}, {"isPrivate": False},
                                    {"currentVersionNumber": 2}, {"id": 124},
                                    {"dockerImage": "other"}, {"datasetDataSources": ["owner/other"]},
                                    {"kernelDataSources": ["owner/extra"]}])
def test_wrong_remote_identity_or_settings_rejected(change):
    args = remote_fixture()
    args[0]["metadata"].update(change)
    with pytest.raises(ValueError):
        life.validate_remote(*args)


def test_code_changes_rejected_even_with_same_remote_identity():
    args = remote_fixture()
    args[0]["blob"]["source"] = json.dumps({"cells": [{"cell_type": "code", "source": "print('changed')"}]})
    with pytest.raises(ValueError, match="executable source"):
        life.validate_remote(*args)
