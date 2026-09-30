"""Terminal scoring status must never create unearned GPU budget headroom."""

import hashlib
import json

import pytest

from rsna_knee.gpu_campaign import (
    GpuCampaignLedgerError,
    begin_gpu_submission_attempt,
    gpu_campaign_capacity,
    initialize_gpu_campaign_ledger,
    record_gpu_submission_result,
    record_kaggle_scoring_terminal_without_allocation,
    reserve_gpu_campaign_request,
)


def active_scoring(tmp_path):
    ledger = tmp_path / "ledger.json"
    checksum = "a" * 64
    initialize_gpu_campaign_ledger(ledger, reserved_comparison_hours=0)
    reserve_gpu_campaign_request(ledger, request_checksum=checksum, experiment_id="scoring",
                                wall_time_minutes=480, gpu_count=2,
                                retry_allowed=False, remaining_attempts=1)
    begin_gpu_submission_attempt(ledger, request_checksum=checksum, attempt_number=1, submitted_epoch=100)
    record_gpu_submission_result(ledger, request_checksum=checksum, attempt_number=1,
                                 job_id="123", exit_code=0, submission_wait=False)
    proof = {"schema_version": "rsna-kaggle-scoring-terminal-evidence-v1",
             "request_checksum": checksum, "job_id": "123", "allocation_timing_available": False,
             "submission": {"ref": 123, "status": "COMPLETE"}}
    return ledger, checksum, proof


def close(ledger, checksum, proof, tmp_path, *, digest=None):
    evidence = tmp_path / "terminal.json"
    evidence.write_text(json.dumps(proof))
    return record_kaggle_scoring_terminal_without_allocation(
        ledger, request_checksum=checksum, job_id="123", evidence_path=evidence,
        evidence_sha256=digest or hashlib.sha256(evidence.read_bytes()).hexdigest())


@pytest.mark.parametrize("status,expected", [("COMPLETE", "completed"), ("ERROR", "failed")])
def test_terminal_job_charges_both_gpus_for_the_entire_unknown_reservation(tmp_path, status, expected):
    ledger, checksum, proof = active_scoring(tmp_path)
    before = gpu_campaign_capacity(ledger)
    proof["submission"]["status"] = status
    entry = close(ledger, checksum, proof, tmp_path)
    after = gpu_campaign_capacity(ledger)
    assert entry["status"] == expected
    assert entry["actual_allocation_known"] is False
    assert entry["actual_allocated_seconds"] == 0
    assert entry["attempts"][0]["allocated_seconds"] is None
    assert entry["unverified_allocation_charge_seconds"] == 57_600
    assert after["outstanding_reserved_seconds"] == 0
    assert after["spent_seconds"] == 57_600
    assert after["remaining_seconds"] == before["remaining_seconds"]
    reserve_gpu_campaign_request(ledger, request_checksum="b" * 64, experiment_id="next-visible-trial",
                                wall_time_minutes=30, gpu_count=2,
                                retry_allowed=False, remaining_attempts=1)


@pytest.mark.parametrize("change", ["pending", "wrong_ref", "wrong_job", "wrong_request", "false_timing"])
def test_unproven_or_mismatched_terminal_record_cannot_release_active_gpu_request(tmp_path, change):
    ledger, checksum, proof = active_scoring(tmp_path)
    if change == "pending": proof["submission"]["status"] = "PENDING"
    if change == "wrong_ref": proof["submission"]["ref"] = 124
    if change == "wrong_job": proof["job_id"] = "124"
    if change == "wrong_request": proof["request_checksum"] = "c" * 64
    if change == "false_timing": proof["allocation_timing_available"] = True
    before = ledger.read_bytes()
    with pytest.raises(GpuCampaignLedgerError):
        close(ledger, checksum, proof, tmp_path)
    assert ledger.read_bytes() == before


def test_changed_terminal_artifact_cannot_release_request(tmp_path):
    ledger, checksum, proof = active_scoring(tmp_path)
    before = ledger.read_bytes()
    with pytest.raises(GpuCampaignLedgerError, match="checksum"):
        close(ledger, checksum, proof, tmp_path, digest="0" * 64)
    assert ledger.read_bytes() == before


def test_terminal_charge_cannot_be_applied_twice(tmp_path):
    ledger, checksum, proof = active_scoring(tmp_path)
    close(ledger, checksum, proof, tmp_path)
    before = ledger.read_bytes()
    with pytest.raises(GpuCampaignLedgerError, match="active submission"):
        close(ledger, checksum, proof, tmp_path)
    assert ledger.read_bytes() == before
