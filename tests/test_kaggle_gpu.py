from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rsna_knee.gpu_campaign import (
    GpuCampaignBudgetError,
    GpuCampaignLedgerError,
    begin_gpu_submission_attempt,
    gpu_campaign_capacity,
    initialize_gpu_campaign_ledger,
    mark_gpu_campaign_request_externally_blocked,
    record_gpu_backend_allocation,
    record_gpu_submission_result,
    reserve_gpu_campaign_request,
)
from rsna_knee.kaggle_gpu import KaggleGpuError, load_kaggle_gpu_request
from rsna_knee.real_inference import KaggleCompetitionSource


def _json_file(path: Path, value: object) -> tuple[Path, str]:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_slurm_blocker_stays_charged_and_allows_bounded_t4_reservation(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    initialize_gpu_campaign_ledger(ledger, reserved_comparison_hours=18)
    slurm_checksum = "a" * 64
    reserve_gpu_campaign_request(
        ledger,
        request_checksum=slurm_checksum,
        experiment_id="ticket33-slurm",
        wall_time_minutes=30,
        retry_allowed=True,
        remaining_attempts=2,
    )
    begin_gpu_submission_attempt(
        ledger,
        request_checksum=slurm_checksum,
        attempt_number=1,
        submitted_epoch=100,
    )
    record_gpu_submission_result(
        ledger,
        request_checksum=slurm_checksum,
        attempt_number=1,
        job_id="7190",
        exit_code=0,
        submission_wait=False,
    )
    evidence, evidence_sha = _json_file(
        tmp_path / "slurm-blocker.json",
        {
            "schema_version": "rsna-ticket33-slurm-external-blocker-v1",
            "job_id": "7190",
            "held_pending_before_cancel": True,
            "cancel_command_exit_code": 0,
            "absent_from_queue_after_cancel": True,
            "sacct_status": "unavailable_connection_refused",
        },
    )
    blocked = mark_gpu_campaign_request_externally_blocked(
        ledger,
        request_checksum=slurm_checksum,
        slurm_job_id="7190",
        evidence_path=evidence,
        evidence_sha256=evidence_sha,
        conservative_charge_seconds=3600,
    )
    assert blocked["status"] == "externally_blocked"
    assert blocked["actual_allocation_known"] is False
    assert blocked["unverified_allocation_charge_seconds"] == 3600
    assert gpu_campaign_capacity(ledger) == {
        "budget_seconds": 72000,
        "spent_seconds": 3600,
        "outstanding_reserved_seconds": 0,
        "comparison_reserved_seconds": 64800,
        "remaining_seconds": 68400,
    }

    reserve_gpu_campaign_request(
        ledger,
        request_checksum="b" * 64,
        experiment_id="ticket33-kaggle-t4",
        wall_time_minutes=30,
        retry_allowed=False,
        remaining_attempts=1,
        gpu_count=2,
    )
    capacity = gpu_campaign_capacity(ledger)
    assert capacity["outstanding_reserved_seconds"] == 3600
    assert capacity["comparison_reserved_seconds"] == 64800
    assert capacity["remaining_seconds"] == 64800
    kaggle_checksum = "b" * 64
    begin_gpu_submission_attempt(
        ledger,
        request_checksum=kaggle_checksum,
        attempt_number=1,
        submitted_epoch=200,
    )
    evidence, evidence_sha = _json_file(
        tmp_path / "kaggle-run.json",
        {
            "schema_version": "rsna-ticket33-kaggle-gpu-evidence-v1",
            "status": "completed",
            "request_checksum": kaggle_checksum,
            "kernel_id": "jitheswar/ticket33-kaggle-free-t4-validation",
            "scope": {"external_spend_usd": 0},
        },
    )
    record_gpu_backend_allocation(
        ledger,
        request_checksum=kaggle_checksum,
        attempt_number=1,
        backend_run_id="jitheswar/ticket33-kaggle-free-t4-validation",
        allocated_start_epoch=300,
        allocated_end_epoch=1200,
        exit_code=0,
        evidence_path=evidence,
        evidence_sha256=evidence_sha,
    )
    with pytest.raises(GpuCampaignBudgetError):
        reserve_gpu_campaign_request(
            ledger,
            request_checksum="f" * 64,
            experiment_id="ticket33-kaggle-over-budget",
            wall_time_minutes=16,
            retry_allowed=False,
            remaining_attempts=1,
            gpu_count=2,
        )


def test_campaign_requires_exact_slurm_cancellation_evidence(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    initialize_gpu_campaign_ledger(ledger, reserved_comparison_hours=18)
    checksum = "c" * 64
    reserve_gpu_campaign_request(
        ledger,
        request_checksum=checksum,
        experiment_id="ticket33-slurm",
        wall_time_minutes=30,
        retry_allowed=True,
        remaining_attempts=2,
    )
    begin_gpu_submission_attempt(
        ledger,
        request_checksum=checksum,
        attempt_number=1,
        submitted_epoch=100,
    )
    record_gpu_submission_result(
        ledger,
        request_checksum=checksum,
        attempt_number=1,
        job_id="7190",
        exit_code=0,
        submission_wait=False,
    )
    evidence, evidence_sha = _json_file(
        tmp_path / "bad-evidence.json",
        {"schema_version": "not-a-ticket33-slurm-blocker"},
    )
    with pytest.raises(GpuCampaignLedgerError, match="exact cancellation"):
        mark_gpu_campaign_request_externally_blocked(
            ledger,
            request_checksum=checksum,
            slurm_job_id="7190",
            evidence_path=evidence,
            evidence_sha256=evidence_sha,
            conservative_charge_seconds=3600,
        )


def test_kaggle_actual_gpu_time_charges_both_assigned_t4_devices(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    initialize_gpu_campaign_ledger(ledger, reserved_comparison_hours=18)
    checksum = "d" * 64
    reserve_gpu_campaign_request(
        ledger,
        request_checksum=checksum,
        experiment_id="ticket33-kaggle-t4",
        wall_time_minutes=30,
        retry_allowed=False,
        remaining_attempts=1,
        gpu_count=2,
    )
    begin_gpu_submission_attempt(
        ledger,
        request_checksum=checksum,
        attempt_number=1,
        submitted_epoch=100,
    )
    evidence, evidence_sha = _json_file(
        tmp_path / "kaggle-run.json",
        {
            "schema_version": "rsna-ticket33-kaggle-gpu-evidence-v1",
            "status": "completed",
            "request_checksum": checksum,
            "kernel_id": "jitheswar/ticket33-kaggle-free-t4-validation",
            "scope": {"external_spend_usd": 0},
        },
    )
    entry = record_gpu_backend_allocation(
        ledger,
        request_checksum=checksum,
        attempt_number=1,
        backend_run_id="jitheswar/ticket33-kaggle-free-t4-validation",
        allocated_start_epoch=150,
        allocated_end_epoch=330,
        exit_code=0,
        evidence_path=evidence,
        evidence_sha256=evidence_sha,
    )
    assert entry["actual_allocated_seconds"] == 360
    assert entry["queue_wait_seconds"] == 50
    assert entry["status"] == "completed"
    assert gpu_campaign_capacity(ledger)["spent_seconds"] == 360


def test_kaggle_request_rejects_an_unapproved_payload(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"schema_version": "rsna-ticket33-kaggle-gpu-request-v1"}),
        encoding="utf-8",
    )
    with pytest.raises(KaggleGpuError, match="checksum"):
        load_kaggle_gpu_request(request)


def test_kaggle_source_attestation_is_limited_to_the_approved_competition() -> None:
    with pytest.raises(ValueError, match="not the approved RSNA competition"):
        KaggleCompetitionSource(
            Path("/kaggle/input/other-competition"),
            "other-competition",
            "jitheswar/ticket33-kaggle-free-t4-validation",
            "e" * 64,
        )
