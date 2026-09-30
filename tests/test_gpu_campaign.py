from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsna_knee.gpu import (
    GPU_APPROVAL_SCHEMA_VERSION,
    GPU_APPROVAL_SCOPE,
    GpuResourceRequest,
    TrainingConfig,
    prepare_gpu_experiment,
    submit_gpu_experiment,
)
from rsna_knee.gpu_cli import main as gpu_cli_main
from rsna_knee.image_variant import TICKET31_RESNET18_VARIANT, image_model_training_config
from rsna_knee.gpu_campaign import (
    GpuCampaignActiveRequestError,
    GpuCampaignBudgetError,
    GpuCampaignLedgerError,
    finish_gpu_campaign_request,
    read_gpu_campaign_ledger,
    begin_gpu_submission_attempt,
    initialize_gpu_campaign_ledger,
    record_gpu_submission_result,
    record_gpu_allocation,
    record_gpu_backend_allocation,
    reserve_gpu_campaign_request,
    reconcile_gpu_unallocated_attempt,
    reconcile_gpu_unallocated_from_scontrol,
)


@pytest.mark.parametrize("schema", ["rsna-ticket33-kaggle-gpu-evidence-v1", "rsna-kaggle-gpu-allocation-evidence-v1"])
def test_kaggle_inference_allocation_charges_both_assigned_gpus(tmp_path: Path, schema: str) -> None:
    ledger = tmp_path / "ledger.json"
    initialize_gpu_campaign_ledger(ledger, budget_hours=20, reserved_comparison_hours=0)
    digest = "d" * 64
    reserve_gpu_campaign_request(ledger, request_checksum=digest, experiment_id="public-inference-preflight", wall_time_minutes=10, retry_allowed=False, remaining_attempts=1, gpu_count=2)
    begin_gpu_submission_attempt(ledger, request_checksum=digest, attempt_number=1, submitted_epoch=100)
    kernel = "owner/private-inference"
    record_gpu_submission_result(ledger, request_checksum=digest, attempt_number=1, job_id=kernel, exit_code=0, submission_wait=False)
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"schema_version": schema, "status": "completed", "request_checksum": digest, "kernel_id": kernel, "scope": {"external_spend_usd": 0}}))
    result = record_gpu_backend_allocation(ledger, request_checksum=digest, attempt_number=1, backend_run_id=kernel, allocated_start_epoch=110, allocated_end_epoch=160, exit_code=0, evidence_path=evidence, evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
    assert result["actual_allocated_seconds"] == 100
    assert result["reserved_seconds"] == 0
from rsna_knee.training import PREFLIGHT_SCHEMA_VERSION
from rsna_knee.pretrained import (
    RESNET18_ARTIFACT_BYTES,
    RESNET18_ARTIFACT_ID,
    RESNET18_DOWNLOAD_URL,
    RESNET18_SHA256,
    RESNET18_SOURCE_REVISION,
    TICKET33_RESOURCE_SCOPE,
)


def _write_preflight(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": PREFLIGHT_SCHEMA_VERSION,
                "passed": True,
                "checks": {
                    "python_311": True,
                    "user_environment_python": True,
                    "cluster1_cpu_slurm": True,
                    "core_dependencies_import": True,
                    "known_roc_auc": True,
                    "finite_model_loss_and_gradients": True,
                    "checkpoint_save_reload": True,
                    "inference_import_isolated": True,
                },
                "environment": {"python_version": "3.11.16"},
                "details": {
                    "required_slurm": True,
                    "required_isolated_python": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _prepare_request(
    tmp_path: Path,
    *,
    experiment_id: str,
    ledger_path: Path,
    retry_allowed: bool = False,
    wall_time_minutes: int = 30,
    comparison_reserve_release_minutes: int = 0,
) -> tuple[object, Path]:
    preflight = _write_preflight(tmp_path / f"{experiment_id}-preflight.json")
    request = prepare_gpu_experiment(
        tmp_path / experiment_id,
        preflight_path=preflight,
        campaign_ledger_path=ledger_path,
        reserved_comparison_hours=18.0,
        resource=GpuResourceRequest(
            partition="cluster1-gpu-add" if wall_time_minutes <= 30 else "cluster1-gpu",
            wall_time_minutes=wall_time_minutes,
            smoke_test=wall_time_minutes <= 30,
        ),
        training_config=TrainingConfig(
            image_size=(16, 16),
            channels=(2, 4, 8),
            max_slices=1,
            max_series=1,
            successful_optimizer_steps=1,
            device="cuda",
        ),
        experiment_id=experiment_id,
        experiment_identity={"scope": "synthetic-boundary-test"},
        retry_allowed=retry_allowed,
        comparison_reserve_release_minutes=comparison_reserve_release_minutes,
    )
    approval = tmp_path / f"{experiment_id}-approval.json"
    approval.write_text(
        json.dumps(
            {
                "schema_version": GPU_APPROVAL_SCHEMA_VERSION,
                "request_checksum": request.request_checksum,
                "approval_scope": GPU_APPROVAL_SCOPE,
                "approved": True,
                "retry_allowed": retry_allowed,
                "approver": "synthetic-test-owner",
                "approved_at_utc": "2026-09-25T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return request, approval


def test_campaign_reserves_eighteen_hours_and_rejects_a_request_that_spends_it(
    tmp_path: Path,
) -> None:
    preflight = _write_preflight(tmp_path / "preflight.json")

    with pytest.raises(GpuCampaignBudgetError, match="comparison reserve"):
        prepare_gpu_experiment(
            tmp_path / "oversized-request",
            preflight_path=preflight,
            campaign_ledger_path=tmp_path / "campaign-ledger.json",
            reserved_comparison_hours=18.0,
            resource=GpuResourceRequest(
                partition="cluster1-gpu",
                wall_time_minutes=360,
            ),
            training_config=TrainingConfig(device="cuda"),
            experiment_id="ticket33-six-hour-request",
            experiment_identity={"scope": "synthetic-boundary-test"},
        )


def test_prepare_command_accepts_an_exact_resnet_training_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preflight = _write_preflight(tmp_path / "preflight.json")
    config_path = tmp_path / "training-config.json"
    config = replace(
        image_model_training_config(TICKET31_RESNET18_VARIANT),
        successful_optimizer_steps=1,
        device="cuda",
    )
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(
        json.dumps(
            {
                "pretrained_resource": {
                    "artifact_id": RESNET18_ARTIFACT_ID,
                    "source_revision": RESNET18_SOURCE_REVISION,
                    "download_url": RESNET18_DOWNLOAD_URL,
                    "sha256": RESNET18_SHA256,
                    "expected_bytes": RESNET18_ARTIFACT_BYTES,
                    "weights_path": str((tmp_path / "weights.pth").resolve()),
                    "resource_approval_path": str(
                        (tmp_path / "resource-approval.json").resolve()
                    ),
                    "scope": TICKET33_RESOURCE_SCOPE,
                    "encoder_frozen": True,
                    "runtime_network": False,
                    "external_spend": 0,
                }
            }
        ),
        encoding="utf-8",
    )

    assert gpu_cli_main(
        [
            "prepare",
            "--preflight",
            str(preflight),
            "--output",
            str(tmp_path / "cli-request"),
            "--training-config-json",
            str(config_path),
            "--execution-json",
            str(execution_path),
            "--campaign-ledger",
            str(tmp_path / "campaign-ledger.json"),
            "--reserved-comparison-hours",
            "18",
            "--partition",
            "cluster1-gpu-add",
            "--smoke-test",
            "--wall-time-minutes",
            "30",
            "--experiment-id",
            "ticket33-cli-smoke",
            "--fold",
            "0",
            "--source-version",
            "fixture-source-v1",
            "--source-fingerprint",
            "fixture-source-sha256",
            "--fold-manifest-checksum",
            "fixture-fold-sha256",
            "--integration-manifest-checksum",
            "fixture-integration-sha256",
        ]
    ) == 0

    request_payload = json.loads(capsys.readouterr().out)
    assert request_payload["experiment"]["training_config"]["model"]["model_id"] == (
        config.model_id
    )
    assert request_payload["campaign"]["remaining_hours_before_request"] == 20.0
    assert request_payload["campaign"]["remaining_hours_after_request_and_comparison_reserve"] == 1.0


def test_one_pending_gpu_request_blocks_a_second_approved_request(tmp_path: Path) -> None:
    ledger_path = tmp_path / "campaign-ledger.json"
    first, first_approval = _prepare_request(
        tmp_path,
        experiment_id="ticket33-first-smoke",
        ledger_path=ledger_path,
    )
    second, second_approval = _prepare_request(
        tmp_path,
        experiment_id="ticket33-second-smoke",
        ledger_path=ledger_path,
    )
    script = Path(__file__).parents[1] / "slurm" / "run_ticket13_gpu_experiment.sbatch"
    submissions: list[list[str]] = []

    def submit(command: list[str], **_: object) -> SimpleNamespace:
        submissions.append(command)
        return SimpleNamespace(returncode=0, stdout="12345\n", stderr="")

    first_result = submit_gpu_experiment(
        first.root,
        first_approval,
        script_path=script,
        wait=False,
        runner=submit,
        scheduler_query=lambda: (),
    )

    assert first_result.status == "submitted"
    with pytest.raises(GpuCampaignActiveRequestError, match="active or pending GPU request"):
        submit_gpu_experiment(
            second.root,
            second_approval,
            script_path=script,
            wait=False,
            runner=submit,
            scheduler_query=lambda: (),
        )
    assert len(submissions) == 1


def test_scheduler_pending_gpu_job_blocks_submission_before_sbatch(tmp_path: Path) -> None:
    ledger_path = tmp_path / "campaign-ledger.json"
    request, approval = _prepare_request(
        tmp_path,
        experiment_id="ticket33-queued-smoke",
        ledger_path=ledger_path,
    )
    script = Path(__file__).parents[1] / "slurm" / "run_ticket13_gpu_experiment.sbatch"
    submissions: list[list[str]] = []

    with pytest.raises(GpuCampaignActiveRequestError, match="job 9921, state=PENDING"):
        submit_gpu_experiment(
            request.root,
            approval,
            script_path=script,
            runner=lambda command, **_: submissions.append(command),
            scheduler_query=lambda: (
                {
                    "job_id": "9921",
                    "partition": "cluster1-gpu",
                    "state": "PENDING",
                },
            ),
        )

    assert submissions == []
    assert read_gpu_campaign_ledger(ledger_path)["requests"] == []


def test_campaign_ledger_separates_queue_wait_from_allocated_gpu_time(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "campaign-ledger.json"
    request, approval = _prepare_request(
        tmp_path,
        experiment_id="ticket33-timed-smoke",
        ledger_path=ledger_path,
    )
    script = Path(__file__).parents[1] / "slurm" / "run_ticket13_gpu_experiment.sbatch"

    submit_gpu_experiment(
        request.root,
        approval,
        script_path=script,
        wait=False,
        runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="23456\n",
            stderr="",
        ),
        scheduler_query=lambda: (),
    )
    submitted_epoch = read_gpu_campaign_ledger(ledger_path)["requests"][0]["attempts"][0][
        "submitted_epoch"
    ]
    record_gpu_allocation(
        ledger_path,
        request_checksum=request.request_checksum,
        attempt_number=1,
        job_id="23456",
        allocated_start_epoch=submitted_epoch + 600,
        allocated_end_epoch=submitted_epoch + 900,
        exit_code=0,
        submission_wait=False,
    )

    ledger = read_gpu_campaign_ledger(ledger_path)
    entry = ledger["requests"][0]
    attempt = entry["attempts"][0]
    assert attempt["queue_wait_seconds"] == 600
    assert attempt["allocated_seconds"] == 300
    assert entry["actual_allocated_seconds"] == 300
    assert entry["status"] == "completed"


def test_comparison_reserve_moves_forward_one_fold_at_a_time(tmp_path: Path) -> None:
    ledger_path = tmp_path / "campaign-ledger.json"
    first, first_approval = _prepare_request(
        tmp_path,
        experiment_id="ticket33-comparison-fold-0",
        ledger_path=ledger_path,
        wall_time_minutes=360,
        comparison_reserve_release_minutes=360,
    )
    assert first.payload["campaign"]["reserved_comparison_seconds"] == 18 * 3600
    assert first.payload["campaign"]["comparison_reserve_after_request_seconds"] == 12 * 3600
    assert first.payload["campaign"]["remaining_hours_after_request_and_comparison_reserve"] == 2.0

    script = Path(__file__).parents[1] / "slurm" / "run_ticket13_gpu_experiment.sbatch"
    submit_gpu_experiment(
        first.root,
        first_approval,
        script_path=script,
        wait=False,
        runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="34567\n",
            stderr="",
        ),
        scheduler_query=lambda: (),
    )
    submitted_epoch = read_gpu_campaign_ledger(ledger_path)["requests"][0]["attempts"][0][
        "submitted_epoch"
    ]
    record_gpu_allocation(
        ledger_path,
        request_checksum=first.request_checksum,
        attempt_number=1,
        job_id="34567",
        allocated_start_epoch=submitted_epoch,
        allocated_end_epoch=submitted_epoch + 3600,
        exit_code=0,
        submission_wait=False,
    )

    ledger = read_gpu_campaign_ledger(ledger_path)
    assert ledger["reserved_comparison_seconds"] == 12 * 3600

    second, _ = _prepare_request(
        tmp_path,
        experiment_id="ticket33-comparison-fold-1",
        ledger_path=ledger_path,
        wall_time_minutes=360,
        comparison_reserve_release_minutes=360,
    )
    assert second.payload["campaign"]["comparison_reserve_after_request_seconds"] == 6 * 3600
    assert second.payload["campaign"]["remaining_hours_after_request_and_comparison_reserve"] == 7.0


def test_unallocated_comparison_request_restores_its_reserve(tmp_path: Path) -> None:
    ledger_path = tmp_path / "campaign-ledger.json"
    initialize_gpu_campaign_ledger(
        ledger_path,
        reserved_comparison_hours=18.0,
    )
    reserve_gpu_campaign_request(
        ledger_path,
        request_checksum="a" * 64,
        experiment_id="comparison-fold-no-allocation",
        wall_time_minutes=360,
        retry_allowed=False,
        remaining_attempts=1,
        comparison_reserve_release_seconds=6 * 3600,
    )
    begin_gpu_submission_attempt(
        ledger_path,
        request_checksum="a" * 64,
        attempt_number=1,
        submitted_epoch=100,
    )
    record_gpu_submission_result(
        ledger_path,
        request_checksum="a" * 64,
        attempt_number=1,
        job_id=None,
        exit_code=1,
        submission_wait=False,
    )
    finish_gpu_campaign_request(
        ledger_path,
        request_checksum="a" * 64,
        status="failed",
    )

    assert read_gpu_campaign_ledger(ledger_path)["reserved_comparison_seconds"] == 18 * 3600


def test_scontrol_reconciliation_closes_cancelled_pending_job_when_sacct_is_down(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "campaign-ledger.json"
    request_checksum = "b" * 64
    initialize_gpu_campaign_ledger(ledger_path, reserved_comparison_hours=18.0)
    reserve_gpu_campaign_request(
        ledger_path,
        request_checksum=request_checksum,
        experiment_id="comparison-fold-pending-cancel-scontrol",
        wall_time_minutes=30,
        retry_allowed=False,
        remaining_attempts=1,
        comparison_reserve_release_seconds=30 * 60,
    )
    begin_gpu_submission_attempt(
        ledger_path,
        request_checksum=request_checksum,
        attempt_number=1,
        submitted_epoch=100,
    )
    record_gpu_submission_result(
        ledger_path,
        request_checksum=request_checksum,
        attempt_number=1,
        job_id="54320",
        exit_code=0,
        submission_wait=False,
    )
    controller_path = tmp_path / "scontrol.txt"
    controller_text = (
        "JobId=54320 JobName=test\n"
        "   JobState=CANCELLED Reason=cancelled\n"
        "   RunTime=00:00:00 TimeLimit=00:30:00\n"
        "   SubmitTime=1970-01-01T00:01:40Z EndTime=1970-01-01T00:03:40Z\n"
        "   NodeList=(null)\n"
    )
    controller_path.write_text(controller_text, encoding="utf-8")
    evidence_path = tmp_path / "scontrol.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": "rsna-slurm-scontrol-unallocated-v1",
                "request_checksum": request_checksum,
                "job_id": "54320",
                "job_state": "CANCELLED",
                "elapsed_seconds": 0,
                "node_list": None,
                "held_pending_before_cancel": True,
                "submitted_epoch": 100,
                "ended_epoch": 220,
                "controller_submit_time": "1970-01-01T00:01:40Z",
                "controller_end_time": "1970-01-01T00:03:40Z",
                "controller_record_path": str(controller_path),
                "controller_record_sha256": hashlib.sha256(
                    controller_path.read_bytes()
                ).hexdigest(),
                "cancel_command_exit_code": 0,
                "absent_from_queue_after_cancel": True,
                "sacct_status": "unavailable_connection_refused",
            }
        ),
        encoding="utf-8",
    )

    entry = reconcile_gpu_unallocated_from_scontrol(
        ledger_path,
        request_checksum=request_checksum,
        attempt_number=1,
        job_id="54320",
        scheduler_evidence_path=evidence_path,
    )

    ledger = read_gpu_campaign_ledger(ledger_path)
    assert entry["status"] == "failed"
    assert entry["actual_allocated_seconds"] == 0
    assert entry["queue_wait_seconds"] == 120
    assert entry["attempts"][0]["scheduler_evidence_source"] == "scontrol"
    assert ledger["reserved_comparison_seconds"] == 18 * 3600


def test_scontrol_reconciliation_closes_externally_terminalized_pending_job(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "campaign-ledger.json"
    request_checksum = "f" * 64
    initialize_gpu_campaign_ledger(ledger_path, reserved_comparison_hours=18.0)
    reserve_gpu_campaign_request(
        ledger_path,
        request_checksum=request_checksum,
        experiment_id="comparison-fold-external-terminal-scontrol",
        wall_time_minutes=30,
        retry_allowed=False,
        remaining_attempts=1,
        comparison_reserve_release_seconds=30 * 60,
    )
    begin_gpu_submission_attempt(
        ledger_path,
        request_checksum=request_checksum,
        attempt_number=1,
        submitted_epoch=100,
    )
    record_gpu_submission_result(
        ledger_path,
        request_checksum=request_checksum,
        attempt_number=1,
        job_id="54321",
        exit_code=0,
        submission_wait=False,
    )

    prior_controller = tmp_path / "prior-pending-scontrol.txt"
    prior_controller.write_text(
        "JobId=54321 JobState=PENDING Reason=Resources RunTime=00:00:00 "
        "SubmitTime=1970-01-01T00:01:40Z NodeList=(null)\n",
        encoding="utf-8",
    )
    prior_observation = tmp_path / "prior-pending-observation.json"
    prior_observation.write_text(
        json.dumps(
            {
                "schema_version": "rsna-slurm-scontrol-pending-observation-v1",
                "request_checksum": request_checksum,
                "job_id": "54321",
                "state": "PENDING",
                "elapsed_seconds": 0,
                "node_list": None,
                "observed_at_epoch": 150,
                "controller_record_path": str(prior_controller),
                "controller_record_sha256": hashlib.sha256(
                    prior_controller.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    controller_path = tmp_path / "terminal-scontrol.txt"
    controller_path.write_text(
        "JobId=54321 JobState=CANCELLED Reason=Resources RunTime=00:00:00 "
        "SubmitTime=1970-01-01T00:01:40Z EndTime=1970-01-01T00:03:40Z "
        "NodeList=(null)\n",
        encoding="utf-8",
    )
    evidence_path = tmp_path / "external-terminal-evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": "rsna-slurm-scontrol-unallocated-external-v1",
                "request_checksum": request_checksum,
                "job_id": "54321",
                "job_state": "CANCELLED",
                "elapsed_seconds": 0,
                "node_list": None,
                "submitted_epoch": 100,
                "ended_epoch": 220,
                "controller_submit_time": "1970-01-01T00:01:40Z",
                "controller_end_time": "1970-01-01T00:03:40Z",
                "controller_record_path": str(controller_path),
                "controller_record_sha256": hashlib.sha256(
                    controller_path.read_bytes()
                ).hexdigest(),
                "prior_pending_observation_path": str(prior_observation),
                "prior_pending_observation_sha256": hashlib.sha256(
                    prior_observation.read_bytes()
                ).hexdigest(),
                "terminalized_without_agent_cancel": True,
                "absent_from_queue_after_terminal": True,
                "sacct_status": "unavailable_connection_refused",
            }
        ),
        encoding="utf-8",
    )

    entry = reconcile_gpu_unallocated_from_scontrol(
        ledger_path,
        request_checksum=request_checksum,
        attempt_number=1,
        job_id="54321",
        scheduler_evidence_path=evidence_path,
    )

    assert entry["status"] == "failed"
    assert entry["actual_allocated_seconds"] == 0
    assert entry["attempts"][0]["scheduler_terminal_state"] == "CANCELLED"
    assert read_gpu_campaign_ledger(ledger_path)["reserved_comparison_seconds"] == 18 * 3600


def test_scontrol_reconciliation_rejects_started_or_unrelated_gpu_job(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "campaign-ledger.json"
    request_checksum = "e" * 64
    initialize_gpu_campaign_ledger(ledger_path, reserved_comparison_hours=18.0)
    reserve_gpu_campaign_request(
        ledger_path,
        request_checksum=request_checksum,
        experiment_id="comparison-fold-invalid-scontrol",
        wall_time_minutes=30,
        retry_allowed=False,
        remaining_attempts=1,
    )
    begin_gpu_submission_attempt(
        ledger_path,
        request_checksum=request_checksum,
        attempt_number=1,
        submitted_epoch=100,
    )
    record_gpu_submission_result(
        ledger_path,
        request_checksum=request_checksum,
        attempt_number=1,
        job_id="54319",
        exit_code=0,
        submission_wait=False,
    )
    evidence_path = tmp_path / "scontrol-invalid.json"
    evidence = {
        "schema_version": "rsna-slurm-scontrol-unallocated-v1",
        "request_checksum": request_checksum,
        "job_id": "54319",
        "job_state": "CANCELLED",
        "elapsed_seconds": 1,
        "node_list": "gpu-node-1",
        "held_pending_before_cancel": True,
        "submitted_epoch": 100,
        "ended_epoch": 220,
        "controller_submit_time": "1970-01-01T00:01:40Z",
        "controller_end_time": "1970-01-01T00:03:40Z",
        "controller_record_path": str(tmp_path / "missing-scontrol.txt"),
        "controller_record_sha256": "0" * 64,
        "cancel_command_exit_code": 0,
        "absent_from_queue_after_cancel": True,
        "sacct_status": "unavailable_connection_refused",
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(GpuCampaignLedgerError, match="does not prove"):
        reconcile_gpu_unallocated_from_scontrol(
            ledger_path,
            request_checksum=request_checksum,
            attempt_number=1,
            job_id="54319",
            scheduler_evidence_path=evidence_path,
        )


def test_transient_no_job_retry_keeps_and_then_restores_one_fold_reserve(tmp_path: Path) -> None:
    ledger_path = tmp_path / "campaign-ledger.json"
    request, approval = _prepare_request(
        tmp_path,
        experiment_id="ticket33-retry-before-allocation",
        ledger_path=ledger_path,
        retry_allowed=True,
        wall_time_minutes=30,
        comparison_reserve_release_minutes=30,
    )
    calls = 0

    def submit(_command: list[str], **_: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(returncode=1, stdout="", stderr="node failure")
        assert read_gpu_campaign_ledger(ledger_path)["reserved_comparison_seconds"] == (
            18 * 3600 - 30 * 60
        )
        return SimpleNamespace(returncode=1, stdout="", stderr="CANCELLED by user")

    script = Path(__file__).parents[1] / "slurm" / "run_ticket13_gpu_experiment.sbatch"
    result = submit_gpu_experiment(
        request.root,
        approval,
        script_path=script,
        runner=submit,
        scheduler_query=lambda: (),
    )

    assert result.status == "failed"
    assert result.retry_count == 1
    assert calls == 2
    assert read_gpu_campaign_ledger(ledger_path)["reserved_comparison_seconds"] == 18 * 3600


def test_pending_slurm_cancel_can_be_reconciled_without_gpu_charge(tmp_path: Path) -> None:
    ledger_path = tmp_path / "campaign-ledger.json"
    initialize_gpu_campaign_ledger(ledger_path, reserved_comparison_hours=18.0)
    request_checksum = "b" * 64
    reserve_gpu_campaign_request(
        ledger_path,
        request_checksum=request_checksum,
        experiment_id="comparison-fold-pending-cancel",
        wall_time_minutes=30,
        retry_allowed=False,
        remaining_attempts=1,
        comparison_reserve_release_seconds=30 * 60,
    )
    begin_gpu_submission_attempt(
        ledger_path,
        request_checksum=request_checksum,
        attempt_number=1,
        submitted_epoch=100,
    )
    record_gpu_submission_result(
        ledger_path,
        request_checksum=request_checksum,
        attempt_number=1,
        job_id="54321",
        exit_code=0,
        submission_wait=False,
    )
    scheduler_evidence = tmp_path / "sacct.txt"
    scheduler_evidence.write_text(
        "54321|CANCELLED by 1000|100|Unknown|220|0\n",
        encoding="utf-8",
    )

    assert gpu_cli_main(
        [
            "reconcile-unallocated",
            "--ledger",
            str(ledger_path),
            "--request-checksum",
            request_checksum,
            "--attempt-number",
            "1",
            "--job-id",
            "54321",
            "--scheduler-evidence",
            str(scheduler_evidence),
        ]
    ) == 0

    entry = read_gpu_campaign_ledger(ledger_path)["requests"][0]
    assert entry["actual_allocated_seconds"] == 0
    assert entry["queue_wait_seconds"] == 120
    assert entry["status"] == "failed"
    assert entry["attempts"][0]["scheduler_evidence_sha256"]
    assert read_gpu_campaign_ledger(ledger_path)["reserved_comparison_seconds"] == 18 * 3600


def test_unallocated_reconciliation_rejects_unrelated_or_started_job_evidence(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "campaign-ledger.json"
    initialize_gpu_campaign_ledger(ledger_path, reserved_comparison_hours=18.0)
    request_checksum = "c" * 64
    reserve_gpu_campaign_request(
        ledger_path,
        request_checksum=request_checksum,
        experiment_id="comparison-fold-invalid-evidence",
        wall_time_minutes=30,
        retry_allowed=False,
        remaining_attempts=1,
        comparison_reserve_release_seconds=30 * 60,
    )
    begin_gpu_submission_attempt(
        ledger_path,
        request_checksum=request_checksum,
        attempt_number=1,
        submitted_epoch=100,
    )
    record_gpu_submission_result(
        ledger_path,
        request_checksum=request_checksum,
        attempt_number=1,
        job_id="54322",
        exit_code=0,
        submission_wait=False,
    )
    evidence = tmp_path / "sacct-invalid.txt"
    evidence.write_text("99999|CANCELLED|100|Unknown|220|0\n", encoding="utf-8")
    with pytest.raises(GpuCampaignLedgerError, match="exact job ID"):
        reconcile_gpu_unallocated_attempt(
            ledger_path,
            request_checksum=request_checksum,
            attempt_number=1,
            job_id="54322",
            scheduler_evidence_path=evidence,
        )

    evidence.write_text("54322|CANCELLED|100|110|220|110\n", encoding="utf-8")
    with pytest.raises(GpuCampaignLedgerError, match="shows that the job started"):
        reconcile_gpu_unallocated_attempt(
            ledger_path,
            request_checksum=request_checksum,
            attempt_number=1,
            job_id="54322",
            scheduler_evidence_path=evidence,
        )


def test_reconciling_missing_timing_preserves_successful_retry_outcome(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "campaign-ledger.json"
    initialize_gpu_campaign_ledger(ledger_path, reserved_comparison_hours=18.0)
    request_checksum = "d" * 64
    reserve_gpu_campaign_request(
        ledger_path,
        request_checksum=request_checksum,
        experiment_id="comparison-fold-successful-retry",
        wall_time_minutes=30,
        retry_allowed=True,
        remaining_attempts=2,
        comparison_reserve_release_seconds=30 * 60,
    )
    begin_gpu_submission_attempt(
        ledger_path,
        request_checksum=request_checksum,
        attempt_number=1,
        submitted_epoch=100,
    )
    record_gpu_submission_result(
        ledger_path,
        request_checksum=request_checksum,
        attempt_number=1,
        job_id="54331",
        exit_code=1,
        submission_wait=True,
    )
    begin_gpu_submission_attempt(
        ledger_path,
        request_checksum=request_checksum,
        attempt_number=2,
        submitted_epoch=200,
    )
    record_gpu_allocation(
        ledger_path,
        request_checksum=request_checksum,
        attempt_number=2,
        job_id="54332",
        allocated_start_epoch=200,
        allocated_end_epoch=500,
        exit_code=0,
        submission_wait=True,
    )
    finish_gpu_campaign_request(
        ledger_path,
        request_checksum=request_checksum,
        status="completed",
    )
    evidence = tmp_path / "sacct-no-start.txt"
    evidence.write_text("54331|CANCELLED|100|Unknown|150|0\n", encoding="utf-8")

    entry = reconcile_gpu_unallocated_attempt(
        ledger_path,
        request_checksum=request_checksum,
        attempt_number=1,
        job_id="54331",
        scheduler_evidence_path=evidence,
    )

    assert entry["status"] == "completed"
    assert entry["actual_allocated_seconds"] == 300
    assert entry["queue_wait_seconds"] == 50
    assert read_gpu_campaign_ledger(ledger_path)["reserved_comparison_seconds"] == (
        18 * 3600 - 30 * 60
    )
