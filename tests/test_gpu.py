from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rsna_knee.gpu import (
    GPU_APPROVAL_SCHEMA_VERSION,
    GPU_APPROVAL_SCOPE,
    GPU_EXPERIMENT_SCHEMA_VERSION,
    GpuApprovalError,
    GpuExecutionError,
    GpuResourceError,
    GpuResourceRequest,
    run_approved_gpu_experiment,
    run_gpu_smoke_test,
    classify_gpu_failure,
    collect_gpu_environment,
    load_gpu_approval,
    load_gpu_experiment_request,
    prepare_gpu_experiment,
    submit_gpu_experiment,
    validate_gpu_environment,
    verify_slurm_gpu_allocation,
)
from rsna_knee.training import (
    PREFLIGHT_SCHEMA_VERSION,
    TrainingConfig,
    TrainingConfigurationError,
    TrainingError,
    TrainingResult,
    save_training_checkpoint,
    train_fixed_budget,
)


def _write_preflight(path: Path, *, passed: bool = True) -> Path:
    checks = {
        "python_311": passed,
        "user_environment_python": passed,
        "cluster1_cpu_slurm": passed,
        "core_dependencies_import": passed,
        "known_roc_auc": passed,
        "finite_model_loss_and_gradients": passed,
        "checkpoint_save_reload": passed,
        "inference_import_isolated": passed,
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": PREFLIGHT_SCHEMA_VERSION,
                "passed": passed,
                "checks": checks,
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


def _config() -> TrainingConfig:
    return TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=1,
        max_series=1,
        successful_optimizer_steps=1,
        device="cuda",
    )


def _slurm_environment(resource: GpuResourceRequest) -> dict[str, str]:
    hours, minutes = divmod(resource.wall_time_minutes, 60)
    return {
        "SLURM_JOB_ID": "1234",
        "SLURM_JOB_PARTITION": resource.partition,
        "SLURM_JOB_NUM_NODES": str(resource.nodes),
        "SLURM_NTASKS": str(resource.ntasks),
        "SLURM_CPUS_PER_TASK": str(resource.cpus_per_task),
        "SLURM_MEM_PER_NODE": f"{resource.memory_gb}G",
        "SLURM_TIMELIMIT": f"{hours:02d}:{minutes:02d}:00",
        "SLURM_JOB_GRES": resource.gres,
        "SLURM_GPUS_ON_NODE": str(resource.gpu_count),
        "CUDA_VISIBLE_DEVICES": "0",
        "SLURM_JOB_SHARED": "shared",
    }


def test_gpu_resource_request_is_one_explicit_noninteractive_gpu() -> None:
    request = GpuResourceRequest(
        partition="cluster1-gpu-add",
        gres="gpu:1",
        wall_time_minutes=30,
        smoke_test=True,
    )

    assert request.gpu_count == 1
    assert request.to_dict()["exclusive"] is False

    with pytest.raises(ValueError, match="one GPU"):
        GpuResourceRequest(gres="gpu:2")
    with pytest.raises(ValueError, match="at most 60 minutes"):
        GpuResourceRequest(smoke_test=True)
    with pytest.raises(ValueError, match="reserved for an approved"):
        GpuResourceRequest(partition="cluster1-gpu-add")


def test_prepare_gpu_experiment_binds_passed_cpu_preflight_and_exact_config(tmp_path: Path) -> None:
    preflight = _write_preflight(tmp_path / "cpu-preflight.json")
    request = prepare_gpu_experiment(
        tmp_path / "request",
        preflight_path=preflight,
        resource=GpuResourceRequest(
            partition="cluster1-gpu-add",
            gres="gpu:1",
            wall_time_minutes=30,
            smoke_test=True,
        ),
        training_config=_config(),
        experiment_id="gpu-smoke-001",
        experiment_identity={"source": "synthetic-fixture", "fold": 0},
    )

    assert request.payload["schema_version"] == GPU_EXPERIMENT_SCHEMA_VERSION
    assert request.payload["status"] == "owner_approval_required"
    assert request.payload["cpu_preflight"]["sha256"]
    assert request.payload["experiment"]["training_config"]["device"] == "cuda"
    assert request.payload["evidence"]["required_fields"]
    assert load_gpu_experiment_request(request.root).request_checksum == request.request_checksum

    approval_path = tmp_path / "owner-approval.json"
    approval_path.write_text(
        json.dumps(
            {
                "schema_version": GPU_APPROVAL_SCHEMA_VERSION,
                "request_checksum": request.request_checksum,
                "approval_scope": GPU_APPROVAL_SCOPE,
                "approved": True,
                "retry_allowed": True,
                "approver": "owner",
                "approved_at_utc": "2026-09-12T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    approval = load_gpu_approval(approval_path, request)
    assert approval.payload["approved"] is True

    approval_path.write_text(
        json.dumps(
            {
                "schema_version": GPU_APPROVAL_SCHEMA_VERSION,
                "request_checksum": "wrong-request",
                "approval_scope": GPU_APPROVAL_SCOPE,
                "approved": True,
                "retry_allowed": True,
                "approver": "owner",
                "approved_at_utc": "2026-09-12T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(GpuApprovalError, match="request"):
        load_gpu_approval(approval_path, request)

    preflight.write_text(preflight.read_text(encoding="utf-8").replace("3.11.16", "3.11.17"), encoding="utf-8")
    with pytest.raises(GpuApprovalError, match="preflight"):
        load_gpu_experiment_request(request.root)


def test_prepare_gpu_experiment_rejects_unpassed_cpu_preflight(tmp_path: Path) -> None:
    preflight = _write_preflight(tmp_path / "cpu-preflight.json", passed=False)

    with pytest.raises(GpuApprovalError, match="CPU preflight"):
        prepare_gpu_experiment(
            tmp_path / "request",
            preflight_path=preflight,
            resource=GpuResourceRequest(smoke_test=True, wall_time_minutes=30),
            training_config=_config(),
        )


def test_slurm_gpu_allocation_must_match_the_approved_single_gpu_request() -> None:
    resource = GpuResourceRequest(
        partition="cluster1-gpu-add",
        gres="gpu:1",
        wall_time_minutes=30,
        smoke_test=True,
    )
    allocation = verify_slurm_gpu_allocation(
        resource,
        _slurm_environment(resource),
    )

    assert allocation["job_id"] == "1234"
    assert allocation["partition"] == "cluster1-gpu-add"
    assert allocation["visible_gpu_count"] == 1
    assert allocation["sharing_behavior"] == "shared"
    assert allocation["nominal_memory_assumed"] is False

    with pytest.raises(GpuResourceError, match="partition"):
        verify_slurm_gpu_allocation(
            resource,
            {**_slurm_environment(resource), "SLURM_JOB_PARTITION": "cluster1-gpu"},
        )


def test_gpu_environment_records_assigned_memory_and_rejects_missing_hardware_facts() -> None:
    resource = GpuResourceRequest(smoke_test=True, wall_time_minutes=30)
    allocation = verify_slurm_gpu_allocation(
        resource,
        _slurm_environment(resource),
    )
    snapshot = collect_gpu_environment(
        resource=resource,
        slurm_environment=_slurm_environment(resource),
        probe=lambda: {
            "device_name": "NVIDIA A40",
            "driver_version": "550.54.15",
            "cuda_runtime_version": "12.6",
            "visible_device_count": 1,
            "device_total_memory_bytes": 45_000_000_000,
            "available_memory_bytes": 20_000_000_000,
            "runtime_visible_memory_limit_bytes": 20_000_000_000,
            "available_memory_source": "synthetic_allocation_fixture",
            "assigned_memory_bytes": 20_000_000_000,
            "assigned_memory_source": "synthetic_allocation_fixture",
            "allocated_vram_bytes": 1_000_000,
            "reserved_vram_bytes": 2_000_000,
            "peak_vram_bytes": 3_000_000,
            "peak_reserved_vram_bytes": 4_000_000,
        },
    )

    validated = validate_gpu_environment(snapshot, resource=resource, allocation=allocation)
    assert validated["device_name"] == "NVIDIA A40"
    assert validated["available_memory_bytes"] == 20_000_000_000
    assert validated["assigned_memory_bytes"] == 20_000_000_000
    assert validated["memory_assignment_verified"] is True
    assert validated["peak_vram_within_assigned_memory"] is True
    assert validated["nominal_device_memory_used"] is False

    unassigned = validate_gpu_environment(
        {key: value for key, value in snapshot.items() if key not in {
            "assigned_memory_bytes",
            "assigned_memory_source",
        }},
        resource=resource,
        allocation=allocation,
    )
    assert unassigned["assigned_memory_bytes"] is None
    assert unassigned["memory_assignment_verified"] is False

    with pytest.raises(GpuResourceError, match="driver"):
        validate_gpu_environment(
            {**snapshot, "driver_version": None}, resource=resource, allocation=allocation
        )
    with pytest.raises(GpuResourceError, match="observed allocation"):
        validate_gpu_environment(
            {**snapshot, "available_memory_source": "nominal"},
            resource=resource,
            allocation=allocation,
        )


def test_approved_gpu_execution_stops_without_observed_memory_assignment(tmp_path: Path) -> None:
    request, approval = _approved_request(tmp_path)
    calls = 0

    def runner(resume_from: Path | None) -> TrainingResult:
        nonlocal calls
        calls += 1
        raise AssertionError("training must not start without assigned-memory evidence")

    snapshot = _gpu_snapshot()
    snapshot.pop("assigned_memory_bytes")
    snapshot.pop("assigned_memory_source")
    with pytest.raises(GpuResourceError, match="assigned to this job"):
        run_approved_gpu_experiment(
            request.root,
            approval,
            output_dir=tmp_path / "gpu-run",
            slurm_environment=_slurm_environment(request.resources),
            gpu_probe=lambda: snapshot,
            training_runner=runner,
        )
    assert calls == 0


def _approved_request(tmp_path: Path):
    preflight = _write_preflight(tmp_path / "cpu-preflight.json")
    request = prepare_gpu_experiment(
        tmp_path / "request",
        preflight_path=preflight,
        resource=GpuResourceRequest(smoke_test=True, wall_time_minutes=30),
        training_config=_config(),
    )
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "schema_version": GPU_APPROVAL_SCHEMA_VERSION,
                "request_checksum": request.request_checksum,
                "approval_scope": GPU_APPROVAL_SCOPE,
                "approved": True,
                "retry_allowed": True,
                "approver": "owner",
                "approved_at_utc": "2026-09-12T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return request, approval


def test_gpu_retry_classification_excludes_user_cancel_and_oom() -> None:
    assert classify_gpu_failure("node failure while waiting for filesystem").retryable is True
    assert classify_gpu_failure("CUDA out of memory").retryable is False
    assert classify_gpu_failure("CANCELLED by user").retryable is False


def test_direct_cuda_training_requires_the_approval_capability() -> None:
    with pytest.raises(TrainingConfigurationError, match="only through the approval-gated"):
        train_fixed_budget((), _config())


def test_submit_gpu_experiment_uses_sbatch_and_retries_one_transient_failure(
    tmp_path: Path,
) -> None:
    request, approval = _approved_request(tmp_path)
    script = tmp_path / "run.sbatch"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    calls: list[tuple[list[str], dict[str, str]]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, dict(kwargs["env"])))
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="node failure")
        return subprocess.CompletedProcess(command, 0, stdout="5678\n", stderr="")

    result = submit_gpu_experiment(
        request.root,
        approval,
        script_path=script,
        environment={"RSNA_TICKET13_OUTPUT": str(tmp_path / "run")},
        runner=runner,
    )

    assert result.status == "completed"
    assert result.retry_count == 1
    assert result.job_ids == ("5678",)
    assert len(calls) == 2
    assert calls[0][0] == calls[1][0]
    assert "--wait" in calls[0][0]
    assert "--partition=cluster1-gpu" in calls[0][0]
    assert "--gres=gpu:1" in calls[0][0]
    assert "srun" not in calls[0][0]
    assert calls[1][1]["RSNA_TICKET13_PREVIOUS_RETRIES"] == "1"


def test_submit_gpu_experiment_durably_consumes_the_approval(tmp_path: Path) -> None:
    request, approval = _approved_request(tmp_path)
    script = tmp_path / "run.sbatch"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    calls = 0

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 0, stdout="5678\n", stderr="")

    environment = {"RSNA_TICKET13_OUTPUT": str(tmp_path / "run")}
    first = submit_gpu_experiment(
        request.root,
        approval,
        script_path=script,
        environment=environment,
        runner=runner,
    )
    assert first.status == "completed"
    with pytest.raises(GpuApprovalError, match="already been consumed"):
        submit_gpu_experiment(
            request.root,
            approval,
            script_path=script,
            environment=environment,
            runner=runner,
        )
    assert calls == 1


def test_submit_gpu_experiment_accepts_standard_sbatch_job_output(tmp_path: Path) -> None:
    request, approval = _approved_request(tmp_path)
    script = tmp_path / "run.sbatch"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    result = submit_gpu_experiment(
        request.root,
        approval,
        script_path=script,
        runner=lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="Submitted batch job 5678\n",
            stderr="",
        ),
    )

    assert result.status == "completed"
    assert result.job_ids == ("5678",)


def test_submit_gpu_experiment_rejects_success_without_a_job_id(tmp_path: Path) -> None:
    request, approval = _approved_request(tmp_path)
    script = tmp_path / "run.sbatch"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    result = submit_gpu_experiment(
        request.root,
        approval,
        script_path=script,
        runner=lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="sbatch completed\n",
            stderr="",
        ),
    )

    assert result.status == "failed"
    assert result.job_ids == ()
    assert result.attempts[0]["returncode"] == 1
    assert result.last_failure["category"] == "unknown"


def test_submit_gpu_experiment_stops_on_non_transient_failure(tmp_path: Path) -> None:
    request, approval = _approved_request(tmp_path)
    script = tmp_path / "run.sbatch"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="CUDA out of memory")

    result = submit_gpu_experiment(request.root, approval, script_path=script, runner=runner)

    assert result.status == "failed"
    assert result.retry_count == 0
    assert result.last_failure["retryable"] is False


def test_submit_gpu_experiment_honors_durable_in_job_failure_before_retry(tmp_path: Path) -> None:
    request, approval = _approved_request(tmp_path)
    script = tmp_path / "run.sbatch"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "gpu-failure.json").write_text(
        json.dumps(
            {
                "schema_version": "rsna-knee-gpu-execution-v1",
                "request_checksum": request.request_checksum,
                "retry_count": 1,
                "retry": {
                    "retryable": True,
                    "category": "transient_infrastructure",
                    "reason": "node failure inside the approved job",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(GpuApprovalError, match="new approved request"):
        submit_gpu_experiment(
            request.root,
            approval,
            script_path=script,
            environment={"RSNA_TICKET13_OUTPUT": str(output_dir)},
            runner=lambda command, **kwargs: subprocess.CompletedProcess(
                command, 1, stdout="", stderr="job failed"
            ),
        )


def test_submit_gpu_experiment_does_not_trust_stale_failure_metadata(tmp_path: Path) -> None:
    request, approval = _approved_request(tmp_path)
    script = tmp_path / "run.sbatch"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "gpu-failure.json").write_text(
        json.dumps(
            {
                "schema_version": "rsna-knee-gpu-execution-v1",
                "request_checksum": "stale-request",
                "retry_count": 0,
                "retry": {
                    "retryable": True,
                    "category": "transient_infrastructure",
                    "reason": "stale node failure",
                },
            }
        ),
        encoding="utf-8",
    )
    calls = 0

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="node failure")

    result = submit_gpu_experiment(
        request.root,
        approval,
        script_path=script,
        environment={"RSNA_TICKET13_OUTPUT": str(output_dir)},
        runner=runner,
    )

    assert result.status == "failed"
    assert result.retry_count == 1
    assert result.last_failure["retryable"] is False
    assert calls == 1


def _gpu_snapshot() -> dict[str, object]:
    return {
        "device_name": "NVIDIA A40",
        "driver_version": "550.54.15",
        "cuda_runtime_version": "12.6",
        "visible_device_count": 1,
        "device_total_memory_bytes": 45_000_000_000,
        "available_memory_bytes": 20_000_000_000,
        "runtime_visible_memory_limit_bytes": 20_000_000_000,
        "available_memory_source": "synthetic_allocation_fixture",
        "assigned_memory_bytes": 20_000_000_000,
        "assigned_memory_source": "synthetic_allocation_fixture",
        "allocated_vram_bytes": 1_000_000,
        "reserved_vram_bytes": 2_000_000,
        "peak_vram_bytes": 3_000_000,
        "peak_reserved_vram_bytes": 4_000_000,
    }


def _training_result_with_reloadable_checkpoint(tmp_path: Path, config: TrainingConfig) -> TrainingResult:
    from rsna_knee.model import HierarchicalStudyModel

    model = HierarchicalStudyModel(config.model_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    checkpoint = tmp_path / "final.pt"
    save_training_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        config,
        successful_optimizer_steps=1,
        attempted_batches=1,
        skipped_batches=0,
        loss_history=(0.5,),
        checkpoint_kind="final_selected",
        overwrite=False,
    )
    return TrainingResult(
        model=model,
        config=config,
        successful_optimizer_steps=1,
        scheduler_steps=1,
        attempted_batches=1,
        skipped_batches=0,
        loss_history=(0.5,),
        final_checkpoint_path=checkpoint,
    )


def test_approved_gpu_execution_records_hardware_checkpoint_and_one_retry(tmp_path: Path) -> None:
    request, approval = _approved_request(tmp_path)
    snapshots = iter((_gpu_snapshot(), _gpu_snapshot()))
    calls: list[Path | None] = []
    training_result = _training_result_with_reloadable_checkpoint(tmp_path, _config())

    def runner(resume_from: Path | None) -> TrainingResult:
        calls.append(resume_from)
        if len(calls) == 1:
            raise RuntimeError("node failure during the approved allocation")
        return training_result

    result = run_approved_gpu_experiment(
        request.root,
        approval,
        output_dir=tmp_path / "gpu-run",
        evidence_path=tmp_path / "gpu-evidence.json",
        slurm_environment=_slurm_environment(request.resources),
        gpu_probe=lambda: next(snapshots),
        training_runner=runner,
    )

    assert calls == [None, None]
    assert result.retry_count == 1
    assert result.evidence["status"] == "completed"
    assert result.evidence["checkpoint"]["final_reload_passed"] is True
    assert result.evidence["hardware"]["peak_vram_bytes"] == 3_000_000
    assert result.evidence["scientific_identity"]["config_fingerprint"] == request.payload[
        "experiment"
    ]["scientific_identity"]["config_fingerprint"]
    retry_attempt = result.evidence["retry"]["attempts"][0]
    assert "error" not in retry_attempt
    assert retry_attempt["retry"]["reason"].startswith("redacted;")
    assert result.evidence_path == tmp_path / "gpu-evidence.json"
    with pytest.raises(GpuExecutionError, match="different output directory"):
        run_approved_gpu_experiment(
            request.root,
            approval,
            output_dir=tmp_path / "second-gpu-run",
            slurm_environment=_slurm_environment(request.resources),
            gpu_probe=_gpu_snapshot,
            training_runner=runner,
        )


def test_approved_gpu_execution_does_not_retry_oom_or_hide_the_failure(tmp_path: Path) -> None:
    request, approval = _approved_request(tmp_path)
    calls = 0

    def runner(resume_from: Path | None) -> TrainingResult:
        nonlocal calls
        calls += 1
        raise TrainingError("CUDA out of memory")

    with pytest.raises(GpuExecutionError, match="out of memory"):
        run_approved_gpu_experiment(
            request.root,
            approval,
            output_dir=tmp_path / "gpu-run",
            slurm_environment=_slurm_environment(request.resources),
            gpu_probe=_gpu_snapshot,
            training_runner=runner,
        )
    assert calls == 1


def test_approved_gpu_execution_consumes_approval_before_lazy_loader(tmp_path: Path) -> None:
    preflight = _write_preflight(tmp_path / "cpu-preflight.json")
    request = prepare_gpu_experiment(
        tmp_path / "request",
        preflight_path=preflight,
        resource=GpuResourceRequest(smoke_test=True, wall_time_minutes=30),
        training_config=_config(),
        fold=0,
        source_version="fixture-v1",
        source_fingerprint="fixture-source",
        fold_manifest_checksum="fixture-fold",
        integration_manifest_checksum="fixture-integration",
    )
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "schema_version": GPU_APPROVAL_SCHEMA_VERSION,
                "request_checksum": request.request_checksum,
                "approval_scope": GPU_APPROVAL_SCOPE,
                "approved": True,
                "retry_allowed": True,
                "approver": "owner",
                "approved_at_utc": "2026-09-12T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    def loader() -> dict[str, object]:
        raise RuntimeError("private study loader failed")

    with pytest.raises(RuntimeError, match="private study loader failed"):
        run_approved_gpu_experiment(
            request.root,
            approval,
            output_dir=tmp_path / "gpu-run",
            slurm_environment=_slurm_environment(request.resources),
            gpu_probe=_gpu_snapshot,
            studies=loader,
            fold_manifest=SimpleNamespace(manifest_checksum="fixture-fold"),
            fold=0,
            source_version="fixture-v1",
            source_fingerprint="fixture-source",
            fold_manifest_checksum="fixture-fold",
            integration_manifest_checksum="fixture-integration",
        )

    with pytest.raises(GpuExecutionError, match="already been consumed"):
        run_approved_gpu_experiment(
            request.root,
            approval,
            output_dir=tmp_path / "gpu-run",
            slurm_environment=_slurm_environment(request.resources),
            gpu_probe=_gpu_snapshot,
            fold=0,
            source_version="fixture-v1",
            source_fingerprint="fixture-source",
            fold_manifest_checksum="fixture-fold",
            integration_manifest_checksum="fixture-integration",
        )


def test_gpu_smoke_test_requires_a_smoke_scoped_approved_request(tmp_path: Path) -> None:
    preflight = _write_preflight(tmp_path / "cpu-preflight.json")
    full_request = prepare_gpu_experiment(
        tmp_path / "full-request",
        preflight_path=preflight,
        training_config=_config(),
    )
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "schema_version": GPU_APPROVAL_SCHEMA_VERSION,
                "request_checksum": full_request.request_checksum,
                "approval_scope": GPU_APPROVAL_SCOPE,
                "approved": True,
                "retry_allowed": True,
                "approver": "owner",
                "approved_at_utc": "2026-09-12T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GpuExecutionError, match="smoke"):
        run_gpu_smoke_test(
            (),
            full_request.root,
            approval,
            output_dir=tmp_path / "gpu-run",
            slurm_environment=_slurm_environment(full_request.resources),
            gpu_probe=_gpu_snapshot,
        )


def test_gpu_smoke_test_rejects_empty_examples_before_allocation(tmp_path: Path) -> None:
    request, approval = _approved_request(tmp_path)

    with pytest.raises(GpuExecutionError, match="at least one example"):
        run_gpu_smoke_test(
            (),
            request.root,
            approval,
            output_dir=tmp_path / "gpu-run",
            slurm_environment={
                "SLURM_JOB_ID": "1234",
                "SLURM_JOB_PARTITION": "cluster1-gpu",
                "SLURM_JOB_GRES": "gpu:1",
                "CUDA_VISIBLE_DEVICES": "0",
            },
            gpu_probe=_gpu_snapshot,
        )
