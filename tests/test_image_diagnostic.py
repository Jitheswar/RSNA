from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from rsna_knee.domain import DicomSlice, SeriesRecord, StudyRecord
from rsna_knee.image_diagnostic import (
    DiagnosticError,
    GPURequestRejected,
    _check_inputs,
    _check_slurm_context,
    _example_summary,
    _example_tensors,
    _measure_tree,
    _pixel_and_geometry_summary,
    admit_mock_gpu_request,
    summarize_gpu_attempts,
)
from rsna_knee.real_inference import IntegrationSeries, LoadedSeries, LoadedStudy
from rsna_knee.training import TrainingConfig, TrainingExample, load_training_checkpoint, train_fixed_budget


def test_gpu_ledger_counts_failed_allocations_and_keeps_queue_wait_separate() -> None:
    ledger = summarize_gpu_attempts(
        (
            {
                "status": "failed",
                "allocated_gpu_count": 1,
                "allocated_seconds": 1800,
                "queue_wait_seconds": 7200,
            },
            {
                "status": "pending",
                "allocated_gpu_count": 0,
                "allocated_seconds": 0,
                "queue_wait_seconds": 3600,
            },
        )
    )

    assert ledger["allocated_attempt_count"] == 1
    assert ledger["allocated_gpu_hours"] == 0.5
    assert ledger["queue_wait_seconds"] == 10800
    assert ledger["remaining_gpu_hours"] == 19.5


def test_mock_scheduler_rejects_concurrent_or_over_budget_gpu_requests() -> None:
    attempts = (
        {
            "status": "failed",
            "allocated_gpu_count": 1,
            "allocated_seconds": 19 * 3600,
            "queue_wait_seconds": 1800,
        },
    )

    with pytest.raises(GPURequestRejected, match="active or pending"):
        admit_mock_gpu_request(
            attempts,
            active_or_pending_requests=1,
            proposed_gpu_count=1,
            proposed_wall_time_seconds=30 * 60,
        )
    with pytest.raises(GPURequestRejected, match="exceed"):
        admit_mock_gpu_request(
            attempts,
            active_or_pending_requests=0,
            proposed_gpu_count=1,
            proposed_wall_time_seconds=2 * 3600,
        )
    with pytest.raises(GPURequestRejected, match="history is incomplete"):
        admit_mock_gpu_request(
            (),
            active_or_pending_requests=0,
            proposed_gpu_count=1,
            proposed_wall_time_seconds=30 * 60,
            accounting_complete=False,
        )
    with pytest.raises(GPURequestRejected, match="one GPU"):
        admit_mock_gpu_request(
            (),
            active_or_pending_requests=0,
            proposed_gpu_count=2,
            proposed_wall_time_seconds=30 * 60,
        )


def test_storage_metadata_walk_stops_at_its_configured_entry_bound(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rsna_knee.image_diagnostic as diagnostic

    (tmp_path / "one.bin").write_bytes(b"x")
    (tmp_path / "two.bin").write_bytes(b"y")
    monkeypatch.setattr(diagnostic, "STORAGE_SCAN_MAX_ENTRIES", 1)

    with pytest.raises(DiagnosticError, match="entry limit"):
        _measure_tree(tmp_path)


def test_mock_scheduler_accepts_a_request_only_when_both_gates_pass() -> None:
    decision = admit_mock_gpu_request(
        (
            {
                "status": "completed",
                "allocated_gpu_count": 1,
                "allocated_seconds": 60 * 60,
                "queue_wait_seconds": 900,
            },
        ),
        active_or_pending_requests=0,
        proposed_gpu_count=1,
        proposed_wall_time_seconds=30 * 60,
    )

    assert decision["admitted"] is True
    assert decision["allocated_gpu_hours"] == 1.0
    assert decision["proposed_reservation_hours"] == 0.5
    assert decision["remaining_gpu_hours"] == 19.0


@pytest.mark.parametrize("time_limit", ("360", "06:00", "6:00:00", "0-06:00:00"))
def test_cpu_slurm_guard_accepts_equivalent_frozen_six_hour_limits(
    monkeypatch: pytest.MonkeyPatch, time_limit: str
) -> None:
    environment = {
        "SLURM_JOB_ID": "7155",
        "SLURM_JOB_PARTITION": "cluster1-cpu",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_MEM_PER_NODE": "32G",
        "SLURM_TIMELIMIT": time_limit,
        "SLURM_JOB_EXCLUSIVE": "0",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    for name in ("SLURM_JOB_GRES", "SLURM_GPUS", "SLURM_JOB_GPUS"):
        monkeypatch.delenv(name, raising=False)

    assert _check_slurm_context()["time_limit"] == time_limit
    assert os.environ["SLURM_TIMELIMIT"] == "360"


def test_cpu_slurm_guard_rejects_a_longer_wall_time(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = {
        "SLURM_JOB_ID": "7155",
        "SLURM_JOB_PARTITION": "cluster1-cpu",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_MEM_PER_NODE": "32G",
        "SLURM_TIMELIMIT": "06:01:00",
        "SLURM_JOB_EXCLUSIVE": "0",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    for name in ("SLURM_JOB_GRES", "SLURM_GPUS", "SLURM_JOB_GPUS"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(DiagnosticError, match="time_limit='06:01:00'"):
        _check_slurm_context()


def test_cpu_slurm_guard_reads_missing_time_limit_from_active_job_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "SLURM_JOB_ID": "7156",
        "SLURM_JOB_PARTITION": "cluster1-cpu",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_MEM_PER_NODE": "32G",
        "SLURM_JOB_EXCLUSIVE": "0",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("SLURM_TIMELIMIT", raising=False)
    for name in ("SLURM_JOB_GRES", "SLURM_GPUS", "SLURM_JOB_GPUS"):
        monkeypatch.delenv(name, raising=False)
    calls: list[tuple[object, ...]] = []

    def fake_scontrol(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="JobId=7156 TimeLimit=06:00:00 State=RUNNING")

    monkeypatch.setattr("rsna_knee.image_diagnostic.subprocess.run", fake_scontrol)

    assert _check_slurm_context()["time_limit"] == "06:00:00"
    assert os.environ["SLURM_TIMELIMIT"] == "360"
    assert calls[0][0] == ["scontrol", "show", "job", "7156"]


def test_cpu_slurm_guard_fails_closed_when_job_time_limit_cannot_be_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "SLURM_JOB_ID": "7156",
        "SLURM_JOB_PARTITION": "cluster1-cpu",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_MEM_PER_NODE": "32G",
        "SLURM_JOB_EXCLUSIVE": "0",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("SLURM_TIMELIMIT", raising=False)
    for name in ("SLURM_JOB_GRES", "SLURM_GPUS", "SLURM_JOB_GPUS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "rsna_knee.image_diagnostic.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
    )

    with pytest.raises(DiagnosticError, match="could not query the active Slurm job TimeLimit"):
        _check_slurm_context()


@pytest.mark.parametrize(
    ("record", "message"),
    (
        ("JobId=7156 XTimeLimit=06:00:00", "could not resolve the active Slurm job TimeLimit"),
        ("JobId=7156 TimeLimit=06:00:00,", "frozen request"),
    ),
)
def test_cpu_slurm_guard_rejects_nonexact_or_malformed_time_limit_tokens(
    monkeypatch: pytest.MonkeyPatch, record: str, message: str
) -> None:
    environment = {
        "SLURM_JOB_ID": "7156",
        "SLURM_JOB_PARTITION": "cluster1-cpu",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_MEM_PER_NODE": "32G",
        "SLURM_JOB_EXCLUSIVE": "0",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("SLURM_TIMELIMIT", raising=False)
    for name in ("SLURM_JOB_GRES", "SLURM_GPUS", "SLURM_JOB_GPUS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "rsna_knee.image_diagnostic.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=record),
    )

    with pytest.raises(DiagnosticError, match=message):
        _check_slurm_context()


@pytest.mark.parametrize("record", ("JobId=7157 TimeLimit=06:00:00", "TimeLimit=06:00:00"))
def test_cpu_slurm_guard_rejects_a_time_limit_from_an_unmatched_job_record(
    monkeypatch: pytest.MonkeyPatch, record: str
) -> None:
    environment = {
        "SLURM_JOB_ID": "7156",
        "SLURM_JOB_PARTITION": "cluster1-cpu",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_MEM_PER_NODE": "32G",
        "SLURM_JOB_EXCLUSIVE": "0",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("SLURM_TIMELIMIT", raising=False)
    for name in ("SLURM_JOB_GRES", "SLURM_GPUS", "SLURM_JOB_GPUS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "rsna_knee.image_diagnostic.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=record),
    )

    with pytest.raises(DiagnosticError, match="did not identify the active CPU diagnostic job"):
        _check_slurm_context()


def test_cpu_slurm_guard_fails_closed_on_controller_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = {
        "SLURM_JOB_ID": "7156",
        "SLURM_JOB_PARTITION": "cluster1-cpu",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_MEM_PER_NODE": "32G",
        "SLURM_JOB_EXCLUSIVE": "0",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("SLURM_TIMELIMIT", raising=False)
    for name in ("SLURM_JOB_GRES", "SLURM_GPUS", "SLURM_JOB_GPUS"):
        monkeypatch.delenv(name, raising=False)

    def controller_timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("scontrol", timeout=10)

    monkeypatch.setattr("rsna_knee.image_diagnostic.subprocess.run", controller_timeout)

    with pytest.raises(DiagnosticError, match="could not query the active Slurm job TimeLimit"):
        _check_slurm_context()


def test_real_input_gate_enforces_three_valid_packed_slices_per_study() -> None:
    summary = {
        "study_identity_alignment": True,
        "series_status": {"accepted": 8},
        "expected_series": 8,
        "studies_with_accepted_series": 8,
        "nonconstant_fraction_of_finite_slices": 1.0,
    }
    examples = {
        "valid_studies_with_series_and_slices": 8,
        "minimum_valid_slice_count_per_study": 3,
        "packed_pixels_finite": True,
        "target_support": [{"positive": 1, "negative": 1} for _ in range(12)],
        "supervised_target_entries": 96,
    }

    assert all(_check_inputs(summary, examples).values())
    summary["study_identity_alignment"] = False
    assert not _check_inputs(summary, examples)["series_study_identities_align_with_loaded_studies"]
    summary["study_identity_alignment"] = True
    examples["minimum_valid_slice_count_per_study"] = 2
    assert not _check_inputs(summary, examples)["all_frozen_studies_meet_minimum_valid_slice_count"]


def test_pixel_and_numerical_gates_accept_actual_real_inference_record_shapes() -> None:
    loaded: list[LoadedStudy] = []
    training_examples: list[TrainingExample] = []
    study_uids: list[str] = []
    for study_index in range(8):
        study_uid = f"fixture-study-{study_index}"
        series_uid = f"fixture-series-{study_index}"
        study_uids.append(study_uid)
        entries = tuple(f"{study_uid}/slice-{slice_index}.dcm" for slice_index in range(3))
        manifest_series = IntegrationSeries(
            study_instance_uid=study_uid,
            series_instance_uid=series_uid,
            entries=entries,
            anatomical_plane="sagittal",
            fluid_sensitive=True,
            fat_suppression=False,
            diagnostic_status="diagnostic",
            expected_entry_count=3,
        )
        source_slices = tuple(
            DicomSlice(
                sop_instance_uid=f"fixture-sop-{study_index}-{slice_index}",
                image_position=(0.0, 0.0, float(slice_index)),
                image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                pixel_array=(np.arange(4, dtype=np.float32).reshape(2, 2) + slice_index),
                study_instance_uid=study_uid,
                series_instance_uid=series_uid,
                filename=entries[slice_index],
                padding_value=None,
            )
            for slice_index in range(3)
        )
        source_series = SeriesRecord(
            study_instance_uid=study_uid,
            series_instance_uid=series_uid,
            slices=source_slices,
            anatomical_plane="sagittal",
            fluid_sensitive=True,
            fat_suppression=False,
            diagnostic_status="diagnostic",
            source_entry_names=entries,
        )
        loaded.append(
            LoadedStudy(
                study=StudyRecord(study_uid, (source_series,)),
                series_results=(
                    LoadedSeries(
                        manifest_series=manifest_series,
                        series=source_series,
                        status="accepted",
                        expected_slice_count=3,
                        decoded_slice_count=3,
                    ),
                ),
            )
        )
        training_examples.append(
            TrainingExample(
                study_instance_uid=study_uid,
                images=np.ones((1, 3, 1, 2, 2), dtype=np.float32),
                slice_mask=np.ones((1, 3), dtype=bool),
                series_mask=np.ones(1, dtype=bool),
                targets=np.asarray(
                    [(study_index + target_index) % 2 for target_index in range(12)],
                    dtype=np.float32,
                ),
                supervision_mask=np.ones(12, dtype=bool),
            )
        )

    pixel_summary, _ = _pixel_and_geometry_summary(loaded)
    example_summary = _example_summary(training_examples, expected_ids=study_uids)
    checks = _check_inputs(pixel_summary, example_summary)

    assert pixel_summary["study_identity_alignment"] is True
    assert pixel_summary["decoded_slices_in_accepted_series"] == 24
    assert pixel_summary["nonconstant_fraction_of_finite_slices"] == 1.0
    assert pixel_summary["anatomical_plane_series_counts"] == {"sagittal": 8}
    assert example_summary["minimum_valid_slice_count_per_study"] == 3
    assert all(checks.values())


@pytest.mark.parametrize("operation", ("summary", "tensors"))
def test_real_image_diagnostic_rejects_feature_only_examples(operation: str) -> None:
    example = TrainingExample(
        study_instance_uid="feature-only-study",
        images=None,
        features=np.zeros((1, 1, 4), dtype=np.float32),
        slice_mask=np.ones((1, 1), dtype=bool),
        series_mask=np.ones(1, dtype=bool),
        targets=np.zeros(12, dtype=np.float32),
        supervision_mask=np.ones(12, dtype=bool),
    )

    with pytest.raises(DiagnosticError, match="requires packed images"):
        if operation == "summary":
            _example_summary((example,), expected_ids=(example.study_instance_uid,))
        else:
            _example_tensors((example,))


def test_two_step_checkpoint_interval_captures_step_250_between_epoch_boundaries(tmp_path) -> None:
    examples = tuple(
        TrainingExample(
            study_instance_uid=f"checkpoint-study-{study_index}",
            images=np.full((1, 1, 1, 16, 16), study_index / 8, dtype=np.float32),
            slice_mask=np.ones((1, 1), dtype=bool),
            series_mask=np.ones(1, dtype=bool),
            targets=np.asarray(
                [(study_index + target_index) % 2 for target_index in range(12)],
                dtype=np.float32,
            ),
            supervision_mask=np.ones(12, dtype=bool),
        )
        for study_index in range(8)
    )
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=1,
        max_series=1,
        batch_size=2,
        successful_optimizer_steps=252,
        checkpoint_interval_steps=2,
        checkpoint_interval_seconds=900,
        seed=29,
        batch_order_seed=30,
        augmentation="none",
        device="cpu",
    )

    result = train_fixed_budget(examples, config, artifact_dir=tmp_path / "interval-two", fold=0)
    observed_steps = set()
    for path in (tmp_path / "interval-two" / "checkpoints" / "history").glob("*.pt"):
        payload = load_training_checkpoint(
            path,
            expected_config=config,
            expected_fold=0,
            expected_checkpoint_kind="latest_resumable",
        )
        progress = payload["progress"]
        if isinstance(progress, dict):
            observed_steps.add(progress["successful_optimizer_steps"])

    assert result.successful_optimizer_steps == 252
    assert {100, 250, 252}.issubset(observed_steps)
