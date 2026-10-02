"""Operational guards prevent duplicate launches and advancing failed stages."""

import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "ticket64_coordinator", Path(__file__).resolve().parents[1]
    / "scripts/coordinate_ticket64_full_backbone_comparison.py",
)
coordinator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coordinator)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def complete_stage(root, *, step=1000, state="COMPLETED", cpus="16"):
    write(root / "accepted-launch.json", dict(job_id=123, plan_sha256="plan", launch_attempt=1))
    controller = root / "controller-123"
    write(controller / "terminal-observation.json", dict(job_id="123", state=state, returncode=0))
    fields = dict(JobId="123", JobState=state, ExitCode="0:0", Partition="cluster1-cpu",
                  NumNodes="1", NumCPUs=cpus, NumTasks="1", MinMemoryNode="64G",
                  TimeLimit="05:30:00", OverSubscribe="OK", Requeue="0", Restarts="0", TRES="cpu=16")
    (controller / "terminal-controller.txt").write_text(" ".join(f"{k}={v}" for k, v in fields.items()))
    write(root / "stage-result.json", dict(status="COMPLETE", plan_sha256="plan", job_id="123",
          candidate_promotion=False, successful_steps=step, intermediate_prediction=False,
          progress=dict(successful_optimizer_steps=step, attempted_batches=step)))


def test_consumed_intent_without_receipt_never_relaunches(tmp_path):
    write(tmp_path / "launch-intent.json", {})
    with pytest.raises(ValueError, match="Consumed intent"):
        coordinator.stage_state(tmp_path, "plan", 1000, "05:30:00")


@pytest.mark.parametrize("changes", [dict(state="FAILED"), dict(cpus="32"), dict(step=999)])
def test_bad_terminal_resource_or_counter_blocks_advancement(tmp_path, changes):
    complete_stage(tmp_path, **changes)
    with pytest.raises(ValueError):
        coordinator.stage_state(tmp_path, "plan", 1000, "05:30:00")


def test_complete_stage_requires_record_and_exact_success(tmp_path):
    complete_stage(tmp_path)
    assert coordinator.stage_state(tmp_path, "plan", 1000, "05:30:00") == "complete"
    (tmp_path / "stage-result.json").unlink()
    with pytest.raises(FileNotFoundError):
        coordinator.stage_state(tmp_path, "plan", 1000, "05:30:00")


def test_registered_schedule_preserves_control_dependency_and_two_slots():
    folds = [["active", "unlaunched"], ["unlaunched", "unlaunched"], ["unlaunched", "unlaunched"]]
    assert coordinator.next_stage("active", folds, 1) is None
    assert coordinator.next_stage("complete", folds, 2) is None
    assert coordinator.next_stage("complete", folds, 1) == (1, 0)
    folds[0] = ["complete", "unlaunched"]
    assert coordinator.next_stage("complete", folds, 0) == (0, 1)
    assert coordinator.next_stage("complete", [["complete", "complete"]] * 3, 0) is None


def test_frozen_launcher_resolves_inputs_in_the_actual_project(tmp_path):
    env = coordinator.launch_environment(dict(source_snapshot=str(tmp_path)))
    assert env["RSNA_PROJECT_ROOT"] == str(coordinator.PROJECT)
    assert env["PYTHONPATH"] == str(tmp_path / "src")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
