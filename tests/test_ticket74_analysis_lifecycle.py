import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("ticket74_lifecycle", Path(__file__).parents[1] / "scripts/ticket74_analysis_lifecycle.py")
life = importlib.util.module_from_spec(spec)
spec.loader.exec_module(life)


def setup_launch(tmp_path, monkeypatch, kind="analysis"):
    root = tmp_path / "run"
    directory = root / kind
    directory.mkdir(parents=True)
    (tmp_path / "artifacts/private").mkdir(parents=True)
    monkeypatch.setattr(life, "ROOT", root)
    monkeypatch.setattr(life, "PROJECT", tmp_path)
    monkeypatch.setattr(life, "admission", lambda: ({}, dict(input_admission_sha256="fixed", launchers=[dict(kind=kind, path="launcher", sha256="fixed")])))
    monkeypatch.setattr(life, "digest", lambda path: "fixed")
    return directory


@pytest.mark.parametrize("kind", ["analysis", "verification"])
def test_consumed_component_never_calls_backend(tmp_path, monkeypatch, kind):
    directory = setup_launch(tmp_path, monkeypatch, kind)
    life.write(directory / "launch-intent.json", dict(plan_sha256="fixed"))
    monkeypatch.setattr(life.subprocess, "check_output", lambda *a, **k: pytest.fail("Consumed launch must not call backend"))
    with pytest.raises(ValueError, match="Consumed component"):
        life.launch(kind)


def test_full_cpu_slots_wait_without_consuming_intent(tmp_path, monkeypatch):
    directory = setup_launch(tmp_path, monkeypatch)
    def queue_only(command, **kwargs):
        assert command[0] == "squeue"
        return "123|cluster1-cpu\n124|cluster1-cpu\n"
    monkeypatch.setattr(life.subprocess, "check_output", queue_only)
    assert life.launch("analysis") is False
    assert not (directory / "launch-intent.json").exists()


def test_verification_cannot_launch_before_actual_primary_success(tmp_path, monkeypatch):
    directory = setup_launch(tmp_path, monkeypatch, "verification")
    def failed_primary(kind):
        assert kind == "analysis"
        raise ValueError("Actual primary success missing")
    monkeypatch.setattr(life, "terminal", failed_primary)
    monkeypatch.setattr(life.subprocess, "check_output", lambda *a, **k: pytest.fail("Incomplete primary must not consume backend"))
    with pytest.raises(ValueError, match="primary success"):
        life.launch("verification")
    assert not (directory / "launch-intent.json").exists()


def test_unregistered_component_rejected(tmp_path, monkeypatch):
    setup_launch(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="registered primary"):
        life.launch("fit")


def parent_decision():
    return life.read(Path(__file__).parents[1] / ".scratch/rsna-leaderboard/resource-approvals/ticket73-configured-continuation-20261002.json")


def test_reserves_cannot_be_released_before_native_inputs_verified():
    with pytest.raises(ValueError, match="actual native inputs"):
        life.remaining_reserves(parent_decision(), complete_inputs_verified=False)


def test_completed_allocations_remain_on_disk_and_whole_future_route_retained():
    from copy import deepcopy
    previous = parent_decision(); original = deepcopy(previous)
    calendar, protected_bytes, credit = life.remaining_reserves(previous, complete_inputs_verified=True)
    assert previous == original
    assert calendar["conditional_final_fit_seconds"] == 158400
    assert calendar["retained_review_seconds"] == 86400
    assert calendar["offline_delivery_seconds"] == 7200
    assert calendar["queue_transition_seconds"] == 21600
    assert calendar["additional_retained_cpu_delivery_seconds"] == 21600
    assert calendar["additional_retained_gpu_delivery_seconds"] == 7200
    assert sum(calendar.values()) == 309600
    per_fit = (previous["projection"]["history_count_upper_bound_per_fit"] + 4) * previous["projection"]["maximum_checkpoint_bytes"]
    assert credit == 6 * per_fit
    assert protected_bytes >= per_fit


def test_changed_storage_envelope_never_releases_assumed_completed_reserve():
    previous = parent_decision()
    previous["projection"]["checkpoint_storage_allowance_bytes"] += 1
    with pytest.raises(ValueError, match="storage envelope"):
        life.remaining_reserves(previous, complete_inputs_verified=True)


def test_observation_timeout_repolls_same_fitting_owner_then_stops_on_actual_failure(tmp_path, monkeypatch):
    root = tmp_path / "analysis";root.mkdir()
    fit = tmp_path / "fit";fit.mkdir()
    life.write(root / "plan.json", dict(fit_root=str(fit)))
    monkeypatch.setattr(life, "ROOT", root)
    monkeypatch.setattr(life, "digest", lambda path: life.PLAN_SHA)
    calls = []
    def observation(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise life.subprocess.TimeoutExpired(command, 15)
        return "ActiveState=failed\nMainPID=0\n"
    monkeypatch.setattr(life.subprocess, "check_output", observation)
    monkeypatch.setattr(life.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(life, "prepare", lambda: pytest.fail("No native proof must not admit resources"))
    with pytest.raises(ValueError, match="controller stopped"):
        life.await_inputs()
    assert len(calls) == 2 and calls[0] == calls[1]
    assert not (root / "input-resource-admission.json").exists()
