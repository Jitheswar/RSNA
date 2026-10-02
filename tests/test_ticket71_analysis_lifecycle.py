import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("ticket71_lifecycle", Path(__file__).parents[1] / "scripts/ticket71_analysis_lifecycle.py")
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
