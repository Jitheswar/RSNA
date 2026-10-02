import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import ticket72_full_backbone_continuation as fit
import coordinate_ticket72_full_backbone_continuation as coordinator
import validate_ticket72_full_backbone_comparison as replay


def terminal_fixture(tmp_path):
    fit.write(tmp_path / "accepted-launch.json", dict(job_id=123, plan_sha256="fixed", launch_attempt=1))
    directory = tmp_path / "controller-123"
    directory.mkdir()
    fit.write(directory / "terminal-observation.json", dict(job_id="123", state="COMPLETED", returncode=0))
    text = "JobId=123 JobState=COMPLETED ExitCode=0:0 Partition=cluster1-cpu NumNodes=1 NumCPUs=16 NumTasks=1 MinMemoryNode=64G TimeLimit=05:30:00 OverSubscribe=OK Requeue=0 Restarts=0 TRES=cpu=16,node=1 StartTime=2026-10-02T10:00:00 EndTime=2026-10-02T11:00:00"
    (directory / "terminal-controller.txt").write_text(text)
    return directory


def test_exact_successful_terminal_interval(tmp_path):
    terminal_fixture(tmp_path)
    assert fit.successful_terminal(tmp_path, "fixed", "05:30:00")[0] == 3600


@pytest.mark.parametrize("old,new", [("COMPLETED", "FAILED"), ("ExitCode=0:0", "ExitCode=1:0"),
    ("NumCPUs=16", "NumCPUs=8"), ("Requeue=0", "Requeue=1"), ("Restarts=0", "Restarts=1"),
    ("TRES=cpu=16,node=1", "TRES=cpu=16,node=1,gres/gpu=1"),
    ("EndTime=2026-10-02T11:00:00", "EndTime=2026-10-03T11:00:00")])
def test_bad_terminal_never_authorizes_continuation(tmp_path, old, new):
    directory = terminal_fixture(tmp_path)
    path = directory / "terminal-controller.txt"
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ValueError):
        fit.successful_terminal(tmp_path, "fixed", "05:30:00")


def test_missing_actual_audit_completion_does_not_create_continuation(tmp_path, monkeypatch):
    monkeypatch.setattr(fit, "ROOT", tmp_path / "new-root")
    monkeypatch.setattr(fit, "DECISION", tmp_path / "decision.json")
    monkeypatch.setattr(fit, "AUDIT", tmp_path / "audit")
    with pytest.raises(FileNotFoundError):
        fit.prepare()
    assert not fit.ROOT.exists()
    assert not fit.DECISION.exists()


def audit_result_fixture(tmp_path, monkeypatch):
    root = tmp_path / "audit"
    root.mkdir()
    monkeypatch.setattr(fit, "AUDIT", root)
    monkeypatch.setattr(fit, "audit_admission", lambda checksum: None)
    monkeypatch.setattr(fit, "digest", lambda path: fit.AUDIT_SHA if Path(path).name == "plan.json" else "fixed")
    monkeypatch.setattr(fit, "successful_terminal", lambda *args: (900, root / "terminal.txt"))
    boundaries = [dict(fold=f, stage=s, successful_steps=n, checkpoint_sha256=str(f),
                       historical_backend_terminal_verified=False) for f, s, n in [(0, 3, 4000), (1, 2, 3000)]]
    fit.write(root / "plan.json", dict(boundaries=boundaries, frozen_files=[]))
    fit.write(root / "accepted-launch.json", dict(job_id=123))
    audited = [dict(row, **{k: True for k in ("model_adam_scheduler_rng_checked",
        "original_full_source_pixels_targets_masks_exposures_checked", "original_initial_tensors_and_normalization_checked",
        "strict_checkpoint_identity_checked")}) for row in boundaries]
    result = dict(status="COMPLETE", plan_sha256=fit.AUDIT_SHA, job_id="123", elapsed_seconds=850,
        original_missing_allocation_charge_seconds=39600, historical_backend_terminals_verified=False,
        original_strict_ancestry_passed=False, fitting_continuation_admitted=False, optimizer_updates_repeated=False,
        predictions_or_metrics_computed=False, candidate_promotion=False, gpu_seconds=0, tpu_seconds=0, external_spend_usd=0,
        boundaries=audited)
    fit.write(root / "checkpoint-audit.json", result)
    return root, result


def test_exact_two_fold_audit_with_actual_terminal_can_admit_state(tmp_path, monkeypatch):
    audit_result_fixture(tmp_path, monkeypatch)
    boundaries, seconds, _ = fit.audited_boundaries()
    assert [r["successful_steps"] for r in boundaries] == [4000, 3000]
    assert seconds == 900
    assert all(r["historical_backend_terminal_verified"] is False for r in boundaries)


@pytest.mark.parametrize("defect", ["job", "elapsed", "charge", "ancestry", "promotion", "onefold", "state", "hash"])
def test_incomplete_or_relabelled_audit_never_authorizes_state(tmp_path, monkeypatch, defect):
    root, result = audit_result_fixture(tmp_path, monkeypatch)
    if defect == "job":
        result["job_id"] = "124"
    elif defect == "elapsed":
        result["elapsed_seconds"] = 1741
    elif defect == "charge":
        result["original_missing_allocation_charge_seconds"] = 0
    elif defect == "ancestry":
        result["original_strict_ancestry_passed"] = True
    elif defect == "promotion":
        result["candidate_promotion"] = True
    elif defect == "onefold":
        result["boundaries"].pop()
    elif defect == "state":
        result["boundaries"][0]["model_adam_scheduler_rng_checked"] = False
    else:
        result["boundaries"][0]["checkpoint_sha256"] = "changed"
    (root / "checkpoint-audit.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        fit.audited_boundaries()


def test_successful_worker_audit_without_backend_success_is_not_admitted(tmp_path, monkeypatch):
    audit_result_fixture(tmp_path, monkeypatch)
    def missing_terminal(*args):
        raise ValueError("No actual successful terminal")
    monkeypatch.setattr(fit, "successful_terminal", missing_terminal)
    with pytest.raises(ValueError, match="actual successful terminal"):
        fit.audited_boundaries()


def test_remaining_stages_preserve_all_three_original_budgets():
    starts = fit.START_STAGES
    totals = [sum(min(7500, (s + 1) * 1000) - s * 1000 for s in range(start, 8)) for start in starts]
    assert totals == [3500, 4500, 6500]
    assert sum(totals) == 14500
    assert sum(len(range(start, 8)) for start in starts) == 16


def test_audited_boundaries_are_located_in_original_roots():
    assert fit.stage_location(0, 0, "new") == (fit.FIRST / "full-backbone/fold-0/stage-0", fit.SCIENCE_SHA)
    assert fit.stage_location(0, 3, "new") == (fit.PRIOR / "full-backbone/fold-0/stage-3", fit.PRIOR_SHA)
    assert fit.stage_location(1, 2, "new") == (fit.PRIOR / "full-backbone/fold-1/stage-2", fit.PRIOR_SHA)
    assert fit.stage_location(0, 4, "new") == (fit.ROOT / "full-backbone/fold-0/stage-4", "new")
    assert fit.stage_location(2, 0, "new") == (fit.PRIOR / "full-backbone/fold-2/stage-0", fit.PRIOR_SHA)


def test_does_not_duplicate_active_fold_or_exceed_slots():
    states = [["active", "unlaunched"], ["unlaunched"] * 5, ["unlaunched"] * 8]
    assert coordinator.next_unused_stage(states, 2) is None
    assert coordinator.next_unused_stage(states, 1) == (1, 3)
    states[0] = ["complete", "unlaunched"]
    assert coordinator.next_unused_stage(states, 1) == (0, 5)


def test_consumed_intent_never_restarts(tmp_path):
    fit.write(tmp_path / "launch-intent.json", dict(plan_sha256="fixed"))
    with pytest.raises(ValueError, match="Consumed intent"):
        coordinator.current_state(tmp_path, "fixed", 5000)


def test_worker_result_without_actual_terminal_is_still_active(tmp_path, monkeypatch):
    fit.write(tmp_path / "accepted-launch.json", dict(job_id=123, plan_sha256="fixed", launch_attempt=1))
    fit.write(tmp_path / "stage-result.json", dict(status="COMPLETE", successful_steps=5000))
    monkeypatch.setattr(coordinator.subprocess, "run", lambda *a, **k: type("Response", (), dict(returncode=0, stdout="JobId=123 JobState=RUNNING", stderr=""))())
    assert coordinator.current_state(tmp_path, "fixed", 5000) == "active"


def test_purged_new_worker_without_terminal_is_not_retried(tmp_path, monkeypatch):
    fit.write(tmp_path / "accepted-launch.json", dict(job_id=123, plan_sha256="fixed", launch_attempt=1))
    monkeypatch.setattr(coordinator.subprocess, "run", lambda *a, **k: type("Response", (), dict(returncode=1, stdout="", stderr="Invalid job id specified"))())
    with pytest.raises(ValueError, match="purged"):
        coordinator.current_state(tmp_path, "fixed", 5000)


def test_new_native_replay_reuses_controls_and_new_full_finals():
    assert replay.final_checkpoint_path("head-continuation", 0) == replay.OLD / "head-continuation/fold-0/training/checkpoints/final.pt"
    assert replay.final_checkpoint_path("head-continuation", 1) == replay.PRIOR / "head-continuation/fold-1/training/checkpoints/final.pt"
    assert replay.final_checkpoint_path("full-backbone", 0) == replay.FIT / "full-backbone/fold-0/training/checkpoints/final.pt"


@pytest.mark.parametrize("actual,expected", [({}, {}), ({"a": [0.] * 12}, {"b": [0.] * 12}),
    ({"a": [float("nan")] * 12}, {"a": [0.] * 12}), ({"a": [0.] * 11}, {"a": [0.] * 12})])
def test_native_proof_rejects_malformed_predictions(actual, expected):
    with pytest.raises(ValueError):
        replay.maximum_error(actual, expected)
