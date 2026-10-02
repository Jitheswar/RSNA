"""Final replay cannot admit unfinished fits or conceal malformed predictions."""

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("ticket64_final_replay", SCRIPTS / "validate_ticket64_full_backbone_comparison.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


@pytest.mark.parametrize("actual, expected", [
    ({}, {}),
    ({"a": [0.] * 12}, {"b": [0.] * 12}),
    ({"a": [0.] * 11}, {"a": [0.] * 12}),
    ({"a": [float("nan")] * 12}, {"a": [0.] * 12}),
])
def test_final_replay_rejects_incomplete_or_nonfinite_predictions(actual, expected):
    with pytest.raises(ValueError):
        replay.maximum_error(actual, expected)


def test_final_replay_preserves_small_real_prediction_difference():
    assert replay.maximum_error({"a": [0.5] * 12}, {"a": [0.500002] * 12}) > 1e-6


def test_unfinished_control_does_not_create_a_replay_request(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.setattr(replay, "FIT", tmp_path / "fit")
    monkeypatch.setattr(replay, "ROOT", tmp_path / "output")
    monkeypatch.setattr(replay, "DECISION", tmp_path / "decision.json")
    replay.FIT.mkdir()
    plan = replay.FIT / "plan.json"
    plan.write_text(json.dumps(dict(projection=dict(independent_replay_wall_seconds=5400))))
    monkeypatch.setattr(replay, "FIT_SHA", replay.digest(plan))
    with pytest.raises(ValueError, match="complete matched control"):
        replay.prepare()
    assert not replay.ROOT.exists()
    assert not replay.DECISION.exists()


def test_replay_uses_retained_fold_zero_finals_and_original_first_stage():
    assert replay.final_checkpoint_path("head-continuation", 0) == (
        replay.OLD / "head-continuation/fold-0/training/checkpoints/final.pt"
    )
    assert replay.final_checkpoint_path("full-backbone", 0) == (
        replay.OLD / "full-backbone/fold-0/training/checkpoints/final.pt"
    )
    assert replay.final_checkpoint_path("full-backbone", 1) == (
        replay.FIT / "full-backbone/fold-1/training/checkpoints/final.pt"
    )
    assert replay.stage_location(0, 0) == (
        replay.OLD / "full-backbone/fold-0/stage-0", replay.SCIENCE_SHA
    )
    assert replay.stage_location(0, 1) == (
        replay.FIT / "full-backbone/fold-0/stage-1", replay.FIT_SHA
    )


def test_replay_rejects_unknown_final_fit():
    with pytest.raises(ValueError, match="Unknown final fit"):
        replay.final_checkpoint_path("unknown", 0)
    with pytest.raises(ValueError, match="Unknown final fit"):
        replay.final_checkpoint_path("full-backbone", 3)


def test_bounded_analysis_generates_only_the_frozen_final_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "FIT", tmp_path)
    records = [dict(arm="full-backbone", fold=0, checkpoint_sha256="final")]
    request = dict(analysis_generated_in_replay_allocation=True, fit_plan_sha256="fit", final_run_records=records)
    called = []

    def generator(checksum):
        called.append(checksum)
        (tmp_path / "analysis.json").write_text(json.dumps(dict(
            status="COMPLETE", plan_sha256=checksum, run_records=records, candidate_promotion=False)))

    assert replay.ensure_final_analysis(request, generator)["run_records"] == records
    assert called == ["fit"]
    with pytest.raises(ValueError, match="Unexpected analysis"):
        replay.ensure_final_analysis(request, generator)
    assert called == ["fit"], "An ambiguous existing result must not trigger another analysis"


def test_bounded_analysis_rejects_a_different_checkpoint_result(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "FIT", tmp_path)
    (tmp_path / "analysis.json").write_text(json.dumps(dict(
        status="COMPLETE", plan_sha256="fit", run_records=[dict(checkpoint_sha256="other")], candidate_promotion=False)))
    request = dict(analysis_generated_in_replay_allocation=False, fit_plan_sha256="fit",
                   final_run_records=[dict(checkpoint_sha256="frozen")])
    with pytest.raises(ValueError, match="Exact complete final comparison"):
        replay.ensure_final_analysis(request, lambda _: pytest.fail("Existing analysis must not be rerun"))
