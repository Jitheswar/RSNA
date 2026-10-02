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
