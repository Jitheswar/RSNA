"""Replay the endpoint-specific URL difference through the real observer CLI."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/monitor_public_frontier_scoring.py"


def load_monitor(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("observer_replay_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def endpoint_submission(*, ref=123, url, status="COMPLETE"):
    data = {"ref": ref, "url": url, "status": status, "publicScore": "0.943", "privateScore": ""}
    return SimpleNamespace(ref=ref, url=url, status=SimpleNamespace(name=status), public_score="0.943",
                           private_score="", error_description="", to_json=lambda **kwargs: json.dumps(data))


def install_fake_sdk(monkeypatch, exact, listing):
    class FakeApi:
        def authenticate(self):
            pass

        @contextmanager
        def build_kaggle_client(self):
            yield SimpleNamespace(competitions=SimpleNamespace(competition_api_client=
                SimpleNamespace(get_submission=lambda request: exact)))

        def competition_submissions(self, competition, page_size):
            return listing

    for name in ["kaggle", "kaggle.api", "kaggle.api.kaggle_api_extended", "kagglesdk",
                 "kagglesdk.competitions", "kagglesdk.competitions.types", "kagglesdk.competitions.types.competition_api_service"]:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["kaggle.api.kaggle_api_extended"].KaggleApi = FakeApi
    sys.modules["kagglesdk.competitions.types.competition_api_service"].ApiGetSubmissionRequest = SimpleNamespace


def prepare(monkeypatch, tmp_path, monitor):
    monkeypatch.setattr(monitor, "local_evidence", lambda root: {"proposal": {"competition": "rsna-knee-abnormality-detection"}})
    (tmp_path / "scoring-submit-intent.json").write_text(json.dumps({"proposal_sha256": monitor.PROPOSAL_SHA}))
    (tmp_path / "scoring-submit-response.json").write_text('{"ref":123}')
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--root", str(tmp_path)])
    calls = []
    monkeypatch.setattr(monitor, "ledger_action", lambda *args: calls.append(args) or {"spent_seconds": 57600})
    return calls


@pytest.mark.parametrize("listing_status", ["COMPLETE", "PENDING"])
def test_completed_exact_submission_raw_url_uses_listing_for_saved_code_identity(tmp_path, monkeypatch, listing_status):
    monitor = load_monitor(monkeypatch)
    exact = endpoint_submission(url="/submissions/123/123.raw")
    listed = endpoint_submission(url=f"/code/{monitor.HANDLE}?scriptVersionId=999", status=listing_status)
    install_fake_sdk(monkeypatch, exact, [listed])
    calls = prepare(monkeypatch, tmp_path, monitor)
    monitor.main()
    status = json.loads((tmp_path / "scoring-status.json").read_text())
    terminal = json.loads((tmp_path / "terminal-scoring-evidence.json").read_text())
    assert status["status"] == "COMPLETE" and status["script_version_id"] == 999
    assert status["source_url"] == listed.url
    assert terminal["submission"]["url"] == exact.url
    assert terminal["source_submission"]["url"] == listed.url
    assert terminal["submission"]["ref"] == terminal["source_submission"]["ref"] == 123
    assert terminal["submission"]["status"] == "COMPLETE"
    assert terminal["source_submission"]["status"] == listing_status
    assert calls == [("terminal", tmp_path)]


@pytest.mark.parametrize("failure", ["wrong_exact_ref", "missing", "duplicate", "wrong_source", "source_drift"])
def test_unproven_source_or_reference_never_creates_terminal_accounting(tmp_path, monkeypatch, failure):
    monitor = load_monitor(monkeypatch)
    exact = endpoint_submission(url="/submissions/123/123.raw", ref=124 if failure == "wrong_exact_ref" else 123)
    listed = endpoint_submission(url=f"/code/{monitor.HANDLE}?scriptVersionId=999")
    listing = [listed]
    if failure == "missing": listing = []
    if failure == "duplicate": listing = [listed, listed]
    if failure == "wrong_source": listing = [endpoint_submission(url="/code/other/model?scriptVersionId=999")]
    if failure == "source_drift":
        monitor.save(tmp_path / "accepted-scoring-source.json", monitor.source_identity(
            endpoint_submission(url=f"/code/{monitor.HANDLE}?scriptVersionId=998"), 123))
    install_fake_sdk(monkeypatch, exact, listing)
    calls = prepare(monkeypatch, tmp_path, monitor)
    with pytest.raises(ValueError):
        monitor.main()
    assert not calls
    assert not (tmp_path / "terminal-scoring-evidence.json").exists()


def test_endpoint_specific_terminal_proof_charges_once_in_real_temporary_ledger(tmp_path, monkeypatch):
    monitor = load_monitor(monkeypatch)
    scoring = sys.modules["score_public_frontier"]
    from rsna_knee.gpu_campaign import (initialize_gpu_campaign_ledger, reserve_gpu_campaign_request,
        begin_gpu_submission_attempt, record_gpu_submission_result, gpu_campaign_capacity)
    ledger = tmp_path / "test-ledger.json"
    initialize_gpu_campaign_ledger(ledger, reserved_comparison_hours=0)
    reserve_gpu_campaign_request(ledger, request_checksum=scoring.PROPOSAL_SHA, experiment_id="test",
        wall_time_minutes=480, gpu_count=2, retry_allowed=False, remaining_attempts=1)
    begin_gpu_submission_attempt(ledger, request_checksum=scoring.PROPOSAL_SHA, attempt_number=1, submitted_epoch=100)
    record_gpu_submission_result(ledger, request_checksum=scoring.PROPOSAL_SHA, attempt_number=1,
        job_id="123", exit_code=0, submission_wait=False)
    monkeypatch.setattr(scoring, "LEDGER", ledger)
    monkeypatch.setattr(scoring, "local_evidence", lambda root: {"proposal": {}})
    monkeypatch.setattr(monitor, "local_evidence", lambda root: {"proposal": {"competition": "test"}})
    exact = endpoint_submission(url="/submissions/123/123.raw")
    listed = endpoint_submission(url=f"/code/{monitor.HANDLE}?scriptVersionId=999")
    install_fake_sdk(monkeypatch, exact, [listed])
    prepare(monkeypatch, tmp_path, monitor)
    monkeypatch.setattr(monitor, "ledger_action", lambda action, root: scoring.apply_ledger_action(
        SimpleNamespace(ledger_action=action, root=root, approval=None)))
    monitor.main()
    assert gpu_campaign_capacity(ledger)["spent_seconds"] == 57600
    before = ledger.read_bytes()
    monitor.main()
    assert ledger.read_bytes() == before
