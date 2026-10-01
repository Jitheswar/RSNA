"""Scoring admission and ambiguous-response handling must prevent extra runs."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/score_public_frontier.py"
spec = importlib.util.spec_from_file_location("frontier_scoring", SCRIPT)
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)


def capacity(**changes):
    return {"budget_seconds": 129600, "spent_seconds": 69348,
            "outstanding_reserved_seconds": 0, "comparison_reserved_seconds": 0,
            "remaining_seconds": 60252, **changes}


def quota(**changes):
    return {"time_used_seconds": 49557.984, "time_reserved_seconds": 0,
            "total_time_allowed_seconds": 108000, "is_pay_to_scale_enabled": False, **changes}


def test_duration_normalization_retains_whole_days_and_fractional_seconds():
    response = SimpleNamespace(gpu_quota=SimpleNamespace(time_used=timedelta(seconds=49557.984),
        time_reserved=timedelta(0), total_time_allowed=timedelta(days=1, hours=6),
        is_pay_to_scale_enabled=False), quota_refresh_time=datetime(2026, 10, 3))
    actual = scoring.normalized_quota(response)
    assert actual["total_time_allowed_seconds"] == 108000
    assert actual["time_used_seconds"] == 49557.984


@pytest.mark.parametrize("change", ["approval", "cap", "space", "reserve", "protected", "paid",
                                    "unknown_paid", "free_quota", "other_session", "daily", "duplicate", "live"])
def test_each_required_admission_gate_blocks_submission(change):
    c, q, allowed, approval, duplicate, live = capacity(), quota(), 4, None, False, False
    if change == "approval": approval = "owner approval missing"
    if change == "cap": c["budget_seconds"] = 72000
    if change == "space": c["remaining_seconds"] = 2652
    if change == "reserve": c["outstanding_reserved_seconds"] = 2400
    if change == "protected": c["comparison_reserved_seconds"] = 3000
    if change == "paid": q["is_pay_to_scale_enabled"] = True
    if change == "unknown_paid": q["is_pay_to_scale_enabled"] = None
    if change == "free_quota": q["total_time_allowed_seconds"] = 50000
    if change == "other_session": q["time_reserved_seconds"] = 1
    if change == "daily": allowed = 0
    if change == "duplicate": duplicate = True
    if change == "live": live = True
    assert scoring.blockers(c, q, allowed, approval, duplicate=duplicate, other_live_scoring=live)


def test_free_quota_does_not_expand_campaign_budget():
    original = capacity(budget_seconds=72000, remaining_seconds=2652)
    before = deepcopy(original)
    assert scoring.blockers(original, quota(total_time_allowed_seconds=1_000_000), 4, None,
                            duplicate=False, other_live_scoring=False)
    assert original == before


def local():
    return {"proposal": {"competition": "rsna-knee-abnormality-detection", "resource_proposal_sha256": "f" * 64}}


def approval(tmp_path):
    value = {"schema_version": "rsna-ticket51-scoring-approval-v1", "decision": "approved",
             "proposal_sha256": scoring.PROPOSAL_SHA, "resource_proposal_sha256": "f" * 64,
             "approved_campaign_cap_gpu_seconds": 129600, "approved_request_gpu_seconds": 57600,
             "kernel_id": scoring.HANDLE, "kernel_version": 3, "output_file": "submission.csv",
             "maximum_attempts": 1, "external_spend_usd": 0, "paid_scaling_permitted": False,
             "final_candidate_selection": False, "owner_decision_text": "Synthetic test authorization",
             "owner_decision_received_at_utc": "2026-09-30T00:00:00+00:00"}
    path = tmp_path / "test-approval.json"
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize("field,value", [("proposal_sha256", "b" * 64),
    ("approved_campaign_cap_gpu_seconds", 72000), ("kernel_version", 4),
    ("owner_decision_text", ""), ("owner_decision_received_at_utc", "2026-09-30T00:00:00"),
    ("maximum_attempts", 2), ("final_candidate_selection", True)])
def test_approval_must_bind_exact_owner_decision_and_scoring_scope(tmp_path, field, value):
    path = approval(tmp_path)
    data = json.loads(path.read_text()); data[field] = value; path.write_text(json.dumps(data))
    assert scoring.approval_error(path, local()["proposal"], datetime(2026, 10, 1, tzinfo=timezone.utc))


def ready():
    return {"blockers": [], "observed_at_utc": datetime.now(timezone.utc).isoformat()}


class FakeApi:
    def __init__(self, *, fail=False):
        self.calls, self.fail = [], fail

    def competition_submit_code(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise TimeoutError("ambiguous response")
        return SimpleNamespace(ref=123, to_json=lambda **unused: '{"ref":123,"message":"accepted"}')


def test_missing_approval_cannot_reserve_or_send_code(tmp_path):
    api = FakeApi()
    ledgers = []
    with pytest.raises(ValueError, match="approval"):
        scoring.submit_once(tmp_path, local(), None, ready(), api, lambda *args: ledgers.append(args))
    assert not api.calls and not ledgers
    assert not (tmp_path / "scoring-submit-intent.json").exists()


def test_admitted_action_sends_one_exact_code_request_after_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "local_evidence", lambda root: local())
    api, calls = FakeApi(), []
    ref = scoring.submit_once(tmp_path, local(), approval(tmp_path), ready(), api, lambda *args: calls.append(args[0]))
    assert ref == 123 and calls == ["reserve", "accepted"]
    assert len(api.calls) == 1
    assert api.calls[0]["kernel"] == scoring.HANDLE
    assert api.calls[0]["kernel_version"] == 3
    assert api.calls[0]["file_name"] == "submission.csv"
    assert api.calls[0]["competition"] == "rsna-knee-abnormality-detection"
    with pytest.raises(ValueError, match="intent"):
        scoring.submit_once(tmp_path, local(), tmp_path / "test-approval.json", ready(), api, lambda *args: calls.append(args[0]))
    assert len(api.calls) == 1


def test_timeout_keeps_reservation_and_forbids_a_second_network_call(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "local_evidence", lambda root: local())
    api, calls = FakeApi(fail=True), []
    path = approval(tmp_path)
    with pytest.raises(TimeoutError):
        scoring.submit_once(tmp_path, local(), path, ready(), api, lambda *args: calls.append(args[0]))
    failure = json.loads((tmp_path / "scoring-submit-ambiguous-or-local-failure.json").read_text())
    assert failure["reservation_released"] is False
    assert failure["automatic_retry_permitted"] is False and calls == ["reserve"]
    with pytest.raises(ValueError, match="intent"):
        scoring.submit_once(tmp_path, local(), path, ready(), api, lambda *args: calls.append(args[0]))
    assert len(api.calls) == 1


def test_stale_preflight_cannot_send_or_reserve(tmp_path):
    check = ready()
    check["observed_at_utc"] = (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat()
    api, calls = FakeApi(), []
    with pytest.raises(ValueError, match="fresh"):
        scoring.submit_once(tmp_path, local(), approval(tmp_path), check, api, lambda *args: calls.append(args))
    assert not calls and not api.calls


@pytest.mark.parametrize("field", ["currentVersionNumber", "enableInternet", "dockerImage", "datasetDataSources", "source"])
def test_saved_source_or_attachment_drift_is_rejected(field):
    remote = {"metadata": {"id": 1, "ref": scoring.HANDLE, "currentVersionNumber": 3, "isPrivate": True,
        "enableGpu": True, "enableInternet": False, "enableTpu": False, "dockerImage": "pinned-image",
        "machineShape": "NvidiaTeslaT4", "language": "python", "kernelType": "notebook", "datasetDataSources": ["a"]},
        "blob": {"source": "frozen source"}}
    expected = {"remote": deepcopy(remote), "resource": {"remote_source_sha256": hashlib.sha256(b"frozen source").hexdigest()}}
    if field == "source": remote["blob"]["source"] = "changed source"
    elif field == "datasetDataSources": remote["metadata"][field].append("b")
    else: remote["metadata"][field] = "changed"
    with pytest.raises(ValueError):
        scoring.validate_remote(expected, remote, "COMPLETE")


def test_changed_local_evidence_blocks_submission_before_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "local_evidence", lambda root: {"changed": True})
    api, calls = FakeApi(), []
    with pytest.raises(ValueError, match="changed"):
        scoring.submit_once(tmp_path, local(), approval(tmp_path), ready(), api, lambda *args: calls.append(args))
    assert not calls and not api.calls


def test_accepted_receipt_survives_local_ledger_failure_and_prevents_resubmission(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "local_evidence", lambda root: local())
    def ledger(action, *args):
        if action == "accepted":
            raise RuntimeError("local persistence unavailable")
    api, path = FakeApi(), approval(tmp_path)
    with pytest.raises(RuntimeError):
        scoring.submit_once(tmp_path, local(), path, ready(), api, ledger)
    assert json.loads((tmp_path / "scoring-submit-response.json").read_text())["ref"] == 123
    with pytest.raises(ValueError, match="intent"):
        scoring.submit_once(tmp_path, local(), path, ready(), api, ledger)
    assert len(api.calls) == 1


@pytest.mark.parametrize("ref,url", [(124, "/code/" + scoring.HANDLE + "?scriptVersionId=999"),
    (123, "/code/someone/another-model?scriptVersionId=999"),
    (123, "/code/" + scoring.HANDLE + "?scriptVersionId=0"),
    (123, "/code/" + scoring.HANDLE + "?other=999")])
def test_monitor_rejects_other_references_and_unbound_source_urls(monkeypatch, ref, url):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    monitor_spec = importlib.util.spec_from_file_location("frontier_monitor_test", SCRIPT.with_name("monitor_public_frontier_scoring.py"))
    monitor = importlib.util.module_from_spec(monitor_spec)
    monitor_spec.loader.exec_module(monitor)
    with pytest.raises(ValueError):
        monitor.source_identity(SimpleNamespace(ref=ref, url=url), 123)
