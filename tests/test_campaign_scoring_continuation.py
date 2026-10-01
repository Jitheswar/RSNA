"""Exercise delayed continuation against terminal accounting and real trace events."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "campaign_continuation", Path(__file__).resolve().parents[1] / "scripts/continue_after_frontier_scoring.py"
)
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def proof(tmp_path):
    source = {"submission_ref": continuation.REFERENCE, "script_version_id": continuation.SCRIPT_VERSION,
              "kernel_id": continuation.HANDLE, "kernel_version": 3,
              "source_url": f"/code/{continuation.HANDLE}?scriptVersionId={continuation.SCRIPT_VERSION}"}
    evidence = {"request_checksum": continuation.PROPOSAL, "job_id": str(continuation.REFERENCE),
                "script_version_id": continuation.SCRIPT_VERSION,
                "submission": {"ref": continuation.REFERENCE, "status": "COMPLETE"},
                "source_submission": {"ref": continuation.REFERENCE, "url": source["source_url"]}}
    write(tmp_path / "terminal-scoring-evidence.json", evidence)
    write(tmp_path / "accepted-scoring-source.json", source)
    write(tmp_path / "scoring-submit-response.json", {"ref": continuation.REFERENCE})
    write(tmp_path / "scoring-status.json", {"submission_ref": continuation.REFERENCE, "status": "COMPLETE"})
    ledger = {"requests": [{"request_checksum": continuation.PROPOSAL, "status": "completed",
        "reserved_seconds": 0, "terminal_accounting": {"job_id": str(continuation.REFERENCE),
        "evidence_sha256": hashlib.sha256((tmp_path / "terminal-scoring-evidence.json").read_bytes()).hexdigest(),
        "conservative_charge_seconds": 57600}}]}
    write(tmp_path / "ledger.json", ledger)
    return tmp_path, ledger


def test_exact_reconciled_terminal_can_continue(proof):
    root, _ = proof
    assert continuation.reconciled_terminal(root, root / "ledger.json")["charged_gpu_seconds"] == 57600


@pytest.mark.parametrize("fault", ["other_reference", "pending", "unaccounted", "unreleased", "wrong_source", "changed_evidence"])
def test_incomplete_or_mismatched_proof_cannot_launch(proof, fault):
    root, ledger = proof
    if fault == "other_reference":
        write(root / "scoring-submit-response.json", {"ref": 56705536})
    elif fault == "pending":
        write(root / "scoring-status.json", {"submission_ref": continuation.REFERENCE, "status": "PENDING"})
    elif fault == "unaccounted":
        ledger["requests"][0].pop("terminal_accounting")
    elif fault == "unreleased":
        ledger["requests"][0]["reserved_seconds"] = 57600
    elif fault == "wrong_source":
        source = json.loads((root / "accepted-scoring-source.json").read_text())
        source["script_version_id"] = 354125287
        write(root / "accepted-scoring-source.json", source)
    else:
        evidence = json.loads((root / "terminal-scoring-evidence.json").read_text())
        evidence["submission"]["publicScore"] = "0.999"
        write(root / "terminal-scoring-evidence.json", evidence)
    write(root / "ledger.json", ledger)
    assert continuation.reconciled_terminal(root, root / "ledger.json") is None


def test_parent_must_finish_after_arming_and_have_no_new_active_turn(tmp_path):
    trace = tmp_path / "rollout.jsonl"
    stamp = "2026-10-01T01:00:00+00:00"
    event = lambda kind, time: json.dumps({"type": "event_msg", "timestamp": time, "payload": {"type": kind}}) + "\n"
    trace.write_text(event("task_complete", "2026-10-01T00:59:59Z"))
    assert not continuation.parent_idle(trace, stamp)
    with trace.open("a") as handle:
        handle.write(event("task_started", "2026-10-01T01:00:01Z"))
        handle.write(event("task_complete", "2026-10-01T01:00:02Z"))
    assert continuation.parent_idle(trace, stamp)
    with trace.open("a") as handle:
        handle.write(event("task_started", "2026-10-01T01:00:03Z"))
    assert not continuation.parent_idle(trace, stamp)


def test_exclusive_intent_prevents_a_second_launch(tmp_path):
    intent = tmp_path / "launch-intent.json"
    continuation.save_once(intent, {"attempts": 1})
    with pytest.raises(FileExistsError):
        continuation.save_once(intent, {"attempts": 2})
    assert json.loads(intent.read_text()) == {"attempts": 1}
