"""Resume the existing campaign thread once its exact scoring run is reconciled.

This coordinator never submits a notebook or releases a GPU reservation.
It reuses existing ChatGPT CLI authentication; API-key billing is refused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


PROJECT = Path(__file__).resolve().parents[1]
SCORING = PROJECT / "artifacts/private/public-frontier-adoption/scored-baseline-v1"
STATE = PROJECT / "artifacts/private/campaign-continuation-frontier56739912"
LEDGER = PROJECT / ".scratch/rsna-leaderboard/gpu-campaign-ledger.json"
REFERENCE = 56739912
SCRIPT_VERSION = 354194017
PROPOSAL = "3490b9fcba9381e4778c70f4e2a4597ff660c7eb7e74dc3f8b0f446ea763317c"
HANDLE = "jitheswar/rsna-public-frontier-gpu-preflight-attributed"


def read(path: Path) -> dict:
    if path.is_symlink():
        raise ValueError("symlinked evidence is not accepted")
    return json.loads(path.read_text())


def save_once(path: Path, value: dict) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)


def reconciled_terminal(scoring: Path, ledger: Path) -> dict | None:
    try:
        status = read(scoring / "scoring-status.json")
        evidence_path = scoring / "terminal-scoring-evidence.json"
        evidence = read(evidence_path)
        source = read(scoring / "accepted-scoring-source.json")
        response = read(scoring / "scoring-submit-response.json")
        data = read(ledger)
    except (OSError, ValueError):
        return None
    terminal = status.get("status")
    if terminal not in {"COMPLETE", "ERROR"}:
        return None
    expected = "completed" if terminal == "COMPLETE" else "failed"
    if (status.get("submission_ref") != REFERENCE or response.get("ref") != REFERENCE
            or source.get("submission_ref") != REFERENCE
            or source.get("kernel_id") != HANDLE or source.get("kernel_version") != 3
            or source.get("source_url") != f"/code/{HANDLE}?scriptVersionId={SCRIPT_VERSION}"
            or source.get("script_version_id") != SCRIPT_VERSION
            or evidence.get("job_id") != str(REFERENCE)
            or evidence.get("script_version_id") != SCRIPT_VERSION
            or evidence.get("request_checksum") != PROPOSAL
            or evidence.get("submission", {}).get("status") != terminal
            or evidence.get("submission", {}).get("ref") != REFERENCE
            or evidence.get("source_submission", {}).get("ref") != REFERENCE
            or evidence.get("source_submission", {}).get("url") != source.get("source_url")):
        return None
    entries = [row for row in data.get("requests", []) if row.get("request_checksum") == PROPOSAL]
    if len(entries) != 1:
        return None
    entry = entries[0]
    accounting = entry.get("terminal_accounting", {})
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    if (entry.get("status") != expected or entry.get("reserved_seconds") != 0
            or accounting.get("job_id") != str(REFERENCE)
            or accounting.get("evidence_sha256") != digest
            or accounting.get("conservative_charge_seconds") != 57600):
        return None
    return {"submission_ref": REFERENCE, "terminal_status": terminal,
            "terminal_evidence_sha256": digest, "charged_gpu_seconds": 57600}


def parent_idle(rollout: Path, armed_at: str) -> bool:
    latest = None
    armed = datetime.fromisoformat(armed_at)
    try:
        with rollout.open() as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                payload = record.get("payload", {})
                kind = payload.get("type")
                if record.get("type") == "event_msg" and kind in {"task_started", "task_complete"}:
                    latest = (kind, datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00")))
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return latest is not None and latest[0] == "task_complete" and latest[1] >= armed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    arm = read(STATE / "armed.json")
    rollout = Path(arm["rollout_path"])
    STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (STATE / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            ready = reconciled_terminal(SCORING, LEDGER)
            idle = parent_idle(rollout, arm["armed_at_utc"])
            if args.check:
                print(json.dumps({"terminal_reconciled": ready is not None,
                                  "parent_idle_since_arming": idle, "launched": (STATE / "launch-intent.json").exists()}))
                return
            if (STATE / "launch-intent.json").exists():
                return
            if ready and idle:
                auth = read(Path("/home/ch.jitheswar/.codex/auth.json"))
                if (auth.get("auth_mode") != "chatgpt" or not auth.get("tokens")
                        or auth.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
                        or os.environ.get("CODEX_API_KEY")):
                    raise RuntimeError("continuation requires existing ChatGPT account auth without API-key billing")
                prompt = STATE / "continuation-prompt.txt"
                if hashlib.sha256(prompt.read_bytes()).hexdigest() != arm["prompt_sha256"]:
                    raise RuntimeError("approved continuation prompt changed")
                save_once(STATE / "launch-intent.json", {**ready, "thread_id": arm["thread_id"],
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "continuation_attempts": 1, "automatic_retry": False})
                command = ["/home/ch.jitheswar/.local/bin/codex", "-a", "never", "-s", "danger-full-access",
                           "-C", str(PROJECT), "exec", "resume", arm["thread_id"], "-",
                           "--output-last-message", str(STATE / "last-message.md")]
                with prompt.open() as input_handle, (STATE / "continuation.log").open("a") as log:
                    result = subprocess.run(command, cwd=PROJECT, stdin=input_handle, stdout=log, stderr=log)
                save_once(STATE / "continuation-result.json", {"exit_code": result.returncode,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(), "automatic_retry": False})
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
