"""Observe the accepted submission and conservatively close terminal accounting.

This script cannot launch, resubmit, retry, cancel or select a candidate.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]


def apply_accounting(evidence_path):
    # The repo environment owns scientific dependencies; the SDK environment
    # owns authentication/API dependencies. Neither environment is modified.
    sys.path.insert(0, str(PROJECT / "src"))
    from rsna_knee.gpu_campaign import (
        gpu_campaign_capacity, read_gpu_campaign_ledger,
        record_kaggle_scoring_terminal_without_allocation,
    )
    evidence_path = Path(evidence_path)
    evidence = json.loads(evidence_path.read_text())
    checksum, job_id = evidence["request_checksum"], evidence["job_id"]
    ledger_path = PROJECT / ".scratch/rsna-leaderboard/gpu-campaign-ledger.json"
    entry = next(e for e in read_gpu_campaign_ledger(ledger_path)["requests"]
                 if e["request_checksum"] == checksum)
    if entry["status"] in {"completed", "failed"}:
        recorded = entry.get("terminal_accounting", {})
        if recorded.get("job_id") != job_id or recorded.get("evidence_sha256") != hashlib.sha256(evidence_path.read_bytes()).hexdigest():
            raise RuntimeError("Existing terminal accounting is not bound to this exact evidence")
        return {"ledger_changed": False, "already_reconciled": True,
                "campaign_capacity": gpu_campaign_capacity(ledger_path)}
    result = record_kaggle_scoring_terminal_without_allocation(
        ledger_path, request_checksum=checksum, job_id=job_id,
        evidence_path=evidence_path,
        evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest())
    return {"ledger_changed": True,
            "conservative_gpu_seconds_charged": result["unverified_allocation_charge_seconds"],
            "actual_allocation_known": False,
            "campaign_capacity": gpu_campaign_capacity(ledger_path)}


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    root = PROJECT / "artifacts/private/public-notebook-adoption/scored-baseline-v1"
    state = json.loads((root / "scoring-status.json").read_text())
    request_path = root / "competition-submission-request.json"
    checksum = hashlib.sha256(request_path.read_bytes()).hexdigest()
    if checksum != state["request_checksum"]:
        raise RuntimeError("Submission request changed")
    request = json.loads(request_path.read_text())
    if request["kernel_id"] != state["kernel_id"] or request["kernel_version"] != state["kernel_version"]:
        raise RuntimeError("Submission source identity differs from immutable request")
    api = KaggleApi()
    api.authenticate()
    matches = [s for s in api.competition_submissions(request["competition"], page_size=100)
               if s.ref == state["submission_ref"]]
    if len(matches) != 1:
        raise RuntimeError("Exact submission missing from authoritative listing")
    submission = matches[0]
    if f"/code/{request['kernel_id']}?scriptVersionId={state['script_version_id']}" != submission.url:
        raise RuntimeError("Accepted submission points to a different saved source")
    record = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "submission_ref": submission.ref, "status": submission.status.name,
              "public_score": submission.public_score, "private_score": submission.private_score,
              "error_description": submission.error_description, "source_url": submission.url}
    with (root / "reconciliation-observations.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    if submission.status.name not in {"COMPLETE", "ERROR"}:
        print(json.dumps({**record, "ledger_changed": False}))
        return
    evidence_path = root / "terminal-scoring-evidence.json"
    if not evidence_path.exists():
        normalized = json.loads(submission.to_json())
        normalized["status"] = submission.status.name
        evidence = {"schema_version": "rsna-kaggle-scoring-terminal-evidence-v1",
                    "request_checksum": checksum, "job_id": str(submission.ref),
                    "observed_at_utc": record["observed_at_utc"],
                    "observer": "authenticated KaggleApi.competition_submissions",
                    "kernel_id": request["kernel_id"], "kernel_version": request["kernel_version"],
                    "script_version_id": state["script_version_id"], "submission": normalized,
                    "allocation_timing_available": False,
                    "accounting_basis": "Entire reserved exposure charged; no allocation interval or zero GPU use inferred"}
        with evidence_path.open("x") as handle:
            handle.write(json.dumps(evidence, indent=2) + "\n")
    result = subprocess.run([str(PROJECT / ".venv/bin/python"), str(Path(__file__).resolve()),
                             "--apply-accounting", str(evidence_path)], cwd=PROJECT,
                            capture_output=True, text=True, check=True)
    print(json.dumps({**record, **json.loads(result.stdout)}))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--apply-accounting":
        print(json.dumps(apply_accounting(sys.argv[2])))
    else:
        main()
