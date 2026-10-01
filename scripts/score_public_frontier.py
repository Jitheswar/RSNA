"""Check or submit Ticket 51 once, after its separate budget approval.

Run with the dedicated Kaggle SDK Python. Ledger operations run in the repo
environment. This command never changes the campaign cap, creates approval,
retries an ambiguous submission, or selects a final candidate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "artifacts/private/public-frontier-adoption/scored-baseline-v1"
LEDGER = PROJECT / ".scratch/rsna-leaderboard/gpu-campaign-ledger.json"
PROPOSAL_SHA = "3490b9fcba9381e4778c70f4e2a4597ff660c7eb7e74dc3f8b0f446ea763317c"
HANDLE = "jitheswar/rsna-public-frontier-gpu-preflight-attributed"
VERSION = 3
CAP = 129600
RESERVATION = 57600


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("evidence must be an existing regular file")
    return json.loads(path.read_text())


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    path.chmod(0o600)


def local_evidence(root: Path) -> dict:
    proposal = read(root / "competition-submission-proposal.json")
    if sha(root / "competition-submission-proposal.json") != PROPOSAL_SHA:
        raise ValueError("exact reviewed proposal changed")
    if (proposal["kernel_id"] != HANDLE or proposal["kernel_version"] != VERSION
            or proposal["output_file"] != "submission.csv" or proposal["maximum_attempts"] != 1
            or proposal["reserved_gpu_seconds"] != RESERVATION or proposal["assigned_gpu_count"] != 2
            or proposal["walltime_reservation_seconds"] != 28800
            or proposal["execution_watchdog_seconds"] != 28000
            or proposal["external_spend_usd"] != 0 or proposal["paid_scaling_permitted"] is not False
            or proposal["final_candidate_selection"] is not False):
        raise ValueError("scoring scope differs from the reviewed proposal")
    for filename, key in [("request.json", "saved_request_checksum"),
                          ("validation.json", "saved_validation_sha256"),
                          ("remote-source-validation.json", "saved_source_validation_sha256"),
                          ("visible-run-validation-and-accounting.json", "saved_accounting_sha256"),
                          ("scoring-resource-proposal.json", "resource_proposal_sha256")]:
        if sha(root / filename) != proposal[key]:
            raise ValueError("reviewed local evidence changed")
    validation = read(root / "validation.json")
    resource = read(root / "scoring-resource-proposal.json")
    request = read(root / "request.json")
    if (validation["status"] != "COMPLETE" or validation["kernel_version"] != VERSION
            or validation["independent_prediction_artifacts_verified"] != 30
            or validation["strict_execution"]["failed_fallback_events"] != 0
            or validation["visible_predictions_byte_identical_to_passed_preflight"] is not True
            or sha(root / "output/submission.csv") != validation["submission_sha256"]
            or sha(root / "baseline.ipynb") != request["notebook_sha256"]
            or sha(root / "kernel-metadata.json") != request["metadata_sha256"]):
        raise ValueError("saved notebook or visible execution proof differs")
    remote = read(root / "remote-kernel.json")
    if hashlib.sha256(remote["blob"]["source"].encode()).hexdigest() != resource["remote_source_sha256"]:
        raise ValueError("saved source snapshot changed")
    configuration = read(root / "kernel-metadata.json")
    source_proof = read(root / "remote-source-validation.json")
    expected_metadata = {"ref": HANDLE, "id": source_proof["kernel_numeric_id"],
                         "currentVersionNumber": VERSION, "machineShape": "NvidiaTeslaT4"}
    for remote_key, config_key in [("isPrivate", "is_private"), ("enableGpu", "enable_gpu"),
        ("enableInternet", "enable_internet"), ("enableTpu", "enable_tpu"),
        ("dockerImage", "docker_image"), ("language", "language"), ("kernelType", "kernel_type")]:
        expected_metadata[remote_key] = configuration[config_key]
    if any(remote["metadata"].get(k) != v for k, v in expected_metadata.items()):
        raise ValueError("saved remote metadata does not match immutable configuration")
    for remote_key, config_key in [("datasetDataSources", "dataset_sources"),
        ("kernelDataSources", "kernel_sources"), ("competitionDataSources", "competition_sources"),
        ("modelDataSources", "model_sources")]:
        if sorted(remote["metadata"].get(remote_key, [])) != sorted(configuration[config_key]):
            raise ValueError("saved remote attachments do not match immutable configuration")
    return {"proposal": proposal, "resource": resource, "remote": remote}


def approval_error(path: Path | None, proposal: dict, now: datetime) -> str | None:
    if path is None:
        return "explicit Ticket 51 budget-extension approval has not been recorded"
    try:
        value = read(path)
        expected = {"schema_version": "rsna-ticket51-scoring-approval-v1", "decision": "approved",
                    "proposal_sha256": PROPOSAL_SHA, "resource_proposal_sha256": proposal["resource_proposal_sha256"],
                    "approved_campaign_cap_gpu_seconds": CAP, "approved_request_gpu_seconds": RESERVATION,
                    "kernel_id": HANDLE, "kernel_version": VERSION, "output_file": "submission.csv",
                    "maximum_attempts": 1, "external_spend_usd": 0, "paid_scaling_permitted": False,
                    "final_candidate_selection": False}
        if any(value.get(k) != v for k, v in expected.items()):
            return "approval does not bind the exact scoring proposal and budget extension"
        if not value.get("owner_decision_text", "").strip():
            return "approval must preserve the owner's actual budget decision"
        stamp = datetime.fromisoformat(value["owner_decision_received_at_utc"].replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp > now:
            return "owner decision timestamp is missing a timezone or is in the future"
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return "approval evidence is missing or invalid"
    return None


def validate_remote(local: dict, remote: dict, status: str) -> None:
    expected = local["remote"]["metadata"]
    actual = remote["metadata"]
    for field in ["id", "ref", "currentVersionNumber", "isPrivate", "enableGpu", "enableInternet",
                  "enableTpu", "dockerImage", "machineShape", "language", "kernelType"]:
        if actual.get(field) != expected.get(field):
            raise ValueError("remote saved version or execution settings drifted")
    for field in ["datasetDataSources", "kernelDataSources", "competitionDataSources", "modelDataSources"]:
        if sorted(actual.get(field, [])) != sorted(expected.get(field, [])):
            raise ValueError("remote data attachments drifted")
    source_sha = hashlib.sha256(remote["blob"]["source"].encode()).hexdigest()
    if source_sha != local["resource"]["remote_source_sha256"] or status != "COMPLETE":
        raise ValueError("remote source changed or saved execution is not complete")


def normalized_quota(response) -> dict:
    # SDK to_json uses timedelta.seconds and drops whole days. Never use its
    # serialized duration strings to admit resource use.
    result = {key + "_seconds": getattr(response.gpu_quota, key).total_seconds()
              for key in ["time_used", "time_reserved", "total_time_allowed"]}
    if any(not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError("invalid free quota duration")
    result["is_pay_to_scale_enabled"] = response.gpu_quota.is_pay_to_scale_enabled
    result["quota_refresh_time"] = response.quota_refresh_time.isoformat()
    return result


def blockers(capacity: dict, quota: dict, allowed: int, approval: str | None,
             *, duplicate: bool, other_live_scoring: bool) -> list[str]:
    result = []
    if approval:
        result.append(approval)
    if capacity["budget_seconds"] != CAP:
        result.append("campaign ledger retains its existing cap; approved 36-hour amendment is not applied")
    if (capacity["outstanding_reserved_seconds"] != 0
            or capacity["remaining_seconds"] - capacity["comparison_reserved_seconds"] < RESERVATION):
        result.append("57,600 GPU-second reservation does not fit the available campaign capacity")
    if quota["is_pay_to_scale_enabled"] is not False:
        result.append("paid scaling must be disabled")
    free = quota["total_time_allowed_seconds"] - quota["time_used_seconds"] - quota["time_reserved_seconds"]
    if quota["time_reserved_seconds"] != 0 or free < 28800:
        result.append("free accelerator quota is insufficient or another session has a reservation")
    if allowed < 1:
        result.append("official submission capacity is exhausted")
    if duplicate:
        result.append("this one-attempt source already has submission or intent evidence")
    if other_live_scoring:
        result.append("another competition scoring request is live")
    return result


def ledger_action(action: str, root: Path, approval: Path | None = None, ref: int | None = None) -> dict:
    args = [str(PROJECT / ".venv/bin/python"), str(Path(__file__).resolve()), "--ledger-action", action,
            "--root", str(root)]
    if approval is not None:
        args += ["--approval", str(approval)]
    if ref is not None:
        args += ["--submission-ref", str(ref)]
    result = subprocess.run(args, check=True, capture_output=True, text=True, cwd=PROJECT)
    return json.loads(result.stdout)


def apply_ledger_action(args) -> dict:
    sys.path.insert(0, str(PROJECT / "src"))
    from rsna_knee.gpu_campaign import (begin_gpu_submission_attempt, check_gpu_campaign_capacity,
        gpu_campaign_capacity, read_gpu_campaign_ledger, record_gpu_submission_result,
        record_kaggle_scoring_terminal_without_allocation, reserve_gpu_campaign_request)
    if args.ledger_action == "capacity":
        return dict(gpu_campaign_capacity(LEDGER))
    local = local_evidence(args.root)
    if args.ledger_action == "terminal":
        path = args.root / "terminal-scoring-evidence.json"
        evidence = read(path)
        receipt = read(args.root / "scoring-submit-response.json")
        source = read(args.root / "accepted-scoring-source.json")
        if (evidence.get("request_checksum") != PROPOSAL_SHA
                or evidence.get("job_id") != str(receipt["ref"])
                or evidence.get("kernel_id") != HANDLE or evidence.get("kernel_version") != VERSION
                or source["submission_ref"] != receipt["ref"]
                or source.get("kernel_id") != HANDLE or source.get("kernel_version") != VERSION
                or evidence.get("script_version_id") != source["script_version_id"]
                or evidence["submission"].get("ref") != receipt["ref"]
                or evidence.get("source_submission", {}).get("ref") != receipt["ref"]
                or evidence.get("source_submission", {}).get("url") != source["source_url"]):
            raise ValueError("terminal proof does not bind the accepted request")
        entry = next(row for row in read_gpu_campaign_ledger(LEDGER)["requests"] if row["request_checksum"] == PROPOSAL_SHA)
        if entry["status"] in {"completed", "failed"}:
            recorded = entry.get("terminal_accounting", {})
            if recorded.get("job_id") != evidence["job_id"] or recorded.get("evidence_sha256") != sha(path):
                raise ValueError("existing terminal charge binds different proof")
        else:
            record_kaggle_scoring_terminal_without_allocation(LEDGER,
                request_checksum=PROPOSAL_SHA, job_id=evidence["job_id"], evidence_path=path,
                evidence_sha256=sha(path))
        return dict(gpu_campaign_capacity(LEDGER))
    problem = approval_error(args.approval, local["proposal"], datetime.now(timezone.utc))
    if problem:
        raise ValueError(problem)
    if args.ledger_action == "reserve":
        if gpu_campaign_capacity(LEDGER)["budget_seconds"] != CAP:
            raise ValueError("explicitly approved cap amendment has not been applied to ledger")
        check_gpu_campaign_capacity(LEDGER, request_reserved_seconds=RESERVATION)
        reserve_gpu_campaign_request(LEDGER, request_checksum=PROPOSAL_SHA, experiment_id="ticket51-exact-frontier-scoring",
            wall_time_minutes=480, gpu_count=2, retry_allowed=False, remaining_attempts=1)
        begin_gpu_submission_attempt(LEDGER, request_checksum=PROPOSAL_SHA, attempt_number=1, submitted_epoch=int(time.time()))
    elif args.ledger_action == "accepted":
        if args.submission_ref is None or args.submission_ref <= 0:
            raise ValueError("exact accepted reference required")
        receipt = read(args.root / "scoring-submit-response.json")
        if receipt.get("ref") != args.submission_ref:
            raise ValueError("accepted reference does not match preserved API receipt")
        record_gpu_submission_result(LEDGER, request_checksum=PROPOSAL_SHA, attempt_number=1,
            job_id=str(args.submission_ref), exit_code=0, submission_wait=False)
    return dict(gpu_campaign_capacity(LEDGER))


def submit_once(root: Path, local: dict, approval: Path | None, check: dict, api, ledger) -> int:
    if check["blockers"]:
        raise ValueError("submission refused: " + "; ".join(check["blockers"]))
    now = datetime.now(timezone.utc)
    observed = datetime.fromisoformat(check["observed_at_utc"].replace("Z", "+00:00"))
    if observed.tzinfo is None or not 0 <= (now - observed).total_seconds() <= 120:
        raise ValueError("fresh authenticated preflight required within 120 seconds")
    problem = approval_error(approval, local["proposal"], now)
    if problem:
        raise ValueError(problem)
    # Recheck the immutable local evidence immediately before reservation.
    if local_evidence(root) != local:
        raise ValueError("local evidence changed after preflight")
    if root.joinpath("scoring-submit-intent.json").exists():
        raise ValueError("an intent already exists; reconcile it instead of submitting again")
    save(root / "scoring-submit-intent.json", {"proposal_sha256": PROPOSAL_SHA,
         "approval_sha256": sha(approval), "preflight": check, "network_call_attempts": 1,
         "policy": "Ambiguous API failure retains reservation; never retry automatically."})
    ledger("reserve", root, approval)
    try:
        response = api.competition_submit_code(file_name="submission.csv",
            message="Ticket 51: exact verified public frontier v3; unchanged recipe; zero spend",
            competition=local["proposal"]["competition"], kernel=HANDLE, kernel_version=VERSION, quiet=True)
        record = json.loads(response.to_json(ignore_defaults=False))
        save(root / "scoring-submit-response.json", record)
        if isinstance(response.ref, bool) or not isinstance(response.ref, int) or response.ref <= 0:
            raise ValueError("submission API returned no valid accepted reference")
        ledger("accepted", root, approval, response.ref)
        save(root / "scoring-status.json", {"submission_ref": response.ref, "kernel_id": HANDLE,
             "kernel_version": VERSION, "competition": local["proposal"]["competition"],
             "request_checksum": PROPOSAL_SHA, "status": "ACCEPTED_AWAITING_OBSERVATION",
             "final_candidate_selection": False})
        return response.ref
    except Exception as exc:
        save(root / "scoring-submit-ambiguous-or-local-failure.json", {"exception_type": type(exc).__name__,
             "reservation_released": False, "automatic_retry_permitted": False,
             "instruction": "Read the preserved API response and authoritative exact submission state; never issue another create-code-submission call."})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--ledger-action", choices=["capacity", "reserve", "accepted", "terminal"], help=argparse.SUPPRESS)
    parser.add_argument("--submission-ref", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.ledger_action:
        print(json.dumps(apply_ledger_action(args)))
        return
    local = local_evidence(args.root)
    now = datetime.now(timezone.utc)
    problem = approval_error(args.approval, local["proposal"], now)
    if args.submit and problem:
        raise ValueError(problem)
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiGetAcceleratorQuotaStatisticsRequest
    api = KaggleApi()
    api.authenticate()
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = HANDLE.split("/")
    with api.build_kaggle_client() as client:
        remote = json.loads(client.kernels.kernels_api_client.get_kernel(request).to_json(ignore_defaults=False))
        quota = normalized_quota(client.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest()))
    status = api.kernels_status(HANDLE).status.name
    validate_remote(local, remote, status)
    limits = api.competition_get_submission_limits(local["proposal"]["competition"])
    submissions = api.competition_submissions(local["proposal"]["competition"], page_size=100)
    duplicate = (args.root.joinpath("scoring-submit-intent.json").exists()
                 or any(f"/code/{HANDLE}?" in s.url for s in submissions))
    live = any(s.status.name not in {"COMPLETE", "ERROR", "CANCELLED"} for s in submissions)
    capacity = ledger_action("capacity", args.root)
    check = {"observed_at_utc": now.isoformat(), "proposal_sha256": PROPOSAL_SHA,
             "remote_source_and_settings_exact": True, "remote_status": status, "campaign_capacity": capacity,
             "quota": quota, "official_submissions_allowed_now": limits.num_allowed_now,
             "duplicate_source_or_intent": duplicate, "other_live_scoring": live,
             "blockers": blockers(capacity, quota, limits.num_allowed_now, problem, duplicate=duplicate, other_live_scoring=live),
             "network_mutations": False, "campaign_ledger_changed": False, "approval_created": False}
    audit = args.root / "scoring-preflight-audits" / now.strftime("%Y%m%dT%H%M%S%fZ")
    save(audit / "remote-source.json", remote)
    save(audit / "readiness.json", check)
    if args.submit:
        reference = submit_once(args.root, local, args.approval, check, api, ledger_action)
        print(json.dumps({"accepted_submission_ref": reference, "automatic_retry_permitted": False}))
    else:
        print(json.dumps(check))


if __name__ == "__main__":
    main()
