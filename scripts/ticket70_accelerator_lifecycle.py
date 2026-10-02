"""Admit, launch once and observe the immutable Ticket70 development request.

This lifecycle cannot submit to the competition. Terminal accounting charges
the complete reservation when controller allocation duration is unavailable.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(PROJECT / "src"))
from rsna_knee.archive import atomic_json_write
from rsna_knee.gpu_campaign import (
    _locked_ledger, begin_gpu_submission_attempt, gpu_campaign_capacity,
    read_gpu_campaign_ledger, record_gpu_submission_result,
    reserve_gpu_campaign_request,
)

ROOT = PROJECT / "artifacts/private/ticket70-scored-anchor-gold58-v3"
LEDGER = PROJECT / ".scratch/rsna-leaderboard/gpu-campaign-ledger.json"
TPU_LEDGER = PROJECT / ".scratch/rsna-leaderboard/tpu-campaign-ledger.json"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket70-exact-anchor-20261002.json"
TERMINAL = {"COMPLETE", "ERROR", "CANCELLED", "CANCEL_ACKNOWLEDGED"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def api_client():
    # The SDK-only environment lacks NumPy required by the project package.
    # Use the existing CPU Python with the already installed official SDK.
    sys.path.insert(0, "/home/ch.jitheswar/.venvs/kaggle-cli-2.2.4/lib/python3.11/site-packages")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def typed_quota(api):
    value = api.quota_view()
    result = {"observed_at_utc": now(), "source": "Authenticated typed timedelta fields",
              "quota_refresh_time": value.quota_refresh_time.isoformat(), "accelerators": {}}
    for name in ("gpu", "tpu"):
        q = getattr(value, name + "_quota")
        used, reserved, total = (getattr(q, key).total_seconds() for key in
                                ("time_used", "time_reserved", "total_time_allowed"))
        if any(not math.isfinite(x) or x < 0 for x in (used, reserved, total)):
            raise ValueError("Quota must contain finite nonnegative typed durations")
        result["accelerators"][name] = {
            "time_used_seconds": used, "time_reserved_seconds": reserved,
            "total_time_allowed_seconds": total,
            "free_seconds": total - used - reserved,
            "paid_scaling_enabled": q.is_pay_to_scale_enabled,
        }
    return result


def require_free_quota(quota):
    for name in ("gpu", "tpu"):
        q = quota["accelerators"][name]
        if q["paid_scaling_enabled"] is not False or q["time_reserved_seconds"] != 0:
            raise ValueError("Paid scaling off and both accelerator reservations vacant required")
    if quota["accelerators"]["gpu"]["free_seconds"] < 14400:
        raise ValueError("Insufficient fresh free GPU quota for full two-device bound")


def remote_kernel(api, handle):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    query = ApiGetKernelRequest()
    query.user_name, query.kernel_slug = handle.split("/")
    with api.build_kaggle_client() as client:
        return json.loads(client.kernels.kernels_api_client.get_kernel(query).to_json(ignore_defaults=False))


def require_frozen():
    request = read(ROOT / "request.json")
    preparation = read(ROOT / "preparation-validation.json")
    if (preparation["status"] != "VERIFIED_RUNTIME_ENVELOPE_PREPARATION_VALIDATED"
            or preparation["request_sha256"] != digest(ROOT / "request.json")
            or preparation["scientific_cells_unchanged"] != 5):
        raise ValueError("Independent immutable v3 preparation proof required")
    if digest(AUTH) != request["authorization_sha256"]:
        raise ValueError("Standing owner authorization changed")
    for key, expected in (("assigned_gpu_count", 2), ("session_timeout_seconds", 7200),
                          ("execution_watchdog_seconds", 7140), ("maximum_launch_attempts", 1),
                          ("proposed_reserved_gpu_seconds", 14400), ("external_spend_usd", 0),
                          ("study_count", 58), ("competition_submission", False), ("model_fitting", False)):
        if request[key] != expected:
            raise ValueError("Fixed scientific/resource contract changed: " + key)
    for name, expected in request["frozen_files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    if digest(ROOT / "original-mri-contract.json") != request["original_mri_contract_sha256"]:
        raise ValueError("Original MRI byte contract changed")
    for name, key in (("development.ipynb", "notebook_sha256"), ("kernel-metadata.json", "metadata_sha256")):
        if digest(ROOT / name) != request[key]:
            raise ValueError("Exact notebook/settings changed")
    metadata = read(ROOT / "kernel-metadata.json")
    if (metadata["id"] != request["kernel_id"] or metadata["is_private"] is not True
            or metadata["enable_gpu"] is not True or metadata["enable_tpu"] is not False
            or metadata["enable_internet"] is not False
            or metadata["docker_image"] != request["docker_image"]
            or metadata["docker_image_pinning_type"] != "original"
            or metadata["dataset_sources"] != request["dataset_version_pins"]
            or metadata["competition_sources"] != ["rsna-knee-abnormality-detection"]):
        raise ValueError("Pinned private offline attachments/settings changed")
    return request


def validate_remote(remote, request, accepted, notebook, metadata):
    actual = remote["metadata"]
    if (actual["ref"] != accepted["kernel_id"] or actual["id"] != accepted["kernel_numeric_id"]
            or actual["currentVersionNumber"] != accepted["kernel_version"]
            or actual["isPrivate"] is not True or actual["enableGpu"] is not True
            or actual["enableTpu"] is not False or actual["enableInternet"] is not False
            or actual["machineShape"] != "NvidiaTeslaT4"
            or actual["dockerImage"] != request["docker_image"]):
        raise ValueError("Exact remote privacy/accelerator/container identity differs")
    # GetKernel canonicalizes dataset version refs; the accepted immutable push
    # pins versions, while execution records mounted payload names/sizes/hashes.
    expected_refs = ["/".join(ref.split("/")[:2]) for ref in metadata["dataset_sources"]]
    for field, expected in (("datasetDataSources", expected_refs),
                            ("competitionDataSources", metadata["competition_sources"]),
                            ("modelDataSources", metadata["model_sources"]),
                            ("kernelDataSources", metadata["kernel_sources"])):
        if actual[field] != expected:
            raise ValueError("Exact remote canonical attachment membership/order differs")
    def cells(value):
        return [(cell["cell_type"], "".join(cell["source"])) for cell in value["cells"]]
    if cells(json.loads(remote["blob"]["source"])) != cells(notebook):
        raise ValueError("Remote executable source differs from frozen request")
    return {"status": "PASS", "observed_at_utc": now(),
            "request_sha256": accepted["request_sha256"], "kernel_id": accepted["kernel_id"],
            "kernel_numeric_id": accepted["kernel_numeric_id"], "kernel_version": accepted["kernel_version"],
            "exact_code_cells": True, "pinned_docker": True, "private_internet_off": True,
            "attachments_canonical_refs_verified": True,
            "remote_source_sha256": hashlib.sha256(remote["blob"]["source"].encode()).hexdigest(),
            "numeric_dataset_mount_versions_exposed": False,
            "limitation": "GetKernel does not expose numeric mounted dataset versions or session timeout; immutable accepted push pins these; runtime records actual input/weight bytes"}


def vacancy(api):
    observations, placeholders, token, seen = [], 0, None, set()
    while True:
        response = json.loads(api.kernels_list_with_response(
            mine=True, user="jitheswar", page_size=100, page_token=token).to_json(ignore_defaults=False))
        for item in response["kernels"]:
            handle = item["ref"]
            if not handle:
                if item["id"] != 0 or item["title"] != "[Private Notebook]" or item["author"]:
                    raise ValueError("Unidentifiable owner notebook cannot establish vacancy")
                placeholders += 1  # SDK redaction sentinel, no executable identity.
                continue
            if handle.lower() in seen or not handle.lower().startswith("jitheswar/"):
                raise ValueError("Owner notebook listing is ambiguous")
            seen.add(handle.lower())
            status = json.loads(api.kernels_status(handle).to_json(ignore_defaults=False))
            observations.append({"handle": handle, "status": status["status"]})
            if status["status"] not in TERMINAL:
                raise ValueError("Owner notebook still pending/active: " + handle)
        token = response.get("nextPageToken")
        if not token:
            break
    queue = subprocess.check_output(["squeue", "-u", "ch.jitheswar", "-h", "-o", "%i|%P|%T|%b"], text=True)
    for line in queue.splitlines():
        _, partition, _, gres = line.split("|", 3)
        if partition != "cluster1-cpu" or "gpu" in gres.lower():
            raise ValueError("Another pending/active owner accelerator allocation exists")
    if gpu_campaign_capacity(LEDGER)["outstanding_reserved_seconds"]:
        raise ValueError("Existing GPU campaign reservation remains open")
    tpu = read(TPU_LEDGER)
    if tpu["reserved_seconds"] or any(x["status"] not in TERMINAL for x in tpu["requests"]):
        raise ValueError("Existing TPU request remains pending/active")
    return {"observed_at_utc": now(), "owner_notebooks": observations,
            "redacted_zero_id_sentinels": placeholders, "owner_slurm_queue": queue,
            "gpu_ledger_sha256": digest(LEDGER), "tpu_ledger_sha256": digest(TPU_LEDGER),
            "limitation": "API redaction sentinels expose no executable identity; both typed reservations must also be zero"}


def prepare():
    request = require_frozen()
    if DECISION.exists() or (ROOT / "lifecycle-admission.json").exists():
        raise FileExistsError("Immutable resource admission exists; do not replace it")
    api = api_client()
    quota = typed_quota(api)
    require_free_quota(quota)
    vacant = vacancy(api)
    write(ROOT / "quota-admission.json", quota)
    write(ROOT / "vacancy-admission.json", vacant)
    # Preserve the previous complete-route allowances; add this entire route.
    previous = read(PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket68-biomedclip-heads-20261002.json")
    protected_seconds = previous["projection"]["retained_prior_calendar_seconds"] + previous["projection"]["additional_calendar_seconds"]
    review_seconds = previous["projection"]["retained_review_seconds"]
    additional_seconds = 7200 + 7200 + 3600 + 3600 + 21600 + 7200
    finish = datetime.now(timezone.utc) + timedelta(seconds=protected_seconds + review_seconds + additional_seconds)
    freeze = datetime.fromisoformat(previous["projection"]["science_freeze_utc"])
    if finish >= freeze:
        raise ValueError("Complete route plus review does not fit before science freeze")
    extra_path = "/home/ch.jitheswar/models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218"
    storage = subprocess.check_output(["du", "-sb", str(PROJECT), extra_path], text=True)
    existing = sum(int(line.split()[0]) for line in storage.splitlines())
    allowance = 80_000_000_000
    protected_bytes = previous["storage"]["retained_prior_allowance_bytes"] + previous["storage"]["additional_allowance_bytes"]
    projected = existing + protected_bytes + allowance
    if projected > 2_000_000_000_000 or shutil.disk_usage(ROOT).free < protected_bytes + allowance:
        raise ValueError("Whole-route apparent storage/free-space bound fails")
    capacity = dict(gpu_campaign_capacity(LEDGER))
    if capacity != {"budget_seconds": 136800, "spent_seconds": 129637,
                   "outstanding_reserved_seconds": 0, "comparison_reserved_seconds": 0,
                   "remaining_seconds": 7163}:
        raise ValueError("Unexpected capacity; make a distinct explicit resource decision")
    snapshot = ROOT / "lifecycle-source-snapshot"
    snapshot.mkdir()
    shutil.copyfile(__file__, snapshot / Path(__file__).name)
    shutil.copyfile(PROJECT / "scripts/verify_ticket70_exact_anchor_outputs.py", snapshot / "verify_ticket70_exact_anchor_outputs.py")
    # Bind accounting implementation too; it must not change underneath this run.
    shutil.copyfile(PROJECT / "src/rsna_knee/gpu_campaign.py", snapshot / "gpu_campaign.py")
    frozen = {str(p.relative_to(PROJECT)): digest(p) for p in snapshot.iterdir()}
    decision = {"schema_version": "rsna-ticket70-resource-admission-v1", "decision": "approved",
                "decision_made_by": "campaign agent under standing owner authorization",
                "new_exact_owner_numeric_approval_claimed": False, "created_at_utc": now(),
                "authorization_sha256": digest(AUTH), "request_sha256": digest(ROOT / "request.json"),
                "resources": {"assigned_gpu_count": 2, "session_timeout_seconds": 7200,
                              "watchdog_seconds": 7140, "reserved_gpu_seconds": 14400,
                              "maximum_launch_attempts": 1, "external_spend_usd": 0},
                "forecast_wall_seconds_including_input_hash": request["forecast_with_input_hash_allowance_seconds"],
                "forecast_limitation": "Conservative source scaling plus180s input-hash allowance, not measured complete Gold58 cost",
                "old_campaign_cap_seconds": 136800, "new_campaign_cap_seconds": 151200,
                "capacity_before": capacity, "quota_sha256": digest(ROOT / "quota-admission.json"),
                "vacancy_sha256": digest(ROOT / "vacancy-admission.json"),
                "calendar": {"protected_prior_seconds": protected_seconds, "review_seconds": review_seconds,
                             "additional_route_seconds": additional_seconds,
                             "conservative_finish_utc": finish.isoformat(), "science_freeze_utc": freeze.isoformat()},
                "storage": {"du_output": storage, "existing_apparent_bytes": existing,
                            "protected_allowance_bytes": protected_bytes, "additional_allowance_bytes": allowance,
                            "projected_apparent_bytes": projected}, "frozen_files": frozen,
                "model_fitting": False, "competition_submission": False,
                "official_requirements_refresh_sha256": digest(PROJECT / "docs/research/competition-requirements-refresh-2026-10-02.md"),
                "pinned_model_inventory_sha256": digest(PROJECT / "artifacts/private/ticket70-model-asset-audit-v1/pinned-files.json"),
                "purpose": "Exact five-cell scored-anchor Gold58 development inference and separate fixed comparison/delivery preparation"}
    write(DECISION, decision)
    write(ROOT / "lifecycle-admission.json", {"resource_decision_sha256": digest(DECISION),
          "request_sha256": digest(ROOT / "request.json"), "frozen_files": frozen})
    print(json.dumps({"status": "RESOURCE_ADMITTED", "request_sha256": decision["request_sha256"],
                      "projected_bytes": projected, "complete_route_finish_utc": finish.isoformat()}))


def admission():
    request = require_frozen()
    plan, decision = read(ROOT / "lifecycle-admission.json"), read(DECISION)
    if (digest(DECISION) != plan["resource_decision_sha256"] or decision["decision"] != "approved"
            or plan["request_sha256"] != digest(ROOT / "request.json")
            or decision["request_sha256"] != plan["request_sha256"]):
        raise ValueError("Immutable lifecycle admission changed")
    for name, sha in plan["frozen_files"].items():
        if digest(PROJECT / name) != sha:
            raise ValueError("Frozen lifecycle source changed")
    if digest(__file__) != digest(ROOT / "lifecycle-source-snapshot" / Path(__file__).name):
        raise ValueError("Run the exact admitted lifecycle snapshot")
    if digest(PROJECT / "src/rsna_knee/gpu_campaign.py") != digest(ROOT / "lifecycle-source-snapshot/gpu_campaign.py"):
        raise ValueError("Admitted accounting implementation changed")
    return request, plan, decision


def launch():
    request, plan, decision = admission()
    if (ROOT / "launch-intent.json").exists():
        raise FileExistsError("Consumed launch intent; observe/reconcile the exact handle, never retry")
    api = api_client()
    quota = typed_quota(api)
    require_free_quota(quota)
    vacant = vacancy(api)
    write(ROOT / "quota-before.json", quota)
    write(ROOT / "vacancy-before.json", vacant)
    with _locked_ledger(LEDGER):
        ledger = dict(read_gpu_campaign_ledger(LEDGER))
        if ledger["budget_seconds"] != decision["old_campaign_cap_seconds"]:
            raise ValueError("Campaign cap changed before amendment")
        ledger["budget_seconds"] = decision["new_campaign_cap_seconds"]
        ledger.setdefault("budget_amendments", []).append({"old_budget_seconds": 136800,
                "new_budget_seconds": 151200, "path": str(DECISION), "sha256": digest(DECISION)})
        atomic_json_write(LEDGER, ledger, overwrite=True)
    checksum = digest(ROOT / "request.json")
    reserve_gpu_campaign_request(LEDGER, request_checksum=checksum,
        experiment_id="ticket70-exact-scored-anchor-gold58-v3", wall_time_minutes=120,
        retry_allowed=False, remaining_attempts=1, gpu_count=2)
    write(ROOT / "launch-intent.json", {"observed_at_utc": now(), "request_sha256": checksum,
        "kernel_id": request["kernel_id"], "resource_decision_sha256": digest(DECISION),
        "quota_before_sha256": digest(ROOT / "quota-before.json"), "gpu_ledger_sha256": digest(LEDGER)})
    begin_gpu_submission_attempt(LEDGER, request_checksum=checksum, attempt_number=1, submitted_epoch=int(time.time()))
    try:
        response = json.loads(api.kernels_push(str(ROOT), timeout="7200", acc="NvidiaTeslaT4").to_json(ignore_defaults=False))
    except BaseException as exc:
        write(ROOT / "launch-exception.json", {"type": type(exc).__name__, "message": str(exc), "retry_allowed": False})
        raise
    write(ROOT / "push-response.json", response)
    handle = response["ref"].removeprefix("/code/")
    if (response.get("error") or handle != request["kernel_id"] or response["versionNumber"] != 1
            or any(response.get(key) for key in ("invalidDatasetSources", "invalidModelSources", "invalidCompetitionSources", "invalidKernelSources"))):
        raise RuntimeError("Unexpected push identity; preserve reserve and reconcile without another push")
    write(ROOT / "accepted-launch.json", {"kernel_id": handle, "kernel_numeric_id": response["kernelId"],
        "kernel_version": 1, "launch_attempt": 1, "request_sha256": checksum, "response_sha256": digest(ROOT / "push-response.json"),
        "accepted_at_utc": now()})
    record_gpu_submission_result(LEDGER, request_checksum=checksum, attempt_number=1,
                                 job_id=handle, exit_code=0, submission_wait=False)
    print(json.dumps({"status": "ACCEPTED", "kernel_id": handle, "version": 1}))


def observe(api):
    request, _, _ = admission()
    accepted = read(ROOT / "accepted-launch.json")
    if accepted["request_sha256"] != digest(ROOT / "request.json"):
        raise ValueError("Accepted request checksum changed")
    status = json.loads(api.kernels_status(accepted["kernel_id"]).to_json(ignore_defaults=False))
    remote = remote_kernel(api, accepted["kernel_id"])
    metadata = remote["metadata"]
    if (metadata["id"] != accepted["kernel_numeric_id"] or metadata["ref"] != accepted["kernel_id"]
            or metadata["currentVersionNumber"] != accepted["kernel_version"]):
        raise ValueError("Exact remote numeric identity/version changed; never retry")
    result = {"observed_at_utc": now(), "kernel_id": accepted["kernel_id"],
              "kernel_numeric_id": accepted["kernel_numeric_id"], "launch_attempt": 1,
              "request_sha256": accepted["request_sha256"],
              "kernel_version": accepted["kernel_version"], "status": status["status"], "response": status}
    try:
        source = validate_remote(remote, request, accepted, read(ROOT / "development.ipynb"), read(ROOT / "kernel-metadata.json"))
        result["source_status"] = "PASS"
        atomic_json_write(ROOT / "remote-source-validation.json", source, overwrite=True)
    except ValueError as exc:
        result["source_status"] = "REJECTED"
        result["source_error"] = str(exc)
    atomic_json_write(ROOT / "kernel-status.json", result, overwrite=True)
    if result["status"] in TERMINAL and not (ROOT / "kernel-terminal.json").exists():
        write(ROOT / "kernel-terminal.json", result)
        write(ROOT / "remote-terminal.json", remote)
    if result["status"] in TERMINAL and not (ROOT / "quota-after.json").exists():
        write(ROOT / "quota-after.json", typed_quota(api))
    return result


def close_terminal():
    admission()
    accepted, terminal = read(ROOT / "accepted-launch.json"), read(ROOT / "kernel-terminal.json")
    if terminal["status"] not in TERMINAL or terminal["kernel_id"] != accepted["kernel_id"] or terminal["kernel_version"] != 1:
        raise ValueError("Matching actual terminal observation required before closure")
    checksum = digest(ROOT / "request.json")
    with _locked_ledger(LEDGER):
        ledger = dict(read_gpu_campaign_ledger(LEDGER))
        entries = [x for x in ledger["requests"] if x["request_checksum"] == checksum]
        if len(entries) != 1:
            raise ValueError("Unique exact reservation required")
        entry = entries[0]
        if entry["status"] in {"completed", "failed"}:
            if entry["terminal_accounting"]["evidence_sha256"] != digest(ROOT / "kernel-terminal.json"):
                raise ValueError("Already closed reservation has different terminal proof")
            return
        if entry["reserved_seconds"] != 14400 or len(entry["attempts"]) != 1 or entry["attempts"][0]["job_id"] != accepted["kernel_id"]:
            raise ValueError("Exact active one-attempt reservation differs")
        status = "completed" if terminal["status"] == "COMPLETE" else "failed"
        proof = {"basis": "Full reservation conservatively charged; notebook elapsed does not prove controller allocation duration",
                 "conservative_charge_seconds": 14400, "evidence_path": str(ROOT / "kernel-terminal.json"),
                 "evidence_sha256": digest(ROOT / "kernel-terminal.json"), "recorded_at_utc": now()}
        entry.update(status=status, reserved_seconds=0, actual_allocation_known=False,
                     unverified_allocation_charge_seconds=14400, terminal_accounting=proof)
        entry["attempts"][0].update(status=status, allocated_seconds=None,
              terminal_backend_status=terminal["status"], conservative_exposure_charge_seconds=14400)
        atomic_json_write(LEDGER, ledger, overwrite=True)


def watch():
    api = api_client()
    while True:
        try:
            result = observe(api)
            if result["status"] in TERMINAL:
                close_terminal()
                print(json.dumps(result), flush=True)
                return
        except Exception as exc:
            result = {"observed_at_utc": now(), "observation_error": str(exc), "retry_launch_allowed": False}
        with (ROOT / "kernel-observations.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch", "observe", "watch", "close"))
    command = parser.parse_args().command
    if command == "observe":
        print(json.dumps(observe(api_client())))
    else:
        {"prepare": prepare, "launch": launch, "watch": watch, "close": close_terminal}[command]()
