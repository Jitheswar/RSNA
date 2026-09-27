"""Shared first-campaign GPU budget and allocation accounting."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping, cast

from .archive import atomic_json_write


GPU_CAMPAIGN_LEDGER_SCHEMA_VERSION = "rsna-gpu-campaign-ledger-v1"
GPU_CAMPAIGN_BUDGET_HOURS = 20.0
ACTIVE_REQUEST_STATUSES = frozenset({"submitting", "submitted", "running", "unknown"})
TERMINAL_REQUEST_STATUSES = frozenset({"completed", "failed"})
EXTERNALLY_BLOCKED_REQUEST_STATUSES = frozenset({"externally_blocked"})


class GpuCampaignError(RuntimeError):
    """Base error for shared GPU campaign accounting."""


class GpuCampaignBudgetError(GpuCampaignError):
    """Raised when a request would spend capacity reserved for comparisons."""


class GpuCampaignActiveRequestError(GpuCampaignError):
    """Raised when another GPU request is already pending or active."""


class GpuCampaignLedgerError(GpuCampaignError):
    """Raised when campaign accounting evidence is missing or inconsistent."""


def _hours_to_seconds(hours: float, field_name: str) -> int:
    if isinstance(hours, bool) or not isinstance(hours, (int, float)):
        raise GpuCampaignBudgetError(f"{field_name} must be a finite number of hours")
    if not math.isfinite(float(hours)) or hours < 0:
        raise GpuCampaignBudgetError(f"{field_name} must be a finite non-negative number of hours")
    return int(round(float(hours) * 3600))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _stored_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise GpuCampaignLedgerError(f"GPU campaign ledger has an invalid {field_name}")
    return value


@contextmanager
def _locked_ledger(path: Path) -> Iterator[None]:
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_ledger(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise GpuCampaignLedgerError(f"GPU campaign ledger does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GpuCampaignLedgerError(f"could not read GPU campaign ledger: {path}") from exc
    if not isinstance(value, Mapping):
        raise GpuCampaignLedgerError("GPU campaign ledger must contain a JSON object")
    payload = dict(cast(Mapping[str, object], value))
    if payload.get("schema_version") != GPU_CAMPAIGN_LEDGER_SCHEMA_VERSION:
        raise GpuCampaignLedgerError("GPU campaign ledger has an unexpected schema")
    budget_seconds = payload.get("budget_seconds")
    reserve_seconds = payload.get("reserved_comparison_seconds")
    if not isinstance(budget_seconds, int) or isinstance(budget_seconds, bool):
        raise GpuCampaignLedgerError("GPU campaign ledger has an invalid budget")
    if (
        not isinstance(reserve_seconds, int)
        or isinstance(reserve_seconds, bool)
        or reserve_seconds < 0
        or reserve_seconds > budget_seconds
    ):
        raise GpuCampaignLedgerError("GPU campaign ledger has an invalid comparison reserve")
    initial_reserve = payload.setdefault(
        "initial_reserved_comparison_seconds", reserve_seconds
    )
    if (
        not isinstance(initial_reserve, int)
        or isinstance(initial_reserve, bool)
        or initial_reserve < reserve_seconds
    ):
        raise GpuCampaignLedgerError("GPU campaign ledger has an invalid initial comparison reserve")
    requests = payload.get("requests")
    if not isinstance(requests, list) or any(not isinstance(item, Mapping) for item in requests):
        raise GpuCampaignLedgerError("GPU campaign ledger has invalid request records")
    for entry in requests:
        if entry.get("status") not in (
            ACTIVE_REQUEST_STATUSES
            | TERMINAL_REQUEST_STATUSES
            | EXTERNALLY_BLOCKED_REQUEST_STATUSES
        ):
            raise GpuCampaignLedgerError("GPU campaign ledger has an invalid request status")
        if not isinstance(entry.get("attempts"), list):
            raise GpuCampaignLedgerError("GPU campaign ledger has invalid attempts")
        if not isinstance(entry.get("actual_allocated_seconds"), int):
            raise GpuCampaignLedgerError("GPU campaign ledger has invalid actual allocation time")
        if not isinstance(entry.get("reserved_seconds"), int):
            raise GpuCampaignLedgerError("GPU campaign ledger has invalid reserved allocation time")
        for field_name in ("gpu_count", "unverified_allocation_charge_seconds"):
            field_value = entry.get(field_name, 1 if field_name == "gpu_count" else 0)
            if (
                not isinstance(field_value, int)
                or isinstance(field_value, bool)
                or field_value < 0
                or (field_name == "gpu_count" and field_value < 1)
            ):
                raise GpuCampaignLedgerError(f"GPU campaign ledger has invalid {field_name}")
        release_seconds = entry.get("comparison_reserve_release_seconds", 0)
        if not isinstance(release_seconds, int) or isinstance(release_seconds, bool):
            raise GpuCampaignLedgerError("GPU campaign ledger has an invalid comparison reserve release")
        if not isinstance(entry.get("comparison_reserve_release_active", False), bool):
            raise GpuCampaignLedgerError("GPU campaign ledger has an invalid comparison reserve release state")
    return payload


def initialize_gpu_campaign_ledger(
    path: str | Path,
    *,
    reserved_comparison_hours: float,
    budget_hours: float = GPU_CAMPAIGN_BUDGET_HOURS,
) -> Mapping[str, object]:
    """Create the shared ledger once and require later requests to retain its budget policy."""

    destination = Path(path).resolve()
    budget_seconds = _hours_to_seconds(budget_hours, "campaign budget")
    reserve_seconds = _hours_to_seconds(reserved_comparison_hours, "comparison reserve")
    if budget_seconds != int(GPU_CAMPAIGN_BUDGET_HOURS * 3600):
        raise GpuCampaignBudgetError("the first campaign GPU budget is fixed at 20 hours")
    if reserve_seconds > budget_seconds:
        raise GpuCampaignBudgetError("comparison reserve cannot exceed the campaign budget")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _locked_ledger(destination):
        if destination.exists():
            payload = _read_ledger(destination)
            if payload["budget_seconds"] != budget_seconds:
                raise GpuCampaignBudgetError("campaign ledger budget differs from this request")
            if payload["initial_reserved_comparison_seconds"] != reserve_seconds:
                raise GpuCampaignBudgetError("campaign ledger comparison reserve differs from this request")
            return payload
        new_payload: dict[str, object] = {
            "schema_version": GPU_CAMPAIGN_LEDGER_SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "budget_seconds": budget_seconds,
            "initial_reserved_comparison_seconds": reserve_seconds,
            "reserved_comparison_seconds": reserve_seconds,
            "requests": [],
        }
        atomic_json_write(destination, new_payload)
        return new_payload


def read_gpu_campaign_ledger(path: str | Path) -> Mapping[str, object]:
    """Return validated campaign accounting evidence."""

    return _read_ledger(Path(path).resolve())


def gpu_campaign_capacity(path: str | Path) -> Mapping[str, int]:
    """Summarize spent, outstanding and still-available GPU seconds."""

    ledger = _read_ledger(Path(path).resolve())
    requests = cast(list[Mapping[str, object]], ledger["requests"])
    spent = sum(
        _stored_int(entry["actual_allocated_seconds"], "actual allocation time")
        + _stored_int(
            entry.get("unverified_allocation_charge_seconds", 0),
            "unverified allocation charge",
        )
        for entry in requests
    )
    outstanding = sum(
        _stored_int(entry["reserved_seconds"], "reserved allocation time")
        for entry in requests
        if entry["status"] in ACTIVE_REQUEST_STATUSES
    )
    budget = int(cast(int, ledger["budget_seconds"]))
    return {
        "budget_seconds": budget,
        "spent_seconds": spent,
        "outstanding_reserved_seconds": outstanding,
        "comparison_reserved_seconds": int(cast(int, ledger["reserved_comparison_seconds"])),
        "remaining_seconds": budget - spent - outstanding,
    }


def check_gpu_campaign_capacity(
    path: str | Path,
    *,
    request_reserved_seconds: int,
    comparison_reserve_release_seconds: int = 0,
) -> Mapping[str, int]:
    """Check a request while protecting comparison capacity not assigned to it."""

    if not isinstance(request_reserved_seconds, int) or isinstance(request_reserved_seconds, bool):
        raise GpuCampaignBudgetError("request reservation must be an integer number of seconds")
    if request_reserved_seconds <= 0:
        raise GpuCampaignBudgetError("request reservation must be positive")
    if (
        not isinstance(comparison_reserve_release_seconds, int)
        or isinstance(comparison_reserve_release_seconds, bool)
        or comparison_reserve_release_seconds < 0
    ):
        raise GpuCampaignBudgetError("comparison reserve release must be a non-negative integer")
    if comparison_reserve_release_seconds > request_reserved_seconds:
        raise GpuCampaignBudgetError("comparison reserve release cannot exceed the request reservation")
    capacity = gpu_campaign_capacity(path)
    if comparison_reserve_release_seconds > capacity["comparison_reserved_seconds"]:
        raise GpuCampaignBudgetError("comparison reserve release exceeds the protected reserve")
    _require_campaign_request_capacity(
        request_reserved_seconds,
        capacity["comparison_reserved_seconds"] - comparison_reserve_release_seconds,
        capacity["remaining_seconds"],
    )
    return capacity


def reserve_gpu_campaign_request(
    path: str | Path,
    *,
    request_checksum: str,
    experiment_id: str,
    wall_time_minutes: int,
    retry_allowed: bool,
    remaining_attempts: int,
    comparison_reserve_release_seconds: int = 0,
    gpu_count: int = 1,
) -> Mapping[str, object]:
    """Reserve budget for one approved request and reject every overlapping request."""

    destination = Path(path).resolve()
    max_attempts = 2 if retry_allowed else 1
    if remaining_attempts not in (1, 2) or remaining_attempts > max_attempts:
        raise GpuCampaignBudgetError("GPU campaign attempt count does not match retry authorization")
    if (
        not isinstance(gpu_count, int)
        or isinstance(gpu_count, bool)
        or gpu_count < 1
        or gpu_count > 2
    ):
        raise GpuCampaignBudgetError("GPU campaign GPU count must be one or two")
    if (
        not isinstance(comparison_reserve_release_seconds, int)
        or isinstance(comparison_reserve_release_seconds, bool)
        or comparison_reserve_release_seconds < 0
    ):
        raise GpuCampaignBudgetError("comparison reserve release must be a non-negative integer")
    wall_seconds = int(wall_time_minutes) * 60
    gpu_seconds_per_attempt = wall_seconds * gpu_count
    if comparison_reserve_release_seconds > wall_seconds:
        raise GpuCampaignBudgetError("comparison reserve release cannot exceed one fold's request limit")
    with _locked_ledger(destination):
        ledger = _read_ledger(destination)
        requests = [dict(cast(Mapping[str, object], item)) for item in cast(list[object], ledger["requests"])]
        active = [entry for entry in requests if entry["status"] in ACTIVE_REQUEST_STATUSES]
        if active:
            raise GpuCampaignActiveRequestError(
                "the campaign already has an active or pending GPU request"
            )
        previous = next(
            (entry for entry in requests if entry.get("request_checksum") == request_checksum),
            None,
        )
        if previous is not None:
            attempts = cast(list[Mapping[str, object]], previous["attempts"])
            if previous["status"] != "failed" or len(attempts) >= max_attempts:
                raise GpuCampaignActiveRequestError(
                    "this GPU request has already been submitted or consumed"
                )
            if _stored_int(previous.get("wall_time_seconds"), "retry wall time") != wall_seconds:
                raise GpuCampaignLedgerError("GPU retry wall time changed for the same request")
            remaining = max_attempts - len(attempts)
            if remaining_attempts != remaining:
                raise GpuCampaignLedgerError("GPU retry count differs from the campaign ledger")
            if _stored_int(previous.get("gpu_count", 1), "GPU count") != gpu_count:
                raise GpuCampaignLedgerError("GPU retry device count changed for the same request")
            max_reservation = remaining * gpu_seconds_per_attempt
            capacity = gpu_campaign_capacity(destination)
            prior_release = _stored_int(
                previous.get("comparison_reserve_release_seconds", 0),
                "comparison reserve release",
            )
            if prior_release != comparison_reserve_release_seconds:
                raise GpuCampaignLedgerError(
                    "GPU retry comparison reserve release changed for the same request"
                )
            release_active = previous.get("comparison_reserve_release_active", False) is True
            if not release_active and prior_release > capacity["comparison_reserved_seconds"]:
                raise GpuCampaignBudgetError("comparison reserve release exceeds the protected reserve")
            protected_reserve = capacity["comparison_reserved_seconds"] - (
                0 if release_active else prior_release
            )
            _require_campaign_request_capacity(
                max_reservation, protected_reserve, capacity["remaining_seconds"]
            )
            previous["status"] = "submitting"
            previous["reserved_seconds"] = max_reservation
            previous["max_attempts"] = max_attempts
            if prior_release and not release_active:
                ledger["reserved_comparison_seconds"] = protected_reserve
                previous["comparison_reserve_release_active"] = True
            ledger["requests"] = requests
            atomic_json_write(destination, ledger, overwrite=True)
            return previous
        max_reservation = remaining_attempts * gpu_seconds_per_attempt
        capacity = gpu_campaign_capacity(destination)
        if comparison_reserve_release_seconds > capacity["comparison_reserved_seconds"]:
            raise GpuCampaignBudgetError("comparison reserve release exceeds the protected reserve")
        protected_reserve = (
            capacity["comparison_reserved_seconds"] - comparison_reserve_release_seconds
        )
        _require_campaign_request_capacity(
            max_reservation, protected_reserve, capacity["remaining_seconds"]
        )
        entry: dict[str, object] = {
            "request_checksum": request_checksum,
            "experiment_id": experiment_id,
            "status": "submitting",
            "created_at_utc": _utc_now(),
            "retry_allowed": retry_allowed,
            "max_attempts": max_attempts,
            "wall_time_seconds": wall_seconds,
            "gpu_count": gpu_count,
            "reserved_seconds": max_reservation,
            "comparison_reserve_release_seconds": comparison_reserve_release_seconds,
            "comparison_reserve_release_active": comparison_reserve_release_seconds > 0,
            "actual_allocated_seconds": 0,
            "unverified_allocation_charge_seconds": 0,
            "actual_allocation_known": True,
            "queue_wait_seconds": 0,
            "attempts": [],
        }
        ledger["reserved_comparison_seconds"] = protected_reserve
        requests.append(entry)
        ledger["requests"] = requests
        atomic_json_write(destination, ledger, overwrite=True)
        return entry


def begin_gpu_submission_attempt(
    path: str | Path,
    *,
    request_checksum: str,
    attempt_number: int,
    submitted_epoch: int,
) -> None:
    destination = Path(path).resolve()
    with _locked_ledger(destination):
        ledger = _read_ledger(destination)
        entry = _request_entry(ledger, request_checksum)
        attempts = cast(list[dict[str, object]], entry["attempts"])
        if entry["status"] not in ACTIVE_REQUEST_STATUSES:
            raise GpuCampaignLedgerError("GPU request is not reserved for submission")
        if attempt_number != len(attempts) + 1:
            raise GpuCampaignLedgerError("GPU submission attempt number is not sequential")
        if attempt_number > _stored_int(entry["max_attempts"], "maximum attempts"):
            raise GpuCampaignLedgerError("GPU request exceeded its approved attempt count")
        attempts.append(
            {
                "attempt_number": attempt_number,
                "submitted_epoch": submitted_epoch,
                "job_id": None,
                "queue_wait_seconds": None,
                "allocated_seconds": None,
                "exit_code": None,
                "status": "submitting",
            }
        )
        atomic_json_write(destination, ledger, overwrite=True)


def record_gpu_submission_result(
    path: str | Path,
    *,
    request_checksum: str,
    attempt_number: int,
    job_id: str | None,
    exit_code: int,
    submission_wait: bool,
) -> None:
    destination = Path(path).resolve()
    with _locked_ledger(destination):
        ledger = _read_ledger(destination)
        entry = _request_entry(ledger, request_checksum)
        attempt = _attempt_entry(entry, attempt_number)
        if job_id is None:
            attempt.update(
                {
                    "exit_code": exit_code,
                    "queue_wait_seconds": 0,
                    "allocated_seconds": 0,
                    "status": "not_allocated",
                }
            )
            _refresh_request_totals(entry)
            _refresh_request_reservation(entry)
        else:
            if attempt.get("job_id") not in (None, job_id):
                raise GpuCampaignLedgerError("Slurm job ID changed for a submission attempt")
            attempt["job_id"] = job_id
            attempt["submission_exit_code"] = exit_code
            if submission_wait and attempt.get("allocated_seconds") is None:
                # The job ended but did not write its allocation trap record.
                # Charge the full approved wall limit and keep the campaign
                # blocked until its accounting can be reconciled.
                attempt["allocated_seconds"] = _stored_int(entry["wall_time_seconds"], "wall time")
                attempt["queue_wait_seconds"] = None
                attempt["allocation_timing_missing"] = True
                attempt["allocation_confirmed"] = True
                attempt["status"] = "accounting_unavailable"
                _refresh_request_totals(entry)
                _refresh_request_reservation(entry)
            if attempt.get("status") == "submitting":
                attempt["status"] = "submitted"
        atomic_json_write(destination, ledger, overwrite=True)


def record_gpu_allocation(
    path: str | Path,
    *,
    request_checksum: str,
    attempt_number: int,
    job_id: str,
    allocated_start_epoch: int,
    allocated_end_epoch: int,
    exit_code: int,
    submission_wait: bool,
) -> Mapping[str, object]:
    """Record queue delay and allocated wall time from a Slurm job's exit trap."""

    destination = Path(path).resolve()
    if allocated_end_epoch < allocated_start_epoch:
        raise GpuCampaignLedgerError("Slurm allocation end precedes its start")
    with _locked_ledger(destination):
        ledger = _read_ledger(destination)
        entry = _request_entry(ledger, request_checksum)
        attempt = _attempt_entry(entry, attempt_number)
        expected_job_id = attempt.get("job_id")
        if expected_job_id not in (None, job_id):
            raise GpuCampaignLedgerError("allocation report names a different Slurm job")
        start = int(allocated_start_epoch)
        end = int(allocated_end_epoch)
        submitted = int(cast(int, attempt["submitted_epoch"]))
        attempt.update(
            {
                "job_id": job_id,
                "queue_wait_seconds": max(0, start - submitted),
                "allocated_start_epoch": start,
                "allocated_end_epoch": end,
                "allocated_seconds": end - start,
                "allocation_confirmed": True,
                "exit_code": int(exit_code),
                "status": "completed" if exit_code == 0 else "failed",
            }
        )
        _refresh_request_totals(entry)
        _refresh_request_reservation(entry)
        if not submission_wait:
            if exit_code == 0:
                entry["status"] = "completed"
            else:
                entry["status"] = "failed"
            entry["reserved_seconds"] = 0
        elif entry["status"] not in TERMINAL_REQUEST_STATUSES:
            entry["status"] = "submitting"
        atomic_json_write(destination, ledger, overwrite=True)
        return dict(entry)


def record_gpu_backend_allocation(
    path: str | Path,
    *,
    request_checksum: str,
    attempt_number: int,
    backend_run_id: str,
    allocated_start_epoch: int,
    allocated_end_epoch: int,
    exit_code: int,
    evidence_path: str | Path,
    evidence_sha256: str,
) -> Mapping[str, object]:
    """Record measured wall time for an approved non-Slurm GPU backend."""

    destination = Path(path).resolve()
    evidence = Path(evidence_path).resolve()
    if not evidence.is_file() or evidence.is_symlink():
        raise GpuCampaignLedgerError("external GPU allocation evidence is unavailable")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    if digest != evidence_sha256 or len(digest) != 64:
        raise GpuCampaignLedgerError("external GPU allocation evidence checksum is invalid")
    try:
        evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise GpuCampaignLedgerError("external GPU allocation evidence is unreadable") from exc
    scope_payload = evidence_payload.get("scope") if isinstance(evidence_payload, Mapping) else None
    if (
        not isinstance(evidence_payload, Mapping)
        or evidence_payload.get("schema_version") != "rsna-ticket33-kaggle-gpu-evidence-v1"
        or evidence_payload.get("status") != "completed"
        or evidence_payload.get("request_checksum") != request_checksum
        or evidence_payload.get("kernel_id") != backend_run_id
        or not isinstance(scope_payload, Mapping)
        or scope_payload.get("external_spend_usd") != 0
    ):
        raise GpuCampaignLedgerError("external GPU evidence does not establish this Kaggle run")
    if not backend_run_id.strip() or allocated_end_epoch < allocated_start_epoch:
        raise GpuCampaignLedgerError("external GPU allocation timestamps or run identity are invalid")
    wall_seconds = int(allocated_end_epoch) - int(allocated_start_epoch)
    if wall_seconds <= 0:
        raise GpuCampaignLedgerError("external GPU allocation duration must be positive")
    with _locked_ledger(destination):
        ledger = _read_ledger(destination)
        entry = _request_entry(ledger, request_checksum)
        if entry["status"] not in ACTIVE_REQUEST_STATUSES:
            raise GpuCampaignLedgerError("external GPU request is not active")
        attempt = _attempt_entry(entry, attempt_number)
        if attempt.get("status") not in {"submitting", "submitted"}:
            raise GpuCampaignLedgerError("external GPU attempt is not awaiting allocation evidence")
        gpu_count = _stored_int(entry.get("gpu_count", 1), "GPU count")
        allocated_gpu_seconds = wall_seconds * gpu_count
        if allocated_gpu_seconds > _stored_int(entry["reserved_seconds"], "reserved allocation time"):
            raise GpuCampaignLedgerError("external GPU allocation exceeded its reserved budget")
        submitted_epoch = attempt.get("submitted_epoch")
        queue_wait_seconds = (
            max(0, int(allocated_start_epoch) - submitted_epoch)
            if isinstance(submitted_epoch, int) and not isinstance(submitted_epoch, bool)
            else None
        )
        attempt.update(
            {
                "backend_run_id": backend_run_id,
                "queue_wait_seconds": queue_wait_seconds,
                "allocated_wall_seconds": wall_seconds,
                "allocated_seconds": allocated_gpu_seconds,
                "allocation_confirmed": True,
                "allocation_evidence_path": str(evidence),
                "allocation_evidence_sha256": evidence_sha256,
                "exit_code": int(exit_code),
                "status": "completed" if exit_code == 0 else "failed",
            }
        )
        _refresh_request_totals(entry)
        _refresh_request_reservation(entry)
        if exit_code == 0:
            entry["status"] = "completed"
        else:
            entry["status"] = "failed"
        entry["reserved_seconds"] = 0
        atomic_json_write(destination, ledger, overwrite=True)
        return dict(entry)


def record_gpu_backend_failure_allocation(
    path: str | Path,
    *,
    request_checksum: str,
    attempt_number: int,
    backend_run_id: str,
    quota_used_wall_seconds: float,
    exit_code: int,
    evidence_path: str | Path,
    evidence_sha256: str,
) -> Mapping[str, object]:
    """Close a failed external GPU run with a conservative quota-backed charge."""

    if (
        not backend_run_id.strip()
        or exit_code == 0
        or isinstance(quota_used_wall_seconds, bool)
        or not isinstance(quota_used_wall_seconds, (int, float))
        or not math.isfinite(float(quota_used_wall_seconds))
        or quota_used_wall_seconds <= 0
    ):
        raise GpuCampaignLedgerError("failed external allocation evidence is invalid")
    destination = Path(path).resolve()
    evidence = Path(evidence_path).resolve()
    if not evidence.is_file() or evidence.is_symlink():
        raise GpuCampaignLedgerError("external GPU failure evidence is unavailable")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    if digest != evidence_sha256 or len(digest) != 64:
        raise GpuCampaignLedgerError("external GPU failure evidence checksum is invalid")
    try:
        evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise GpuCampaignLedgerError("external GPU failure evidence is unreadable") from exc
    if (
        not isinstance(evidence_payload, Mapping)
        or evidence_payload.get("schema_version") != "rsna-ticket33-kaggle-gpu-failure-v1"
        or evidence_payload.get("status") != "failed"
        or evidence_payload.get("request_checksum") != request_checksum
        or evidence_payload.get("kernel_id") != backend_run_id
        or evidence_payload.get("quota_used_wall_seconds") != quota_used_wall_seconds
    ):
        raise GpuCampaignLedgerError("external GPU failure evidence does not establish this Kaggle run")
    wall_seconds = math.ceil(float(quota_used_wall_seconds))
    with _locked_ledger(destination):
        ledger = _read_ledger(destination)
        entry = _request_entry(ledger, request_checksum)
        if entry["status"] not in ACTIVE_REQUEST_STATUSES:
            raise GpuCampaignLedgerError("external GPU request is not active")
        attempt = _attempt_entry(entry, attempt_number)
        if attempt.get("status") not in {"submitting", "submitted", "running"}:
            raise GpuCampaignLedgerError("external GPU attempt is not awaiting failure evidence")
        gpu_count = _stored_int(entry.get("gpu_count", 1), "GPU count")
        allocated_gpu_seconds = wall_seconds * gpu_count
        if allocated_gpu_seconds > _stored_int(entry["reserved_seconds"], "reserved allocation time"):
            raise GpuCampaignLedgerError("external GPU failure exceeded its reserved budget")
        attempt.update(
            {
                "backend_run_id": backend_run_id,
                "queue_wait_seconds": None,
                "allocated_wall_seconds": wall_seconds,
                "allocated_seconds": allocated_gpu_seconds,
                "allocation_confirmed": True,
                "allocation_evidence_path": str(evidence),
                "allocation_evidence_sha256": evidence_sha256,
                "quota_used_wall_seconds": float(quota_used_wall_seconds),
                "exit_code": int(exit_code),
                "status": "failed",
            }
        )
        _refresh_request_totals(entry)
        _refresh_request_reservation(entry)
        entry["status"] = "failed"
        entry["reserved_seconds"] = 0
        _close_terminal_request(ledger, entry, "failed")
        atomic_json_write(destination, ledger, overwrite=True)
        return dict(entry)


def mark_gpu_campaign_request_externally_blocked(
    path: str | Path,
    *,
    request_checksum: str,
    slurm_job_id: str,
    evidence_path: str | Path,
    evidence_sha256: str,
    conservative_charge_seconds: int,
) -> Mapping[str, object]:
    """Close a canceled Slurm request while retaining its full unknown exposure charge."""

    destination = Path(path).resolve()
    evidence = Path(evidence_path).resolve()
    if not evidence.is_file() or evidence.is_symlink():
        raise GpuCampaignLedgerError("external blocker evidence is unavailable")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    if digest != evidence_sha256 or len(digest) != 64:
        raise GpuCampaignLedgerError("external blocker evidence checksum is invalid")
    try:
        evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise GpuCampaignLedgerError("external blocker evidence is unreadable") from exc
    if (
        not isinstance(evidence_payload, Mapping)
        or evidence_payload.get("schema_version") != "rsna-ticket33-slurm-external-blocker-v1"
        or evidence_payload.get("job_id") != slurm_job_id
        or evidence_payload.get("held_pending_before_cancel") is not True
        or evidence_payload.get("cancel_command_exit_code") != 0
        or evidence_payload.get("absent_from_queue_after_cancel") is not True
        or evidence_payload.get("sacct_status") != "unavailable_connection_refused"
    ):
        raise GpuCampaignLedgerError("external blocker evidence does not establish the exact cancellation")
    if not isinstance(conservative_charge_seconds, int) or isinstance(
        conservative_charge_seconds, bool
    ) or conservative_charge_seconds <= 0:
        raise GpuCampaignLedgerError("conservative external allocation charge must be positive")
    with _locked_ledger(destination):
        ledger = _read_ledger(destination)
        entry = _request_entry(ledger, request_checksum)
        if entry["status"] not in ACTIVE_REQUEST_STATUSES:
            raise GpuCampaignLedgerError("only a pending or active request can be externally blocked")
        if conservative_charge_seconds < _stored_int(entry["reserved_seconds"], "reserved allocation time"):
            raise GpuCampaignLedgerError("external blocker charge must cover every unobserved attempt")
        attempts = cast(list[dict[str, object]], entry["attempts"])
        matching = [attempt for attempt in attempts if attempt.get("job_id") == slurm_job_id]
        if len(matching) != 1:
            raise GpuCampaignLedgerError("external blocker evidence must name the exact Slurm job")
        matching[0].update(
            {
                "status": "cancelled_for_alternate_backend",
                "cancellation_evidence_path": str(evidence),
                "cancellation_evidence_sha256": evidence_sha256,
            }
        )
        entry.update(
            {
                "status": "externally_blocked",
                "reserved_seconds": 0,
                "actual_allocation_known": False,
                "unverified_allocation_charge_seconds": conservative_charge_seconds,
                "external_blocker": {
                    "reason": "Slurm accounting unavailable after the exact pending request was canceled",
                    "job_id": slurm_job_id,
                    "evidence_path": str(evidence),
                    "evidence_sha256": evidence_sha256,
                    "conservative_charge_seconds": conservative_charge_seconds,
                },
            }
        )
        atomic_json_write(destination, ledger, overwrite=True)
        return dict(entry)


def finish_gpu_campaign_request(
    path: str | Path,
    *,
    request_checksum: str,
    status: str,
) -> Mapping[str, object]:
    """Close a blocking submission or retain its full reserve when accounting is missing."""

    if status not in TERMINAL_REQUEST_STATUSES | {"unknown"}:
        raise GpuCampaignLedgerError("GPU campaign request needs a terminal status")
    destination = Path(path).resolve()
    with _locked_ledger(destination):
        ledger = _read_ledger(destination)
        entry = _request_entry(ledger, request_checksum)
        attempts = cast(list[Mapping[str, object]], entry["attempts"])
        has_unaccounted_job = any(
            (attempt.get("job_id") is not None and attempt.get("allocated_seconds") is None)
            or attempt.get("allocation_timing_missing") is True
            for attempt in attempts
        )
        final_status = "unknown" if has_unaccounted_job else status
        entry["status"] = final_status
        if final_status in TERMINAL_REQUEST_STATUSES:
            _close_terminal_request(ledger, entry, final_status)
        atomic_json_write(destination, ledger, overwrite=True)
        return dict(entry)


def reconcile_gpu_unallocated_attempt(
    path: str | Path,
    *,
    request_checksum: str,
    attempt_number: int,
    job_id: str,
    scheduler_evidence_path: str | Path,
) -> Mapping[str, object]:
    """Close a terminal Slurm job verified to have ended before GPU allocation."""

    evidence_path = Path(scheduler_evidence_path).resolve()
    try:
        evidence = evidence_path.read_bytes()
    except OSError as exc:
        raise GpuCampaignLedgerError("could not read Slurm no-allocation evidence") from exc
    if not evidence or len(evidence) > 1_000_000:
        raise GpuCampaignLedgerError("Slurm no-allocation evidence must be non-empty and at most 1 MB")
    state, queue_wait_seconds, submitted_epoch, ended_epoch = _parse_unallocated_scheduler_evidence(
        evidence.decode("utf-8", errors="strict"), expected_job_id=job_id
    )

    destination = Path(path).resolve()
    with _locked_ledger(destination):
        ledger = _read_ledger(destination)
        entry = _request_entry(ledger, request_checksum)
        attempt = _attempt_entry(entry, attempt_number)
        if attempt.get("job_id") != job_id:
            raise GpuCampaignLedgerError("Slurm no-allocation evidence names a different job")
        if attempt.get("allocated_seconds") is not None and attempt.get("allocation_timing_missing") is not True:
            raise GpuCampaignLedgerError("GPU allocation time is already recorded for this job")
        attempt.update(
            {
                "allocated_seconds": 0,
                "allocation_confirmed": False,
                "queue_wait_seconds": queue_wait_seconds,
                "allocation_timing_missing": False,
                "status": "not_allocated",
                "scheduler_terminal_state": state,
                "scheduler_submit_epoch": submitted_epoch,
                "scheduler_end_epoch": ended_epoch,
                "scheduler_evidence_path": str(evidence_path),
                "scheduler_evidence_sha256": hashlib.sha256(evidence).hexdigest(),
                "reconciled_at_utc": _utc_now(),
            }
        )
        _refresh_request_totals(entry)
        attempts = cast(list[Mapping[str, object]], entry["attempts"])
        has_unaccounted_job = any(
            (item.get("job_id") is not None and item.get("allocated_seconds") is None)
            or item.get("allocation_timing_missing") is True
            for item in attempts
        )
        has_successful_allocation = any(
            item.get("status") == "completed" and item.get("allocation_confirmed") is True
            for item in attempts
        )
        if has_unaccounted_job:
            entry["status"] = "unknown"
        else:
            _close_terminal_request(
                ledger,
                entry,
                "completed" if has_successful_allocation else "failed",
            )
        atomic_json_write(destination, ledger, overwrite=True)
        return dict(entry)


def _parse_unallocated_scheduler_evidence(
    content: str,
    *,
    expected_job_id: str,
) -> tuple[str, int, int, int]:
    """Parse one ``sacct -n -P`` row and derive its no-allocation queue time."""

    rows = [line.strip().split("|") for line in content.splitlines() if line.strip()]
    matching = [row for row in rows if row and row[0] == expected_job_id]
    if len(matching) != 1 or len(matching[0]) != 6:
        raise GpuCampaignLedgerError(
            "Slurm evidence must contain one six-field sacct row for the exact job ID"
        )
    _, raw_state, submitted_text, started_text, ended_text, elapsed_text = matching[0]
    state = raw_state.strip().upper().split(maxsplit=1)[0].rstrip("+")
    if state not in {"CANCELLED", "PREEMPTED", "TIMEOUT", "FAILED", "NODE_FAIL", "BOOT_FAIL"}:
        raise GpuCampaignLedgerError("Slurm evidence does not show an eligible terminal state")
    if started_text.strip().upper() not in {"", "UNKNOWN", "NONE", "N/A", "NOTSET"}:
        raise GpuCampaignLedgerError("Slurm evidence shows that the job started")
    if _slurm_elapsed_seconds(elapsed_text) != 0:
        raise GpuCampaignLedgerError("Slurm evidence shows a non-zero allocation duration")
    submitted_epoch = _slurm_epoch(submitted_text, "submit time")
    ended_epoch = _slurm_epoch(ended_text, "end time")
    if ended_epoch < submitted_epoch:
        raise GpuCampaignLedgerError("Slurm evidence end time precedes its submit time")
    return state, ended_epoch - submitted_epoch, submitted_epoch, ended_epoch


def _slurm_epoch(value: str, field_name: str) -> int:
    raw = value.strip()
    try:
        if raw.isdigit():
            return int(raw)
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GpuCampaignLedgerError(f"Slurm evidence has an invalid {field_name}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def _slurm_elapsed_seconds(value: str) -> int:
    raw = value.strip()
    if raw.isdigit():
        return int(raw)
    parts = raw.split(":")
    if len(parts) not in (2, 3) or any(not part.isdigit() for part in parts):
        raise GpuCampaignLedgerError("Slurm evidence has an invalid elapsed duration")
    numbers = [int(part) for part in parts]
    if len(numbers) == 2:
        hours, minutes, seconds = 0, numbers[0], numbers[1]
    else:
        hours, minutes, seconds = numbers
    if minutes > 59 or seconds > 59:
        raise GpuCampaignLedgerError("Slurm evidence has an invalid elapsed duration")
    return hours * 3600 + minutes * 60 + seconds


def _request_entry(ledger: Mapping[str, object], request_checksum: str) -> dict[str, object]:
    requests = cast(list[dict[str, object]], ledger["requests"])
    entry = next(
        (item for item in requests if item.get("request_checksum") == request_checksum),
        None,
    )
    if entry is None:
        raise GpuCampaignLedgerError("GPU request is not registered in the campaign ledger")
    return entry


def _attempt_entry(entry: Mapping[str, object], attempt_number: int) -> dict[str, object]:
    attempts = cast(list[dict[str, object]], entry["attempts"])
    attempt = next(
        (item for item in attempts if item.get("attempt_number") == attempt_number),
        None,
    )
    if attempt is None:
        raise GpuCampaignLedgerError("GPU allocation report has no matching submission attempt")
    return attempt


def _refresh_request_totals(entry: dict[str, object]) -> None:
    attempts = cast(list[Mapping[str, object]], entry["attempts"])
    entry["actual_allocated_seconds"] = sum(
        _stored_int(attempt["allocated_seconds"], "attempt allocation time")
        for attempt in attempts
        if isinstance(attempt.get("allocated_seconds"), int)
    )
    entry["queue_wait_seconds"] = sum(
        _stored_int(attempt["queue_wait_seconds"], "attempt queue wait")
        for attempt in attempts
        if isinstance(attempt.get("queue_wait_seconds"), int)
    )


def _completed_allocation_attempts(entry: Mapping[str, object]) -> int:
    return sum(
        1
        for attempt in cast(list[Mapping[str, object]], entry["attempts"])
        if (
            attempt.get("allocation_confirmed") is True
            or attempt.get("allocation_timing_missing") is True
            or (
                attempt.get("allocation_confirmed") is None
                and attempt.get("allocated_seconds") is not None
                and attempt.get("status") != "not_allocated"
            )
        )
    )


def _refresh_request_reservation(entry: dict[str, object]) -> None:
    remaining_attempts = (
        _stored_int(entry["max_attempts"], "maximum attempts")
        - _completed_allocation_attempts(entry)
    )
    entry["reserved_seconds"] = max(
        0,
        remaining_attempts
        * _stored_int(entry["wall_time_seconds"], "wall time")
        * _stored_int(entry.get("gpu_count", 1), "GPU count"),
    )


def _require_campaign_request_capacity(
    request_reserved_seconds: int,
    protected_comparison_seconds: int,
    remaining_seconds: int,
) -> None:
    if request_reserved_seconds + protected_comparison_seconds > remaining_seconds:
        raise GpuCampaignBudgetError(
            "GPU request would consume the protected comparison reserve "
            f"(request={request_reserved_seconds / 3600:.3f}h, "
            f"reserve={protected_comparison_seconds / 3600:.3f}h, "
            f"remaining={remaining_seconds / 3600:.3f}h)"
        )


def _restore_unspent_comparison_reserve(
    ledger: dict[str, object], entry: dict[str, object]
) -> None:
    """Return a fold's reserve when no attempt received a GPU allocation."""

    release_seconds = _stored_int(
        entry.get("comparison_reserve_release_seconds", 0),
        "comparison reserve release",
    )
    if (
        release_seconds > 0
        and entry.get("comparison_reserve_release_active") is True
        and _stored_int(entry["actual_allocated_seconds"], "actual allocation time") == 0
    ):
        ledger["reserved_comparison_seconds"] = (
            _stored_int(ledger["reserved_comparison_seconds"], "comparison reserve")
            + release_seconds
        )
        entry["comparison_reserve_release_active"] = False


def _close_terminal_request(
    ledger: dict[str, object],
    entry: dict[str, object],
    status: str,
) -> None:
    if status not in TERMINAL_REQUEST_STATUSES:
        raise GpuCampaignLedgerError("terminal GPU request status is invalid")
    entry["status"] = status
    entry["reserved_seconds"] = 0
    _restore_unspent_comparison_reserve(ledger, entry)
