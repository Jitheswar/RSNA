from __future__ import annotations

import pytest

from rsna_knee.pseudo_label_cli import _populate_ticket21_slurm_timelimit
from rsna_knee.training import PreflightError


def test_ticket21_slurm_timelimit_is_populated_from_scontrol_details() -> None:
    environment: dict[str, str] = {"SLURM_JOB_ID": "7108"}

    _populate_ticket21_slurm_timelimit(
        environment,
        "JobId=7108 JobName=t21prefprobe TimeLimit=00:05:00 State=RUNNING",
    )

    assert environment["SLURM_TIMELIMIT"] == "00:05:00"


def test_ticket21_slurm_timelimit_preserves_existing_value() -> None:
    environment = {"SLURM_TIMELIMIT": "06:00:00"}

    _populate_ticket21_slurm_timelimit(environment, "TimeLimit=00:05:00")

    assert environment["SLURM_TIMELIMIT"] == "06:00:00"


def test_ticket21_slurm_timelimit_rejects_unresolved_job_details() -> None:
    with pytest.raises(PreflightError, match="TimeLimit"):
        _populate_ticket21_slurm_timelimit({}, "JobId=7108 State=RUNNING")
