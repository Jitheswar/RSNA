"""A mistyped observer ID must not capture a different scheduler allocation."""

import json
import os
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).parents[1] / "scripts/monitor_ticket48_cpu_pilot.py"


def test_wrong_accepted_job_is_rejected_before_controller_query(tmp_path):
    (tmp_path / "accepted-launch.json").write_text(json.dumps({"job_id": 123}))
    tools = tmp_path / "tools"
    tools.mkdir()
    marker = tmp_path / "controller-was-queried"
    control = tools / "scontrol"
    control.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    control.chmod(0o700)
    result = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), "124"],
                            env={**os.environ, "PATH": str(tools)}, capture_output=True, text=True)
    assert result.returncode != 0
    assert "actual accepted launch" in result.stderr
    assert not marker.exists()
    assert not (tmp_path / "controller-124").exists()


def test_exact_accepted_job_retains_matching_terminal_record(tmp_path):
    (tmp_path / "accepted-launch.json").write_text(json.dumps({"job_id": 123}))
    tools = tmp_path / "tools"
    tools.mkdir()
    control = tools / "scontrol"
    control.write_text("#!/bin/sh\nprintf '%s\\n' 'JobId=123 JobState=COMPLETED ExitCode=0:0'\n")
    control.chmod(0o700)
    result = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), "123"],
                            env={**os.environ, "PATH": str(tools)}, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    terminal = json.loads((tmp_path / "controller-123/terminal-observation.json").read_text())
    assert terminal["job_id"] == "123"
    assert terminal["state"] == "COMPLETED"
    assert (tmp_path / "controller-123/terminal-controller.txt").read_text().startswith("JobId=123 ")
