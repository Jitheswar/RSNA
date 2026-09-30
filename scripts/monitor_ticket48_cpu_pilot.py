"""Retain Slurm terminal evidence for one exact already submitted CPU job."""

import datetime as dt
import json
from pathlib import Path
import re
import subprocess
import sys
import time


root = Path(sys.argv[1]).resolve()
job_id = sys.argv[2]
if not job_id.isdecimal():
    raise ValueError("exact numeric Slurm job required")
evidence = root / f"controller-{job_id}"
evidence.mkdir(mode=0o700, exist_ok=True)
while True:
    result = subprocess.run(["scontrol", "show", "job", job_id, "-o"],
                            capture_output=True, text=True, timeout=15)
    text = result.stdout.strip()
    match = re.search(r"\bJobState=(\S+)", text)
    state = match.group(1) if match else "unavailable"
    observation = {"job_id": job_id, "state": state,
                   "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "returncode": result.returncode, "stderr": result.stderr.strip()}
    temporary = evidence / "latest.json.partial"
    temporary.write_text(json.dumps(observation, indent=2) + "\n")
    temporary.replace(evidence / "latest.json")
    if text:
        (evidence / "latest-controller.txt").write_text(text + "\n")
    with (evidence / "observations.jsonl").open("a") as handle:
        handle.write(json.dumps(observation) + "\n")
    if state in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                 "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}:
        with (evidence / "terminal-controller.txt").open("x") as handle:
            handle.write(text + "\n")
        with (evidence / "terminal-observation.json").open("x") as handle:
            json.dump(observation, handle, indent=2)
            handle.write("\n")
        break
    time.sleep(5)
