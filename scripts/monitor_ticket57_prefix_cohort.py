"""Observe accepted prefix jobs and preserve terminal evidence before purge."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def observe(root):
    plan_sha = digest(root / "plan.json")
    observations = []
    for partition in range(2):
        accepted = read(root / f"accepted-launch-{partition}.json")
        if accepted["partition"] != partition or accepted["plan_sha256"] != plan_sha:
            raise ValueError("Accepted extraction request identity differs")
        job = accepted["job_id"]
        directory = root / f"controller-{job}"
        directory.mkdir(mode=0o700, exist_ok=True)
        terminal_path = directory / "terminal-observation.json"
        if terminal_path.exists():
            observation = read(terminal_path)
            if observation["job_id"] != job or observation["plan_sha256"] != plan_sha:
                raise ValueError("Retained terminal identity differs")
        else:
            response = subprocess.run(["scontrol", "show", "job", str(job), "-o"],
                                      capture_output=True, text=True)
            observation = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
                           "job_id": job, "partition": partition, "plan_sha256": plan_sha,
                           "returncode": response.returncode, "controller": response.stdout,
                           "stderr": response.stderr}
            fields = dict(token.split("=", 1) for token in response.stdout.split() if "=" in token)
            if response.returncode == 0 and fields.get("JobId") == str(job):
                observation.update(state=fields.get("JobState"), exit_code=fields.get("ExitCode"),
                                   runtime=fields.get("RunTime"))
                if observation["state"] in TERMINAL:
                    with (directory / "terminal-controller.txt").open("x") as handle:
                        handle.write(response.stdout)
                    write(terminal_path, observation)
            else:
                observation["observation_error"] = True
                observation["terminal_not_inferred"] = True
            with (directory / "observations.jsonl").open("a") as handle:
                handle.write(json.dumps(observation, sort_keys=True) + "\n")
        progress = root / f"partition-{partition}-job-{job}/progress.jsonl"
        # Ignore an incomplete final journal line while the writer is active.
        content = progress.read_bytes() if progress.exists() else b""
        lines = content.split(b"\n")[:-1]
        rows = [json.loads(line) for line in lines if line]
        observation.update(completed_studies=len(rows), valid_slices=sum(row["valid_slices"] for row in rows),
                           feature_bytes=sum(row["file_bytes"] for row in rows),
                           maximum_native_feature_error=max((row["native_feature_max_abs_error"] for row in rows), default=None))
        observations.append(observation)
    status = {"schema_version": "rsna-ticket57-controller-status-v1",
              "observed_at_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": plan_sha,
              "partitions": observations, "all_terminal": all(o.get("state") in TERMINAL for o in observations),
              "all_completed": all(o.get("state") == "COMPLETED" and o.get("exit_code") == "0:0" for o in observations),
              "completed_studies": sum(o["completed_studies"] for o in observations),
              "valid_slices": sum(o["valid_slices"] for o in observations),
              "new_launches": False, "retries": False, "cohort_verified": False}
    temporary = root / "controller-status.json.partial"
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    temporary.replace(root / "controller-status.json")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        status = observe(args.root.resolve())
        print(json.dumps({"observed_at_utc": status["observed_at_utc"],
                          "partitions": [{k: o.get(k) for k in ("job_id", "state", "completed_studies", "valid_slices")}
                                         for o in status["partitions"]], "all_terminal": status["all_terminal"]}), flush=True)
        if status["all_terminal"] or not args.watch:
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
