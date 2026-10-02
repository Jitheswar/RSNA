"""Freeze one full-corpus residual CoAt input scan from the measured pilot."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from prepare_ticket59_coat_input_probe import PROJECT, FULL, AUTH, digest, read, write


PARENT = PROJECT / "artifacts/private/public-frontier-adoption/coat-cpu-input-probe-v1"
ROOT = PROJECT / "artifacts/private/public-frontier-adoption/full-coat-cpu-input-probe-v1"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket59-full-coat-cpu-input-probe-20261002.json"


def main():
    if ROOT.exists() or DECISION.exists():
        raise FileExistsError("Immutable full CoAt probe already prepared")
    parent = read(PARENT / "plan.json")
    validation = read(PARENT / "independent-summary-validation.json")
    if (validation["status"] != "COMPLETE" or validation["study_count"] != 64
            or validation["input_failure_count"] or validation["dense_sampler_fallback_count"]
            or validation["preparation_warning_studies"] or validation["plan_sha256"] != digest(PARENT / "plan.json")
            or validation["result_sha256"] != digest(PARENT / "probe-result.json")):
        raise ValueError("Exact successful independently checked CoAt pilot required")
    for row in parent["frozen_files"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Consumed original-MRI/runtime identities changed")
    forecast = 2 * validation["allocated_cpu_node_seconds"] / 64 * 4407 + 900
    if not 0 < forecast < 21540:
        raise ValueError("Twice measured full-corpus cost plus allowance exceeds bounded watchdog")
    ROOT.mkdir(mode=0o700)
    source = ROOT / "source-snapshot"
    shutil.copytree(Path(parent["runtime_source"]), source)
    shutil.copyfile(__file__, source / Path(__file__).name)
    worker = source / "ticket59_coat_input_probe.py"
    text = worker.read_text()
    replacements = {'"TimeLimit": "01:00:00"': '"TimeLimit": "06:00:00"',
                    'threading.Timer(3540,': 'threading.Timer(21540,'}
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError("Exact measured worker resource substitution is ambiguous")
        text = text.replace(old, new)
    compile(text, str(worker), "exec")
    worker.write_text(text)
    studies = {}
    for index in range(2):
        directory = FULL / f"partition-{index}"
        members = read(directory / "plan.json")["selected_study_uids"]
        counts = {}
        with (directory / "headers.csv").open() as handle:
            for row in csv.DictReader(handle):
                uid = row["StudyInstanceUID"]
                counts[uid] = counts.get(uid, 0) + int(row["n_slices"])
        for uid in members:
            if uid in studies:
                raise ValueError("Full original partition membership overlaps")
            studies[uid] = {"study_uid": uid, "original_slice_count": counts[uid], "records": [],
                            "series_root": str(directory / "input-view/test_series")}
        with (directory / "input-view/test_series.csv").open() as handle:
            for row in csv.DictReader(handle):
                studies[row["StudyInstanceUID"]]["records"].append(row)
    if len(studies) != 4407 or any(not row["records"] for row in studies.values()):
        raise ValueError("Full original input coverage differs")
    resources = {**parent["resources"], "walltime_seconds": 21600, "code_watchdog_seconds": 21540}
    inputs = [Path(row["path"]) for row in parent["frozen_files"] if Path(row["path"]).parent != Path(parent["runtime_source"])]
    inputs.extend([PARENT / "plan.json", PARENT / "probe-result.json", PARENT / "independent-summary-validation.json"])
    inputs.extend(path for path in source.iterdir() if path.is_file())
    plan = {**parent, "schema_version": "rsna-ticket59-full-coat-cpu-input-probe-plan-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_root": str(ROOT), "decision_path": str(DECISION),
            "runtime_source": str(source), "resources": resources, "studies": [studies[uid] for uid in sorted(studies)],
            "study_count": 4407, "selection": "All 4,407 original training MRI study UIDs in sorted order; labels/reports not read.",
            "prospective_cpu_node_seconds": forecast, "worker_changes": replacements,
            "frozen_files": [{"path": str(path), "sha256": digest(path)} for path in inputs]}
    write(ROOT / "plan.json", plan)
    checksum = digest(ROOT / "plan.json")
    write(DECISION, {"schema_version": "rsna-campaign-autonomous-decision-v1", "decision": "approved",
                    "plan_sha256": checksum, "resources": resources, "authorization_sha256": digest(AUTH),
                    "decision_made_by": "campaign agent under standing owner authorization",
                    "basis": "Exact CPU CoAt pilot passes 64 study complexity order statistics and both extremes with no warnings/failures/fallbacks. Twice measured cost plus 900 seconds fits six-hour bound. Reuse original MRI read-only; preserve exact sampler/renderer and two workers; no model/GPU.",
                    "new_exact_owner_numeric_approval_claimed": False})
    launcher = (PARENT / "run.sbatch").read_text().replace(str(PARENT), str(ROOT)).replace(
        "--job-name=rsna-ticket59-coat-input", "--job-name=rsna-ticket59-full-coat-input").replace(
        "--time=01:00:00", "--time=06:00:00").replace(digest(PARENT / "plan.json"), checksum)
    (ROOT / "run.sbatch").write_text(launcher)
    subprocess.run(["bash", "-n", str(ROOT / "run.sbatch")], check=True)
    print(json.dumps({"study_count": 4407, "plan_sha256": checksum, "forecast_seconds": forecast, "launched": False}))


if __name__ == "__main__":
    main()
