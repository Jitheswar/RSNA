"""Freeze a label-blind 64-study exact residual CoAt preparation probe."""

import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "artifacts/private/public-frontier-adoption/coat-cpu-input-probe-v1"
FULL = PROJECT / "artifacts/private/public-frontier-adoption/full-cpu-input-probe-v2"
ADMISSION = PROJECT / "artifacts/private/public-frontier-adoption/coat-input-runtime-admission-v1"
CELLS = PROJECT / "artifacts/private/public-frontier-adoption/scoring-failure-diagnosis-v1/upstream-cells"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket59-coat-cpu-input-probe-20261002.json"
PYTHON = PROJECT / "artifacts/private/ticket48-convnext-cpu/runtime/bin/python"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def definition(source, name):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError("Exact upstream helper is ambiguous")
    return ast.get_source_segment(source, nodes[0]) + "\n"


def main():
    if ROOT.exists() or DECISION.exists():
        raise FileExistsError("Immutable CoAt CPU probe already prepared")
    validation = read(FULL / "independent-summary-validation.json")
    if validation["status"] != "COMPLETE" or validation["study_count"] != 4407 or not validation["disjoint_complete_membership"]:
        raise ValueError("Complete independent source-cohort check required")
    admission = read(ADMISSION / "admission.json")
    for row in admission["files"]:
        if digest(ADMISSION / row["name"]) != row["sha256"]:
            raise ValueError("Actual CPU-mounted public source identity differs")
    wheels = list((ADMISSION / "wheels").glob("*.whl"))
    if len(wheels) != 1 or digest(wheels[0]) != "236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9":
        raise ValueError("Exact notebook OpenCV wheel required")
    metadata, slice_counts, partition_for_uid = [], Counter(), {}
    inputs = [AUTH, FULL / "plan.json", FULL / "independent-summary-validation.json", ADMISSION / "admission.json",
              CELLS / "cell-3.py", CELLS / "cell-4.py", wheels[0]]
    for index in range(2):
        directory = FULL / f"partition-{index}"
        inputs.extend([directory / "plan.json", directory / "probe-result.json", directory / "headers.csv", directory / "input-view/test_series.csv"])
        plan = read(directory / "plan.json")
        with (directory / "input-view/test_series.csv").open() as handle:
            metadata.extend(csv.DictReader(handle))
        with (directory / "headers.csv").open() as handle:
            for row in csv.DictReader(handle):
                slice_counts[row["StudyInstanceUID"]] += int(row["n_slices"])
        partition_for_uid.update({uid: index for uid in plan["selected_study_uids"]})
    ordered = sorted(slice_counts, key=lambda uid: (slice_counts[uid], hashlib.sha256(uid.encode()).digest()))
    selected = [ordered[round(i * (len(ordered) - 1) / 63)] for i in range(64)]
    if len(set(selected)) != 64 or len(ordered) != 4407:
        raise ValueError("Original study/count cohort differs")
    ROOT.mkdir(mode=0o700)
    source = ROOT / "source-snapshot"
    source.mkdir(mode=0o700)
    for row in admission["files"]:
        shutil.copyfile(ADMISSION / row["name"], source / row["name"])
    for name in ("ticket59_coat_input_probe.py", "prepare_ticket59_coat_input_probe.py"):
        shutil.copyfile(PROJECT / "scripts" / name, source / name)
    cells = {i: (CELLS / f"cell-{i}.py").read_text() for i in (3, 4)}
    helpers = "import numpy as np\n" + "\n".join(definition(cells[4], name) for name in
              ("_dense_allocate", "_dense_unique_linspace", "_dense_coat_specs"))
    environment = {}
    exec(compile(definition(cells[3], "_asset_write_coat_runtime"), "unchanged-source-writer", "exec"), environment)
    generated = source / "input_resgated_runtime.py"
    expected_runtime_sha = environment["_asset_write_coat_runtime"](source, generated, helpers)
    records = {uid: [] for uid in selected}
    for row in metadata:
        if row["StudyInstanceUID"] in records:
            records[row["StudyInstanceUID"]].append(row)
    studies = [{"study_uid": uid, "original_slice_count": slice_counts[uid], "records": records[uid],
                "series_root": str(FULL / f"partition-{partition_for_uid[uid]}" / "input-view/test_series")} for uid in selected]
    inputs.extend(path for path in source.iterdir() if path.is_file())
    inputs.extend(path for path in (ADMISSION / "cv2-site/cv2").rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    versions = json.loads(subprocess.check_output([str(PYTHON), "-c",
        "import json,importlib.metadata; print(json.dumps({n:importlib.metadata.version(n) for n in ('numpy','pandas','pydicom','torch','torchvision','timm')}))"], text=True))
    resources = {"partition": "cluster1-cpu", "nodes": 1, "cpus_per_task": 16, "memory_gib": 64,
                 "walltime_seconds": 3600, "code_watchdog_seconds": 3540, "maximum_launch_attempts": 1,
                 "prep_workers": 2, "torch_threads_per_worker": 1, "gpu_count": 0, "external_spend_usd": 0,
                 "exclusive": False, "requeue": False, "additional_retained_mri_bytes": 0}
    plan = {"schema_version": "rsna-ticket59-coat-cpu-input-probe-plan-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_root": str(ROOT), "decision_path": str(DECISION), "resources": resources,
            "selection": "64 equally spaced order statistics of original study DICOM slice count across all 4,407 studies; SHA256(UID) breaks ties. Label-blind; includes both extremes.",
            "studies": studies, "study_count": 64, "runtime_source": str(source), "cv2_site": str(ADMISSION / "cv2-site"),
            "python": str(PYTHON), "runtime_versions": versions, "patched_runtime_sha256": expected_runtime_sha,
            "frozen_files": [{"path": str(path), "sha256": digest(path)} for path in inputs],
            "source_rules_changed": False, "model_inference": False, "labels_read": False, "network_during_execution": False,
            "hidden_error_attribution": False, "candidate_promotion": False, "competition_submission": False}
    write(ROOT / "plan.json", plan)
    checksum = digest(ROOT / "plan.json")
    write(DECISION, {"schema_version": "rsna-campaign-autonomous-decision-v1", "decision": "approved",
                    "plan_sha256": checksum, "resources": resources, "authorization_sha256": digest(AUTH),
                    "decision_made_by": "campaign agent under standing owner authorization",
                    "basis": "Exercise an untested actual original-MRI CoAt preparation seam after complete DINO input success; measure fixed label-blind original-input complexity extremes, preserving exact renderer and sampler. No model or GPU.",
                    "new_exact_owner_numeric_approval_claimed": False})
    launcher = ROOT / "run.sbatch"
    launcher.write_text(f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket59-coat-input
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output={ROOT}/slurm-%j.out
#SBATCH --error={ROOT}/slurm-%j.err
#SBATCH --chdir={PROJECT}
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
{PYTHON} {source}/ticket59_coat_input_probe.py {ROOT}/plan.json {checksum}
''')
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    print(json.dumps({"plan_sha256": checksum, "study_count": 64, "original_slice_range": [min(slice_counts.values()), max(slice_counts.values())], "launched": False}))


if __name__ == "__main__":
    main()
