"""Finish the existing sole medical verifier after controller interruption.

No launch, retry, metric computation or full-fit scheduling is performed here.
The missing historical full-fit terminal records remain missing.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
ROOT = PROJECT / "artifacts/private/ticket68-biomedclip-matched-heads-v1"
CONTROLLER = PROJECT / "artifacts/private/ticket68-biomedclip-head-controller-v1"
PLAN_SHA = "16ee8b86a75aa30a0c6a1f25e68ee65028a084914b7863584e1abb540c8bc666"
JOB = 8106


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    if sha(ROOT / "plan.json") != PLAN_SHA:
        raise ValueError("Exact previously admitted medical plan required")
    accepted = read(ROOT / "verification/accepted-launch.json")
    if accepted != {"job_id": JOB, "launch_attempt": 1, "plan_sha256": PLAN_SHA}:
        raise ValueError("Only the existing sole native verifier is observable")
    admission = read(CONTROLLER / "admission.json")
    for row in admission["frozen_files"]:
        if sha(row["path"]) != row["sha256"]:
            raise ValueError("Original controller/helper source changed")
    snapshot = Path(admission["source_snapshot"])
    os.environ["RSNA_PROJECT_ROOT"] = str(PROJECT)
    sys.path.insert(0, str(snapshot))
    sys.path.insert(0, str(snapshot / "src"))
    from coordinate_ticket68_biomedclip_heads import component
    from validate_ticket68_biomedclip_heads import successful_terminal
    if component("fit", PLAN_SHA)[0] != "complete":
        raise ValueError("Existing fit requires actual successful terminal")
    while True:
        status, job = component("verification", PLAN_SHA)
        if status == "complete":
            validation = read(ROOT / "verification/independent-validation.json")
            if not (job == JOB and validation["verified_final_heads"] == 6
                    and validation["original_mri_studies"] == 58
                    and validation["verified_medical_predictions"] == validation["verified_resnet_predictions"] == 116
                    and validation["maximum_prediction_abs_error"] <= 1e-6
                    and validation["maximum_medical_feature_abs_error"] <= 1e-6
                    and validation["all_fixed_native_comparisons_reproduced"] is True
                    and validation["all_fitted_targets_masks_init_exposures_adam_checked"] is True
                    and validation["analysis_sha256"] == sha(ROOT / "analysis.json")
                    and validation["candidate_promotion"] is False):
                raise ValueError("Complete original independent MRI/state/source proof required")
            seconds, terminal_sha = successful_terminal(ROOT / "verification", JOB, "03:00:00")
            result = dict(status="COMPLETE", passed=True, plan_sha256=PLAN_SHA, verification_job_id=JOB,
                    allocated_verification_cpu_node_seconds=seconds, verification_terminal_sha256=terminal_sha,
                    independent_validation_sha256=sha(ROOT / "verification/independent-validation.json"),
                    analysis_sha256=sha(ROOT / "analysis.json"), disposition=validation["disposition"],
                    candidate_promotion=False, gpu_seconds=0, tpu_seconds=0, external_spend_usd=0)
            if (ROOT / "terminal-verification.json").exists():
                if read(ROOT / "terminal-verification.json") != result:
                    raise ValueError("Existing terminal proof differs; never replace it")
            else:
                write(ROOT / "terminal-verification.json", result)
            print(json.dumps(result), flush=True)
            return
        observation = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), job_id=JOB, phase="wait_for_existing_native_terminal")
        try:
            queue = subprocess.check_output(["squeue", "-j", str(JOB), "-h", "-o", "%i %T"], text=True, timeout=15).strip()
            observation["actual_queue"] = queue
            if not queue:
                observation["missing_from_queue"] = True
                observation["terminal_not_inferred_from_absence"] = True
        except Exception as exc:
            observation["observation_error"] = str(exc)
        print(json.dumps(observation), flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
