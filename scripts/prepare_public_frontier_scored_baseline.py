"""Prepare full-workload delivery from the exact validated visible recipe."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "artifacts/private/public-frontier-adoption/gpu-inference-preflight-v3"
DEST = ROOT / "artifacts/private/public-frontier-adoption/scored-baseline-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise SystemExit("Immutable full-workload request already exists")
    validation = json.loads((PARENT / "validation.json").read_text())
    terminal = json.loads((PARENT / "kernel-terminal.json").read_text())
    assert validation["status"] == terminal["status"] == "COMPLETE"
    assert validation["independent_prediction_artifacts_verified"] == 30
    assert validation["strict_execution"]["failed_fallback_events"] == 0
    assert validation["submission_sha256"] == sha(PARENT / "output/submission.csv")
    notebook = json.loads((PARENT / "gpu-preflight.ipynb").read_text())
    cell = notebook["cells"][-1]
    code = "".join(cell["source"])
    # Each replacement is an execution-envelope change. The embedded upstream
    # cells and strict reader/prediction/schema checks remain byte-for-byte.
    replacements = {
        "print('Visible preflight exceeded 900 seconds', flush=True)":
            "print('Full-workload execution watchdog expired', flush=True)",
        "watchdog=threading.Timer(900,expire)": "watchdog=None",
        "watchdog.daemon=True\nwatchdog.start()\n": "",
        "if not 1 <= len(sample) <= 3 or len(test)!=len(sample):\n"
        "    raise RuntimeError('This exact request permits visible studies only; hidden scoring is prohibited')":
            "if len(sample)<1 or len(test)!=len(sample):\n"
            "    raise RuntimeError('Competition test/sample cohort mismatch')\n"
            "visible_only=len(sample)<=3\n"
            "execution_limit=900 if visible_only else 28000\n"
            "watchdog=threading.Timer(execution_limit,expire)\n"
            "watchdog.daemon=True\nwatchdog.start()",
        "'schema_version':'rsna-public-frontier-visible-gpu-validation-v1'":
            "'schema_version':'rsna-public-frontier-full-workload-validation-v1'",
        "'visible_only':True": "'visible_only':visible_only",
        "work/'strict_visible_validation.json'": "work/'strict_execution_validation.json'",
    }
    for old, new in replacements.items():
        assert code.count(old) == 1, repr(old)
        code = code.replace(old, new)
    compile(code, "full-workload-frontier", "exec")
    cell["source"] = code.splitlines(keepends=True)
    notebook["cells"][0]["source"].insert(0,
        "Full-workload private adaptation of the verified four-reader recipe. "
        "Changes visible-only guard and execution watchdog; preserves all "
        "upstream code, checkpoints, blend weights and strict output checks. "
        "Saving this version does not issue a competition submission.\n")
    metadata = json.loads((PARENT / "kernel-metadata.json").read_text())
    metadata["code_file"] = "baseline.ipynb"
    DEST.mkdir(parents=True)

    def write(name, obj):
        (DEST / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

    write("baseline.ipynb", notebook)
    write("kernel-metadata.json", metadata)
    for name in ("NOTICE.md", "LICENSE-APACHE-2.0.txt"):
        shutil.copyfile(PARENT / name, DEST / name)
    with (DEST / "NOTICE.md").open("a") as handle:
        handle.write("\nFull-workload adaptation: visible-only guard replaced by "
                     "cohort validation; 900-second visible / 28,000-second "
                     "hidden watchdog. Scientific recipe unchanged.\n")
    write("request.json", {
        "schema_version": "rsna-public-frontier-scored-baseline-request-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kernel_id": metadata["id"],
        "parent_request_checksum": sha(PARENT / "request.json"),
        "parent_validation_sha256": sha(PARENT / "validation.json"),
        "parent_submission_sha256": validation["submission_sha256"],
        "notebook_sha256": sha(DEST / "baseline.ipynb"),
        "metadata_sha256": sha(DEST / "kernel-metadata.json"),
        "assigned_gpu_count": 2,
        "visible_execution_cap_seconds": 900,
        "visible_run_reservation_wall_seconds": 1200,
        "visible_reserved_gpu_seconds": 2400,
        "hidden_execution_cap_seconds": 28000,
        "hidden_run_reservation_wall_seconds": 28800,
        "hidden_reserved_gpu_seconds": 57600,
        "external_spend_usd": 0,
        "model_fitting": False,
        "competition_submission_issued": False,
        "final_candidate_selection": False,
        "authorization_basis": "Existing broad owner zero-spend authorization "
            "for separate public-baseline evaluation; saved-version execution "
            "fits current campaign. Hidden scoring requires separate reservation "
            "and sufficient approved campaign capacity; no budget extension inferred.",
    })
    (DEST / "request-checksum.txt").write_text(sha(DEST / "request.json") + "\n")
    print(DEST)


if __name__ == "__main__":
    main()
