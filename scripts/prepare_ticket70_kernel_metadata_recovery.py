"""Repair the diagnosed notebook envelope while preserving every inference byte."""

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[1]
PARENT = PROJECT / "artifacts/private/ticket70-scored-anchor-gold58-v3"
ROOT = PROJECT / "artifacts/private/ticket70-scored-anchor-gold58-v4"
PARENT_SHA = "255eff3d3ea66deb926a9a4507391322bd19bd66b8a9edd061a470a518cd2924"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repair(notebook, successful_metadata):
    result = copy.deepcopy(notebook)
    spec = successful_metadata.get("kernelspec", {})
    if spec.get("name") != "python3" or spec.get("language") != "python":
        raise ValueError("Exact known successful Python kernelspec required")
    if len(result["cells"]) != 1 or result["cells"][0]["cell_type"] != "code":
        raise ValueError("The single immutable runtime envelope is required")
    result["metadata"]["kernelspec"] = copy.deepcopy(spec)
    result["cells"][0]["id"] = "exact-anchor-runtime"
    if result["cells"][0]["source"] != notebook["cells"][0]["source"]:
        raise ValueError("Notebook repair changed executable source")
    return result


def main():
    if ROOT.exists():
        raise FileExistsError("Immutable recovery already exists")
    if digest(PARENT / "request.json") != PARENT_SHA:
        raise ValueError("Exact failed parent required")
    parent = json.loads((PARENT / "request.json").read_text())
    terminal = json.loads((PARENT / "kernel-terminal.json").read_text())
    if terminal["status"] != "ERROR" or terminal["source_status"] != "PASS":
        raise ValueError("Actual matching failed parent terminal required")
    log = PARENT / "failure-output/rsna-exact-scored-anchor-gold58-development.log"
    if "No kernel name found in notebook and no override provided." not in log.read_text():
        raise ValueError("Exact observed Papermill startup failure required")
    baseline = PROJECT / "artifacts/private/public-notebook-adoption/scored-baseline-v1/baseline.ipynb"
    if digest(baseline) != parent["original_notebook_sha256"]:
        raise ValueError("Known successful scored notebook changed")
    for name, expected in parent["frozen_files"].items():
        if digest(PARENT / name) != expected:
            raise ValueError("Immutable parent source changed")
    if digest(PARENT / "development.ipynb") != parent["notebook_sha256"]:
        raise ValueError("Failed executable envelope changed")
    notebook = repair(json.loads((PARENT / "development.ipynb").read_text()), json.loads(baseline.read_text())["metadata"])
    metadata = json.loads((PARENT / "kernel-metadata.json").read_text())
    metadata.update(id="jitheswar/rsna-exact-scored-anchor-gold58-v4", title="RSNA Exact Scored Anchor Gold58 V4")
    ROOT.mkdir(mode=0o700)
    shutil.copytree(PARENT / "source-snapshot", ROOT / "source-snapshot")
    shutil.copyfile(__file__, ROOT / "source-snapshot" / Path(__file__).name)
    for name in ("original-mri-contract.json", "NOTICE.md", "LICENSE-APACHE-2.0.txt"):
        shutil.copyfile(PARENT / name, ROOT / name)
    def write(name, value):
        (ROOT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    write("development.ipynb", notebook)
    write("kernel-metadata.json", metadata)
    request = {**parent, "schema_version": "rsna-ticket70-kernel-metadata-recovery-v4",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "parent_request_sha256": PARENT_SHA,
        "supersedes": str(PARENT), "kernel_id": metadata["id"],
        "notebook_sha256": digest(ROOT / "development.ipynb"), "metadata_sha256": digest(ROOT / "kernel-metadata.json"),
        "proposed_campaign_cap_seconds": 165600,
        "diagnosed_failure_log_sha256": digest(log), "parent_terminal_sha256": digest(PARENT / "kernel-terminal.json"),
        "only_recovery_changes": "Known successful kernelspec and deterministic cell id; matching unique kernel title/slug. All executable source unchanged.",
        "frozen_files": {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "source-snapshot").iterdir()},
        "resource_decision_status": "Prepared diagnosed recovery only; separate admission/reservation/launch required"}
    write("request.json", request)
    sha = digest(ROOT / "request.json")
    (ROOT / "request-checksum.txt").write_text(sha + "\n")
    write("preparation-validation.json", {"status": "VERIFIED_RUNTIME_ENVELOPE_PREPARATION_VALIDATED",
        "request_sha256": sha, "parent_request_sha256": PARENT_SHA, "scientific_cells_unchanged": 5,
        "entire_executable_envelope_unchanged": True, "successful_kernelspec_restored": True,
        "matching_title_slug": True, "model_fitting": False, "competition_submission": False,
        "launched": False, "resource_decision_admitted": False})
    print(json.dumps({"status": "PREPARED_DIAGNOSED_RECOVERY", "request_sha256": sha, "code_unchanged": True}))


if __name__ == "__main__":
    main()
