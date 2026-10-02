"""Validate exact-anchor preparation without inference, reservation or launch."""

import ast
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from rsna_knee.scored_anchor_recipe import (
    extract_scored_recipe, notebook_code, read_notebook_code, relocate_scored_recipe,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = PROJECT / "artifacts/private/ticket70-scored-anchor-gold58-v1"
    parent = PROJECT / "artifacts/private/public-notebook-adoption/scored-baseline-v1"
    request = json.loads((root / "request.json").read_text())
    if digest(root / "request.json") != (root / "request-checksum.txt").read_text().strip():
        raise ValueError("Exact request bytes changed")
    for key, path in {
        "notebook_sha256": root / "development.ipynb",
        "metadata_sha256": root / "kernel-metadata.json",
        "original_notebook_sha256": parent / "baseline.ipynb",
        "remote_notebook_sha256": parent / "remote-source/rsna-public-ensemble-gpu-preflight-attributed.ipynb",
        "remote_capture_sha256": parent / "remote-version-2.json",
        "scoring_request_sha256": parent / "competition-submission-request.json",
        "scoring_terminal_sha256": parent / "terminal-scoring-evidence.json",
        "visible_execution_evidence_sha256": parent / "output/gpu_allocation_evidence.json",
    }.items():
        if digest(path) != request[key]:
            raise ValueError(f"Changed preparation/source evidence: {key}")
    for name, expected in request["frozen_files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError("Frozen source bytes differ")
    original = extract_scored_recipe(read_notebook_code(parent / "baseline.ipynb"))
    remote = extract_scored_recipe(read_notebook_code(parent / "remote-source/rsna-public-ensemble-gpu-preflight-attributed.ipynb"))
    capture = json.loads((parent / "remote-version-2.json").read_text())
    if original != remote or original != extract_scored_recipe(notebook_code(json.loads(capture["blob"]["source"]))):
        raise ValueError("Local/remote scored recipe differs")
    code = read_notebook_code(root / "development.ipynb")
    assignments = {}
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("ORIGINAL_CELLS", "SOURCE_CELLS"):
                if name in assignments:
                    raise ValueError("Ambiguous embedded source constants")
                assignments[name] = tuple(ast.literal_eval(node.value))
    if assignments.get("ORIGINAL_CELLS") != original or assignments.get("SOURCE_CELLS") != relocate_scored_recipe(original):
        raise ValueError("Prepared recipe has changes beyond dataset roots")
    for i, (a, b) in enumerate(zip(original, assignments["SOURCE_CELLS"])):
        entry = request["source_cells"][i]
        if entry != {"index": i, "original_sha256": hashlib.sha256(a.encode()).hexdigest(),
                     "relocated_sha256": hashlib.sha256(b.encode()).hexdigest()}:
            raise ValueError("Recipe source identity differs")
    metadata = json.loads((root / "kernel-metadata.json").read_text())
    expected = json.loads((parent / "kernel-metadata.json").read_text())
    expected.update(id="jitheswar/rsna-exact-scored-anchor-gold58-v1",
                    title="RSNA exact scored anchor Gold58 development", code_file="development.ipynb",
                    is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=False)
    if metadata != expected or request["kernel_id"] != metadata["id"]:
        raise ValueError("Exact private offline asset/runtime settings differ")
    if (request["assigned_gpu_count"] != 2 or request["study_count"] != 58
            or request["execution_watchdog_seconds"] != 7140 or request["session_timeout_seconds"] != 7200
            or request["proposed_reserved_gpu_seconds"] != 14400 or request["maximum_launch_attempts"] != 1
            or request["model_fitting"] or request["competition_submission"] or request["final_selection"]
            or request["external_spend_usd"] != 0 or request["forecast_wall_seconds"] >= 7140):
        raise ValueError("Prepared scientific/resource scope differs")
    compile(code, "independent-ticket70-preparation", "exec")
    record = {"status": "PREPARATION_VALIDATED", "request_sha256": digest(root / "request.json"),
              "scored_submission_ref": 56705536, "recipe_cells_verified": 5,
              "root_literals_changed": 4, "all_other_recipe_bytes_unchanged": True,
              "original_mri_predictions_verified": False, "launched": False,
              "resource_decision_admitted": False, "competition_submission": False}
    path = root / "preparation-validation.json"
    serialized = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != serialized:
        raise ValueError("Immutable preparation validation differs")
    if not path.exists():
        path.write_text(serialized)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
