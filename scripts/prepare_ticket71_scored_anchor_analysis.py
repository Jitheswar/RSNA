"""Freeze the prospective Ticket71 protocol without reading predictions or labels.

This writes an immutable preparation. It does not admit final inputs, allocate
resources, compute a metric, promote a candidate or contact Kaggle.
"""

from datetime import datetime, timezone
import argparse
import ast
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys

PROJECT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))
from rsna_knee.scored_anchor_comparison import PROTOCOL, digest, read, require


def analysis_source_dependencies(source_root):
    """Freeze the static local import closure, leaving unrelated adapters editable."""
    source_root = Path(source_root).resolve()
    package = source_root / "rsna_knee"
    pending = [package / "__init__.py", package / "scored_anchor_comparison.py"]
    found = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        require(path.is_file(), "Local analysis dependency source is missing")
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.removeprefix("rsna_knee.") for alias in node.names
                         if alias.name.startswith("rsna_knee.")]
            if isinstance(node, ast.ImportFrom):
                if node.level == 1:
                    names = [node.module] if node.module else [alias.name for alias in node.names]
                elif not node.level and node.module and node.module.startswith("rsna_knee."):
                    names = [node.module.removeprefix("rsna_knee.")]
                elif not node.level and node.module == "rsna_knee":
                    names = [alias.name for alias in node.names]
                elif node.level > 1:
                    raise ValueError("Unresolved relative analysis import")
            for name in names:
                candidate = package.joinpath(*name.split(".")).with_suffix(".py")
                if candidate.is_file():
                    pending.append(candidate)
    return sorted(found)


def prepare(medical_root, anchor_root, destination, output_directory, anchor_verifier):
    medical_root = Path(medical_root).resolve()
    anchor_root = Path(anchor_root).resolve()
    destination = Path(destination).resolve()
    output_directory = Path(output_directory).resolve()
    anchor_verifier = Path(anchor_verifier).resolve()
    require(not destination.exists(), "Immutable Ticket71 preparation already exists")
    require(output_directory.is_relative_to(anchor_root), "Exact anchor run output directory required")
    medical_plan = read(medical_root / "plan.json")
    anchor_request = read(anchor_root / "request.json")
    require(anchor_request["study_count"] == 58 and anchor_request["model_fitting"] is False
            and anchor_request["competition_submission"] is False, "Complete inference-only exact anchor preparation required")
    require(medical_plan["candidate_promotion"] is False, "Medical plan must remain provisional")
    # Read the fold file's hash only. Its labels are loaded after full admission.
    fold_path = Path(medical_plan["fold_manifest_path"])
    paths = [Path(__file__), PROJECT / "scripts/ticket71_scored_anchor_analysis.py", anchor_verifier]
    paths.extend(analysis_source_dependencies(PROJECT / "src"))
    issue_path = PROJECT / ".scratch/rsna-leaderboard/issues/71-compare-medical-heads-with-scored-anchor.md"
    destination.mkdir(mode=0o700)
    snapshot = destination / "source-snapshot"
    snapshot.mkdir()
    frozen = []
    for path in sorted(set(paths)):
        require(path.is_relative_to(PROJECT), "Analysis/verifier sources must belong to this repository")
        copy = snapshot / path.relative_to(PROJECT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copy)
        frozen.extend(dict(path=str(value), sha256=digest(value)) for value in (path, copy))
    plan = dict(
        schema_version="rsna-ticket71-analysis-plan-v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol=PROTOCOL, prospective_issue_sha256=digest(issue_path), run_root=str(destination),
        medical_root=str(medical_root), medical_plan=dict(path=str(medical_root / "plan.json"), sha256=digest(medical_root / "plan.json")),
        anchor_root=str(anchor_root), anchor_request=dict(path=str(anchor_root / "request.json"), sha256=digest(anchor_root / "request.json")),
        anchor_output_directory=str(output_directory), anchor_verifier_path=str(anchor_verifier),
        fold_manifest=dict(path=str(fold_path), sha256=digest(fold_path)),
        fold_manifest_checksum=medical_plan["fold_manifest_checksum"], source_snapshot=str(snapshot), frozen_files=frozen,
        frozen_local_module_paths=[str(path.relative_to(PROJECT)) for path in paths if path.is_relative_to(PROJECT / "src")],
        analysis_runtime=dict(python=platform.python_version(), numpy=importlib.metadata.version("numpy"),
                              torch=importlib.metadata.version("torch")),
        final_inputs_admitted=False, resource_decision_admitted=False, launched=False,
        candidate_promotion=False, competition_submission=False,
    )
    with (destination / "plan.json").open("x") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    checksum = digest(destination / "plan.json")
    (destination / "plan-checksum.txt").write_text(checksum + "\n")
    print(json.dumps(dict(status="PREPARED_ONLY", plan_sha256=checksum, root=str(destination),
                         final_inputs_admitted=False, resource_decision_admitted=False, launched=False)))
    return checksum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("medical_root", type=Path)
    parser.add_argument("anchor_root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("anchor_verifier", type=Path)
    args = parser.parse_args()
    prepare(args.medical_root, args.anchor_root, args.destination, args.output_directory, args.anchor_verifier)
