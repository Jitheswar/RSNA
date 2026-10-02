"""Freeze Ticket74 code and protocol without loading final predictions or labels."""
from datetime import datetime, timezone
import ast
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys

PROJECT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(PROJECT / "src"))
from rsna_knee.full_backbone_anchor_comparison import PROTOCOL, PROTOCOL_SHA, digest, read, require

ROOT = PROJECT / "artifacts/private/ticket74-full-backbone-anchor-analysis-v1"
PROSPECTIVE = PROJECT / ".scratch/rsna-leaderboard/protocols/ticket74-full-backbone-anchor-v1.json"

def analysis_source_dependencies(source_root):
    """Freeze the static local import closure, leaving unrelated adapters editable."""
    source_root = Path(source_root).resolve()
    package = source_root / "rsna_knee"
    pending = [package / "__init__.py", package / "full_backbone_anchor_comparison.py"]
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



def prepare():
    require(not ROOT.exists(), "Immutable Ticket74 preparation already exists")
    require(digest(PROSPECTIVE) == PROTOCOL_SHA and read(PROSPECTIVE) == PROTOCOL, "Exact prospective protocol required")
    fit = PROJECT / "artifacts/private/ticket64-full-backbone-comparison-v4"
    anchor = PROJECT / "artifacts/private/ticket70-scored-anchor-gold58-v4"
    require(digest(fit / "plan.json") == PROTOCOL["fit_plan_sha256"]
            and digest(anchor / "request.json") == PROTOCOL["anchor_request_sha256"], "Exact registered routes required")
    fit_plan = read(fit / "plan.json")
    cohort = read(fit_plan["cohort_parent_plan"])
    manifest = next(Path(r["path"]) for r in cohort["frozen_files"] if Path(r["path"]).name == "fold-manifest.json")
    require(digest(manifest) == next(r["sha256"] for r in cohort["frozen_files"] if Path(r["path"]) == manifest), "Original manifest hash differs")
    independent = read(anchor / "independent-verification.json")
    require(digest(anchor / "independent-verification.json") == PROTOCOL["anchor_independent_verification_sha256"]
            and independent["passed"] is True, "Actual exact anchor proof required")
    output = Path(independent["output_directory"])
    require(output.is_relative_to(anchor), "Exact anchor output directory required")
    paths = [Path(__file__), PROJECT / "scripts/ticket74_full_backbone_anchor_analysis.py",
        PROJECT / "scripts/verify_ticket70_exact_anchor_outputs_v4.py"]
    paths.extend(analysis_source_dependencies(PROJECT / "src"))
    ROOT.mkdir(mode=0o700)
    snapshot = ROOT / "source-snapshot"
    snapshot.mkdir()
    frozen = []
    for path in sorted(set(paths)):
        require(path.is_relative_to(PROJECT), "Local frozen analysis sources required")
        copy = snapshot / path.relative_to(PROJECT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copy)
        frozen.extend(dict(path=str(p), sha256=digest(p)) for p in (path, copy))
    plan = dict(schema_version="rsna-ticket74-analysis-plan-v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol=PROTOCOL, prospective_protocol=dict(path=str(PROSPECTIVE), sha256=PROTOCOL_SHA),
        run_root=str(ROOT), fit_root=str(fit), fit_plan=dict(path=str(fit / "plan.json"), sha256=digest(fit / "plan.json")),
        anchor_root=str(anchor), anchor_request=dict(path=str(anchor / "request.json"), sha256=digest(anchor / "request.json")),
        anchor_output_directory=str(output), anchor_verifier_path=str(PROJECT / "scripts/verify_ticket70_exact_anchor_outputs_v4.py"),
        fold_manifest=dict(path=str(manifest), sha256=digest(manifest)), fold_manifest_checksum=cohort["fold_manifest_checksum"],
        source_snapshot=str(snapshot), frozen_files=frozen,
        frozen_local_module_paths=[str(p.relative_to(PROJECT)) for p in paths if p.is_relative_to(PROJECT / "src")],
        analysis_runtime=dict(python=platform.python_version(), numpy=importlib.metadata.version("numpy"), torch=importlib.metadata.version("torch")),
        final_inputs_admitted=False, resource_decision_admitted=False, launched=False, candidate_promotion=False, competition_submission=False)
    with (ROOT / "plan.json").open("x") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status="PREPARED_ONLY", plan_sha256=digest(ROOT / "plan.json"),
        final_inputs_admitted=False, resource_decision_admitted=False, launched=False)))
    return digest(ROOT / "plan.json")


if __name__ == "__main__":
    prepare()
