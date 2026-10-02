"""Guard native MRI byte membership and complete original ensemble execution."""

import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("ticket70_runtime", Path(__file__).resolve().parents[1] / "scripts/prepare_ticket70_verified_runtime_envelope.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def proof_namespace(tmp_path, inputs=None):
    env = {"_RSNA70_EXPECTED_INPUTS": inputs}
    exec(module.RUNTIME_PROOF, env)
    env["_rsna70_work"] = tmp_path
    env["_rsna70_log"] = tmp_path / "development-execution.jsonl"
    return env


@pytest.mark.parametrize("line", [
    "MEMBER model-a failed on cuda:0; retrying on peer device",
    "MEMBER model-a failed on cuda:1; dropped -- costs one vote, not the run",
    "model-a: degenerate predictions; not banked",
    "public-frontier vote omitted because only 1 / 4 windows completed",
    "surrendering 3 member(s); not one more fits",
    "public-frontier fallback not emitted: 3 / 4 required public members completed",
    "study failed: RuntimeError: input",
    "CoAtNet branch failed; retaining transformer submission",
    "[DINOsaur V4.2] legacy checkpoint disabled",
    "[DINOsaur V4.2] legacy complement unavailable",
    "[DINOsaur V4.2] a study FALLBACK",
    "[DINOsaur V4.2] legacy study 2 fallback: error",
    "[V18] calibration skipped safely",
    "[DINOsaur V4.2] final fusion failed",
    "CoAtNet output unavailable; submission.csv remains",
])
def test_rejects_partial_members_and_studies_on_either_stream_without_newline(tmp_path, line):
    env = proof_namespace(tmp_path)
    for name in ("stdout", "stderr"):
        monitor = env["_Rsna70Monitor"](io.StringIO(), name)
        monitor.write(line[:10])
        monitor.flush()
        assert monitor.fallbacks == 0
        monitor.write(line[10:])
        monitor.finish()
        assert monitor.fallbacks == 1
    with pytest.raises(RuntimeError, match="failed/partial"):
        env["_rsna70_verify_execution"]()


def test_complete_execution_requires_exact_planned_and_public_member_membership(tmp_path):
    env = proof_namespace(tmp_path)
    monitor = env["_Rsna70Monitor"](io.StringIO(), "stdout")
    monitor.write("banked all members\n")
    monitor.finish()
    actual = dict(planned=["a", "b", "legacy"], completed=["a", "b", "legacy"],
                  expected_public=["a", "b"], completed_public=["a", "b"])
    env["_rsna70_dino"].append(actual)
    assert env["_rsna70_verify_execution"]()["dino"] == actual
    actual["completed"].pop()
    with pytest.raises(RuntimeError, match="member completion"):
        env["_rsna70_verify_execution"]()


@pytest.mark.parametrize("change", ["missing_public", "duplicate", "multiple_calls", "no_calls"])
def test_incomplete_or_ambiguous_native_completion_cannot_pass(tmp_path, change):
    env = proof_namespace(tmp_path)
    monitor = env["_Rsna70Monitor"](io.StringIO(), "stderr")
    monitor.write("normal execution\n"); monitor.finish()
    proof = dict(planned=["a"], completed=["a"], expected_public=["a"], completed_public=["a"])
    if change == "missing_public": proof["completed_public"] = []
    if change == "duplicate": proof["planned"] *= 2; proof["completed"] *= 2
    if change != "no_calls": env["_rsna70_dino"].append(proof)
    if change == "multiple_calls": env["_rsna70_dino"].append(proof)
    with pytest.raises(RuntimeError): env["_rsna70_verify_execution"]()


def native_fixture(tmp_path):
    root = tmp_path / "competition" / "train_series" / "test-study" / "test-series"
    root.mkdir(parents=True)
    (root / "a.dcm").write_bytes(b"original-a")
    (root / "b.dcm").write_bytes(b"original-b")
    rows = [["test-series/" + p.name, p.stat().st_size, module.digest(p)] for p in root.iterdir()]
    study = {"study_uid": "test-study", **module.input_aggregate(rows)}
    identity = {"schema_version": "rsna-ticket70-frozen-original-mri-identity-v3", "studies": [study]}
    inputs = {**identity, "identity_sha256": hashlib.sha256(module.canonical(identity).encode()).hexdigest()}
    return root, inputs


def test_native_mri_proof_rehashes_all_original_bytes_with_exact_membership(tmp_path):
    root, inputs = native_fixture(tmp_path)
    env = proof_namespace(tmp_path, inputs)
    result = env["_rsna70_verify_inputs"](root.parents[2], ["test-study"], module.digest)
    assert result["identity_sha256"] == inputs["identity_sha256"]
    assert result["dicom_files"] == 2
    assert json.loads((tmp_path / result["file"]).read_text())["target_labels_uploaded"] is False


@pytest.mark.parametrize("change", ["same_length_tamper", "missing", "extra", "renamed", "wrong_order", "symlink"])
def test_original_mri_tamper_or_membership_changes_are_rejected(tmp_path, change):
    root, inputs = native_fixture(tmp_path)
    ids = ["test-study"]
    if change == "same_length_tamper": (root / "a.dcm").write_bytes(b"modified-a")
    if change == "missing": (root / "b.dcm").unlink()
    if change == "extra": (root / "c.dcm").write_bytes(b"extra")
    if change == "renamed": (root / "a.dcm").rename(root / "c.dcm")
    if change == "wrong_order": ids = ["other-study"]
    if change == "symlink":
        (root / "a.dcm").unlink(); (root / "a.dcm").symlink_to(root / "b.dcm")
    env = proof_namespace(tmp_path, inputs)
    with pytest.raises(RuntimeError): env["_rsna70_verify_inputs"](root.parents[2], ids, module.digest)


@pytest.mark.parametrize("name", ["../a.dcm", "/series/a.dcm", "series//a.dcm", "series/../a.dcm", "series/a.txt", "series\\a.dcm"])
def test_unsafe_original_member_identity_is_rejected(name):
    with pytest.raises(ValueError): module.input_aggregate([[name, 1, "a" * 64]])


def test_duplicate_mri_entries_are_rejected():
    with pytest.raises(ValueError): module.input_aggregate([["series/a.dcm", 1, "a" * 64]] * 2)


def test_additive_wrapper_preserves_scientific_cells_and_rejects_ambiguous_insertion():
    preparer = Path(__file__).resolve().parents[1] / "scripts/prepare_ticket70_scored_anchor_development.py"
    values = [node for node in ast.parse(preparer.read_text()).body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "RUNTIME_CODE" for target in node.targets)]
    assert len(values) == 1
    recipe = tuple(f"original_cell_{index} = True\n" for index in range(5))
    code = f"ORIGINAL_CELLS = {recipe!r}\nSOURCE_CELLS = {recipe!r}\n" + ast.literal_eval(values[0].value)
    cells = module.recipe_constants(code)
    changed = module.instrument(code, {"studies": []})
    assert module.recipe_constants(changed) == cells
    with pytest.raises(ValueError, match="missing or ambiguous"):
        module.instrument(code.replace("    view = Path('/tmp/rsna_exact_scored_gold58')\n", ""), {})
    with pytest.raises(ValueError, match="missing or ambiguous"):
        module.instrument(code + "\n    view = Path('/tmp/rsna_exact_scored_gold58')\n", {})
