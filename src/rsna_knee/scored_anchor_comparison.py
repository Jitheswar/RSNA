"""The single prospective Ticket71 medical vote and strict evidence admission.

Only held-out-fold medical predictions belong here. The public anchor is an
adaptively selected development prediction, so this module admits review only.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation import _average_ranks, evaluate_oof_predictions
from .folds import FOLD_IDS, FoldManifest, load_fold_manifest
from .image_model_comparison import compare_image_model_predictions
from .targets import NUM_TARGETS, TARGET_COLUMNS


SCHEMA = "rsna-ticket71-fixed-scored-anchor-comparison-v1"
PROTOCOL = {
    "candidate": "rankpct(0.90*rankpct(scored_anchor)+0.10*rankpct(medical_report_OOF))",
    "medical_weight": 0.10,
    "targets": list(TARGET_COLUMNS),
    "study_count": 58,
    "rank_rule": "one-based average ties divided by complete cohort size, separately per target",
    "bootstrap_resamples": 2000,
    "bootstrap_seed": 20261002,
    "confidence_level": 0.95,
    "bootstrap_unit": "paired study",
    "primary": "medical_report_vote_minus_exact_scored_anchor",
    "diagnostics": [
        "reference_vote_minus_anchor", "reference_vote_minus_primary",
        "zero_image_vote_minus_anchor", "primary_minus_zero_image_vote",
        "standalone_medical_report", "standalone_medical_reference", "standalone_medical_report_zero_image",
    ],
    "delivery_review_gates": [
        "primary_paired_lower_bound_positive", "primary_all_original_fold_directions_positive",
        "ticket68_report_image_paired_lower_bound_and_all_fold_directions_positive",
        "primary_minus_zero_image_vote_paired_lower_bound_positive",
    ],
    "adaptive_development_cohort": True,
    "public_parent_selected": True,
    "patient_group_ids_unavailable": True,
    "candidate_promotion": False,
    "competition_submission": False,
    "offline_delivery_verified": False,
}
ARMS = ("reference-only", "report-supervision")
MEDICAL_HEAD_ID = "biomedclip-vitb16-projected-l2-plane-pooled-mlp32-v1"


class ScoredAnchorComparisonError(ValueError):
    """A prospective protocol or complete native prediction proof is missing."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise ScoredAnchorComparisonError(message)


def digest(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "Evidence must be a JSON object")
    return value


def input_files_digest(files: Mapping[str, Any]) -> str:
    """Bind a resource decision to inputs without a cyclic admission-file hash."""
    serialized = json.dumps(files, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def bound_file(binding: Mapping[str, Any], expected_path: str | Path) -> Path:
    """A content digest cannot redirect a role to a different run's evidence."""
    path = Path(binding["path"]).resolve()
    require(path == Path(expected_path).resolve(), "Exact evidence role path required")
    require(digest(path) == binding["sha256"], "Exact admitted evidence bytes required")
    return path


def _matrix(predictions: Mapping[str, Sequence[float]], order: Sequence[str]) -> np.ndarray:
    require(len(order) == 58 and len(set(order)) == 58 and set(predictions) == set(order),
            "Complete fixed58 prediction membership required")
    try:
        values = np.asarray([predictions[uid] for uid in order], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ScoredAnchorComparisonError("Finite bounded twelve-target predictions required") from exc
    require(values.shape == (58, NUM_TARGETS) and np.isfinite(values).all()
            and (values >= 0).all() and (values <= 1).all(), "Finite bounded twelve-target predictions required")
    return values


def rank_percentiles(values: np.ndarray) -> np.ndarray:
    """Preserve ties; never rank separately inside a fold or bootstrap sample."""
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (58, NUM_TARGETS) and np.isfinite(values).all(),
            "Ranks require the complete finite58-by12 cohort")
    return np.column_stack([_average_ranks(values[:, target]) / 58 for target in range(NUM_TARGETS)])


def fixed_vote(anchor: Mapping[str, Sequence[float]], medical: Mapping[str, Sequence[float]],
               order: Sequence[str]) -> dict[str, list[float]]:
    result = rank_percentiles(0.90 * rank_percentiles(_matrix(anchor, order))
                              + 0.10 * rank_percentiles(_matrix(medical, order)))
    return {uid: result[index].tolist() for index, uid in enumerate(order)}


def lower_bound_positive(comparison: Mapping[str, Any]) -> bool:
    lower = comparison["paired_study_bootstrap"]["macro_auc"]["lower"]
    return lower is not None and np.isfinite(lower) and lower > 0


def all_fold_directions_positive(comparison: Mapping[str, Any]) -> bool:
    folds = comparison["delta"]["folds"]
    return (len(folds) == 3 and [row["fold"] for row in folds] == list(FOLD_IDS)
            and all(row["delta_macro_auc"] is not None and np.isfinite(row["delta_macro_auc"])
                    and row["delta_macro_auc"] > 0 for row in folds))


def analyze(anchor: Mapping[str, Sequence[float]], report: Mapping[str, Sequence[float]],
            reference: Mapping[str, Sequence[float]], zero_image: Mapping[str, Sequence[float]],
            manifest: FoldManifest, *, expected_report_image: Mapping[str, Any]) -> dict[str, Any]:
    """Run the fixed primary and diagnostics once; diagnostics cannot be selected."""
    require(not manifest.superseded and len(manifest.gold_memberships) == 58,
            "Original corrected complete58 fold manifest required")
    order = sorted(member.study_instance_uid for member in manifest.gold_memberships)
    for arm in (anchor, report, reference, zero_image):
        _matrix(arm, order)
    primary = fixed_vote(anchor, report, order)
    reference_vote = fixed_vote(anchor, reference, order)
    zero_vote = fixed_vote(anchor, zero_image, order)
    pairs = {
        "primary": (anchor, primary),
        "reference_vote_minus_anchor": (anchor, reference_vote),
        "reference_vote_minus_primary": (primary, reference_vote),
        "zero_image_vote_minus_anchor": (anchor, zero_vote),
        "primary_minus_zero_image_vote": (zero_vote, primary),
        "report_image_signal": (zero_image, report),
    }
    comparisons = {
        name: compare_image_model_predictions(baseline, candidate, manifest,
                 bootstrap_resamples=2000, bootstrap_seed=20261002, confidence_level=0.95)
        for name, (baseline, candidate) in pairs.items()
    }
    require(comparisons["report_image_signal"] == expected_report_image,
            "Ticket68's complete fixed native report-image comparison must reproduce exactly")
    gates = {
        "primary_paired_lower_bound_positive": lower_bound_positive(comparisons["primary"]),
        "primary_all_original_fold_directions_positive": all_fold_directions_positive(comparisons["primary"]),
        "ticket68_report_image_paired_lower_bound_and_all_fold_directions_positive": (
            lower_bound_positive(comparisons["report_image_signal"])
            and all_fold_directions_positive(comparisons["report_image_signal"])),
        "primary_minus_zero_image_vote_paired_lower_bound_positive": lower_bound_positive(
            comparisons["primary_minus_zero_image_vote"]),
    }
    standalone = {name: evaluate_oof_predictions(predictions, manifest, include_diagnostics=False).to_public_dict()
                  for name, predictions in (("medical_report", report), ("medical_reference", reference),
                                            ("medical_report_zero_image", zero_image))}
    return {
        "schema_version": SCHEMA, "protocol": PROTOCOL, "comparisons": comparisons,
        "standalone": standalone, "gates": gates,
        "disposition": "provisional_delivery_review" if all(gates.values()) else "reject_fixed_medical_overlay",
        "primary_predictions": primary, "reference_vote_predictions": reference_vote,
        "zero_image_vote_predictions": zero_vote, "candidate_promotion": False,
        "competition_submission": False, "offline_delivery_verified": False,
        "anchor_is_clean_oof": False, "patient_group_ids_unavailable": True,
        "limitation": "Adaptively reused Gold58 and publicly selected parent; review only, no untouched generalization or leaderboard claim.",
    }


def successful_cpu_terminal(directory: Path, job: int | str, limit: str,
                            bindings: Mapping[str, Any], prefix: str) -> tuple[int, str]:
    terminal = bound_file(bindings[prefix + "_terminal"], directory / f"controller-{job}/terminal-controller.txt")
    observation = read(bound_file(bindings[prefix + "_observation"], directory / f"controller-{job}/terminal-observation.json"))
    fields = dict(token.split("=", 1) for token in terminal.read_text().split() if "=" in token)
    expected = dict(JobId=str(job), JobState="COMPLETED", ExitCode="0:0", Partition="cluster1-cpu",
                    NumNodes="1", NumCPUs="16", NumTasks="1", MinMemoryNode="64G", TimeLimit=limit,
                    OverSubscribe="OK", Requeue="0", Restarts="0")
    require(str(observation["job_id"]) == str(job) and observation["state"] == "COMPLETED"
            and observation["returncode"] == 0 and all(fields.get(key) == value for key, value in expected.items())
            and "gpu" not in fields.get("TRES", "").lower(), "Exact successful sole CPU allocation required")
    hours, minutes, seconds = map(int, fields["RunTime"].split(":"))
    elapsed = hours * 3600 + minutes * 60 + seconds
    h, m, s = map(int, limit.split(":"))
    require(0 <= elapsed <= h * 3600 + m * 60 + s, "Terminal allocation exceeded its bound")
    return elapsed, digest(terminal)


def load_verified_medical(plan: Mapping[str, Any], bindings: Mapping[str, Any],
                          manifest: FoldManifest) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Native verifier and both successful terminal receipts precede run reads."""
    root = Path(plan["medical_root"])
    medical_plan_path = bound_file(plan["medical_plan"], root / "plan.json")
    medical_plan = read(medical_plan_path)
    checksum = digest(medical_plan_path)
    require(medical_plan["fold_manifest_checksum"] == manifest.manifest_checksum,
            "Medical and comparison folds differ")
    for row in medical_plan["frozen_files"]:
        require(digest(row["path"]) == row["sha256"], "Frozen medical source/input bytes changed")
    fit = read(bound_file(bindings["medical_fit_accepted"], root / "fit/accepted-launch.json"))
    verification = read(bound_file(bindings["medical_verification_accepted"], root / "verification/accepted-launch.json"))
    for receipt in (fit, verification):
        require(receipt["plan_sha256"] == checksum and receipt["launch_attempt"] == 1,
                "Both exact sole medical allocations required")
    fit_seconds, fit_terminal_sha = successful_cpu_terminal(root / "fit", fit["job_id"], "00:30:00", bindings, "medical_fit")
    verification_seconds, verification_terminal_sha = successful_cpu_terminal(
        root / "verification", verification["job_id"], "03:00:00", bindings, "medical_verification")
    independent = read(bound_file(bindings["medical_independent"], root / "verification/independent-validation.json"))
    terminal = read(bound_file(bindings["medical_terminal_verification"], root / "terminal-verification.json"))
    analysis_path = bound_file(bindings["medical_analysis"], root / "analysis.json")
    require(independent["status"] == terminal["status"] == "COMPLETE" and terminal["passed"] is True
            and independent["plan_sha256"] == terminal["plan_sha256"] == checksum
            and str(independent["fit_job_id"]) == str(fit["job_id"])
            and str(independent["job_id"]) == str(terminal["verification_job_id"]) == str(verification["job_id"])
            and independent["fit_terminal_sha256"] == fit_terminal_sha
            and independent["fit_allocated_cpu_node_seconds"] == fit_seconds
            and terminal["verification_terminal_sha256"] == verification_terminal_sha
            and terminal["allocated_verification_cpu_node_seconds"] == verification_seconds
            and terminal["independent_validation_sha256"] == digest(root / "verification/independent-validation.json")
            and terminal["analysis_sha256"] == independent["analysis_sha256"] == digest(analysis_path),
            "Complete bound medical native and terminal chain required")
    require(independent["verified_final_heads"] == 6 and independent["original_mri_studies"] == 58
            and independent["verified_medical_predictions"] == independent["verified_resnet_predictions"] == 116
            and np.isfinite(independent["maximum_prediction_abs_error"])
            and 0 <= independent["maximum_prediction_abs_error"] <= 1e-6
            and np.isfinite(independent["maximum_medical_feature_abs_error"])
            and 0 <= independent["maximum_medical_feature_abs_error"] <= 1e-6
            and independent["all_fitted_targets_masks_init_exposures_adam_checked"] is True
            and independent["all_fixed_native_comparisons_reproduced"] is True
            and not independent["candidate_promotion"] and not terminal["candidate_promotion"],
            "All58 original-MRI medical/parent/zero predictions and six fitted heads must be verified")
    expected = {member.study_instance_uid: member.fold for member in manifest.gold_memberships}
    source_audit = independent["source_audit"]
    require(len(source_audit) == 58 and {row["study_instance_uid"] for row in source_audit} == set(expected)
            and independent["original_mri_valid_slices"] == sum(row["valid_slices"] for row in source_audit)
            and independent["original_mri_valid_slices"] > 0, "Complete original-MRI source audit required")
    head_audits = independent["audits"]
    require(len(head_audits) == 6 and {(row["fold"], row["arm"]) for row in head_audits}
            == {(fold, arm) for fold in FOLD_IDS for arm in ARMS}
            and all(row["all_targets_masks_init_exposures_adam_checked"] is True for row in head_audits),
            "Exactly six independently audited final medical heads required")
    analysis = read(analysis_path)
    require(analysis["status"] == "COMPLETE" and analysis["plan_sha256"] == checksum
            and str(analysis["job_id"]) == str(fit["job_id"]) and len(analysis["run_records"]) == 6
            and analysis["disposition"] == independent["disposition"] == terminal["disposition"]
            and not analysis["candidate_promotion"], "All six bound medical run records required")
    medical: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    zero: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    seen: set[tuple[int, str]] = set()
    for registered in analysis["run_records"]:
        fold, arm = registered["fold"], registered["arm"]
        require(fold in FOLD_IDS and arm in ARMS and (fold, arm) not in seen, "Exactly one medical head per original fold and arm required")
        seen.add((fold, arm))
        directory = root / arm / f"fold-{fold}"
        record = read(bound_file(registered, directory / "run.json"))
        held = {uid for uid, original_fold in expected.items() if original_fold == fold}
        fold_row = next(row for row in medical_plan["folds"] if row["fold"] == fold)
        identity = dict(plan_sha256=checksum, fold=fold, arm=arm, head_id=MEDICAL_HEAD_ID,
                        feature_identity_sha256=medical_plan["feature_identity_sha256"],
                        fold_manifest_checksum=manifest.manifest_checksum, pooled_sha256=medical_plan["medical_pooled_sha256"])
        head_audit = next(row for row in head_audits if row["fold"] == fold and row["arm"] == arm)
        require(record["status"] == "COMPLETE" and record["identity"] == identity
                and str(record["job_id"]) == str(fit["job_id"])
                and set(record["held_out_study_uids"]) == held and len(record["held_out_study_uids"]) == len(held)
                and record["held_out_study_uids"] == fold_row["gold_heldout_study_uids"]
                and record["training_study_uids"] == fold_row["gold_training_study_uids"]
                    + (fold_row["weak_training_study_uids"] if arm == ARMS[1] else [])
                and set(record["predictions"]) == set(record["zero_image_predictions"]) == held
                and not set(record["training_study_uids"]) & held and record["held_out_targets_masked"] is True
                and record["counts"]["successful_optimizer_steps"] == record["counts"]["attempted_batches"] == 7500
                and record["counts"]["skipped_batches"] == 0 and not record["candidate_promotion"]
                and digest(directory / "checkpoints/final.pt") == record["checkpoint_sha256"] == head_audit["checkpoint_sha256"],
                "Use only the original held-out-fold final medical and zero-image predictions")
        require(not set(medical[arm]) & held, "Duplicate held-out medical predictions")
        medical[arm].update(record["predictions"])
        zero[arm].update(record["zero_image_predictions"])
    require(seen == {(fold, arm) for fold in FOLD_IDS for arm in ARMS}, "All six final medical heads required")
    for predictions in (*medical.values(), *zero.values()):
        _matrix(predictions, sorted(expected))
    return medical[ARMS[1]], medical[ARMS[0]], zero[ARMS[1]], analysis["comparisons"]["report_image_signal"]


def load_anchor_csv(path: Path, order: Sequence[str]) -> dict[str, list[float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == ["StudyInstanceUID", *TARGET_COLUMNS], "Exact anchor CSV target order required")
        rows = list(reader)
    require([row["StudyInstanceUID"] for row in rows] == list(order), "Exact original anchor58 membership/order required")
    predictions = {row["StudyInstanceUID"]: [float(row[target]) for target in TARGET_COLUMNS] for row in rows}
    _matrix(predictions, order)
    return predictions


def validate_anchor_runtime_proofs(root: Path, output: Path, request: Mapping[str, Any],
                                   receipt: Mapping[str, Any], independent: Mapping[str, Any]) -> None:
    """Recheck genuine MRI/runtime/member proof contents, beyond success flags."""
    proof_paths = {}
    for role, field in (("original_mri_evidence", "input_receipt"), ("runtime_evidence", "runtime_receipt")):
        descriptor = receipt[field]
        path = (output / descriptor["file"]).resolve()
        require(path.is_relative_to(output.resolve())
                and path == Path(independent[role]["path"]).resolve()
                and digest(path) == descriptor["sha256"] == independent[role]["sha256"],
                "Genuine runtime receipt must bind its independent MRI/runtime proof files")
        proof_paths[role] = path
    input_receipt = receipt["input_receipt"]
    require(input_receipt["identity_sha256"] == request["original_mri_identity_sha256"]
            and input_receipt["dicom_files"] == request["original_mri_dicom_files"] > 0
            and input_receipt["dicom_bytes"] == request["original_mri_dicom_bytes"] > 0,
            "All original-MRI DICOM members and bytes must match the fixed contract")
    contract = read(root / "original-mri-contract.json")
    actual = read(proof_paths["original_mri_evidence"])
    identity = dict(schema_version="rsna-ticket70-frozen-original-mri-identity-v3", studies=actual["studies"])
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(actual["schema_version"] == "rsna-ticket70-original-mri-byte-proof-v3" and actual["status"] == "COMPLETE"
            and actual["study_count"] == len(actual["studies"]) == 58
            and actual["studies"] == contract["studies"] and len({row["study_uid"] for row in actual["studies"]}) == 58
            and actual["identity_sha256"] == identity_sha == contract["identity_sha256"] == request["original_mri_identity_sha256"]
            and actual["dicom_files"] == sum(row["dicom_files"] for row in actual["studies"]) == request["original_mri_dicom_files"]
            and actual["dicom_bytes"] == sum(row["dicom_bytes"] for row in actual["studies"]) == request["original_mri_dicom_bytes"]
            and actual["target_labels_uploaded"] is False, "Fresh all58 mounted DICOM bytes must reproduce the exact original contract")
    runtime = read(proof_paths["runtime_evidence"])
    require(runtime["schema_version"] == "rsna-ticket70-actual-runtime-v3" and isinstance(runtime["python"], str)
            and bool(runtime["python"]) and runtime["torch_cuda"] is not None and runtime["cudnn"] is not None
            and runtime["gpu_count"] == 2 and len(runtime["gpus"]) == 2
            and [gpu["index"] for gpu in runtime["gpus"]] == [0, 1]
            and all("T4" in gpu["name"] and gpu["capability"] == [7, 5]
                    and gpu["total_memory_bytes"] > 0 for gpu in runtime["gpus"])
            and all(isinstance(runtime["packages"][name], str) and runtime["packages"][name]
                    for name in ("numpy", "pandas", "pydicom", "torch", "torchvision", "timm", "scipy", "Pillow")),
            "Genuine installed runtime and two actual CUDA T4 devices must be recorded")
    execution = receipt["execution_receipt"]
    execution_log = (output / execution["execution_log_file"]).resolve()
    dino = execution["dino"]
    require(execution["schema_version"] == "rsna-ticket70-complete-member-proof-v3"
            and execution["rejection_events"] == [] and execution["stdout_and_stderr"] is True
            and execution["unterminated_tails_checked"] is True and execution["logged_lines"] > 0
            and execution_log.is_relative_to(output.resolve()) and digest(execution_log) == execution["execution_log_sha256"]
            and bool(dino["planned"]) and len(set(dino["planned"])) == len(dino["planned"])
            and bool(dino["expected_public"]) and len(set(dino["expected_public"])) == len(dino["expected_public"])
            and dino["planned"] == dino["completed"] and dino["expected_public"] == dino["completed_public"],
            "Every planned ensemble member and both output streams must pass the no-drop proof")
    with execution_log.open() as handle:
        lines = [json.loads(line) for line in handle]
    require(len(lines) == execution["logged_lines"] and all(row["stream"] in ("stdout", "stderr")
            and isinstance(row["line"], str) and row["rejection"] is None for row in lines),
            "Complete bound execution log must contain no rejected or invalid records")


def load_verified_anchor(plan: Mapping[str, Any], bindings: Mapping[str, Any],
                         manifest: FoldManifest) -> dict[str, list[float]]:
    """Check the exact accepted remote version and native receipt before CSV reads.

    The final input admission must bind the independent verifier's MRI and
    runtime evidence files. Neither a prepared notebook nor a COMPLETE status
    alone can admit predictions. Their detailed schema is checked by Ticket70's
    frozen independent verifier, whose own bytes are frozen in the analysis plan.
    """
    root = Path(plan["anchor_root"])
    output = Path(plan["anchor_output_directory"])
    require(output.resolve().is_relative_to(root.resolve()), "Anchor output must belong to its exact admitted run")
    request_path = bound_file(plan["anchor_request"], root / "request.json")
    request = read(request_path)
    checksum = digest(request_path)
    require(request["study_count"] == 58 and request["scored_submission_ref"] == 56705536
            and request["scored_script_version_id"] == 354125287 and request["scored_public_score"] == "0.936"
            and request["model_fitting"] is False and request["competition_submission"] is False,
            "Exact scored anchor and inference-only scope required")
    require(digest(root / "development.ipynb") == request["notebook_sha256"]
            and digest(root / "kernel-metadata.json") == request["metadata_sha256"],
            "Exact admitted source and asset/runtime metadata bytes required")
    for name, checksum_value in request["frozen_files"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()) and digest(path) == checksum_value,
                "Frozen anchor source bytes changed")
    require(digest(root / "original-mri-contract.json") == request["original_mri_contract_sha256"],
            "Exact original-MRI contract bytes required")
    accepted = read(bound_file(bindings["anchor_accepted"], root / "accepted-launch.json"))
    terminal_path = bound_file(bindings["anchor_terminal"], root / "kernel-terminal.json")
    terminal = read(terminal_path)
    independent = read(bound_file(bindings["anchor_independent"], root / "independent-verification.json"))
    require(accepted["request_sha256"] == checksum and accepted["launch_attempt"] == 1
            and accepted["kernel_id"] == request["kernel_id"] and type(accepted["kernel_version"]) is int
            and accepted["kernel_version"] > 0 and type(accepted["kernel_numeric_id"]) is int
            and accepted["kernel_numeric_id"] > 0, "Exact sole anchor remote version required")
    identity = {key: accepted[key] for key in ("kernel_id", "kernel_version")}
    require(terminal["status"] == terminal["response"]["status"] == "COMPLETE"
            and terminal["request_sha256"] == checksum and terminal["launch_attempt"] == 1
            and all(terminal.get(key) == value for key, value in identity.items())
            and independent["status"] == "COMPLETE" and independent["passed"] is True
            and independent["terminal_state"] == "COMPLETE" and independent["request_sha256"] == checksum
            and all(independent.get(key) == value for key, value in identity.items())
            and independent["kernel_numeric_id"] == accepted["kernel_numeric_id"]
            and independent["accepted_launch_sha256"] == digest(root / "accepted-launch.json")
            and independent["terminal_sha256"] == digest(terminal_path),
            "Exact successful remote terminal and independent identity chain required")
    require(independent["original_mri_studies"] == 58
            and all(independent[key] is True for key in ("remote_source_verified", "remote_settings_verified",
                    "asset_runtime_envelope_verified", "exact_recipe_verified", "no_fallback_verified")),
            "Native MRI, exact source/settings/assets/runtime and no-fallback verification required")
    receipt_path = bound_file(bindings["anchor_receipt"], output / "exact-scored-anchor-receipt.json")
    predictions_path = bound_file(bindings["anchor_predictions"], output / "development-exact-scored-anchor.csv")
    require(independent["output_receipt_sha256"] == digest(receipt_path)
            and independent["output_sha256"] == digest(predictions_path), "Exact independently verified anchor output bytes required")
    # These are explicit byte bindings, never paths supplied by a prediction CSV.
    for role in ("original_mri_evidence", "runtime_evidence"):
        descriptor = independent[role]
        path = Path(descriptor["path"]).resolve()
        require(path.is_relative_to(output.resolve()), "Anchor proof must belong to downloaded terminal output")
        bound_file(bindings["anchor_" + role], path)
        require(digest(path) == descriptor["sha256"], "Exact independent MRI/runtime proof bytes required")
    receipt = read(receipt_path)
    require(receipt["schema_version"] == "rsna-ticket70-exact-scored-anchor-output-v1"
            and receipt["status"] == "COMPLETE" and receipt["study_count"] == 58
            and receipt["development_uid_sha256"] == request["development_uid_sha256"]
            and receipt["final_sha256"] == digest(predictions_path)
            and receipt["assigned_gpu_count"] == 2 and receipt["model_fitting"] is False
            and receipt["competition_submission"] is False and receipt["fallback_events"] == 0
            and receipt["coat_receipt"]["completed"] is True and receipt["coat_receipt"]["legacy_success_count"] == 58
            and np.isfinite(receipt["elapsed_seconds"]) and 0 < receipt["elapsed_seconds"] <= request["execution_watchdog_seconds"],
            "Exact complete58 runtime receipt with both Raptor arms and no fallback required")
    validate_anchor_runtime_proofs(root, output, request, receipt, independent)
    require(receipt["original_cells_sha256"] == [row["original_sha256"] for row in request["source_cells"]]
            and len(receipt["stages"]) == 5 and [row["index"] for row in receipt["stages"]] == list(range(5))
            and len(request["source_cells"]) == 5, "All five exact scored source stages required")
    with predictions_path.open(newline="") as handle:
        order = [row["StudyInstanceUID"] for row in csv.DictReader(handle)]
    uid_sha = hashlib.sha256("".join(uid + "\n" for uid in order).encode()).hexdigest()
    require(len(order) == 58 and len(set(order)) == 58 and set(order) == {
        member.study_instance_uid for member in manifest.gold_memberships}
        and uid_sha == request["development_uid_sha256"], "Exact unchanged original-MRI Gold58 cohort/order required")
    for row, source in zip(receipt["stages"], request["source_cells"], strict=True):
        stage_path = output / f"development-stage-{row['index']}.csv"
        require(digest(stage_path) == row["sha256"] and row["recipe_sha256"] == source["relocated_sha256"],
                "Every exact source stage output must be bound")
        load_anchor_csv(stage_path, order)
    require(receipt["stages"][-1]["sha256"] == digest(predictions_path), "Final anchor must be the unchanged fifth stage")
    require(bool(receipt["loaded_weights"]) and all(type(row["bytes"]) is int and row["bytes"] > 0
            and isinstance(row["sha256"], str) and len(row["sha256"]) == 64 for row in receipt["loaded_weights"].values()),
            "Actual loaded model byte evidence required")
    return load_anchor_csv(predictions_path, order)


def load_verified_inputs(plan: Mapping[str, Any], admission: Mapping[str, Any],
                         plan_sha256: str) -> tuple[FoldManifest, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(plan["schema_version"] == "rsna-ticket71-analysis-plan-v1" and plan["protocol"] == PROTOCOL
            and admission["schema_version"] == "rsna-ticket71-analysis-input-admission-v1"
            and admission["plan_sha256"] == plan_sha256 and admission["all_inputs_verified"] is True
            and admission["candidate_promotion"] is False and admission["competition_submission"] is False,
            "Exact fixed protocol and complete input admission required")
    for row in plan["frozen_files"]:
        require(digest(row["path"]) == row["sha256"], "Frozen analysis/verifier/runtime source changed")
    manifest_path = bound_file(plan["fold_manifest"], plan["fold_manifest"]["path"])
    manifest = load_fold_manifest(manifest_path)
    require(not manifest.superseded and manifest.manifest_checksum == plan["fold_manifest_checksum"]
            and len(manifest.gold_memberships) == 58, "Exact original corrected58 folds required")
    bindings = admission["files"]
    report, reference, zero, image_signal = load_verified_medical(plan, bindings, manifest)
    anchor = load_verified_anchor(plan, bindings, manifest)
    return manifest, anchor, report, reference, zero, image_signal
