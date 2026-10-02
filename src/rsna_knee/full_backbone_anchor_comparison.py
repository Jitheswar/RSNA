"""Fixed prospective whole-backbone vote, with complete native input admission."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluation import evaluate_oof_predictions
from .folds import FoldManifest, load_fold_manifest
from .image_model_comparison import compare_image_model_predictions
from .scored_anchor_comparison import (
    _matrix, all_fold_directions_positive, bound_file, digest, fixed_vote,
    input_files_digest, load_verified_anchor, lower_bound_positive, read,
    require, successful_cpu_terminal,
)

PROTOCOL_SHA = "bfb077403cc3c2edbc445be099a94df6a7edf461b01396d1027553f4dc258416"
PROTOCOL: dict[str, Any] = {'adaptive_development_cohort': True,
 'analysis_resource_projection': {'additional_storage_bytes': 100000000,
                                  'external_spend_usd': 0,
                                  'fresh_complete_route_admission_required': True,
                                  'gpu_seconds': 0,
                                  'independent_cpu_node_seconds': 3600,
                                  'maximum_launches_each': 1,
                                  'primary_cpu_node_seconds': 3600,
                                  'tpu_seconds': 0},
 'anchor_independent_verification_sha256': '5a9682f7b2efbeeabf47f0f6c441edacb59f338f3e8661bf5dc7b94251a2d670',
 'anchor_native_prediction_sha256': '326819fa2c3649e08e00ccb62d3ae2888e1bfa3b179545f6e12af1b9f54c5c59',
 'anchor_request_sha256': 'ff7ae7ad97578582a5fd7a3b6e4058fb02fb8d3208c64fb3c5d782d596e64b4a',
 'basis': 'Evaluate whether the unchanged original whole-backbone treatment contributes '
          'complementary image information to the exact public-scored anchor. Define one candidate '
          'before any final full-backbone predictions or metrics exist.',
 'bootstrap_resamples': 2000,
 'bootstrap_seed': 20261002,
 'bootstrap_unit': 'paired study',
 'candidate': 'rankpct(0.90*rankpct(exact_scored_anchor)+0.10*rankpct(full_backbone_OOF))',
 'candidate_promotion': False,
 'competition_submission': False,
 'confidence_level': 0.95,
 'created_at_utc': '2026-10-02T18:01:00+00:00',
 'delivery_review_gates': ['actual_complete_independent_original_MRI_full_fit_verification',
                           'original_full_vs_matched_control_gate_passed',
                           'primary_paired_macro_AUC_lower_bound_positive',
                           'primary_all_three_original_fold_macro_AUC_directions_positive',
                           'full_image_signal_paired_macro_AUC_lower_bound_positive',
                           'full_vote_minus_zero_image_full_vote_paired_macro_AUC_lower_bound_positive'],
 'diagnostics_cannot_be_promoted': True,
 'final_full_predictions_present_when_declared': False,
 'fit_plan_sha256': '9844fd6a7e62f3f1aaf5d0678784f197b8373d702e14c19e669ccde5244a36c8',
 'fixed_diagnostics': ['standalone_full_backbone',
                       'standalone_matched_head_control',
                       'head_control_vote_minus_anchor',
                       'full_vote_minus_head_control_vote',
                       'zero_image_full_vote_minus_anchor',
                       'full_vote_minus_zero_image_full_vote'],
 'full_backbone_weight': 0.1,
 'historical_backend_terminals_verified': False,
 'medical_overlay_rejection_preserved': True,
 'no_weight_or_target_sweep': True,
 'offline_delivery_verified': False,
 'original_missing_allocation_charge_seconds': 39600,
 'original_strict_ancestry_passed': False,
 'patient_group_ids_unavailable': True,
 'primary': 'fixed_full_backbone_vote_minus_exact_scored_anchor',
 'public_parent_selected': True,
 'rank_rule': 'one-based average ties divided by complete58, per target; fix cohort ranks before '
              'paired bootstrap; no within-fold ranks',
 'resource_launch_admitted': False,
 'schema_version': 'rsna-ticket74-prospective-full-backbone-anchor-v1',
 'scientific_plan_sha256': '557230fec672f6db9252b1895a4d174652fe217c0388a8c9598bd749183aa985',
 'study_count': 58,
 'target_count': 12}
ARMS = ("head-continuation", "full-backbone")
SCHEMA = "rsna-ticket74-fixed-full-backbone-anchor-v1"


def analyze(anchor: Mapping[str, Sequence[float]], full: Mapping[str, Sequence[float]],
            control: Mapping[str, Sequence[float]], zero: Mapping[str, Sequence[float]],
            manifest: FoldManifest, *, expected_full_image: Mapping[str, Any],
            original_full_control_gate_passed: bool) -> dict[str, Any]:
    """Only the declared full vote may reach delivery review; no diagnostic selection."""
    require(not manifest.superseded and len(manifest.gold_memberships) == 58,
            "Original complete corrected58 folds required")
    require(type(original_full_control_gate_passed) is bool, "Actual original comparison gate required")
    order = sorted(m.study_instance_uid for m in manifest.gold_memberships)
    for values in (anchor, full, control, zero):
        _matrix(values, order)
    primary, head_vote, zero_vote = (fixed_vote(anchor, x, order) for x in (full, control, zero))
    pairs = dict(primary=(anchor, primary), head_control_vote_minus_anchor=(anchor, head_vote),
        full_vote_minus_head_control_vote=(head_vote, primary), zero_image_full_vote_minus_anchor=(anchor, zero_vote),
        full_vote_minus_zero_image_full_vote=(zero_vote, primary), full_image_signal=(zero, full))
    comparisons = {name: compare_image_model_predictions(base, candidate, manifest,
        bootstrap_resamples=2000, bootstrap_seed=20261002, confidence_level=.95)
        for name, (base, candidate) in pairs.items()}
    require(comparisons["full_image_signal"] == expected_full_image,
            "Original independently verified full image-information comparison must reproduce")
    gates = dict(actual_complete_independent_original_MRI_full_fit_verification=True,
        original_full_vs_matched_control_gate_passed=original_full_control_gate_passed,
        primary_paired_macro_AUC_lower_bound_positive=lower_bound_positive(comparisons["primary"]),
        primary_all_three_original_fold_macro_AUC_directions_positive=all_fold_directions_positive(comparisons["primary"]),
        full_image_signal_paired_macro_AUC_lower_bound_positive=lower_bound_positive(comparisons["full_image_signal"]),
        full_vote_minus_zero_image_full_vote_paired_macro_AUC_lower_bound_positive=lower_bound_positive(comparisons["full_vote_minus_zero_image_full_vote"]))
    return dict(schema_version=SCHEMA, protocol=PROTOCOL, comparisons=comparisons,
        standalone={name: evaluate_oof_predictions(value, manifest, include_diagnostics=False).to_public_dict()
                    for name, value in (("full_backbone", full), ("matched_head_control", control))},
        gates=gates, disposition="provisional_delivery_review" if all(gates.values()) else "reject_fixed_full_backbone_overlay",
        primary_predictions=primary, head_control_vote_predictions=head_vote, zero_image_vote_predictions=zero_vote,
        candidate_promotion=False, competition_submission=False, offline_delivery_verified=False,
        anchor_is_clean_oof=False, patient_group_ids_unavailable=True,
        historical_backend_terminals_verified=False, original_strict_ancestry_passed=False,
        original_missing_allocation_charge_seconds=39600,
        limitation="Adaptive Gold58 and public-selected parent; review only, with historical allocation gaps explicitly retained.")


def load_verified_backbone(plan: Mapping[str, Any], bindings: Mapping[str, Any], manifest: FoldManifest
                           ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    """Require actual independent native success before reading final predictions."""
    root = Path(plan["fit_root"])
    fit_path = bound_file(plan["fit_plan"], root / "plan.json")
    fit_sha = digest(fit_path)
    require(fit_sha == PROTOCOL["fit_plan_sha256"], "Exact configured whole-backbone route required")
    fit = read(fit_path)
    for row in fit["frozen_files"]:
        require(digest(row["path"]) == row["sha256"], "Frozen fitting input/configuration/source changed")
    native_root = root / "independent-original-mri-replay-v1"
    request_path = bound_file(bindings["fit_native_request"], native_root / "plan.json")
    request = read(request_path)
    require(request["fit_plan_sha256"] == fit_sha and request["resources"]["walltime_seconds"] == 5400,
            "Exact independent original-MRI request required")
    for row in request["frozen_files"]:
        require(digest(row["path"]) == row["sha256"], "Frozen native inputs/final states/source changed")
    accepted = read(bound_file(bindings["fit_native_accepted"], native_root / "accepted-launch.json"))
    request_sha = digest(request_path)
    require(accepted["plan_sha256"] == request_sha and accepted["launch_attempt"] == 1,
            "Exact sole independent native allocation required")
    _, terminal_sha = successful_cpu_terminal(native_root, accepted["job_id"], "01:30:00", bindings, "fit_native")
    native_path = bound_file(bindings["fit_native_validation"], native_root / "validation.json")
    native = read(native_path)
    terminal = read(bound_file(bindings["fit_terminal_verification"], root / "terminal-verification.json"))
    require(terminal["status"] == native["status"] == "COMPLETE" and terminal["passed"] is True
            and terminal["plan_sha256"] == native["fit_plan_sha256"] == fit_sha
            and terminal["native_request_sha256"] == native["plan_sha256"] == request_sha
            and str(terminal["verification_job_id"]) == str(native["job_id"]) == str(accepted["job_id"])
            and terminal["native_terminal_sha256"] == terminal_sha
            and terminal["native_validation_sha256"] == digest(native_path)
            and native["original_mri_studies"] == 58 and native["final_runs_verified"] == 6
            and native["final_predictions_verified"] == 116
            and native["original_pixels_targets_masks_exposures_verified"] is True
            and native["all_registered_comparisons_recomputed"] is True
            and 0 <= native["maximum_prediction_error"] <= 1e-6
            and native["gpu_count"] == native["external_spend_usd"] == 0,
            "Complete actual original-MRI states/source/prediction/metric proof required")
    for row in (native, terminal):
        require(row["historical_backend_terminals_verified"] is False and row["original_strict_ancestry_passed"] is False
                and row["original_missing_allocation_charge_seconds"] == 39600 and row["candidate_promotion"] is False,
                "Explicit historic terminal limitations required")
    expected_uids = {m.study_instance_uid for m in manifest.gold_memberships}
    require(len(native["source_audit"]) == 58
            and {r["study_instance_uid"] for r in native["source_audit"]} == expected_uids
            and len(native["audits"]) == 6
            and {(r["arm"], r["fold"]) for r in native["audits"]} == {(a, f) for a in ARMS for f in range(3)},
            "Every original MRI and all six models need native proof")
    analysis_path = bound_file(bindings["fit_analysis"], root / "analysis.json")
    analysis = read(analysis_path)
    require(terminal["analysis_sha256"] == digest(analysis_path) and analysis["status"] == "COMPLETE"
            and analysis["plan_sha256"] == fit_sha and analysis["scientific_plan_sha256"] == PROTOCOL["scientific_plan_sha256"]
            and type(analysis["provisional_review_gate_passed"]) is bool
            and native["provisional_review_gate_passed"] == analysis["provisional_review_gate_passed"]
            and analysis["candidate_promotion"] is False,
            "Exact complete independently recomputed original comparison required")
    cohort_path = Path(fit["cohort_parent_plan"])
    cohort = read(cohort_path)
    require(cohort["fold_manifest_checksum"] == manifest.manifest_checksum, "Original fitted and comparison folds differ")
    by_fold = {r["fold"]: r for r in cohort["folds"]}
    rows = analysis["run_records"]
    require(rows == request["final_run_records"] and len(rows) == 6
            and {(r["arm"], r["fold"]) for r in rows} == {(a, f) for a in ARMS for f in range(3)},
            "All six actual final models required")
    result: dict[str, dict[str, Any]] = {"full": {}, "control": {}, "zero": {}}
    prior = root.with_name("ticket64-full-backbone-comparison-v2")
    first = root.with_name("ticket64-full-backbone-comparison-v1")
    prior_sha = digest(prior / "plan.json")
    for row in rows:
        fold, arm = row["fold"], row["arm"]
        base = prior if arm == ARMS[0] else root
        run_path = base / arm / f"fold-{fold}" / ("run.json" if arm == ARMS[0] else "stage-7/stage-result.json")
        record = read(bound_file(bindings[f"fit_{arm}_{fold}"], run_path))
        checkpoint_base = (first if fold == 0 else prior) if arm == ARMS[0] else root
        checkpoint = checkpoint_base / arm / f"fold-{fold}/training/checkpoints/final.pt"
        require(Path(row["checkpoint_path"]) == checkpoint and Path(record["checkpoint_path"]) == checkpoint
                and digest(checkpoint) == row["checkpoint_sha256"] == record["checkpoint_sha256"]
                and record["status"] == "COMPLETE" and record["plan_sha256"] == (prior_sha if arm == ARMS[0] else fit_sha)
                and record["scientific_plan_sha256"] == PROTOCOL["scientific_plan_sha256"]
                and record["progress"]["successful_optimizer_steps"] == record["progress"]["attempted_batches"] == 7500,
                "Complete unchanged final budget/state identity required")
        held = {m.study_instance_uid for m in manifest.gold_memberships if m.fold == fold}
        require(set(record["predictions"]) == set(record["encoded_image_ablation"]) == held
                and set(by_fold[fold]["gold_heldout_study_uids"]) == held
                and not held & set(by_fold[fold]["gold_training_study_uids"]),
                "Exact original disjoint held-out prediction membership required")
        native_audit = [v for v in native["audits"] if v["fold"] == fold and v["arm"] == arm]
        require(len(native_audit) == 1 and native_audit[0]["checkpoint_sha256"] == row["checkpoint_sha256"]
                and all(0 <= native_audit[0][k] <= 1e-6 for k in ("native_prediction_error", "native_ablation_error", "parent_error")),
                "Each final model needs independent MRI prediction/ablation evidence")
        key = "control" if arm == ARMS[0] else "full"
        require(not set(result[key]) & held, "Duplicate held-out predictions")
        result[key].update(record["predictions"])
        if arm == ARMS[1]:
            result["zero"].update(record["encoded_image_ablation"])
    order = sorted(m.study_instance_uid for m in manifest.gold_memberships)
    for value in result.values():
        _matrix(value, order)
    return result["full"], result["control"], result["zero"], analysis["comparisons"]["full-backbone-image-signal"], analysis["provisional_review_gate_passed"]


def load_verified_inputs(plan: Mapping[str, Any], admission: Mapping[str, Any], plan_sha256: str
                         ) -> tuple[FoldManifest, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    require(plan["schema_version"] == "rsna-ticket74-analysis-plan-v1" and plan["protocol"] == PROTOCOL
            and admission["schema_version"] == "rsna-ticket74-analysis-input-admission-v1"
            and admission["plan_sha256"] == plan_sha256 and admission["all_inputs_verified"] is True
            and admission["candidate_promotion"] is False and admission["competition_submission"] is False,
            "Exact prospective protocol and complete native input admission required")
    protocol_path = bound_file(plan["prospective_protocol"], plan["prospective_protocol"]["path"])
    require(digest(protocol_path) == PROTOCOL_SHA and read(protocol_path) == PROTOCOL, "Prospective protocol changed")
    for row in plan["frozen_files"]:
        require(digest(row["path"]) == row["sha256"], "Frozen analysis/runtime/source changed")
    manifest = load_fold_manifest(bound_file(plan["fold_manifest"], plan["fold_manifest"]["path"]))
    require(not manifest.superseded and manifest.manifest_checksum == plan["fold_manifest_checksum"], "Original folds required")
    full, control, zero, image, gate = load_verified_backbone(plan, admission["files"], manifest)
    require(plan["anchor_request"]["sha256"] == PROTOCOL["anchor_request_sha256"]
            and admission["files"]["anchor_independent"]["sha256"] == PROTOCOL["anchor_independent_verification_sha256"]
            and admission["files"]["anchor_predictions"]["sha256"] == PROTOCOL["anchor_native_prediction_sha256"],
            "Exact prospectively declared scored anchor required")
    anchor = load_verified_anchor(plan, admission["files"], manifest)
    return manifest, anchor, full, control, zero, image, gate
