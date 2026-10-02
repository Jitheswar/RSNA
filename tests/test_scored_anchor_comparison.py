"""Synthetic guard coverage for the single prospective medical overlay."""

from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from rsna_knee import scored_anchor_comparison as comparison
from rsna_knee.folds import freeze_folds
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.targets import TARGET_COLUMNS


@pytest.fixture
def manifest():
    source = SourceManifest("train", tuple(SourceStudy(f"synthetic-{index:02}", True,
                           tuple(index % 2 for _ in TARGET_COLUMNS)) for index in range(58)), ())
    return freeze_folds(source, source_version="ticket71-synthetic-only", seed=31)


def predictions(manifest, value=.5):
    return {row.study_instance_uid: [value] * 12 for row in manifest.gold_memberships}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return {"path": str(path), "sha256": comparison.digest(path)}


def test_vote_has_full_cohort_average_ties_and_fixed_uniform_weight(manifest):
    order = sorted(predictions(manifest))
    anchor = {uid: [index // 2 / 29] * 12 for index, uid in enumerate(order)}
    medical = {uid: [1 - index / 58] * 12 for index, uid in enumerate(order)}
    actual = comparison.fixed_vote(anchor, medical, order)
    # Anchor tied pairs are split by the declared small medical vote. The large
    # gap between pairs retains their order; the final complete-cohort ranks
    # are therefore [2,1,4,3,...,58,57] / 58 in every target.
    expected = np.asarray([index + (2 if index % 2 == 0 else 0) for index in range(58)]) / 58
    assert np.array_equal(np.asarray([actual[uid][0] for uid in order]), expected)
    assert all(len(row) == 12 for row in actual.values())
    ties = comparison.rank_percentiles(np.full((58, 12), .5))
    assert np.array_equal(ties, np.full((58, 12), 29.5 / 58))


@pytest.mark.parametrize("problem", ["missing", "extra", "nan", "out_of_range", "wrong_target_count"])
def test_vote_rejects_reselected_or_invalid_cohorts(manifest, problem):
    order = sorted(predictions(manifest))
    anchor = predictions(manifest)
    medical = deepcopy(anchor)
    if problem == "missing": medical.pop(order[0])
    if problem == "extra": medical["extra"] = [.5] * 12
    if problem == "nan": medical[order[0]][0] = float("nan")
    if problem == "out_of_range": medical[order[0]][0] = 1.1
    if problem == "wrong_target_count": medical[order[0]].pop()
    with pytest.raises(comparison.ScoredAnchorComparisonError):
        comparison.fixed_vote(anchor, medical, order)


def test_real_fixed_metric_calls_and_diagnostics_cannot_become_candidate(manifest, monkeypatch):
    anchor = report = zero = predictions(manifest)
    reference = {row.study_instance_uid: list(row.reference_labels) for row in manifest.gold_memberships}
    original = comparison.compare_image_model_predictions
    image = original(zero, report, manifest, bootstrap_resamples=2000,
                     bootstrap_seed=20261002, confidence_level=.95)
    calls = []
    def observed(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(comparison, "compare_image_model_predictions", observed)
    actual = comparison.analyze(anchor, report, reference, zero, manifest, expected_report_image=image)
    assert len(calls) == 6 and all(call == dict(bootstrap_resamples=2000, bootstrap_seed=20261002,
                                               confidence_level=.95) for call in calls)
    assert actual["comparisons"]["reference_vote_minus_anchor"]["delta"]["pooled_macro_auc"] == .5
    assert actual["comparisons"]["primary"]["delta"]["pooled_macro_auc"] == 0
    assert actual["disposition"] == "reject_fixed_medical_overlay"
    assert not any(actual["gates"].values())
    assert actual["primary_predictions"] == comparison.fixed_vote(anchor, report, sorted(anchor))
    assert actual["candidate_promotion"] is actual["competition_submission"] is actual["offline_delivery_verified"] is False
    for name in ("primary", "reference_vote_minus_anchor", "reference_vote_minus_primary", "zero_image_vote_minus_anchor",
                 "primary_minus_zero_image_vote", "report_image_signal"):
        metric = actual["comparisons"][name]
        assert [row["fold"] for row in metric["delta"]["folds"]] == [0, 1, 2]
        assert all(row["positive_count"] == row["negative_count"] == 29 for row in metric["per_target"])
        interval = metric["paired_study_bootstrap"]["macro_auc"]
        assert interval["defined_resamples"] + interval["undefined_resamples"] == 2000


def test_unsupported_fold_or_interval_never_passes_gate():
    metric = dict(paired_study_bootstrap={"macro_auc": {"lower": .01}},
                  delta={"folds": [{"fold": fold, "delta_macro_auc": .02} for fold in range(3)]})
    assert comparison.lower_bound_positive(metric) and comparison.all_fold_directions_positive(metric)
    metric["paired_study_bootstrap"]["macro_auc"]["lower"] = None
    assert not comparison.lower_bound_positive(metric)
    metric["delta"]["folds"][1]["delta_macro_auc"] = None
    assert not comparison.all_fold_directions_positive(metric)
    metric["delta"]["folds"].pop()
    assert not comparison.all_fold_directions_positive(metric)


def medical_fixture(tmp_path, manifest):
    root = tmp_path / "medical"
    rows = []
    order = sorted(predictions(manifest))
    for fold in range(3):
        held = [row.study_instance_uid for row in manifest.gold_memberships if row.fold == fold]
        rows.append(dict(fold=fold, gold_heldout_study_uids=held,
                         gold_training_study_uids=[uid for uid in order if uid not in held], weak_training_study_uids=[]))
    medical_plan = dict(fold_manifest_checksum=manifest.manifest_checksum, frozen_files=[], folds=rows,
                        feature_identity_sha256="a" * 64, medical_pooled_sha256="b" * 64)
    plan_binding = write(root / "plan.json", medical_plan)
    sha = plan_binding["sha256"]
    bindings = {}
    terminal_shas = {}
    for kind, job, limit in (("fit", 101, "00:30:00"), ("verification", 102, "03:00:00")):
        prefix = "medical_" + kind
        bindings[prefix + "_accepted"] = write(root / kind / "accepted-launch.json",
               dict(plan_sha256=sha, launch_attempt=1, job_id=job))
        directory = root / kind / f"controller-{job}"
        directory.mkdir()
        path = directory / "terminal-controller.txt"
        path.write_text(f"JobId={job} JobState=COMPLETED ExitCode=0:0 Partition=cluster1-cpu NumNodes=1 NumCPUs=16 "
                        f"NumTasks=1 MinMemoryNode=64G TimeLimit={limit} OverSubscribe=OK Requeue=0 Restarts=0 RunTime=00:00:02")
        bindings[prefix + "_terminal"] = dict(path=str(path), sha256=comparison.digest(path))
        terminal_shas[kind] = comparison.digest(path)
        bindings[prefix + "_observation"] = write(directory / "terminal-observation.json",
                dict(job_id=job, state="COMPLETED", returncode=0))
    records, audits = [], []
    for row in rows:
        fold, held = row["fold"], row["gold_heldout_study_uids"]
        for arm in comparison.ARMS:
            directory = root / arm / f"fold-{fold}"
            checkpoint = directory / "checkpoints/final.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(f"synthetic-only-{fold}-{arm}".encode())
            checkpoint_sha = comparison.digest(checkpoint)
            identity = dict(plan_sha256=sha, fold=fold, arm=arm, head_id=comparison.MEDICAL_HEAD_ID,
                            feature_identity_sha256="a" * 64, fold_manifest_checksum=manifest.manifest_checksum,
                            pooled_sha256="b" * 64)
            record = dict(status="COMPLETE", identity=identity, job_id=101,
                          held_out_study_uids=held, training_study_uids=row["gold_training_study_uids"],
                          predictions={uid:[.6] * 12 for uid in held}, zero_image_predictions={uid:[.5] * 12 for uid in held},
                          held_out_targets_masked=True, counts=dict(successful_optimizer_steps=7500, attempted_batches=7500, skipped_batches=0),
                          candidate_promotion=False, checkpoint_sha256=checkpoint_sha)
            binding = write(directory / "run.json", record)
            records.append(dict(fold=fold, arm=arm, **binding))
            audits.append(dict(fold=fold, arm=arm, checkpoint_sha256=checkpoint_sha, all_targets_masks_init_exposures_adam_checked=True))
    analysis = dict(status="COMPLETE", plan_sha256=sha, job_id=101, run_records=records,
                    disposition="inconclusive_no_promotion", candidate_promotion=False,
                    comparisons={"report_image_signal":{"synthetic_only":True}})
    bindings["medical_analysis"] = write(root / "analysis.json", analysis)
    independent = dict(status="COMPLETE", plan_sha256=sha, job_id=102, fit_job_id=101,
                 fit_allocated_cpu_node_seconds=2, fit_terminal_sha256=terminal_shas["fit"],
                 analysis_sha256=bindings["medical_analysis"]["sha256"], verified_final_heads=6, original_mri_studies=58,
                 verified_medical_predictions=116, verified_resnet_predictions=116, maximum_prediction_abs_error=0.,
                 maximum_medical_feature_abs_error=0., all_fitted_targets_masks_init_exposures_adam_checked=True,
                 all_fixed_native_comparisons_reproduced=True, original_mri_valid_slices=58,
                 source_audit=[dict(study_instance_uid=uid, valid_slices=1) for uid in order], audits=audits,
                 disposition=analysis["disposition"], candidate_promotion=False)
    bindings["medical_independent"] = write(root / "verification/independent-validation.json", independent)
    terminal = dict(status="COMPLETE", passed=True, plan_sha256=sha, verification_job_id=102,
             allocated_verification_cpu_node_seconds=2, verification_terminal_sha256=terminal_shas["verification"],
             independent_validation_sha256=bindings["medical_independent"]["sha256"], analysis_sha256=bindings["medical_analysis"]["sha256"],
             disposition=analysis["disposition"], candidate_promotion=False)
    bindings["medical_terminal_verification"] = write(root / "terminal-verification.json", terminal)
    return dict(medical_root=str(root), medical_plan=plan_binding), bindings


def test_medical_loader_uses_each_studys_single_held_out_fold(tmp_path, manifest):
    plan, bindings = medical_fixture(tmp_path, manifest)
    report, reference, zero, image = comparison.load_verified_medical(plan, bindings, manifest)
    assert len(report) == len(reference) == len(zero) == 58
    assert image == {"synthetic_only":True}
    assert all(row == [.6] * 12 for row in report.values())


@pytest.mark.parametrize("problem", ["missing_terminal", "premature_native", "wrong_plan", "native_error", "partial_mri", "wrong_job", "changed_checkpoint", "training_prediction"])
def test_medical_loader_rejects_incomplete_or_unbound_results(tmp_path, manifest, problem):
    plan, bindings = medical_fixture(tmp_path, manifest)
    root = Path(plan["medical_root"])
    if problem == "missing_terminal": Path(bindings["medical_verification_terminal"]["path"]).unlink()
    elif problem == "changed_checkpoint": (root / "report-supervision/fold-0/checkpoints/final.pt").write_bytes(b"different")
    elif problem == "training_prediction":
        path = root / "report-supervision/fold-0/run.json"
        record = comparison.read(path)
        record["training_study_uids"].append(record["held_out_study_uids"][0])
        write(path, record)
        # Even the unchanged registered digest must reject this substitution.
    else:
        path = root / "verification/independent-validation.json"
        receipt = comparison.read(path)
        if problem == "premature_native": receipt["status"] = "RUNNING"
        if problem == "wrong_plan": receipt["plan_sha256"] = "c" * 64
        if problem == "native_error": receipt["maximum_prediction_abs_error"] = 1e-4
        if problem == "partial_mri": receipt["original_mri_studies"] = 57
        if problem == "wrong_job": receipt["job_id"] = 999
        bindings["medical_independent"] = write(path, receipt)
        terminal_path = root / "terminal-verification.json"
        terminal = comparison.read(terminal_path)
        terminal["independent_validation_sha256"] = bindings["medical_independent"]["sha256"]
        bindings["medical_terminal_verification"] = write(terminal_path, terminal)
    with pytest.raises((comparison.ScoredAnchorComparisonError, FileNotFoundError)):
        comparison.load_verified_medical(plan, bindings, manifest)


def test_bound_digest_cannot_redirect_to_other_runs_evidence(tmp_path):
    first = write(tmp_path / "first.json", {"status":"COMPLETE"})
    write(tmp_path / "other.json", {"status":"COMPLETE"})
    with pytest.raises(comparison.ScoredAnchorComparisonError, match="role path"):
        comparison.bound_file(first, tmp_path / "other.json")


def test_resource_input_digest_is_deterministic_without_cyclic_reference():
    left = {"b":{"path":"b", "sha256":"2"}, "a":{"path":"a", "sha256":"1"}}
    right = {"a":{"sha256":"1", "path":"a"}, "b":{"sha256":"2", "path":"b"}}
    assert comparison.input_files_digest(left) == comparison.input_files_digest(right)
    right["b"]["sha256"] = "changed"
    assert comparison.input_files_digest(left) != comparison.input_files_digest(right)


def anchor_fixture(tmp_path, manifest):
    root, output = tmp_path / "anchor", tmp_path / "anchor/terminal-output"
    output.mkdir(parents=True)
    order = sorted(predictions(manifest))
    studies = [dict(study_uid=uid, dicom_files=1, dicom_bytes=100, aggregate_sha256="d" * 64) for uid in order]
    identity = dict(schema_version="rsna-ticket70-frozen-original-mri-identity-v3", studies=studies)
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    contract = write(root / "original-mri-contract.json", {**identity, "identity_sha256":identity_sha})
    native = write(output / "development-original-mri-proof.json", dict(schema_version="rsna-ticket70-original-mri-byte-proof-v3",
           status="COMPLETE", study_count=58, dicom_files=58, dicom_bytes=5800, identity_sha256=identity_sha,
           studies=studies, target_labels_uploaded=False))
    runtime = write(output / "development-runtime.json", dict(schema_version="rsna-ticket70-actual-runtime-v3",
           python="3.11.0-synthetic", packages={name:"synthetic" for name in
              ("numpy", "pandas", "pydicom", "torch", "torchvision", "timm", "scipy", "Pillow")},
           torch_cuda="synthetic", cudnn=1, gpu_count=2, gpus=[dict(index=index, name="synthetic T4",
               capability=[7,5], total_memory_bytes=1) for index in range(2)]))
    log = output / "development-execution.jsonl"
    log.write_text(json.dumps(dict(stream="stdout", line="synthetic complete", rejection=None)) + "\n")
    sources = [dict(index=index, original_sha256=str(index) * 64, relocated_sha256=str(index) * 64) for index in range(5)]
    stages = []
    for index in range(5):
        path = output / f"development-stage-{index}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["StudyInstanceUID", *TARGET_COLUMNS])
            writer.writerows([[uid] + [.5] * 12 for uid in order])
        stages.append(dict(index=index, sha256=comparison.digest(path), recipe_sha256=sources[index]["relocated_sha256"]))
    final = output / "development-exact-scored-anchor.csv"
    final.write_bytes((output / "development-stage-4.csv").read_bytes())
    (root / "development.ipynb").write_text("synthetic notebook")
    (root / "kernel-metadata.json").write_text("synthetic metadata")
    request = dict(study_count=58, scored_submission_ref=56705536, scored_script_version_id=354125287,
                   scored_public_score="0.936", model_fitting=False, competition_submission=False,
                   notebook_sha256=comparison.digest(root / "development.ipynb"), metadata_sha256=comparison.digest(root / "kernel-metadata.json"),
                   frozen_files={}, original_mri_contract_sha256=contract["sha256"], source_cells=sources,
                   original_mri_identity_sha256=identity_sha, original_mri_dicom_files=58, original_mri_dicom_bytes=5800,
                   development_uid_sha256=hashlib.sha256("".join(uid+"\n" for uid in order).encode()).hexdigest(),
                   execution_watchdog_seconds=7140, kernel_id="synthetic/anchor")
    request_binding = write(root / "request.json", request)
    sha = request_binding["sha256"]
    receipt = dict(schema_version="rsna-ticket70-exact-scored-anchor-output-v1", status="COMPLETE", study_count=58,
                   development_uid_sha256=request["development_uid_sha256"], final_sha256=comparison.digest(final),
                   assigned_gpu_count=2, model_fitting=False, competition_submission=False, fallback_events=0,
                   coat_receipt=dict(completed=True, legacy_success_count=58), elapsed_seconds=1.,
                   original_cells_sha256=[source["original_sha256"] for source in sources], stages=stages,
                   loaded_weights={"synthetic/checkpoint.pt":{"bytes":1,"sha256":"a"*64}},
                   input_receipt=dict(file="development-original-mri-proof.json", sha256=native["sha256"],
                       identity_sha256=identity_sha, dicom_files=58, dicom_bytes=5800),
                   runtime_receipt=dict(file="development-runtime.json", sha256=runtime["sha256"]),
                   execution_receipt=dict(schema_version="rsna-ticket70-complete-member-proof-v3", rejection_events=[],
                       stdout_and_stderr=True, unterminated_tails_checked=True, logged_lines=1,
                       execution_log_file=log.name, execution_log_sha256=comparison.digest(log),
                       dino=dict(planned=["synthetic-member"], completed=["synthetic-member"],
                                 expected_public=["synthetic-member"], completed_public=["synthetic-member"])))
    bindings = {}
    bindings["anchor_accepted"] = write(root / "accepted-launch.json", dict(request_sha256=sha, launch_attempt=1,
         kernel_id="synthetic/anchor", kernel_version=1, kernel_numeric_id=10))
    bindings["anchor_terminal"] = write(root / "kernel-terminal.json", dict(request_sha256=sha, launch_attempt=1,
         kernel_id="synthetic/anchor", kernel_version=1, status="COMPLETE", response={"status":"COMPLETE"}))
    bindings["anchor_receipt"] = write(output / "exact-scored-anchor-receipt.json", receipt)
    bindings["anchor_predictions"] = dict(path=str(final), sha256=comparison.digest(final))
    bindings["anchor_original_mri_evidence"], bindings["anchor_runtime_evidence"] = native, runtime
    independent = dict(status="COMPLETE", passed=True, terminal_state="COMPLETE", request_sha256=sha,
       kernel_id="synthetic/anchor", kernel_version=1, kernel_numeric_id=10,
       accepted_launch_sha256=bindings["anchor_accepted"]["sha256"], terminal_sha256=bindings["anchor_terminal"]["sha256"],
       original_mri_studies=58, remote_source_verified=True, remote_settings_verified=True,
       asset_runtime_envelope_verified=True, exact_recipe_verified=True, no_fallback_verified=True,
       output_sha256=comparison.digest(final), output_receipt_sha256=bindings["anchor_receipt"]["sha256"],
       original_mri_evidence=native, runtime_evidence=runtime)
    bindings["anchor_independent"] = write(root / "independent-verification.json", independent)
    return dict(anchor_root=str(root), anchor_output_directory=str(output), anchor_request=request_binding), bindings


def test_anchor_loader_requires_and_accepts_full_native_terminal_source_chain(tmp_path, manifest):
    plan, bindings = anchor_fixture(tmp_path, manifest)
    anchor = comparison.load_verified_anchor(plan, bindings, manifest)
    assert len(anchor) == 58 and all(row == [.5] * 12 for row in anchor.values())


@pytest.mark.parametrize("problem", ["queued", "missing_native", "wrong_remote_version", "dropped_member", "cuda_fallback", "changed_dicom", "partial_legacy", "fallback"])
def test_anchor_loader_rejects_premature_or_partial_results_before_predictions(tmp_path, manifest, monkeypatch, problem):
    plan, bindings = anchor_fixture(tmp_path, manifest)
    root, output = Path(plan["anchor_root"]), Path(plan["anchor_output_directory"])
    independent = comparison.read(root / "independent-verification.json")
    receipt = comparison.read(output / "exact-scored-anchor-receipt.json")
    if problem == "queued":
        terminal = comparison.read(root / "kernel-terminal.json")
        terminal["status"] = terminal["response"]["status"] = "QUEUED"
        bindings["anchor_terminal"] = write(root / "kernel-terminal.json", terminal)
        independent["terminal_sha256"] = bindings["anchor_terminal"]["sha256"]
    if problem == "missing_native": independent["original_mri_studies"] = 57
    if problem == "wrong_remote_version": independent["kernel_version"] = 2
    if problem == "dropped_member": receipt["execution_receipt"]["dino"]["completed"] = []
    if problem == "partial_legacy": receipt["coat_receipt"]["legacy_success_count"] = 57
    if problem == "fallback": receipt["fallback_events"] = 1
    if problem in ("cuda_fallback", "changed_dicom"):
        role, field = ("runtime_evidence", "runtime_receipt") if problem == "cuda_fallback" else ("original_mri_evidence", "input_receipt")
        path = Path(independent[role]["path"])
        proof = comparison.read(path)
        if problem == "cuda_fallback": proof["torch_cuda"] = None
        else: proof["studies"][0]["aggregate_sha256"] = "f" * 64
        binding = write(path, proof)
        independent[role] = binding
        bindings["anchor_" + role] = binding
        receipt[field]["sha256"] = binding["sha256"]
    bindings["anchor_receipt"] = write(output / "exact-scored-anchor-receipt.json", receipt)
    independent["output_receipt_sha256"] = bindings["anchor_receipt"]["sha256"]
    bindings["anchor_independent"] = write(root / "independent-verification.json", independent)
    monkeypatch.setattr(comparison, "load_anchor_csv", lambda *_: pytest.fail("Premature anchor prediction read"))
    with pytest.raises(comparison.ScoredAnchorComparisonError):
        comparison.load_verified_anchor(plan, bindings, manifest)


def test_preparation_freezes_transitive_analysis_dependencies_only(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_ticket71_scored_anchor_analysis.py"
    spec = importlib.util.spec_from_file_location("ticket71_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = tmp_path / "rsna_knee"
    package.mkdir()
    (package / "__init__.py").write_text('"""Synthetic package."""')
    (package / "scored_anchor_comparison.py").write_text("from .metric import score\n")
    (package / "metric.py").write_text("from . import labels\nfrom rsna_knee.support import check\n")
    (package / "labels.py").write_text("LABELS = (0,1)\n")
    (package / "support.py").write_text("def check(): pass\n")
    (package / "biomedclip_inference.py").write_text("editable_adapter = True\n")
    paths = module.analysis_source_dependencies(tmp_path)
    assert {file.name for file in paths} == {"__init__.py", "scored_anchor_comparison.py", "metric.py", "labels.py", "support.py"}


def worker_module():
    path = Path(__file__).resolve().parents[1] / "scripts/ticket71_scored_anchor_analysis.py"
    spec = importlib.util.spec_from_file_location("ticket71_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def primary_fixture(tmp_path):
    root = tmp_path / "ticket71"
    write(root / "analysis/accepted-launch.json", dict(plan_sha256="plan", admission_sha256="inputs", launch_attempt=1, job_id=201))
    directory = root / "analysis/controller-201"
    directory.mkdir()
    (directory / "terminal-controller.txt").write_text("JobId=201 JobState=COMPLETED ExitCode=0:0 Partition=cluster1-cpu "
         "NumNodes=1 NumCPUs=16 NumTasks=1 MinMemoryNode=64G TimeLimit=00:30:00 OverSubscribe=OK Requeue=0 Restarts=0 RunTime=00:00:02")
    write(directory / "terminal-observation.json", dict(job_id=201, state="COMPLETED", returncode=0))
    result = dict(status="COMPLETE", plan_sha256="plan", input_admission_sha256="inputs", resource_decision_sha256="decision",
                  job_id=201, gpu_seconds=0, tpu_seconds=0, external_spend_usd=0, elapsed_seconds=1.)
    write(root / "analysis/fixed-paired-analysis.json", result)
    return root, {"components":{"analysis":{"time_limit":"00:30:00","watchdog_seconds":1740}}}


def test_independent_replay_requires_primary_successful_bound_terminal(tmp_path):
    worker = worker_module()
    root, decision = primary_fixture(tmp_path)
    actual, audit = worker.verified_primary(root, decision, "plan", "inputs", "decision")
    assert actual["job_id"] == audit["analysis_job_id"] == 201
    assert audit["analysis_allocated_cpu_node_seconds"] == 2
    assert audit["analysis_sha256"] == comparison.digest(root / "analysis/fixed-paired-analysis.json")


@pytest.mark.parametrize("problem", ["running", "nonzero", "wrong_observation", "retry", "wrong_inputs", "overrun"])
def test_independent_replay_rejects_premature_primary_and_consumed_retries(tmp_path, problem):
    worker = worker_module()
    root, decision = primary_fixture(tmp_path)
    terminal = root / "analysis/controller-201/terminal-controller.txt"
    if problem == "running": terminal.write_text(terminal.read_text().replace("COMPLETED", "RUNNING"))
    if problem == "nonzero": terminal.write_text(terminal.read_text().replace("ExitCode=0:0", "ExitCode=1:0"))
    if problem == "overrun": terminal.write_text(terminal.read_text().replace("RunTime=00:00:02", "RunTime=00:31:00"))
    if problem == "wrong_observation": write(terminal.parent / "terminal-observation.json", dict(job_id=999, state="COMPLETED", returncode=0))
    if problem in ("retry", "wrong_inputs"):
        path = root / "analysis/accepted-launch.json"
        accepted = comparison.read(path)
        accepted["launch_attempt" if problem == "retry" else "admission_sha256"] = 2 if problem == "retry" else "other"
        write(path, accepted)
    with pytest.raises(comparison.ScoredAnchorComparisonError):
        worker.verified_primary(root, decision, "plan", "inputs", "decision")


def test_analysis_allocation_requires_exact_cpu_and_bounded_watchdog(monkeypatch):
    worker = worker_module()
    monkeypatch.setenv("SLURM_JOB_ID", "201")
    fields = "JobId=201 JobState=RUNNING Partition=cluster1-cpu NumNodes=1 NumCPUs=16 NumTasks=1 " \
             "MinMemoryNode=64G TimeLimit=00:30:00 OverSubscribe=OK Requeue=0 Restarts=0"
    monkeypatch.setattr(worker.subprocess, "check_output", lambda *_args, **_kwargs: fields)
    assert worker.check_allocation(dict(time_limit="00:30:00", watchdog_seconds=1740)) == fields
    with pytest.raises(comparison.ScoredAnchorComparisonError):
        worker.check_allocation(dict(time_limit="00:30:00", watchdog_seconds=1800))
    fields = fields.replace("NumCPUs=16", "NumCPUs=8")
    with pytest.raises(comparison.ScoredAnchorComparisonError):
        worker.check_allocation(dict(time_limit="00:30:00", watchdog_seconds=1740))
