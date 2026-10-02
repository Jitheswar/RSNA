"""Synthetic guard coverage, not actual MRI performance evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from rsna_knee import full_backbone_anchor_comparison as comparison
from rsna_knee.folds import freeze_folds
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.targets import TARGET_COLUMNS


@pytest.fixture
def manifest():
    source = SourceManifest('train', tuple(SourceStudy(f'synthetic-{i:02}', True,
        tuple(i % 2 for _ in TARGET_COLUMNS)) for i in range(58)), ())
    return freeze_folds(source, source_version='ticket74-synthetic-only', seed=31)


def predictions(manifest, value=.5):
    return {m.study_instance_uid: [value] * 12 for m in manifest.gold_memberships}


def image_signal(zero, full, manifest):
    return comparison.compare_image_model_predictions(zero, full, manifest, bootstrap_resamples=2000,
        bootstrap_seed=20261002, confidence_level=.95)


def test_protocol_matches_pre_outcome_frozen_declaration():
    path = Path(__file__).parents[1] / '.scratch/rsna-leaderboard/protocols/ticket74-full-backbone-anchor-v1.json'
    assert comparison.digest(path) == comparison.PROTOCOL_SHA
    assert comparison.read(path) == comparison.PROTOCOL
    assert comparison.PROTOCOL['full_backbone_weight'] == .1


def test_positive_control_diagnostic_cannot_replace_failed_primary(manifest):
    anchor = full = zero = predictions(manifest)
    control = {m.study_instance_uid: list(m.reference_labels) for m in manifest.gold_memberships}
    result = comparison.analyze(anchor, full, control, zero, manifest,
        expected_full_image=image_signal(zero, full, manifest), original_full_control_gate_passed=True)
    assert result['comparisons']['head_control_vote_minus_anchor']['delta']['pooled_macro_auc'] == .5
    assert result['comparisons']['primary']['delta']['pooled_macro_auc'] == 0
    assert result['disposition'] == 'reject_fixed_full_backbone_overlay'
    assert result['primary_predictions'] == comparison.fixed_vote(anchor, full, sorted(anchor))
    assert result['candidate_promotion'] is result['competition_submission'] is result['offline_delivery_verified'] is False


def test_fixed_primary_all_stats_and_original_gate_preserved(manifest, monkeypatch):
    anchor = zero = predictions(manifest)
    full = {m.study_instance_uid: list(m.reference_labels) for m in manifest.gold_memberships}
    expected = image_signal(zero, full, manifest)
    original = comparison.compare_image_model_predictions
    calls = []
    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(comparison, 'compare_image_model_predictions', record)
    result = comparison.analyze(anchor, full, anchor, zero, manifest,
        expected_full_image=expected, original_full_control_gate_passed=False)
    assert len(calls) == 6
    assert all(x == dict(bootstrap_resamples=2000, bootstrap_seed=20261002, confidence_level=.95) for x in calls)
    assert result['gates']['primary_paired_macro_AUC_lower_bound_positive']
    assert result['gates']['primary_all_three_original_fold_macro_AUC_directions_positive']
    assert not result['gates']['original_full_vs_matched_control_gate_passed']
    assert result['disposition'] == 'reject_fixed_full_backbone_overlay'
    for pair in result['comparisons'].values():
        assert [r['fold'] for r in pair['delta']['folds']] == [0, 1, 2]
        interval = pair['paired_study_bootstrap']['macro_auc']
        assert interval['defined_resamples'] + interval['undefined_resamples'] == 2000
        assert all(r['positive_count'] == r['negative_count'] == 29 for r in pair['per_target'])


def test_mismatched_native_image_stats_rejected(manifest):
    anchor = predictions(manifest)
    with pytest.raises(ValueError, match='reproduce'):
        comparison.analyze(anchor, anchor, anchor, anchor, manifest,
            expected_full_image={}, original_full_control_gate_passed=True)


def test_unfinished_fit_never_reads_candidate_predictions(tmp_path, manifest):
    root = tmp_path / 'fit'
    root.mkdir()
    path = root / 'plan.json'
    path.write_text('{}')
    plan = dict(fit_root=str(root), fit_plan=dict(path=str(path), sha256=comparison.digest(path)))
    with pytest.raises(ValueError, match='Exact configured'):
        comparison.load_verified_backbone(plan, {}, manifest)


def test_fit_source_tampering_stops_before_native_or_predictions(tmp_path, manifest, monkeypatch):
    root = tmp_path / 'fit'; root.mkdir()
    source = root / 'source.py'; source.write_text('old')
    plan_file = root / 'plan.json'
    plan_file.write_text(json.dumps(dict(frozen_files=[dict(path=str(source), sha256=comparison.digest(source))])))
    monkeypatch.setitem(comparison.PROTOCOL, 'fit_plan_sha256', comparison.digest(plan_file))
    source.write_text('changed')
    plan = dict(fit_root=str(root), fit_plan=dict(path=str(plan_file), sha256=comparison.digest(plan_file)))
    with pytest.raises(ValueError, match='Frozen fitting'):
        comparison.load_verified_backbone(plan, {}, manifest)


@pytest.mark.parametrize('defect', ['missing', 'nan', 'target_count'])
def test_partial_or_invalid_final_predictions_never_compared(manifest, defect):
    anchor = predictions(manifest); full = deepcopy(anchor); uid = sorted(full)[0]
    if defect == 'missing': full.pop(uid)
    if defect == 'nan': full[uid][0] = float('nan')
    if defect == 'target_count': full[uid].pop()
    with pytest.raises(ValueError):
        comparison.analyze(anchor, full, anchor, anchor, manifest, expected_full_image={}, original_full_control_gate_passed=True)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return dict(path=str(path), sha256=comparison.digest(path))


def backbone_fixture(tmp_path, manifest, monkeypatch):
    root = tmp_path / 'ticket64-full-backbone-comparison-v4'
    prior = root.with_name('ticket64-full-backbone-comparison-v2')
    first = root.with_name('ticket64-full-backbone-comparison-v1')
    prior_binding = write(prior / 'plan.json', dict(synthetic=True))
    cohort_path = root / 'cohort.json'
    cohort = dict(fold_manifest_checksum=manifest.manifest_checksum, folds=[])
    for fold in range(3):
        held = [m.study_instance_uid for m in manifest.gold_memberships if m.fold == fold]
        cohort['folds'].append(dict(fold=fold, gold_heldout_study_uids=held,
            gold_training_study_uids=[m.study_instance_uid for m in manifest.gold_memberships if m.fold != fold]))
    write(cohort_path, cohort)
    fit_binding = write(root / 'plan.json', dict(frozen_files=[], cohort_parent_plan=str(cohort_path)))
    monkeypatch.setitem(comparison.PROTOCOL, 'fit_plan_sha256', fit_binding['sha256'])
    plan = dict(fit_root=str(root), fit_plan=fit_binding)
    bindings = {}; rows = []; audits = []
    for fold in range(3):
        held = [m.study_instance_uid for m in manifest.gold_memberships if m.fold == fold]
        for arm in comparison.ARMS:
            base = prior if arm == 'head-continuation' else root
            checkpoint_base = (first if fold == 0 else prior) if arm == 'head-continuation' else root
            cp = checkpoint_base / arm / f'fold-{fold}/training/checkpoints/final.pt'
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_bytes(f'synthetic-only-{arm}-{fold}'.encode())
            sha = comparison.digest(cp)
            record = dict(status='COMPLETE', plan_sha256=prior_binding['sha256'] if arm == 'head-continuation' else fit_binding['sha256'],
                scientific_plan_sha256=comparison.PROTOCOL['scientific_plan_sha256'],
                checkpoint_path=str(cp), checkpoint_sha256=sha,
                progress=dict(successful_optimizer_steps=7500, attempted_batches=7500),
                predictions={u: [.5]*12 for u in held}, encoded_image_ablation={u: [.5]*12 for u in held})
            path = base / arm / f'fold-{fold}' / ('run.json' if arm == 'head-continuation' else 'stage-7/stage-result.json')
            bindings[f'fit_{arm}_{fold}'] = write(path, record)
            rows.append(dict(arm=arm, fold=fold, checkpoint_path=str(cp), checkpoint_sha256=sha))
            audits.append(dict(arm=arm, fold=fold, checkpoint_sha256=sha, native_prediction_error=0., native_ablation_error=0., parent_error=0.))
    analysis = dict(status='COMPLETE', plan_sha256=fit_binding['sha256'], scientific_plan_sha256=comparison.PROTOCOL['scientific_plan_sha256'],
        run_records=rows, provisional_review_gate_passed=False, candidate_promotion=False,
        comparisons={'full-backbone-image-signal': {}})
    bindings['fit_analysis'] = write(root / 'analysis.json', analysis)
    native_root = root / 'independent-original-mri-replay-v1'
    request = dict(fit_plan_sha256=fit_binding['sha256'], resources=dict(walltime_seconds=5400), frozen_files=[], final_run_records=rows)
    bindings['fit_native_request'] = write(native_root / 'plan.json', request)
    request_sha = bindings['fit_native_request']['sha256']
    bindings['fit_native_accepted'] = write(native_root / 'accepted-launch.json', dict(job_id=999, plan_sha256=request_sha, launch_attempt=1))
    controller = native_root / 'controller-999';controller.mkdir()
    terminal = controller / 'terminal-controller.txt'
    terminal.write_text('JobId=999 JobState=COMPLETED ExitCode=0:0 Partition=cluster1-cpu NumNodes=1 NumCPUs=16 NumTasks=1 MinMemoryNode=64G TimeLimit=01:30:00 OverSubscribe=OK Requeue=0 Restarts=0 RunTime=00:10:00')
    bindings['fit_native_terminal'] = dict(path=str(terminal), sha256=comparison.digest(terminal))
    bindings['fit_native_observation'] = write(controller / 'terminal-observation.json', dict(job_id=999, state='COMPLETED', returncode=0))
    native = dict(status='COMPLETE', fit_plan_sha256=fit_binding['sha256'], plan_sha256=request_sha, job_id='999',
        original_mri_studies=58, final_runs_verified=6, final_predictions_verified=116,
        original_pixels_targets_masks_exposures_verified=True, all_registered_comparisons_recomputed=True,
        maximum_prediction_error=0., gpu_count=0, external_spend_usd=0, audits=audits,
        source_audit=[dict(study_instance_uid=m.study_instance_uid) for m in manifest.gold_memberships],
        provisional_review_gate_passed=False, candidate_promotion=False, historical_backend_terminals_verified=False,
        original_strict_ancestry_passed=False, original_missing_allocation_charge_seconds=39600)
    bindings['fit_native_validation'] = write(native_root / 'validation.json', native)
    proof = dict(status='COMPLETE', passed=True, plan_sha256=fit_binding['sha256'], native_request_sha256=request_sha,
        verification_job_id=999, native_terminal_sha256=comparison.digest(terminal),
        native_validation_sha256=bindings['fit_native_validation']['sha256'], analysis_sha256=bindings['fit_analysis']['sha256'],
        candidate_promotion=False, historical_backend_terminals_verified=False, original_strict_ancestry_passed=False,
        original_missing_allocation_charge_seconds=39600)
    bindings['fit_terminal_verification'] = write(root / 'terminal-verification.json', proof)
    return plan, bindings, native, proof


def test_complete_native_chain_admits_exact_disjoint_oof_inputs(tmp_path, manifest, monkeypatch):
    plan, bindings, _, _ = backbone_fixture(tmp_path, manifest, monkeypatch)
    full, control, zero, image, gate = comparison.load_verified_backbone(plan, bindings, manifest)
    assert len(full) == len(control) == len(zero) == 58
    assert full == control == zero == predictions(manifest)
    assert image == {} and gate is False


@pytest.mark.parametrize('defect', ['partial_mri', 'partial_models', 'wrong_native_job', 'error', 'source_check', 'ancestry', 'native_audit', 'source_membership'])
def test_partial_or_relabelled_native_chain_never_admits_predictions(tmp_path, manifest, monkeypatch, defect):
    plan, bindings, native, proof = backbone_fixture(tmp_path, manifest, monkeypatch)
    if defect == 'partial_mri': native['original_mri_studies'] = 57
    if defect == 'partial_models': native['final_runs_verified'] = 5
    if defect == 'wrong_native_job': native['job_id'] = '998'
    if defect == 'error': native['maximum_prediction_error'] = 1e-4
    if defect == 'source_check': native['original_pixels_targets_masks_exposures_verified'] = False
    if defect == 'ancestry': native['original_strict_ancestry_passed'] = True
    if defect == 'native_audit': native['audits'].pop()
    if defect == 'source_membership': native['source_audit'][0]['study_instance_uid'] = 'another-study'
    path = Path(bindings['fit_native_validation']['path'])
    bindings['fit_native_validation'] = write(path, native)
    proof['native_validation_sha256'] = bindings['fit_native_validation']['sha256']
    bindings['fit_terminal_verification'] = write(Path(bindings['fit_terminal_verification']['path']), proof)
    with pytest.raises(ValueError):
        comparison.load_verified_backbone(plan, bindings, manifest)


def test_worker_proof_cannot_replace_actual_backend_terminal(tmp_path, manifest, monkeypatch):
    plan, bindings, _, _ = backbone_fixture(tmp_path, manifest, monkeypatch)
    terminal = Path(bindings['fit_native_terminal']['path'])
    terminal.write_text(terminal.read_text().replace('JobState=COMPLETED', 'JobState=FAILED'))
    bindings['fit_native_terminal']['sha256'] = comparison.digest(terminal)
    with pytest.raises(ValueError, match='successful sole CPU'):
        comparison.load_verified_backbone(plan, bindings, manifest)
