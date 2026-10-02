"""Distinct Ticket 64 recovery: gold-only final prediction, no repeated updates.

The v1 science/config/checkpoint identity remains fixed. New execution receipts
use a separate v2 resource plan. The failed control and live first full stage
are preserved. Only completed final weights are evaluated.
"""

from datetime import datetime, timezone
import gc
import importlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

import ticket64_full_backbone_comparison as original

PROJECT = original.PROJECT
OLD = PROJECT / 'artifacts/private/ticket64-full-backbone-comparison-v1'
SCIENCE_SHA = '557230fec672f6db9252b1895a4d174652fe217c0388a8c9598bd749183aa985'
ROOT = PROJECT / 'artifacts/private/ticket64-full-backbone-comparison-v2'
DECISION = PROJECT / '.scratch/rsna-leaderboard/resource-approvals/ticket64-full-backbone-recovery-20261002.json'
SCRIPT = Path(__file__).name
original.ROOT = ROOT
read, write, digest = original.read, original.write, original.digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scientific_identity(plan, checksum, fold, arm):
    return dict(plan_sha256=SCIENCE_SHA, fold=fold, arm=arm,
        input_fingerprints=plan['input_fingerprints'][str(fold)],
        pixel_manifest_sha256=digest(original.PIXELS / 'pixel-manifest.json'),
        source_feature_manifest_sha256=digest(original.t36.FEATURE_ROOT / 'feature-manifest.json'),
        warm_checkpoint_sha256=digest(original.PARENT / f'report-supervision/fold-{fold}/checkpoints/final.pt'))


original.identity = scientific_identity


def prior_stage(fold, stage, checksum):
    require(stage > 0, 'Only resumed stages have a predecessor')
    if fold == 0 and stage == 1:
        return OLD / 'full-backbone/fold-0/stage-0', SCIENCE_SHA
    return ROOT / f'full-backbone/fold-{fold}/stage-{stage-1}', checksum


def training_root(fold):
    base = OLD if fold == 0 else ROOT
    return base / f'full-backbone/fold-{fold}/training'


def final_predictions(model, plan, fold, arm):
    manifest = original.load_fold_manifest(original.t36.FOLD_MANIFEST_PATH)
    gold = {r.study_instance_uid: r for r in manifest.gold_memberships}
    uids = sorted(set(manifest.held_out_study_uids(fold)) & set(gold))
    require(len(uids) == (20 if fold == 0 else 19), 'Exact held-out gold prediction cohort required')
    bank = original.load_bank(plan, set(uids))
    features = original.feature_bank(bank) if arm == 'head-continuation' else None
    examples = [original.TrainingExample(uid,
        **{**bank[uid], 'images': None if features is not None else bank[uid]['images']},
        targets=np.asarray(gold[uid].reference_labels, dtype=np.float32),
        supervision_mask=np.ones(12, dtype=np.bool_), provenance=original.SupervisionProvenance(),
        features=None if features is None else features[uid]) for uid in uids]
    tick = time.monotonic()
    predictions = {uid: value.tolist() for uid, value in
        original.predict_training_examples(model, examples, batch_size=2).items()}
    native_seconds = time.monotonic()-tick
    ablated = {}; model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), 2):
            batch = examples[start:start+2]
            tensors = [torch.from_numpy(np.stack([getattr(e, k) for e in batch]))
                for k in ('slice_mask', 'series_mask', 'series_descriptors')]
            values = model.predict_proba_from_features(torch.zeros((len(batch), 16, 8, 512)), *tensors).numpy()
            ablated.update({e.study_instance_uid: v.tolist() for e, v in zip(batch, values, strict=True)})
    return dict(predictions=predictions, encoded_image_ablation=ablated, held_out_inference_seconds=native_seconds)


def failed_control():
    directory = OLD / 'control'; receipt = read(directory / 'accepted-launch.json')
    require(receipt['job_id'] == 8033 and receipt['plan_sha256'] == SCIENCE_SHA, 'Exact failed control required')
    controller = directory / 'controller-8033/terminal-controller.txt'
    fields = dict(t.split('=', 1) for t in controller.read_text().split() if '=' in t)
    observation = read(controller.with_name('terminal-observation.json'))
    expected = dict(JobId='8033', JobState='FAILED', ExitCode='1:0', NumNodes='1', NumCPUs='16',
        NumTasks='1', MinMemoryNode='64G', TimeLimit='02:00:00', Partition='cluster1-cpu',
        OverSubscribe='OK', Requeue='0', Restarts='0')
    require(all(fields.get(k) == v for k, v in expected.items()) and observation['state'] == 'FAILED'
        and observation['job_id'] == '8033' and observation['returncode'] == 0, 'Failed execution terminal differs')
    trace = directory / 'slurm-8033.err'
    require('Selected source cohort incomplete' in trace.read_text() and 'final_predictions' in trace.read_text(),
        'Exact diagnosed call-site failure required')
    return controller, trace, int((datetime.fromisoformat(fields['EndTime'])-datetime.fromisoformat(fields['StartTime'])).total_seconds())


def prepare():
    require(not ROOT.exists() and not DECISION.exists(), 'Distinct immutable recovery already exists')
    old_plan = read(OLD / 'plan.json')
    require(digest(OLD / 'plan.json') == SCIENCE_SHA, 'Original prospective science plan changed')
    require(all(digest(r['path']) == r['sha256'] for r in old_plan['frozen_files']), 'Original frozen admission changed')
    old_decision = read(old_plan['decision_path'])
    require(old_decision['decision'] == 'approved' and old_decision['plan_sha256'] == SCIENCE_SHA,
        'Original standing fitting authorization required')
    controller, trace, charged = failed_control()
    checkpoint_path = OLD / 'head-continuation/fold-0/training/checkpoints/final.pt'
    checkpoint = original.load_training_checkpoint(checkpoint_path,
        expected_config=original.config_for_arm('head-continuation'), expected_fold=0,
        expected_identity=scientific_identity(old_plan, SCIENCE_SHA, 0, 'head-continuation'))
    progress = checkpoint['progress']
    require(progress['successful_optimizer_steps'] == progress['attempted_batches'] == 7500
        and progress['skipped_all_masked_batches'] == 0, 'Reuse requires the exact completed fold-0 fit')
    full_receipt = OLD / 'full-backbone/fold-0/stage-0/accepted-launch.json'
    require(read(full_receipt)['job_id'] == 8032 and read(full_receipt)['plan_sha256'] == SCIENCE_SHA,
        'Only the existing full stage can be inherited')
    cost_admission = read(original.COST / 'admission-result.json')
    costs = [read(original.COST / f'{s}-result.json') for s in ('interrupt', 'resume', 'continuous')]
    projection = original.resource_projection(cost_admission, costs, datetime.now(timezone.utc))
    projection['maximum_reserved_cpu_node_seconds'] += old_plan['projection']['control_wall_seconds']
    projection['maximum_projected_calendar_seconds'] += old_plan['projection']['control_wall_seconds']
    projection['known_failed_cpu_node_seconds'] = charged
    require(datetime.now(timezone.utc).timestamp() + projection['maximum_projected_calendar_seconds'] + 86400
        < datetime.fromisoformat(projection['science_freeze_utc']).timestamp(), 'Complete recovery lacks calendar headroom')
    usage = subprocess.check_output(['du', '--apparent-size', '-s', '-B1', str(PROJECT),
        str(PROJECT.parent / 'models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218')], text=True)
    existing = sum(int(r.split()[0]) for r in usage.splitlines())
    additional = projection['checkpoint_storage_allowance_bytes'] + 5_000_000_000
    require(existing + additional < 2_000_000_000_000 and shutil.disk_usage(PROJECT).free > additional,
        'Fresh complete recovery storage required')
    ROOT.mkdir(mode=0o700, parents=True)
    snapshot = ROOT / 'source-snapshot'
    shutil.copytree(old_plan['source_snapshot'], snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(__file__, snapshot / SCRIPT)
    frozen = [OLD / 'plan.json', Path(old_plan['decision_path']), original.AUTH, controller,
        controller.with_name('terminal-observation.json'), trace, OLD / 'control/accepted-launch.json',
        checkpoint_path, full_receipt, PROJECT / 'tests/test_full_backbone_gold_prediction_cohort.py']
    frozen += [p for p in snapshot.rglob('*') if p.is_file()]
    plan = dict(old_plan, schema_version='rsna-ticket64-gold-cohort-recovery-plan-v2',
        run_root=str(ROOT), source_snapshot=str(snapshot), decision_path=str(DECISION),
        created_at_utc=datetime.now(timezone.utc).isoformat(), scientific_plan_sha256=SCIENCE_SHA,
        prior_plan_path=str(OLD / 'plan.json'), projection=projection,
        recovery=dict(cause='all held-out memberships included report-only studies in gold prediction',
            prior_failed_job=8033, prior_failed_cpu_node_seconds=charged,
            reused_control_fold0_checkpoint=str(checkpoint_path), reused_control_fold0_sha256=digest(checkpoint_path),
            inherited_full_fold0_stage0_job=8032, inherited_stage0_completion_required_before_resume=True,
            optimizer_updates_repeated=False, scientific_recipe_changed=False),
        storage=dict(du_output=usage, existing_apparent_bytes=existing, additional_allowance_bytes=additional,
            projected_campaign_peak_bytes=existing+additional, soft_cap_bytes=2_000_000_000_000),
        frozen_files=[dict(path=str(p), sha256=digest(p)) for p in frozen])
    write(ROOT / 'plan.json', plan); checksum = digest(ROOT / 'plan.json')
    write(DECISION, dict(decision='approved', plan_sha256=checksum, resources=plan['resources'], projection=projection,
        authorization_sha256=digest(original.AUTH), decision_made_by='campaign agent under standing owner authorization',
        basis='Diagnosed gold-only prediction recovery preserves the completed 7500-step control, live 1000-step full stage and original science/config/optimizer identity; all remaining route costs conservatively admitted.',
        new_exact_owner_numeric_approval_claimed=False, gpu_ledger_changed=False, tpu_ledger_changed=False,
        scientific_recipe_changed=False, automatic_retry=False, candidate_promotion=False))
    launchers = []
    for fold in range(3):
        for stage in range(1 if fold == 0 else 0, 8):
            directory = ROOT / f'full-backbone/fold-{fold}/stage-{stage}'
            directory.mkdir(mode=0o700, parents=True)
            launchers.append(make_launcher(directory, 'full', checksum, 19800, fold, stage))
    directory = ROOT / 'control'; directory.mkdir(mode=0o700)
    launchers.append(make_launcher(directory, 'control', checksum, 7200))
    write(ROOT / 'prepared-validation.json', dict(plan_sha256=checksum, launchers=launchers, launched=False))
    print(json.dumps(dict(plan_sha256=checksum, reused_updates=7500, inherited_live_job=8032, launched=False)))


def make_launcher(directory, action, checksum, wall, fold=None, stage=None):
    launcher = directory / 'run.sbatch'; args = f' {fold} {stage}' if action == 'full' else ''
    launcher.write_text(f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket64-v2-{action}-{fold}-{stage}
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time={original.time_string(wall)}
#SBATCH --no-requeue
#SBATCH --output={directory}/slurm-%j.out
#SBATCH --error={directory}/slurm-%j.err
#SBATCH --chdir={PROJECT}
set -euo pipefail
umask 077
export RSNA_PROJECT_ROOT={PROJECT}
export PYTHONPATH={ROOT}/source-snapshot/src
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
{PROJECT}/.venv/bin/python {ROOT}/source-snapshot/{SCRIPT} {action} {checksum}{args}
''')
    subprocess.run(['bash', '-n', str(launcher)], check=True)
    return dict(path=str(launcher), sha256=digest(launcher), action=action, fold=fold, stage=stage)


def verify_prior(fold, stage, checksum):
    previous, prior_sha = prior_stage(fold, stage, checksum)
    original.terminal(previous, prior_sha, '05:30:00')
    record = read(previous / 'stage-result.json')
    path = training_root(fold) / 'checkpoints/latest.pt'
    require(record['status'] == 'COMPLETE' and record['plan_sha256'] == prior_sha
        and record['successful_steps'] == stage*1000 and digest(path) == record['checkpoint_sha256'],
        'Exact completed predecessor and durable checkpoint required')
    return record


def launch(checksum, action, fold=None, stage=None):
    plan = read(ROOT / 'plan.json'); decision = read(DECISION)
    require(digest(ROOT / 'plan.json') == checksum and decision['decision'] == 'approved'
        and decision['plan_sha256'] == checksum and decision['resources'] == plan['resources']
        and decision['projection'] == plan['projection'] and digest(original.AUTH) == plan['authorization_sha256'],
        'Exact distinct recovery admission required')
    require(all(digest(r['path']) == r['sha256'] for r in plan['frozen_files'])
        and all(digest(r['path']) == r['sha256'] for r in read(OLD / 'plan.json')['frozen_files']),
        'Frozen original/recovery sources changed')
    if action == 'full':
        require(fold in (0, 1, 2) and stage in range(1 if fold == 0 else 0, 8), 'Unknown recovery stage')
        directory = ROOT / f'full-backbone/fold-{fold}/stage-{stage}'
        prior = verify_prior(fold, stage, checksum) if stage else None
    elif action == 'control':
        directory = ROOT / 'control'; prior = None
    else:
        raise ValueError('Only registered recovery arms can launch')
    prepared = read(ROOT / 'prepared-validation.json')
    row = next(r for r in prepared['launchers'] if r['path'] == str(directory / 'run.sbatch'))
    require(prepared['plan_sha256'] == checksum and digest(row['path']) == row['sha256'], 'Frozen recovery launcher changed')
    jobs = subprocess.check_output(['squeue', '-u', os.environ['USER'], '-h', '-o', '%i %P %t'], text=True).splitlines()
    require(sum(r.split()[1] == 'cluster1-cpu' for r in jobs) < 2, 'Both CPU slots occupied')
    write(directory / 'launch-intent.json', dict(plan_sha256=checksum, launcher_sha256=row['sha256'], launch_attempt=1,
        prior_checkpoint_sha256=None if prior is None else prior['checkpoint_sha256'], at_utc=datetime.now(timezone.utc).isoformat()))
    job = int(subprocess.check_output(['sbatch', '--parsable', row['path']], text=True).strip().split(';')[0])
    write(directory / 'accepted-launch.json', dict(job_id=job, plan_sha256=checksum, launcher_sha256=row['sha256'], launch_attempt=1))
    subprocess.run(['systemd-run', '--user', f'--unit=rsna-ticket64-v2-{action}-{fold}-{stage}-{job}-observer', '--collect',
        str(PROJECT / '.venv/bin/python'), str(PROJECT / 'scripts/monitor_ticket48_cpu_pilot.py'), str(directory), str(job)], check=True)
    print(json.dumps(dict(job_id=job, plan_sha256=checksum, action=action, fold=fold, stage=stage)))


def runtime(checksum, wall):
    plan, controller = original.runtime(checksum, wall)
    require(Path(__file__).resolve().parent == Path(plan['source_snapshot']), 'Frozen recovery executable required')
    require(digest(OLD / 'plan.json') == plan['scientific_plan_sha256'] == SCIENCE_SHA, 'Original scientific identity changed')
    for row in read(OLD / 'plan.json')['frozen_files']:
        require(digest(row['path']) == row['sha256'], 'Original admitted input changed')
    return plan, controller


def control_run(checksum):
    plan, controller = runtime(checksum, 7200); start = time.monotonic(); records = []
    for fold in range(3):
        model = original.initial_for_arm(fold, 'head-continuation'); initial_sha = original._model_state_sha256(model)
        encoder = {k: v.clone() for k, v in model.encoder.state_dict().items()}
        directory = ROOT / f'head-continuation/fold-{fold}'; directory.mkdir(mode=0o700, parents=True)
        if fold == 0:
            path = Path(plan['recovery']['reused_control_fold0_checkpoint'])
            require(digest(path) == plan['recovery']['reused_control_fold0_sha256'], 'Completed control checkpoint changed')
        else:
            bank, gold, weak = original.fold_inputs(plan, fold); features = original.feature_bank(bank)
            gold, weak = [[original.replace(e, images=None, features=features[e.study_instance_uid]) for e in pool] for pool in (gold, weak)]
            result = original.train_fixed_budget(gold, original.config_for_arm('head-continuation'), additional_examples=weak,
                additional_batch_order_seed=20260912+34034034, model=model, artifact_dir=directory / 'training', fold=fold,
                identity=scientific_identity(plan, checksum, fold, 'head-continuation'), requested_resources=plan['resources'])
            require(result.successful_optimizer_steps == 7500 and result.skipped_batches == 0, 'Complete fixed control duration required')
            del bank, gold, weak, features, result; gc.collect()
            path = directory / 'training/checkpoints/final.pt'
        checkpoint = original.load_training_checkpoint(path, model=model, expected_config=original.config_for_arm('head-continuation'),
            expected_fold=fold, expected_identity=scientific_identity(plan, checksum, fold, 'head-continuation'))
        require(checkpoint['progress']['successful_optimizer_steps'] == checkpoint['progress']['attempted_batches'] == 7500
            and all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in encoder.items()), 'Exact fixed control state required')
        predictions = final_predictions(model, plan, fold, 'head-continuation')
        write(directory / 'run.json', dict(status='COMPLETE', plan_sha256=checksum, scientific_plan_sha256=SCIENCE_SHA,
            job_id=os.environ['SLURM_JOB_ID'], fold=fold, controller=controller, initial_model_state_sha256=initial_sha,
            checkpoint_path=str(path), checkpoint_sha256=digest(path), progress=checkpoint['progress'], encoder_unchanged=True,
            reused_completed_fit=fold == 0, reused_failed_job=8033 if fold == 0 else None, **predictions))
        records.append(dict(fold=fold, run_sha256=digest(directory / 'run.json')))
        del model, encoder, checkpoint; gc.collect()
    write(ROOT / 'control/stage-result.json', dict(status='COMPLETE', plan_sha256=checksum, job_id=os.environ['SLURM_JOB_ID'],
        controller=controller, folds=records, elapsed_seconds=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, candidate_promotion=False))


def full_stage(checksum, fold, stage):
    require(fold in (0, 1, 2) and stage in range(1 if fold == 0 else 0, 8), 'Unknown recovery treatment stage')
    plan, controller = runtime(checksum, 19800); start = time.monotonic()
    stage_root = ROOT / f'full-backbone/fold-{fold}/stage-{stage}'
    require(not (stage_root / 'stage-result.json').exists(), 'Consumed recovery result')
    bank, gold, weak = original.fold_inputs(plan, fold)
    model = original.initial_for_arm(fold, 'full-backbone'); initial_sha = original._model_state_sha256(model)
    buffers = {k: v.clone() for k, v in model.state_dict().items() if k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    parity = None
    if stage == 0:
        selected = {pool[i].study_instance_uid for pool, seed in ((gold, 20260912), (weak, 20260912+34034034))
            for i in original._batch_indices(len(pool), 2, 0, seed)}
        parity = original.verify_initial_batch(model, gold, weak, original.feature_bank({k: v for k, v in bank.items() if k in selected}))
    previous = verify_prior(fold, stage, checksum) if stage else None
    path = training_root(fold) / 'checkpoints/latest.pt'; stop = min(7500, (stage+1)*1000)
    tick = time.monotonic(); stopped = False
    try:
        original.train_fixed_budget(gold, original.config_for_arm('full-backbone'), additional_examples=weak,
            additional_batch_order_seed=20260912+34034034, model=model, artifact_dir=training_root(fold), fold=fold,
            identity=scientific_identity(plan, checksum, fold, 'full-backbone'), resume_from=path if stage else None,
            requested_resources=plan['resources'], interrupt_after_steps=stop if stop < 7500 else None)
    except original.TrainingInterrupted:
        if stop == 7500:
            raise
        stopped = True
    trainer_seconds = time.monotonic()-tick
    path = path if stopped else training_root(fold) / 'checkpoints/final.pt'
    checkpoint = original.load_training_checkpoint(path, model=model, expected_config=original.config_for_arm('full-backbone'),
        expected_fold=fold, expected_identity=scientific_identity(plan, checksum, fold, 'full-backbone'))
    require(checkpoint['progress']['successful_optimizer_steps'] == checkpoint['progress']['attempted_batches'] == stop
        and all(torch.equal(model.state_dict()[k], v) for k, v in buffers.items()), 'Exact resumed stage/fixed buffers required')
    changed = None
    if stop == 7500:
        initial = original.initial_model(fold).state_dict()
        changed = {prefix: any(not torch.equal(v, initial[k]) for k, v in model.state_dict().items() if k.startswith(prefix))
            for prefix in ('encoder.conv1', 'encoder.layer1', 'encoder.layer2', 'encoder.layer3', 'encoder.layer4', 'head')}
        require(all(changed.values()), 'All registered treatment stages must update')
    predictions = final_predictions(model, plan, fold, 'full-backbone') if stop == 7500 else {}
    write(stage_root / 'stage-result.json', dict(status='COMPLETE', plan_sha256=checksum, scientific_plan_sha256=SCIENCE_SHA,
        job_id=os.environ['SLURM_JOB_ID'], fold=fold, stage=stage, controller=controller, successful_steps=stop,
        planned_boundary=stopped, scientific_config_fingerprint=original.config_for_arm('full-backbone').config_fingerprint,
        checkpoint_path=str(path), checkpoint_sha256=digest(path), prior_checkpoint_sha256=None if previous is None else previous['checkpoint_sha256'],
        initial_model_state_sha256=initial_sha, initial_native_feature_logit_error=parity, trainer_seconds=trainer_seconds,
        elapsed_seconds=time.monotonic()-start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        progress=checkpoint['progress'], normalization_buffers_unchanged=True, all_encoder_stages_changed=changed,
        intermediate_prediction=False, candidate_promotion=False, **predictions))


def analyze(checksum):
    require(digest(ROOT / 'plan.json') == checksum, 'Exact completed recovery plan required')
    original.terminal(ROOT / 'control', checksum, '02:00:00')
    values = {a: {} for a in original.ARMS}; zero = {a: {} for a in original.ARMS}; parent = {}; records = []
    for fold in range(3):
        for stage in range(8):
            directory, sha = (OLD / 'full-backbone/fold-0/stage-0', SCIENCE_SHA) if fold == stage == 0 else (ROOT / f'full-backbone/fold-{fold}/stage-{stage}', checksum)
            original.terminal(directory, sha, '05:30:00')
        full = read(ROOT / f'full-backbone/fold-{fold}/stage-7/stage-result.json')
        control = read(ROOT / f'head-continuation/fold-{fold}/run.json')
        for arm, record in zip(original.ARMS, (control, full), strict=True):
            require(record['plan_sha256'] == checksum and record['progress']['successful_optimizer_steps'] == 7500
                and digest(record['checkpoint_path']) == record['checkpoint_sha256'], 'Exact six final recovery fits required')
            values[arm].update(record['predictions']); zero[arm].update(record['encoded_image_ablation'])
            records.append(dict(arm=arm, fold=fold, checkpoint_path=record['checkpoint_path'], checkpoint_sha256=record['checkpoint_sha256']))
        require(control['initial_model_state_sha256'] == full['initial_model_state_sha256'], 'Matched initial tensors differ')
        require(all(control['progress'][k] == full['progress'][k] for k in
            ('attempted_batches', 'sample_exposures', 'supervised_entry_exposures', 'exposures_by_source')), 'Matched exposures differ')
        parent.update(read(original.PARENT / f'report-supervision/fold-{fold}/run.json')['predictions'])
    manifest = original.load_fold_manifest(original.t36.FOLD_MANIFEST_PATH)
    def compare(base, new):
        return original.compare_image_model_predictions(base, new, manifest, bootstrap_resamples=2000, bootstrap_seed=20261002)
    comparisons = dict(primary=compare(values['head-continuation'], values['full-backbone']))
    for arm in original.ARMS:
        comparisons[f'{arm}-vs-parent'] = compare(parent, values[arm])
        comparisons[f'{arm}-image-signal'] = compare(zero[arm], values[arm])
    primary = comparisons['primary']; lower = primary['paired_study_bootstrap']['macro_auc']['lower']
    passed = lower is not None and lower > 0 and all(r['delta_macro_auc'] is not None and r['delta_macro_auc'] > 0 for r in primary['delta']['folds'])
    passed = passed and all(comparisons[f'{a}-image-signal']['paired_study_bootstrap']['macro_auc']['lower'] is not None
        and comparisons[f'{a}-image-signal']['paired_study_bootstrap']['macro_auc']['lower'] > 0 for a in original.ARMS)
    write(ROOT / 'analysis.json', dict(status='COMPLETE', plan_sha256=checksum, scientific_plan_sha256=SCIENCE_SHA,
        comparisons=comparisons, run_records=records, provisional_review_gate_passed=passed,
        independent_original_mri_replay_still_required=True, candidate_promotion=False, automatic_promotion=False,
        first_place_established=False))


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'prepare':
        prepare()
    elif action == 'launch':
        launch(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else None, int(sys.argv[5]) if len(sys.argv) > 5 else None)
    elif action == 'control':
        control_run(sys.argv[2])
    elif action == 'full':
        full_stage(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif action == 'analyze':
        analyze(sys.argv[2])
    else:
        raise ValueError('Unknown recovery action')
