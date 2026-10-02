"""Independently bind final control states and exposures without reading AUC."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

from coordinate_ticket64_full_backbone_comparison import stage_state
from ticket64_full_backbone_recovery_v2 import ROOT, OLD, SCIENCE_SHA, original, scientific_identity
from rsna_knee.ticket34 import admit_ticket30_fold_supervision
from rsna_knee.training import load_training_checkpoint, _model_state_sha256, _batch_indices


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate():
    plan = original.read(ROOT / 'plan.json'); checksum = original.digest(ROOT / 'plan.json')
    require(stage_state(ROOT / 'control', checksum, None, '02:00:00') == 'complete', 'Successful exact control terminal required')
    original.terminal(ROOT / 'control', checksum, '02:00:00')
    require(all(original.digest(r['path']) == r['sha256'] for r in plan['frozen_files'])
        and all(original.digest(r['path']) == r['sha256'] for r in original.read(OLD / 'plan.json')['frozen_files']),
        'Frozen control science, inputs or executed sources changed')
    decision = original.read(plan['decision_path'])
    require(decision['decision'] == 'approved' and decision['plan_sha256'] == checksum
        and decision['resources'] == plan['resources'], 'Exact standing control admission required')
    manifest = original.load_fold_manifest(original.t36.FOLD_MANIFEST_PATH)
    rows = original.read(original.PIXELS / 'pixel-manifest.json')['pixels']
    uids = [r['study_instance_uid'] for r in rows]
    gold = {r.study_instance_uid: r for r in manifest.gold_memberships}
    folds = original.read(plan['cohort_parent_plan'])['folds']
    summaries = []; seen = set()
    result = original.read(ROOT / 'control/stage-result.json')
    require([r['fold'] for r in result['folds']] == [0, 1, 2], 'Three exact control folds required')
    torch.set_num_threads(1)
    for fold_row in folds:
        fold = fold_row['fold']; path = ROOT / f'head-continuation/fold-{fold}/run.json'
        record = original.read(path)
        require(original.digest(path) == result['folds'][fold]['run_sha256'], 'Terminal result/run identity differs')
        checkpoint_path = Path(record['checkpoint_path'])
        expected = (OLD if fold == 0 else ROOT) / f'head-continuation/fold-{fold}/training/checkpoints/final.pt'
        require(checkpoint_path == expected and original.digest(checkpoint_path) == record['checkpoint_sha256'],
            'Exact original/recovery final checkpoint required')
        initial = original.initial_for_arm(fold, 'head-continuation')
        initial_sha = _model_state_sha256(initial); initial_state = initial.state_dict()
        require(initial_sha == record['initial_model_state_sha256'], 'Original warm initial state differs')
        checkpoint = load_training_checkpoint(checkpoint_path, expected_config=original.config_for_arm('head-continuation'),
            expected_fold=fold, expected_identity=scientific_identity(plan, checksum, fold, 'head-continuation'))
        state = checkpoint['model_state_dict']
        require(all(torch.isfinite(v).all() for v in state.values())
            and all(torch.equal(v, initial_state[k]) for k, v in state.items() if k.startswith('encoder.'))
            and any(not torch.equal(v, initial_state[k]) for k, v in state.items() if k.startswith('head.')),
            'Finite changed head and unchanged entire control encoder required')
        optimizer = checkpoint['optimizer_state_dict']
        require([g['lr'] for g in optimizer['param_groups']] == [1e-4]
            and bool(optimizer['state']) and all(float(v['step']) == 7500
                and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all()
                for v in optimizer['state'].values()), 'Complete original Adam state/rate/duration required')
        labels = admit_ticket30_fold_supervision(manifest, fold, original.t36.TICKET30_ARTIFACT,
            expected_artifact_sha256=original.t36.TICKET30_AUDITED_ARTIFACT_SHA256).for_image_studies(uids)
        require(labels.to_public_dict() == fold_row['supervision_admission'], 'Original target/mask admission differs')
        train, reports = [fold_row[k] for k in ('gold_training_study_uids', 'weak_training_study_uids')]
        held = set(manifest.held_out_study_uids(fold))
        require(not set(train+reports) & held and not set(train) & set(reports), 'Held-out or source overlap entered fitting')
        exposures = {kind: dict(sample_exposures=0, supervised_entry_exposures=0)
            for kind in ('reference', 'weak', 'mixed', 'pseudo')}
        for attempt in range(7500):
            for kind, pool, seed in (('reference', train, 20260912), ('weak', reports, 20260912+34034034)):
                for index in _batch_indices(len(pool), 2, attempt, seed):
                    mask = labels.by_study[pool[index]].supervision_mask
                    require(any(mask), 'Empty training supervision')
                    exposures[kind]['sample_exposures'] += 1
                    exposures[kind]['supervised_entry_exposures'] += sum(mask)
        progress = checkpoint['progress']
        require(progress == record['progress'] and progress['attempted_batches'] == progress['successful_optimizer_steps'] == 7500
            and progress['sample_exposures'] == 30000 and progress['exposures_by_source'] == exposures
            and progress['supervised_entry_exposures'] == sum(v['supervised_entry_exposures'] for v in exposures.values())
            and progress['skipped_all_masked_batches'] == 0, 'Exact independently reconstructed exposures differ')
        expected_held = held & set(gold)
        require(set(record['predictions']) == set(record['encoded_image_ablation']) == expected_held
            and not seen & expected_held and all(np.asarray(v).shape == (12,) and np.isfinite(v).all()
                for field in ('predictions', 'encoded_image_ablation') for v in record[field].values()),
            'Complete disjoint finite gold predictions required')
        seen.update(expected_held)
        require(record['reused_completed_fit'] == (fold == 0), 'Only the completed original fold may be reused')
        if fold == 0:
            require(record['checkpoint_sha256'] == plan['recovery']['reused_control_fold0_sha256']
                and record['reused_failed_job'] == 8033, 'Reused final checkpoint differs')
        summaries.append(dict(fold=fold, successful_updates=7500, gold_predictions=len(expected_held),
            checkpoint_sha256=record['checkpoint_sha256'], unchanged_encoder=True,
            original_sources_targets_masks_exposures_verified=True, initial_model_verified=True,
            completed_fit_reused=fold == 0))
        del checkpoint, initial, state, initial_state, optimizer
    require(len(seen) == 58, 'All 58 gold predictions required')
    _, terminal, seconds = original.terminal(ROOT / 'control', checksum, '02:00:00')
    return dict(passed=True, observed_at_utc=datetime.now(timezone.utc).isoformat(), job_id=8034,
        plan_sha256=checksum, scientific_plan_sha256=SCIENCE_SHA, terminal_sha256=original.digest(terminal),
        allocated_cpu_node_seconds=int(seconds), status='COMPLETED', exit_code='0:0', folds=summaries,
        original_control_encoder_head_optimizer_and_exposures_verified=True, auc_read=False,
        original_mri_prediction_replay_still_required=True, candidate_promotion=False)


if __name__ == '__main__':
    output = ROOT / 'control/independent-state-validation.json'
    require(not output.exists(), 'Immutable independent control validation already exists')
    result = validate(); original.write(output, result)
    print(json.dumps(result))
