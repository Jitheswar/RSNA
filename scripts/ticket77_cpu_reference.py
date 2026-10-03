"""Prepare the original real-MRI TPU pilot reference in one admitted CPU job.

Implementation only: this executable cannot create its request, reserve a slot,
launch a job, upload an attachment or score a competition submission. The future
request must close joint T73/T74/pilot resource ownership before it can run.
"""
import argparse
from dataclasses import replace
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

SELECTION_SHA = 'b08d52294efac12435fb49717ffac8b6f51bfb53c22fb1dc7658fb834e2ea8ce'
FIT_SHA = '9844fd6a7e62f3f1aaf5d0678784f197b8373d702e14c19e669ccde5244a36c8'
SCHEMA = 'rsna-ticket75-cpu-reference-request-v1'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def validate_request(request):
    require(request['schema_version'] == SCHEMA and request['fit_plan_sha256'] == FIT_SHA
            and request['selection_sha256'] == SELECTION_SHA, 'Exact prospective request/source selection required')
    expected = dict(partition='cluster1-cpu', nodes=1, tasks=1, cpus_per_task=16, memory_gib=64,
                    walltime_seconds=1800, code_watchdog_seconds=1740, maximum_launch_attempts=1,
                    exclusive=False, requeue=False, gpu_count=0, tpu_count=0, external_spend_usd=0)
    require(request['resources'] == expected, 'Sole bounded zero-spend CPU preparation required')
    require(request['joint_route_ownership_closed'] is True
            and request['joint_route_proof_sha256'] == digest(request['joint_route_proof_path']),
            'Joint T73/T74/pilot resource ownership must be closed before execution')
    require(all(request[k] is False for k in ('candidate_promotion', 'competition_submission',
                'original_fit_updates_credited', 'accelerator_launch_admitted')), 'Disposable preparation scope required')
    proof = read(request['joint_route_proof_path'])
    require(proof['status'] == 'ADMITTED' and proof['fit_plan_sha256'] == FIT_SHA
            and proof['T74_future_analysis_cpu_node_seconds'] == 7200
            and proof['pilot_complete_calendar_seconds'] == 28800
            and proof['pilot_retained_assets_bytes'] == 8_000_000_000
            and proof['T74_future_admission_accounts_outstanding_pilot'] is True,
            'Joint proof must retain T74 and the complete pilot route')
    bindings = request['frozen_files']
    require(bindings and len({r['path'] for r in bindings}) == len(bindings)
            and all(digest(r['path']) == r['sha256'] for r in bindings), 'Complete immutable input/source closure required')
    bound_paths = {str(Path(r['path']).resolve()) for r in bindings}
    require(str(Path(__file__).resolve()) in bound_paths
            and str(Path(request['selection_path']).resolve()) in bound_paths
            and str(Path(request['joint_route_proof_path']).resolve()) in bound_paths,
            'Executable, selection and joint admission must be frozen')
    return expected


def run(path, checksum):
    started = time.monotonic()
    require(digest(path) == checksum, 'Exact immutable request required')
    request = read(path)
    resources = validate_request(request)
    root = Path(request['run_root'])
    require(Path(__file__).resolve().parent == Path(request['source_snapshot']).resolve(),
            'Only the frozen executable may run')
    require(not (root / 'reference-started.json').exists(), 'Consumed preparation cannot be retried')
    accepted = read(root / 'accepted-launch.json')
    job = os.environ['SLURM_JOB_ID']
    require(str(accepted['job_id']) == job and accepted['plan_sha256'] == checksum
            and accepted['launch_attempt'] == 1, 'Exact sole accepted job required')
    raw = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True, timeout=15)
    fields = dict(v.split('=', 1) for v in raw.split() if '=' in v)
    expected = dict(JobId=job, JobState='RUNNING', Partition='cluster1-cpu', NumNodes='1', NumCPUs='16',
                    NumTasks='1', MinMemoryNode='64G', TimeLimit='00:30:00', OverSubscribe='OK', Requeue='0', Restarts='0')
    require(all(fields.get(k) == v for k, v in expected.items()) and 'gpu' not in fields.get('TRES', '').lower(),
            'Actual bounded CPU allocation required')
    write(root / 'reference-started.json', dict(plan_sha256=checksum, job_id=job, controller=raw))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU reference watchdog')))
    signal.alarm(resources['code_watchdog_seconds'])
    require(all(importlib.metadata.version(n) == v for n, v in request['runtime_versions'].items()),
            'Original pinned CPU runtime required')
    sys.path.insert(0, str(Path(request['source_snapshot']) / 'src'))
    import numpy as np
    import torch
    from rsna_knee.training import TrainingInterrupted, _model_state_sha256, _rng_state, load_training_checkpoint, train_fixed_budget
    from rsna_knee.tpu_pilot_inputs import select_original_pilot_batches, validate_pilot_study_arrays, attachment_upper_bound
    from ticket73_configured_full_continuation import admission, helpers
    from ticket62_mri_pixel_materialize import iter_packed_sources
    from rsna_knee.pixel_cache import load_pixel_entry
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    require(torch.version.cuda is None and not torch.cuda.is_available(), 'Original CPU-only runtime required')
    selection = read(request['selection_path'])
    require(digest(request['selection_path']) == SELECTION_SHA
            and all(digest(r['path']) == r['sha256'] for r in selection['frozen_metadata_sources']),
            'Original metadata selection changed')
    plan = admission(FIT_SHA)
    original, _, _ = helpers()
    bank, gold, weak = original.fold_inputs(plan, 0)
    regenerated = select_original_pilot_batches([e.study_instance_uid for e in gold],
        [e.study_instance_uid for e in weak], request['original_held_out_study_uids']).to_private_dict()
    require(all(selection[k] == v for k, v in regenerated.items()), 'Complete original sampler/pools changed')
    chosen = selection['study_order']
    require(request['replay_training_study_indices'] == list(range(8)), 'Fixed eight training replay studies required')
    examples = {e.study_instance_uid: e for e in (*gold, *weak)}
    attachment = root / 'attachment'
    attachment.mkdir(mode=0o700)
    arrays_root = attachment / 'studies'
    arrays_root.mkdir(mode=0o700)
    pixels_root = Path(plan['pixel_root'])
    pixel_plan = read(pixels_root / 'plan.json')
    manifest = read(pixels_root / 'pixel-manifest.json')
    rows = {r['study_instance_uid']: r for r in manifest['pixels']}
    archive = Path(pixel_plan['source_archive_path']).stat()
    require(archive.st_size == pixel_plan['source_archive_bytes']
            and archive.st_mtime_ns == pixel_plan['source_archive_mtime_ns'], 'Original MRI archive changed')
    source_audit = []
    bindings = {r['study_instance_uid']: r for r in selection['selected_original_pixel_bindings']}
    for uid, packed, source, prior, _ in iter_packed_sources(pixel_plan, sorted(chosen)):
        row, binding = rows[uid], bindings[uid]
        require(source == binding['source_identity_sha256'] and row['pixel_file_sha256'] == binding['pixel_file_sha256'],
                'Original selected source/cache identity changed')
        entry = load_pixel_entry(Path(binding['pixel_path']), expected_sha256=binding['pixel_file_sha256'],
            expected_study_uid=uid, expected_identity_sha256=pixel_plan['pixel_identity_sha256'],
            expected_source_identity_sha256=source, expected_plan_sha256=digest(pixels_root / 'plan.json'))
        require(np.array_equal(packed.images, entry.packed_images())
                and all(np.array_equal(getattr(packed, k), getattr(entry, k))
                        for k in ('slice_mask', 'series_mask', 'series_descriptors'))
                and entry.metadata['sampled_source_indices'] == [list(v) for v in packed.sampled_source_indices],
                'Native MRI packing must exactly reproduce original cache')
        example = examples[uid]
        arrays = dict(images=packed.images, slice_mask=packed.slice_mask, series_mask=packed.series_mask,
            series_descriptors=packed.series_descriptors, targets=example.targets, supervision_mask=example.supervision_mask)
        validate_pilot_study_arrays(arrays)
        directory = arrays_root / f'{chosen.index(uid):02d}'
        directory.mkdir(mode=0o700)
        for name, value in arrays.items():
            np.save(directory / f'{name}.npy', value, allow_pickle=False)
        source_audit.append(dict(study_instance_uid=uid, source_identity_sha256=source,
            pixel_file_sha256=row['pixel_file_sha256'], original_sampler_index=chosen.index(uid)))
        del arrays, packed, entry
    require({r['study_instance_uid'] for r in source_audit} == set(chosen), 'All64 original training MRI must replay')
    write(root / 'source-replay.json', dict(original_MRI_verified=True, training_studies=source_audit,
        held_out_predictions=False, targets_from_complete_original_admission=True))
    config = replace(original.config_for_arm('full-backbone'), successful_optimizer_steps=16)
    model = original.initial_for_arm(0, 'full-backbone')
    initial_sha = _model_state_sha256(model)
    normalization = {k: v.clone() for k, v in model.state_dict().items()
        if k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    torch.save(dict(model_state_dict=model.state_dict(), rng_state=_rng_state(), config=config.to_dict(),
        initial_model_state_sha256=initial_sha, fit_plan_sha256=FIT_SHA, disposable=True), attachment / 'initial.pt')
    reference = root / 'reference'
    reference.mkdir(mode=0o700)
    identity = dict(pilot_request_sha256=checksum, disposable=True, original_fit_updates_credited=False,
                    input_fingerprints=plan['input_fingerprints']['0'])
    def training_logits(current):
        current.eval()
        output = []
        with torch.inference_mode():
            for uid in chosen[:8]:
                ex = examples[uid]
                tensors = [torch.from_numpy(np.asarray(getattr(ex, k))[None])
                    for k in ('images', 'slice_mask', 'series_mask', 'series_descriptors')]
                output.append(current(*tensors).numpy()[0])
        value = np.stack(output)
        require(value.shape == (8, 12) and np.isfinite(value).all(), 'Finite fixed training replay logits required')
        return value

    np.save(attachment / 'cpu-step-0-training-logits.npy', training_logits(model), allow_pickle=False)
    del model, bank
    gc.collect()
    checkpoints = []
    for boundary in (3, 8, 16):
        model = original.initial_for_arm(0, 'full-backbone')
        try:
            result = train_fixed_budget(gold, config, additional_examples=weak,
                additional_batch_order_seed=config.batch_order_seed + 34034034, model=model,
                artifact_dir=reference, fold=0, identity=identity, requested_resources=resources,
                resume_from=reference / 'checkpoints/latest.pt' if boundary > 3 else None,
                interrupt_after_steps=boundary if boundary < 16 else None)
            require(boundary == 16 and result.successful_optimizer_steps == 16, 'Exact final reference boundary required')
        except TrainingInterrupted:
            require(boundary < 16, 'Final reference cannot stop early')
        state_path = reference / ('checkpoints/latest.pt' if boundary < 16 else 'checkpoints/final.pt')
        optimizer = torch.optim.Adam([dict(params=model.encoder.parameters(), lr=1e-5),
            dict(params=model.head.parameters(), lr=1e-4)], lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        state = load_training_checkpoint(state_path, model=model, optimizer=optimizer, scheduler=scheduler,
            expected_config=config, expected_fold=0)
        from ticket72_checkpoint_recovery_audit import validate_adam
        validate_adam(optimizer, boundary)
        require(scheduler.last_epoch == boundary, 'Exact scheduler duration required')
        progress = state['progress']
        require(all(progress[k] == boundary for k in ('successful_optimizer_steps', 'attempted_batches', 'scheduler_steps', 'next_batch_index'))
                and progress['sample_exposures'] == boundary * 4 and progress['skipped_all_masked_batches'] == 0,
                'Exact original real-source update/exposure boundary required')
        expected_exposures = {role: dict(sample_exposures=2 * boundary, supervised_entry_exposures=0)
            for role in ('reference', 'weak')}
        expected_exposures.update({role: dict(sample_exposures=0, supervised_entry_exposures=0)
            for role in ('mixed', 'pseudo')})
        for batch in selection['batch_study_indices'][:boundary]:
            for position, index in enumerate(batch):
                role = 'reference' if position < 2 else 'weak'
                expected_exposures[role]['supervised_entry_exposures'] += int(examples[chosen[index]].supervision_mask.sum())
        require(progress['exposures_by_source'] == expected_exposures
                and progress['supervised_entry_exposures'] == sum(v['supervised_entry_exposures'] for v in expected_exposures.values()),
                'Exact original source masks/exposures required')
        require(all(torch.equal(model.state_dict()[k], v) for k, v in normalization.items()), 'Normalization buffers changed')
        destination = (attachment if boundary == 3 else root) / f'cpu-step-{boundary}.pt'
        shutil.copyfile(state_path, destination)
        logits_path = (attachment if boundary == 3 else root) / f'cpu-step-{boundary}-training-logits.npy'
        np.save(logits_path, training_logits(model), allow_pickle=False)
        checkpoints.append(dict(step=boundary, path=str(destination), sha256=digest(destination), progress=progress,
            logits_path=str(logits_path), logits_sha256=digest(logits_path)))
        del state, model, optimizer, scheduler
        gc.collect()
    write(attachment / 'selection.json', selection)
    files = [p for p in attachment.rglob('*') if p.is_file()]
    attachment_bytes = sum(p.stat().st_size for p in files)
    state_bytes = sum((attachment / n).stat().st_size for n in ('initial.pt', 'cpu-step-3.pt'))
    pixel_bytes = 64 * 16 * 8 * 224 * 224 * 4
    envelope = attachment_upper_bound(initial_state_bytes=(attachment / 'initial.pt').stat().st_size,
        step3_state_bytes=(attachment / 'cpu-step-3.pt').stat().st_size,
        code_and_metadata_bytes=attachment_bytes - state_bytes - pixel_bytes)
    require(attachment_bytes == envelope <= 2_000_000_000, 'Actual lossless attachment envelope exceeded')
    require(sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) <= 8_000_000_000, 'Complete retained-assets envelope exceeded')
    write(root / 'reference-result.json', dict(status='COMPLETE', plan_sha256=checksum, job_id=job,
        initial_model_state_sha256=initial_sha, native_MRI_replayed=64, checkpoints=checkpoints,
        attachment_bytes=attachment_bytes, attachment_files=[dict(path=str(p), sha256=digest(p)) for p in files],
        staged_reference_boundaries=[3, 8, 16], fresh_process_replay_verified=False,
        continuous_reference_verified=False, CPU28_verified=False, TPU_verified=False,
        held_out_predictions=False, candidate_promotion=False, competition_submission=False,
        original_fit_updates_credited=False, elapsed_seconds=time.monotonic()-started))
    signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('request', type=Path)
    parser.add_argument('sha256')
    args = parser.parse_args()
    run(args.request, args.sha256)
