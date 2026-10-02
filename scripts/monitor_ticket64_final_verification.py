"""Run the reserved final analysis/replay once all six fixed fits finish.

No training, retries, candidate promotion, submission or final selection. The
monitor freezes its own code now and freezes final inputs only after successful
fit terminals. Analysis and independent replay share one 90-minute CPU job.
"""

from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from coordinate_ticket64_full_backbone_comparison import stage_state
import validate_ticket64_full_backbone_comparison as replay

PROJECT = replay.PROJECT
ROOT = replay.FIT / 'postfit-controller-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def all_fits_complete():
    entries = [(replay.FIT / 'control', replay.FIT_SHA, None, '02:00:00')]
    entries += [(replay.stage_location(f, s)[0], replay.stage_location(f, s)[1],
                 min(7500, (s+1)*1000), '05:30:00') for f in range(3) for s in range(8)]
    states = []
    for directory, sha, step, limit in entries:
        # A different controller may currently be inside its one sbatch call.
        # An intent alone cannot establish that the submission stopped.
        if (directory / 'launch-intent.json').exists() and not (directory / 'accepted-launch.json').exists():
            states.append('waiting_for_receipt'); continue
        try:
            states.append(stage_state(directory, sha, step, limit))
        except (json.JSONDecodeError, FileNotFoundError):
            states.append('waiting_for_evidence')
    return all(state == 'complete' for state in states)


def action_for_state(*, fits_complete, occupied_slots, prepared, intent, accepted, terminal):
    if intent and not accepted:
        raise ValueError('Consumed replay intent lacks an accepted receipt; never retry')
    if accepted:
        return 'reconcile' if terminal else 'watch'
    if not fits_complete:
        return 'wait_for_fits'
    if occupied_slots >= 2:
        return 'wait_for_cpu_slot'
    return 'launch' if prepared else 'prepare'


def admit():
    require(not ROOT.exists(), 'Immutable postfit monitor already admitted')
    require(replay.digest(replay.FIT / 'plan.json') == replay.FIT_SHA, 'Exact admitted fitting plan required')
    plan = replay.read(replay.FIT / 'plan.json')
    decision = replay.read(plan['decision_path'])
    require(decision['decision'] == 'approved' and decision['plan_sha256'] == replay.FIT_SHA
        and decision['resources'] == plan['resources'] and decision['projection'] == plan['projection']
        and replay.digest(replay.AUTH) == plan['authorization_sha256']
        and plan['projection']['independent_replay_wall_seconds'] == 5400,
        'Exact standing-authorized replay reserve required')
    ROOT.mkdir(mode=0o700)
    snapshot = ROOT / 'source-snapshot'; snapshot.mkdir(mode=0o700)
    sources = (Path(__file__), Path(replay.__file__),
        Path(sys.modules['coordinate_ticket64_full_backbone_comparison'].__file__))
    for path in sources:
        shutil.copyfile(path, snapshot / path.name)
    request = dict(project_path=str(PROJECT), fit_plan_sha256=replay.FIT_SHA,
        source_snapshot=str(snapshot), reserved_replay_cpu_node_seconds=5400,
        authorization_sha256=replay.digest(replay.AUTH), runtime_versions=plan['runtime_versions'],
        source_files=[dict(path=str(p), sha256=replay.digest(p)) for p in snapshot.iterdir()],
        maximum_replay_launches=1, maximum_concurrent_owner_cpu_jobs=2,
        external_spend_usd=0, gpu_count=0, automatic_retry=False, fitting=False,
        candidate_promotion=False, competition_submission=False)
    replay.write(ROOT / 'admission.json', request)
    print(json.dumps(dict(admission_sha256=replay.digest(ROOT / 'admission.json'), launched=False)))


def reconcile():
    root = replay.ROOT
    request_sha = replay.digest(root / 'plan.json'); accepted = replay.read(root / 'accepted-launch.json')
    job = str(accepted['job_id'])
    require(accepted['plan_sha256'] == request_sha and accepted['launch_attempt'] == 1,
        'Exact accepted replay request required')
    controller = root / f'controller-{job}/terminal-controller.txt'
    fields = dict(t.split('=', 1) for t in controller.read_text().split() if '=' in t)
    observation = replay.read(controller.with_name('terminal-observation.json'))
    expected = dict(JobId=job, JobState='COMPLETED', ExitCode='0:0', Partition='cluster1-cpu',
        NumNodes='1', NumCPUs='16', NumTasks='1', MinMemoryNode='64G', TimeLimit='01:30:00',
        OverSubscribe='OK', Requeue='0', Restarts='0')
    require(observation['job_id'] == job and observation['state'] == 'COMPLETED' and observation['returncode'] == 0
        and all(fields.get(k) == v for k, v in expected.items()) and 'gpu' not in fields.get('TRES', '').lower(),
        'Exact successful bounded replay terminal required')
    request = replay.read(root / 'plan.json'); decision = replay.read(request['decision_path'])
    require(request['fit_plan_sha256'] == replay.FIT_SHA and decision['decision'] == 'approved'
        and decision['plan_sha256'] == request_sha and decision['resources'] == request['resources']
        and request['authorization_sha256'] == replay.digest(replay.AUTH), 'Replay admission changed')
    require(all(replay.digest(r['path']) == r['sha256'] for r in request['frozen_files']), 'Frozen replay inputs changed')
    validation = replay.read(root / 'validation.json')
    require(validation['status'] == 'COMPLETE' and validation['job_id'] == job
        and validation['plan_sha256'] == request_sha and validation['fit_plan_sha256'] == replay.FIT_SHA
        and validation['original_mri_studies'] == 58 and validation['final_runs_verified'] == 6
        and validation['final_predictions_verified'] == 116
        and validation['original_pixels_targets_masks_exposures_verified']
        and validation['all_registered_comparisons_recomputed']
        and math.isfinite(validation['maximum_prediction_error']) and 0 <= validation['maximum_prediction_error'] <= 1e-6
        and not validation['fitting'] and not validation['candidate_promotion'] and not validation['first_place_established']
        and validation['gpu_count'] == validation['external_spend_usd'] == 0,
        'Complete actual original-MRI validation required')
    require({(r['arm'], r['fold']) for r in validation['audits']} == {(a, f) for a in replay.ARMS for f in range(3)}
        and len(validation['audits']) == 6, 'Exact six audited final fits required')
    seconds = (datetime.fromisoformat(fields['EndTime'])-datetime.fromisoformat(fields['StartTime'])).total_seconds()
    require(0 <= seconds <= 5400, 'Actual replay exceeds its registered CPU-node bound')
    result = dict(passed=True, observed_at_utc=datetime.now(timezone.utc).isoformat(), job_id=accepted['job_id'],
        plan_sha256=request_sha, fit_plan_sha256=replay.FIT_SHA, terminal_sha256=replay.digest(controller),
        validation_sha256=replay.digest(root / 'validation.json'), allocated_cpu_node_seconds=int(seconds),
        original_mri_studies=58, final_runs_verified=6, final_predictions_verified=116,
        maximum_prediction_error=validation['maximum_prediction_error'],
        provisional_review_gate_passed=validation['provisional_review_gate_passed'],
        candidate_promotion=False, first_place_established=False)
    receipt = root / 'terminal-verification.json'
    if receipt.exists():
        existing = replay.read(receipt)
        require(all(existing.get(k) == v for k, v in result.items() if k != 'observed_at_utc'),
                'Existing independent terminal reconciliation differs')
        return existing
    replay.write(receipt, result)
    return result


def watch(admission_sha):
    require(replay.digest(ROOT / 'admission.json') == admission_sha, 'Frozen postfit admission changed')
    admission = replay.read(ROOT / 'admission.json')
    require(Path(__file__).resolve().parent == Path(admission['source_snapshot'])
        and os.environ.get('RSNA_PROJECT_ROOT') == admission['project_path'], 'Frozen postfit source/project required')
    with (ROOT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        missing_since = None
        while True:
            require(replay.digest(replay.FIT / 'plan.json') == admission['fit_plan_sha256']
                and replay.digest(replay.AUTH) == admission['authorization_sha256']
                and all(replay.digest(r['path']) == r['sha256'] for r in admission['source_files'])
                and all(importlib.metadata.version(n) == v for n, v in admission['runtime_versions'].items()),
                'Postfit source, scientific admission or runtime changed')
            fits_complete = all_fits_complete()
            try:
                queue = subprocess.run(['squeue', '-u', os.environ['USER'], '-h', '-o', '%i %P'],
                    capture_output=True, text=True, timeout=15)
            except subprocess.TimeoutExpired:
                print('Queue observation timed out; retaining the same request and waiting', flush=True)
                time.sleep(30); continue
            if queue.returncode:
                print('Queue observation failed; retaining the same request and waiting', flush=True)
                time.sleep(30); continue
            jobs = [r.split()[0] for r in queue.stdout.splitlines() if r.split()[1] == 'cluster1-cpu']
            root = replay.ROOT; receipt = root / 'accepted-launch.json'
            prepared = (root / 'prepared-validation.json').exists(); intent = (root / 'launch-intent.json').exists()
            accepted = receipt.exists(); terminal = None
            if accepted:
                job = str(replay.read(receipt)['job_id'])
                terminal = root / f'controller-{job}/terminal-observation.json'
                if job in jobs or terminal.exists():
                    missing_since = None
                else:
                    missing_since = missing_since or time.monotonic()
                    require(time.monotonic()-missing_since <= 300, 'Accepted replay has no live job or retained terminal; no retry')
            require(not root.exists() or prepared, 'Partial replay preparation retained; no automatic recovery')
            action = action_for_state(fits_complete=fits_complete, occupied_slots=len(jobs), prepared=prepared,
                intent=intent, accepted=accepted, terminal=terminal is not None and terminal.exists())
            state = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), phase=action, fits_complete=fits_complete,
                occupied_cpu_jobs=jobs, automatic_retry=False, candidate_promotion=False)
            temp = ROOT / 'state.partial'; temp.write_text(json.dumps(state, indent=2)+'\n'); temp.replace(ROOT / 'state.json')
            if action == 'prepare':
                replay.prepare(); continue
            if action == 'launch':
                replay.launch(replay.digest(root / 'plan.json')); continue
            if action == 'reconcile':
                print(json.dumps(reconcile()), flush=True)
                return
            if not fits_complete and datetime.now(timezone.utc) >= datetime(2026, 10, 11, tzinfo=timezone.utc):
                raise ValueError('Fitting did not complete before science freeze; no verification launch')
            time.sleep(30)


if __name__ == '__main__':
    if sys.argv[1] == 'admit':
        admit()
    elif sys.argv[1] == 'watch':
        watch(sys.argv[2])
    else:
        raise ValueError('Unknown postfit monitor action')
