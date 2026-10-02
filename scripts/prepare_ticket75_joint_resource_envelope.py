"""Reserve the whole disposable pilot route without admitting an accelerator.

Completed control checkpoints stay on disk. Only their duplicate future fit
allowances are credited. A live amended T74 waiter owns the added future reserve.
"""
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ticket75_cpu_reference import digest, read, require, write, FIT_SHA

PROJECT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
PARENT = PROJECT / '.scratch/rsna-leaderboard/resource-approvals/ticket73-configured-continuation-20261002.json'
ROUTE = PROJECT / '.scratch/rsna-leaderboard/resource-approvals/ticket75-joint-pilot-route-20261003.json'
T74 = PROJECT / 'artifacts/private/ticket74-full-backbone-anchor-analysis-v1'
AUTH = PROJECT / '.scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json'
UNIT = 'rsna-ticket74-await-native-inputs-v2.service'


def projection(previous, *, completed_control_fits_verified, existing, free, now):
    require(completed_control_fits_verified is True, 'Actual completed original control fits required before future storage credit')
    cost = previous['projection']
    per_fit = (cost['history_count_upper_bound_per_fit'] + 4) * cost['maximum_checkpoint_bytes']
    require(cost['checkpoint_storage_allowance_bytes'] == 7 * per_fit, 'Original seven-fit storage envelope required')
    credit = 3 * per_fit
    protected = previous['storage']['protected_allowance_bytes'] + previous['storage']['additional_allowance_bytes'] - credit
    require(protected >= 4 * per_fit, 'Three outstanding treatment and conditional final fits must remain protected')
    calendar = dict(previous['remaining_calendar'])
    calendar['additional_T74_fixed_analysis_seconds'] = 7200
    calendar['additional_complete_pilot_route_seconds'] = 28800
    finish = now + timedelta(seconds=sum(calendar.values()))
    require(finish < datetime.fromisoformat(previous['science_freeze_utc']), 'Whole final-fit/delivery/review/pilot route exceeds science freeze')
    extra = 8_000_000_000 + 100_000_000
    require(existing + protected + extra < 2_000_000_000_000 and free > protected + extra,
            'Entire retained storage plus analysis/pilot assets exceeds bound')
    return dict(remaining_calendar=calendar, complete_route_finish_utc=finish.isoformat(),
        science_freeze_utc=previous['science_freeze_utc'], completed_control_future_allowance_credit_bytes=credit,
        storage=dict(existing_apparent_bytes=existing, protected_allowance_bytes=protected,
            additional_analysis_bytes=100_000_000, additional_pilot_bytes=8_000_000_000,
            projected_apparent_bytes=existing + protected + extra), completed_files_deleted=False)


def live_waiter():
    frozen = read(T74 / 'await-source-v2-freeze.json')
    require(all(digest(r['path']) == r['sha256'] for r in frozen['source_bindings']), 'Frozen joint lifecycle changed')
    source = Path(frozen['frozen_source_path'])
    raw = subprocess.check_output(['systemctl', '--user', 'show', UNIT, '--property=ActiveState', '--property=MainPID'], text=True, timeout=15)
    fields = dict(r.split('=', 1) for r in raw.splitlines() if '=' in r)
    pid = int(fields.get('MainPID', '0'))
    require(fields.get('ActiveState') == 'active' and pid > 0, 'Actual live amended resource owner required')
    os.kill(pid, 0)
    command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    require(str(source).encode() in command and b'await-inputs' in command, 'Actual PID must run the exact frozen waiter')
    started = read(T74 / 'await-inputs-v2-started.json')
    require(started['pid'] == pid and started['source_sha256'] == digest(source), 'Actual waiter start identity required')
    require(not Path(f'/proc/{frozen["prior_pid"]}').exists(), 'Prior waiter must be stopped before new ownership')
    return source, pid, frozen


def prepare():
    with (PROJECT / 'artifacts/private/campaign-resource-admission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        require(not ROUTE.exists() and not (T74 / 'input-resource-admission.json').exists()
                and not (PROJECT / '.scratch/rsna-leaderboard/resource-approvals/ticket74-fixed-analysis-20261002.json').exists(),
                'Unused joint route must precede final analysis admission')
        source, pid, frozen = live_waiter()
        from ticket73_configured_full_continuation import admission, successful_terminal, PRIOR, PRIOR_SHA
        admission(FIT_SHA)
        elapsed, terminal = successful_terminal(PRIOR / 'control', PRIOR_SHA, '02:00:00')
        control = read(PRIOR / 'control/stage-result.json')
        accepted = read(PRIOR / 'control/accepted-launch.json')
        require(control['status'] == 'COMPLETE' and control['plan_sha256'] == PRIOR_SHA
                and str(control['job_id']) == str(accepted['job_id']) and len(control['folds']) == 3,
                'Actual completed sole original control worker required')
        files = [PARENT, AUTH, source, T74 / 'await-source-v2-freeze.json', T74 / 'await-inputs-v2-started.json',
            PRIOR / 'control/stage-result.json', PRIOR / 'control/accepted-launch.json', terminal,
            terminal.with_name('terminal-observation.json')]
        controls = []
        require({r['fold'] for r in control['folds']} == {0, 1, 2}, 'All three original controls required')
        for row in control['folds']:
            path = PRIOR / f'head-continuation/fold-{row["fold"]}/run.json'
            require(digest(path) == row['run_sha256'], 'Original completed control record changed')
            record = read(path)
            progress = record['progress']
            checkpoint = Path(record['checkpoint_path'])
            require(record['status'] == 'COMPLETE' and record['fold'] == row['fold'] and record['plan_sha256'] == PRIOR_SHA
                    and all(progress[k] == 7500 for k in ('successful_optimizer_steps', 'scheduler_steps', 'attempted_batches', 'next_batch_index'))
                    and progress['sample_exposures'] == 30000 and progress['skipped_all_masked_batches'] == 0
                    and digest(checkpoint) == record['checkpoint_sha256'], 'Actual complete original 7500-step control checkpoint required')
            controls.append(dict(fold=row['fold'], run_sha256=digest(path), checkpoint_sha256=digest(checkpoint),
                                 completed_fit_future_allowance_credited=True, files_deleted=False))
            files.extend((path, checkpoint))
        previous = read(PARENT)
        require(previous['authorization_sha256'] == digest(AUTH), 'Standing owner authorization identity required')
        usage = subprocess.check_output(['du', '--apparent-size', '-s', '-B1', str(PROJECT),
            str(PROJECT.parent / 'models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218')], text=True)
        envelope = projection(previous, completed_control_fits_verified=True,
            existing=sum(int(r.split()[0]) for r in usage.splitlines()), free=shutil.disk_usage(PROJECT).free,
            now=datetime.now(timezone.utc))
        write(ROUTE, dict(schema_version='rsna-ticket75-joint-pilot-route-v1', status='ADMITTED',
            created_at_utc=datetime.now(timezone.utc).isoformat(), fit_plan_sha256=FIT_SHA,
            T74_future_analysis_cpu_node_seconds=7200, pilot_complete_calendar_seconds=28800,
            pilot_retained_assets_bytes=8_000_000_000, T74_future_admission_accounts_outstanding_pilot=True,
            T74_lifecycle_source_sha256=digest(source), T74_live_pid=pid,
            parent_resource_decision_sha256=digest(PARENT), authorization_sha256=digest(AUTH),
            decision_made_by='campaign agent under standing owner authorization', new_exact_owner_numeric_approval_claimed=False,
            cpu_preparation_node_seconds=1800, independent_cpu_verification_node_seconds=1800,
            proposed_TPU_node_seconds=3600, queue_calendar_seconds=21600,
            accelerator_launch_admitted=False, accelerator_quota_reserved=False, competition_submission=False,
            candidate_promotion=False, conditional_final_fit_authorized=False, automatic_retry=False,
            original_control_terminal_cpu_node_seconds=elapsed, completed_controls=controls,
            old_missing_terminal_cpu_node_seconds_retained=39600, old_failed_8111_cpu_node_seconds_retained=349,
            external_spend_usd=0, envelope=envelope, frozen_files=[dict(path=str(p), sha256=digest(p)) for p in files]))
        print(json.dumps(dict(status='JOINT_ROUTE_RESERVED_NO_ACCELERATOR_ADMISSION', route_sha256=digest(ROUTE), **envelope)))


if __name__ == '__main__':
    prepare()
