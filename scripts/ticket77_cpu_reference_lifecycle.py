"""Wait for one free owner CPU slot, launch once, retain the actual job handle."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil
from datetime import datetime, timezone

from ticket77_cpu_reference import digest, read, require, write, validate_request
from prepare_ticket77_cpu_reference_recovery import PROJECT, live_waiter

ROOT = PROJECT / 'artifacts/private/ticket75-cpu-reference-v3'


def admission(checksum):
    require(digest(ROOT / 'plan.json') == checksum, 'Exact original CPU preparation plan required')
    plan, prepared = read(ROOT / 'plan.json'), read(ROOT / 'prepared-validation.json')
    validate_request(plan)
    require(prepared['plan_sha256'] == checksum and digest(ROOT / 'run.sbatch') == prepared['launcher_sha256']
            and digest(__file__) == prepared['lifecycle_source_sha256'], 'Exact frozen launcher/lifecycle required')
    return plan


def launch(checksum):
    admission(checksum)
    require(not (ROOT / 'launch-intent.json').exists(), 'Consumed sole preparation; never retry')
    live_waiter()
    with (PROJECT / 'artifacts/private/cpu-scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        jobs = subprocess.check_output(['squeue', '-u', 'ch.jitheswar', '-h', '-o', '%i|%P'], text=True, timeout=15).splitlines()
        if sum(r.split('|')[1] == 'cluster1-cpu' for r in jobs) >= 2:
            return False
        from prepare_ticket77_cpu_reference_recovery import fresh_projection
        fresh_projection()
        write(ROOT / 'launch-intent.json', dict(plan_sha256=checksum, launch_attempt=1))
        job = int(subprocess.check_output(['sbatch', '--parsable', str(ROOT / 'run.sbatch')], text=True, timeout=30).strip().split(';')[0])
        write(ROOT / 'accepted-launch.json', dict(plan_sha256=checksum, job_id=job, launch_attempt=1))
    subprocess.run(['systemd-run', '--user', '--collect', f'--unit=rsna-ticket77-reference-{job}-observer',
        str(PROJECT / '.venv/bin/python'), str(PROJECT / 'scripts/monitor_ticket48_cpu_pilot.py'), str(ROOT), str(job)], check=True, timeout=30)
    return True


def watch(checksum):
    plan = admission(checksum)
    require(Path(__file__).resolve() == Path(read(ROOT / 'prepared-validation.json')['lifecycle_source_path']).resolve(),
            'Frozen controller only')
    with (ROOT / 'lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(ROOT / 'lifecycle-started.json', dict(plan_sha256=checksum, pid=os.getpid(), source_sha256=digest(__file__)))
        while not (ROOT / 'accepted-launch.json').exists():
            require(not (ROOT / 'launch-intent.json').exists(), 'Consumed intent without receipt; preserve and stop')
            try:
                launch(checksum)
            except subprocess.TimeoutExpired:
                # Only observations can be repolled. Consumed launch intent
                # prevents any ambiguous sbatch acceptance from being repeated.
                require(not (ROOT / 'launch-intent.json').exists(), 'Backend acceptance ambiguous; never retry')
            time.sleep(15)
        accepted = read(ROOT / 'accepted-launch.json')
        job = str(accepted['job_id'])
        terminal = ROOT / f'controller-{job}/terminal-controller.txt'
        while not terminal.exists():
            try:
                raw = subprocess.run(['scontrol', 'show', 'job', job, '-o'], capture_output=True, text=True, timeout=15)
            except subprocess.TimeoutExpired:
                time.sleep(15)
                continue
            require(not (raw.returncode != 0 and 'Invalid job id' in raw.stderr), 'Actual handle purged without terminal; never retry')
            time.sleep(15)
        from ticket73_configured_full_continuation import successful_terminal
        seconds, actual_terminal = successful_terminal(ROOT, checksum, '00:30:00')
        result = read(ROOT / 'reference-result.json')
        require(result['status'] == 'COMPLETE' and result['plan_sha256'] == checksum and str(result['job_id']) == job
                and result['native_MRI_replayed'] == 64 and result['attachment_bytes'] <= 2_000_000_000
                and result['staged_reference_boundaries'] == [3, 8, 16]
                and result['elapsed_seconds'] <= plan['resources']['code_watchdog_seconds']
                and all(result[k] is False for k in ('TPU_verified', 'CPU28_verified', 'competition_submission', 'candidate_promotion')),
                'Actual original-CPU-only source/reference result required')
        for row in result['checkpoints'] + result['attachment_files']:
            require(digest(row['path']) == row['sha256'], 'Frozen output bytes changed')
        write(ROOT / 'terminal-verification.json', dict(status='COMPLETE', plan_sha256=checksum, job_id=job,
            actual_cpu_node_seconds=seconds, terminal_sha256=digest(actual_terminal),
            result_sha256=digest(ROOT / 'reference-result.json'), actual_TPU_execution=False,
            competition_submission=False, candidate_promotion=False))
        print(json.dumps(dict(status='CPU_REFERENCE_COMPLETE', job_id=job, actual_cpu_node_seconds=seconds)), flush=True)


if __name__ == '__main__':
    watch(sys.argv[1])
