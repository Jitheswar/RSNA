"""Separately bounded recovery of the no-update8139 JSON cache-path failure."""
from datetime import datetime, timedelta, timezone
import ast
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time

from ticket75_cpu_reference import digest, read, require, write, FIT_SHA
from prepare_ticket75_joint_resource_envelope import PROJECT, T74, PARENT, ROUTE
from prepare_ticket76_cpu_reference_recovery import failed_input as earlier_failed_input, fresh_projection as root_projection

OLD = PROJECT / 'artifacts/private/ticket75-cpu-reference-v2'
ROOT = PROJECT / 'artifacts/private/ticket75-cpu-reference-v3'
PREVIOUS = PROJECT / '.scratch/rsna-leaderboard/resource-approvals/ticket76-cpu-root-recovery-20261003.json'
DECISION = PROJECT / '.scratch/rsna-leaderboard/resource-approvals/ticket77-cpu-cache-recovery-20261003.json'
OLD_SHA = '45f34a3ff1de8570b71bbae25695e3237013834da5c5e369a2a40a0242f8c35d'
T74_SHA = '187599d49ac25b90fd9fdd5c8dcf830c115481cbb59916771c94bc53edaa6470'
OLD_LIFE_SHA = '8a4414e501f0275627fb98770ea62d21f8d58a20587c8578fdefd8dade72fb59'
UNIT = 'rsna-ticket74-await-native-inputs-v4.service'
OLD_UNIT = 'rsna-ticket74-await-native-inputs-v3.service'


def failed_input():
    earlier_failed_input()
    require(digest(OLD/'plan.json') == OLD_SHA, 'Exact failed corrected-root request required')
    accepted = read(OLD/'accepted-launch.json')
    terminal = OLD/'controller-8139/terminal-controller.txt'
    observation = read(terminal.with_name('terminal-observation.json'))
    fields = dict(v.split('=',1) for v in terminal.read_text().split() if '=' in v)
    expected = dict(JobId='8139',JobState='FAILED',ExitCode='1:0',Partition='cluster1-cpu',NumNodes='1',NumCPUs='16',
        NumTasks='1',MinMemoryNode='64G',TimeLimit='00:30:00',Requeue='0',Restarts='0',OverSubscribe='OK')
    seconds = (datetime.fromisoformat(fields['EndTime'])-datetime.fromisoformat(fields['StartTime'])).total_seconds()
    require(accepted['job_id'] == 8139 and accepted['plan_sha256'] == OLD_SHA and accepted['launch_attempt'] == 1
        and observation['job_id'] == '8139' and observation['state'] == 'FAILED' and seconds == 374
        and all(fields.get(k) == v for k,v in expected.items()), 'Actual exact374-second failed terminal required')
    error = (OLD/'slurm-8139.err').read_text()
    require("AttributeError: 'str' object has no attribute 'is_file'" in error
        and "entry = load_pixel_entry(binding['pixel_path']" in error
        and (OLD/'attachment/studies').is_dir() and not any((OLD/'attachment').rglob('*.npy'))
        and not any(OLD.rglob('*.pt')) and not (OLD/'source-replay.json').exists()
        and not (OLD/'reference-result.json').exists(), 'Exact first-cache failure before array/model/update export required')
    require(all(digest(r['path']) == r['sha256'] for r in read(OLD/'plan.json')['frozen_files']), 'Preserve all consumed source bytes')
    return terminal,seconds


def live_waiter():
    frozen = read(T74/'await-source-v4-freeze.json')
    require(all(digest(r['path']) == r['sha256'] for r in frozen['source_bindings']), 'Exact frozen cache-recovery reserve owner required')
    source = Path(frozen['frozen_source_path'])
    raw = subprocess.check_output(['systemctl','--user','show',UNIT,'--property=ActiveState','--property=MainPID'],text=True,timeout=15)
    fields = dict(v.split('=',1) for v in raw.splitlines() if '=' in v)
    pid = int(fields.get('MainPID','0'))
    require(fields.get('ActiveState') == 'active' and pid > 0, 'Actual live cache-recovery owner required')
    os.kill(pid,0)
    command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    require(str(source).encode() in command and b'await-inputs' in command, 'Exact frozen source must own actual PID')
    started = read(T74/'await-inputs-v4-started.json')
    require(started['pid'] == pid and started['source_sha256'] == digest(source)
        and not Path(f'/proc/{frozen["prior_pid"]}').exists(), 'Exact new ownership and stopped old process required')
    return source,pid,frozen


def original_worker_only_path_change(source):
    original = (Path(read(OLD/'plan.json')['source_snapshot'])/'ticket75_cpu_reference.py').read_text()
    needle = "load_pixel_entry(binding['pixel_path'],"
    require(original.count(needle) == 1 and Path(source).read_text() == original.replace(needle,"load_pixel_entry(Path(binding['pixel_path']),"),
        'Entire original worker must differ only by the cache Path conversion')


def install_owner():
    failed_input()
    with (PROJECT/'artifacts/private/campaign-resource-admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        require(digest(T74/'plan.json') == T74_SHA and not (T74/'await-source-v4-freeze.json').exists()
            and not (T74/'input-resource-admission.json').exists() and not any(T74.glob('*/launch-intent.json')),
            'Only unused analysis waiting lifecycle can be separately amended')
        old = read(T74/'await-source-v3-freeze.json')
        require(all(digest(r['path']) == r['sha256'] for r in old['source_bindings'])
            and digest(old['frozen_source_path']) == OLD_LIFE_SHA, 'Original v3 lifecycle remains unchanged')
        prior_pid = read(T74/'await-inputs-v3-started.json')['pid']
        raw = subprocess.check_output(['systemctl','--user','show',OLD_UNIT,'--property=ActiveState','--property=MainPID'],text=True,timeout=15)
        require('ActiveState=active' in raw and f'MainPID={prior_pid}\n' in raw, 'Exact old resource owner must be live')
        os.kill(prior_pid,0)
        require(str(old['frozen_source_path']).encode() in Path(f'/proc/{prior_pid}/cmdline').read_bytes().split(b'\0'), 'Exact old PID/source required')
        source = PROJECT/'scripts/ticket74_analysis_lifecycle_v4.py'
        def functions(p):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(p).read_text()).body if isinstance(n,ast.FunctionDef)}
        a,b = functions(old['frozen_source_path']),functions(source)
        require(all(a[n] == b[n] for n in ('remaining_reserves','terminal','launch','watch')), 'Scientific/terminal/one-attempt functions stay identical')
        target = T74/'await-source-snapshot-v4'/source.name
        target.parent.mkdir(mode=0o700);shutil.copyfile(source,target)
        write(T74/'await-source-v4-freeze.json',dict(plan_sha256=T74_SHA,prior_pid=prior_pid,
            prior_source_sha256=OLD_LIFE_SHA,frozen_source_path=str(target),protocol_retuned=False,
            source_bindings=[dict(path=str(p),sha256=digest(p)) for p in (source,target)],
            changes='Retain separate1800-second cache recovery and100MB history; preserve original pilot and root-recovery reserves'))
        subprocess.run(['systemctl','--user','stop',OLD_UNIT],check=True,timeout=30)
        require(not Path(f'/proc/{prior_pid}').exists(), 'Old waiter must stop before new owner starts')
        write(T74/'await-inputs-v3-stopped-for-v4.json',dict(prior_pid=prior_pid,actual_process_missing=True,
            source_preserved=True,scientific_plan_sha256=T74_SHA,protocol_retuned=False))
        subprocess.run(['systemd-run','--user','--collect',f'--unit={UNIT.removesuffix(".service")}',
            str(PROJECT/'.venv/bin/python'),str(target),'await-inputs'],check=True,timeout=30)
        for _ in range(20):
            if (T74/'await-inputs-v4-started.json').exists():
                _,pid,_ = live_waiter();print(dict(status='CACHE_RECOVERY_RESERVE_OWNER_LIVE',pid=pid));return
            time.sleep(.5)
        raise TimeoutError('Observe the same unit; expiry cannot authorize restart')


def add_cache_reserve(result,now,free):
    require(result['remaining_calendar'].get('additional_root_recovery_cpu_seconds') == 1800
        and result['remaining_calendar'].get('additional_complete_pilot_route_seconds') == 28800
        and 'additional_cache_recovery_cpu_seconds' not in result['remaining_calendar'],
        'Original pilot/root reserves and unused separate cache recovery required')
    result['remaining_calendar']['additional_cache_recovery_cpu_seconds'] = 1800
    finish = now+timedelta(seconds=sum(result['remaining_calendar'].values()))
    result['complete_route_finish_utc'] = finish.isoformat()
    result['storage']['additional_cache_recovery_bytes'] = 100_000_000
    result['storage']['projected_apparent_bytes'] += 100_000_000
    require(finish < datetime.fromisoformat(result['science_freeze_utc'])
        and result['storage']['projected_apparent_bytes'] < 2_000_000_000_000
        and free > result['storage']['protected_allowance_bytes']+8_300_000_000,
        'Whole original/pilot/root/cache recovery route exceeds calendar/storage')
    return result


def fresh_projection():
    return add_cache_reserve(root_projection(),datetime.now(timezone.utc),shutil.disk_usage(PROJECT).free)


def preflight(snapshot,worker):
    original_worker_only_path_change(worker)
    environment = dict(os.environ,RSNA_PROJECT_ROOT=str(PROJECT),PYTHONPATH=str(snapshot/'src'),PYTHONDONTWRITEBYTECODE='1')
    for arguments in ([str(PROJECT/'scripts/check_ticket76_snapshot_paths.py'),str(snapshot)],
        [str(PROJECT/'scripts/check_ticket77_cache_path.py'),'--worker',str(worker),'--expect','pass']):
        subprocess.run([str(PROJECT/'.venv/bin/python'),*arguments],env=environment,check=True,timeout=30)


def prepare():
    terminal,seconds = failed_input()
    source,pid,_ = live_waiter()
    require(not ROOT.exists() and not DECISION.exists(), 'Unused separate cache recovery request required')
    from ticket73_configured_full_continuation import admission
    admission(FIT_SHA)
    old = read(OLD/'plan.json')
    worker = PROJECT/'scripts/ticket77_cpu_reference.py'
    preflight(Path(old['source_snapshot']),worker)
    with (PROJECT/'artifacts/private/campaign-resource-admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        require(not (T74/'input-resource-admission.json').exists(), 'Recovery must precede unused analysis final admission')
        source,pid,_ = live_waiter()
        envelope = fresh_projection()
        previous = read(PREVIOUS)
        require(all(digest(r['path']) == r['sha256'] for r in previous['frozen_files']), 'Original root-recovery admission bindings changed')
        paths = [PROJECT/'scripts'/name for name in ('ticket77_cpu_reference.py','ticket77_cpu_reference_lifecycle.py',
            'prepare_ticket77_cpu_reference_recovery.py','check_ticket77_cache_path.py','check_ticket76_snapshot_paths.py')]
        files = [ROUTE,PARENT,PREVIOUS,OLD/'plan.json',OLD/'accepted-launch.json',OLD/'launch-intent.json',terminal,
            terminal.with_name('terminal-observation.json'),OLD/'slurm-8139.err',source,T74/'await-source-v4-freeze.json',*paths]
        write(DECISION,dict(previous,schema_version='rsna-ticket77-cpu-cache-recovery-v1',
            created_at_utc=datetime.now(timezone.utc).isoformat(),status='ADMITTED',prior_root_recovery_sha256=digest(PREVIOUS),
            prior_failed_cpu_job_id=8139,prior_failed_plan_sha256=OLD_SHA,prior_actual_failed_cpu_node_seconds=seconds,
            prior_full_failed_bound_cpu_node_seconds_retained=1800,earlier_failed_8131_actual_cpu_node_seconds_retained=88,
            earlier_failed_8131_bound_cpu_node_seconds_retained=1800,additional_cpu_preparation_node_seconds=1800,
            additional_recovery_storage_bytes=100_000_000,T74_lifecycle_source_sha256=digest(source),T74_live_pid=pid,
            original_consumed_intent_reused=False,original_worker_bytes_changed=True,original_scientific_worker_logic_changed=False,
            optimizer_updates_repeated=False,accelerator_launch_admitted=False,competition_submission=False,envelope=envelope,
            reason='Actual no-update first-cache JSON string/Path failure; exact call passes after sole Path conversion',
            frozen_files=[dict(path=str(p),sha256=digest(p)) for p in sorted(set(files))]))
        ROOT.mkdir(mode=0o700)
        snapshot = ROOT/'source-snapshot'
        shutil.copytree(old['source_snapshot'],snapshot,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for path in paths: shutil.copyfile(path,snapshot/path.name)
        preflight(snapshot,snapshot/worker.name)
        files = [Path(r['path']) for r in old['frozen_files']]+[DECISION,*paths]
        files.extend(p for p in snapshot.rglob('*') if p.is_file())
        plan = dict(old,created_at_utc=datetime.now(timezone.utc).isoformat(),run_root=str(ROOT),source_snapshot=str(snapshot),
            joint_route_proof_path=str(DECISION),joint_route_proof_sha256=digest(DECISION),
            operational_recovery=dict(prior_plan_sha256=OLD_SHA,prior_job_id=8139,prior_actual_failed_cpu_node_seconds=seconds,
                earlier_failed_job_id=8131,earlier_failed_cpu_node_seconds=88,original_worker_bytes_changed=True,
                original_scientific_worker_logic_changed=False,only_worker_change='JSON pixel_path to pathlib.Path',
                optimizer_updates_repeated=False,workspace_root_environment=str(PROJECT)),
            frozen_files=[dict(path=str(p),sha256=digest(p)) for p in sorted(set(files))])
        write(ROOT/'plan.json',plan);checksum = digest(ROOT/'plan.json')
        launcher = (OLD/'run.sbatch').read_text().replace(str(OLD),str(ROOT)).replace(OLD_SHA,checksum)
        launcher = launcher.replace('/ticket75_cpu_reference.py','/ticket77_cpu_reference.py')
        (ROOT/'run.sbatch').write_text(launcher)
        subprocess.run(['bash','-n',str(ROOT/'run.sbatch')],check=True)
        lifecycle = snapshot/'ticket77_cpu_reference_lifecycle.py'
        write(ROOT/'prepared-validation.json',dict(plan_sha256=checksum,launcher_sha256=digest(ROOT/'run.sbatch'),
            lifecycle_source_path=str(lifecycle),lifecycle_source_sha256=digest(lifecycle),
            actual_root_import_preflight_passed=True,actual_selected_cache_call_preflight_passed=True,
            original_scientific_worker_logic_changed=False,launched=False,competition_submission=False))
        print(dict(status='DISTINCT_CPU_CACHE_RECOVERY_PREPARED',plan_sha256=checksum,envelope=envelope))


if __name__ == '__main__':
    import sys
    {'install-owner':install_owner,'prepare':prepare}[sys.argv[1]]()
