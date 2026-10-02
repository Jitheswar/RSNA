"""Distinct no-update recovery after the genuine8131 root-environment failure."""
from datetime import datetime, timedelta, timezone
import ast
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time

from ticket75_cpu_reference import digest, read, require, write, FIT_SHA
from prepare_ticket75_joint_resource_envelope import PROJECT, T74, PARENT, ROUTE, projection

OLD = PROJECT / 'artifacts/private/ticket75-cpu-reference-v1'
ROOT = PROJECT / 'artifacts/private/ticket75-cpu-reference-v2'
DECISION = PROJECT / '.scratch/rsna-leaderboard/resource-approvals/ticket76-cpu-root-recovery-20261003.json'
OLD_SHA = 'f9977c944803733e6565cc5b23c30b4d27b04128ff7ef51c335d3d6a65be4428'
T74_SHA = '187599d49ac25b90fd9fdd5c8dcf830c115481cbb59916771c94bc53edaa6470'
OLD_LIFE_SHA = 'aaee49c67f06c76c51dbe6b5c880f8e41a6bf3ccf21414ad887a23cfbeef32ba'
UNIT = 'rsna-ticket74-await-native-inputs-v3.service'


def failed_input():
    require(digest(OLD / 'plan.json') == OLD_SHA, 'Exact original failed request required')
    accepted = read(OLD / 'accepted-launch.json')
    terminal = OLD / 'controller-8131/terminal-controller.txt'
    observation = read(terminal.with_name('terminal-observation.json'))
    fields = dict(v.split('=',1) for v in terminal.read_text().split() if '=' in v)
    expected = dict(JobId='8131',JobState='FAILED',ExitCode='1:0',Partition='cluster1-cpu',NumNodes='1',NumCPUs='16',
        NumTasks='1',MinMemoryNode='64G',TimeLimit='00:30:00',Requeue='0',Restarts='0',OverSubscribe='OK')
    seconds = (datetime.fromisoformat(fields['EndTime'])-datetime.fromisoformat(fields['StartTime'])).total_seconds()
    require(accepted['job_id'] == 8131 and accepted['plan_sha256'] == OLD_SHA and accepted['launch_attempt'] == 1
        and observation['job_id'] == '8131' and observation['state'] == 'FAILED' and seconds == 88
        and all(fields.get(k) == v for k,v in expected.items()), 'Actual exact88-second failed terminal required')
    error = (OLD / 'slurm-8131.err').read_text()
    require('FoldManifestIntegrityError' in error and 'source-snapshot/artifacts/private/ticket-04-real-v2' in error
        and not (OLD / 'attachment').exists() and not (OLD / 'reference').exists()
        and not (OLD / 'reference-result.json').exists(), 'Failure before input export or any reference optimizer update required')
    require(all(digest(r['path']) == r['sha256'] for r in read(OLD / 'plan.json')['frozen_files']), 'Preserve all original consumed source bytes')
    return terminal, seconds


def live_waiter():
    frozen = read(T74 / 'await-source-v3-freeze.json')
    require(all(digest(r['path']) == r['sha256'] for r in frozen['source_bindings']), 'Exact frozen recovery reserve owner required')
    source = Path(frozen['frozen_source_path'])
    raw = subprocess.check_output(['systemctl','--user','show',UNIT,'--property=ActiveState','--property=MainPID'],text=True,timeout=15)
    fields = dict(v.split('=',1) for v in raw.splitlines() if '=' in v)
    pid = int(fields.get('MainPID','0'))
    require(fields.get('ActiveState') == 'active' and pid > 0, 'Actual live recovery resource owner required')
    os.kill(pid,0)
    require(str(source).encode() in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0'), 'Exact source must own actual PID')
    started = read(T74 / 'await-inputs-v3-started.json')
    require(started['pid'] == pid and started['source_sha256'] == digest(source), 'Actual frozen waiter start required')
    return source,pid,frozen


def install_owner():
    failed_input()
    require(digest(T74 / 'plan.json') == T74_SHA and not (T74 / 'await-source-v3-freeze.json').exists()
        and not (T74 / 'input-resource-admission.json').exists() and not any(T74.glob('*/launch-intent.json')),
        'Only unused analysis waiting lifecycle can be separately amended')
    old = read(T74 / 'await-source-v2-freeze.json')
    require(all(digest(r['path']) == r['sha256'] for r in old['source_bindings'])
        and digest(old['frozen_source_path']) == OLD_LIFE_SHA, 'Original v2 lifecycle must remain unchanged')
    prior_pid = read(T74 / 'await-inputs-v2-started.json')['pid'];os.kill(prior_pid,0)
    source = PROJECT / 'scripts/ticket74_analysis_lifecycle_v3.py'
    def functions(p):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(p).read_text()).body if isinstance(n,ast.FunctionDef)}
    a,b = functions(old['frozen_source_path']),functions(source)
    require(all(a[n] == b[n] for n in ('remaining_reserves','terminal','launch','watch')), 'Original scientific/terminal/one-attempt functions must stay identical')
    target = T74 / 'await-source-snapshot-v3' / source.name
    target.parent.mkdir(mode=0o700);shutil.copyfile(source,target)
    write(T74 / 'await-source-v3-freeze.json',dict(plan_sha256=T74_SHA, prior_pid=prior_pid,
        prior_source_sha256=OLD_LIFE_SHA,frozen_source_path=str(target),protocol_retuned=False,
        source_bindings=[dict(path=str(p),sha256=digest(p)) for p in (source,target)],
        changes='Retain1800-second distinct CPU recovery and100MB source/history allowance in future analysis admission'))
    subprocess.run(['systemctl','--user','stop','rsna-ticket74-await-native-inputs-v2.service'],check=True,timeout=30)
    require(not Path(f'/proc/{prior_pid}').exists(), 'Actual old waiter must stop before new owner starts')
    write(T74 / 'await-inputs-v2-stopped-for-v3.json',dict(prior_pid=prior_pid,actual_process_missing=True,
        source_preserved=True,scientific_plan_sha256=T74_SHA,protocol_retuned=False))
    subprocess.run(['systemd-run','--user','--collect',f'--unit={UNIT.removesuffix(".service")}',
        str(PROJECT / '.venv/bin/python'),str(target),'await-inputs'],check=True,timeout=30)
    for _ in range(20):
        if (T74 / 'await-inputs-v3-started.json').exists():
            live_waiter();return
        time.sleep(.5)
    raise TimeoutError('Observe the same unit; expiry cannot authorize restart')


def fresh_projection():
    usage = subprocess.check_output(['du','--apparent-size','-s','-B1',str(PROJECT),
        str(PROJECT.parent / 'models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218')],text=True,timeout=30)
    now = datetime.now(timezone.utc)
    result = projection(read(PARENT),completed_control_fits_verified=True,
        existing=sum(int(v.split()[0]) for v in usage.splitlines()),free=shutil.disk_usage(PROJECT).free,now=now)
    result['remaining_calendar']['additional_root_recovery_cpu_seconds'] = 1800
    finish = now+timedelta(seconds=sum(result['remaining_calendar'].values()))
    result['complete_route_finish_utc'] = finish.isoformat()
    result['storage']['additional_root_recovery_bytes'] = 100_000_000
    result['storage']['projected_apparent_bytes'] += 100_000_000
    require(finish < datetime.fromisoformat(result['science_freeze_utc'])
        and result['storage']['projected_apparent_bytes'] < 2_000_000_000_000
        and shutil.disk_usage(PROJECT).free > result['storage']['protected_allowance_bytes']+8_200_000_000,
        'Whole existing route plus distinct recovery exceeds calendar/storage')
    return result


def prepare():
    terminal,seconds = failed_input()
    source,pid,_ = live_waiter()
    require(not ROOT.exists() and not DECISION.exists(), 'Unused separate recovery request required')
    from ticket73_configured_full_continuation import admission
    admission(FIT_SHA)
    old = read(OLD / 'plan.json')
    # Prove the actual copied import path before allocating another node.
    environment = dict(os.environ,RSNA_PROJECT_ROOT=str(PROJECT),PYTHONPATH=str(Path(old['source_snapshot']) / 'src'))
    check = PROJECT / 'scripts/check_ticket76_snapshot_paths.py'
    subprocess.run([str(PROJECT / '.venv/bin/python'),str(check),old['source_snapshot']],env=environment,check=True,timeout=30)
    with (PROJECT / 'artifacts/private/campaign-resource-admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        require(not (T74 / 'input-resource-admission.json').exists(), 'Recovery must precede unused analysis final admission')
        envelope = fresh_projection()
        files = [ROUTE,PARENT,OLD / 'plan.json',OLD / 'accepted-launch.json',OLD / 'launch-intent.json',terminal,
            terminal.with_name('terminal-observation.json'),OLD / 'slurm-8131.err',source,T74 / 'await-source-v3-freeze.json',check,Path(__file__)]
        previous = read(ROUTE)
        require(all(digest(r['path']) == r['sha256'] for r in previous['frozen_files']), 'Original joint route bindings changed')
        write(DECISION,dict(previous,schema_version='rsna-ticket76-cpu-root-recovery-v1',
            created_at_utc=datetime.now(timezone.utc).isoformat(),status='ADMITTED',prior_joint_route_sha256=digest(ROUTE),
            prior_failed_cpu_job_id=8131,prior_failed_plan_sha256=OLD_SHA,prior_actual_failed_cpu_node_seconds=seconds,
            prior_full_failed_bound_cpu_node_seconds_retained=1800,additional_cpu_preparation_node_seconds=1800,
            additional_recovery_storage_bytes=100_000_000,T74_lifecycle_source_sha256=digest(source),T74_live_pid=pid,
            original_consumed_intent_reused=False,original_worker_bytes_changed=False,optimizer_updates_repeated=False,
            accelerator_launch_admitted=False,competition_submission=False,envelope=envelope,
            reason='Actual copied helpers require the omitted RSNA_PROJECT_ROOT; actual no-update failure and red/green metadata repro retained',
            frozen_files=[dict(path=str(p),sha256=digest(p)) for p in sorted(set(files))]))
        ROOT.mkdir(mode=0o700)
        snapshot = ROOT / 'source-snapshot'
        shutil.copytree(old['source_snapshot'],snapshot,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        paths = [PROJECT / 'scripts/ticket76_cpu_reference_lifecycle.py',Path(__file__),check]
        for path in paths: shutil.copyfile(path,snapshot / path.name)
        require(digest(snapshot / 'ticket75_cpu_reference.py') == digest(Path(old['source_snapshot']) / 'ticket75_cpu_reference.py'), 'Entire original science/reference worker must be unchanged')
        files = [Path(r['path']) for r in old['frozen_files']]+[DECISION,*paths]
        files.extend(p for p in snapshot.rglob('*') if p.is_file())
        plan = dict(old,created_at_utc=datetime.now(timezone.utc).isoformat(),run_root=str(ROOT),source_snapshot=str(snapshot),
            joint_route_proof_path=str(DECISION),joint_route_proof_sha256=digest(DECISION),
            operational_recovery=dict(prior_plan_sha256=OLD_SHA,prior_job_id=8131,prior_actual_failed_cpu_node_seconds=seconds,
                original_worker_bytes_changed=False,optimizer_updates_repeated=False,workspace_root_environment=str(PROJECT)),
            frozen_files=[dict(path=str(p),sha256=digest(p)) for p in sorted(set(files))])
        write(ROOT / 'plan.json',plan);checksum = digest(ROOT / 'plan.json')
        launcher = (OLD / 'run.sbatch').read_text().replace(str(OLD),str(ROOT)).replace(OLD_SHA,checksum)
        launcher = launcher.replace('export PYTHONPATH=',f'export RSNA_PROJECT_ROOT={PROJECT}\nexport PYTHONPATH=')
        (ROOT / 'run.sbatch').write_text(launcher)
        subprocess.run(['bash','-n',str(ROOT / 'run.sbatch')],check=True)
        lifecycle = snapshot / 'ticket76_cpu_reference_lifecycle.py'
        write(ROOT / 'prepared-validation.json',dict(plan_sha256=checksum,launcher_sha256=digest(ROOT / 'run.sbatch'),
            lifecycle_source_path=str(lifecycle),lifecycle_source_sha256=digest(lifecycle),
            actual_root_import_preflight_passed=True,launched=False,competition_submission=False))
        print(dict(status='DISTINCT_CPU_ROOT_RECOVERY_PREPARED',plan_sha256=checksum,envelope=envelope))


if __name__ == '__main__':
    import sys
    {'install-owner':install_owner,'prepare':prepare}[sys.argv[1]]()
