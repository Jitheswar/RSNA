"""Replace only the unadmitted waiting lifecycle, preserving all frozen science."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import time

from ticket75_cpu_reference import digest, read, require, write
from prepare_ticket75_joint_resource_envelope import PROJECT, T74, UNIT

OLD_UNIT = 'rsna-ticket74-await-native-inputs-v1.service'
OLD_SHA = '3b21a40d7ad2d7cfd0683d87e61a316da76feede83f3092f455bfbb77516c665'
PLAN_SHA = '187599d49ac25b90fd9fdd5c8dcf830c115481cbb59916771c94bc53edaa6470'


def install():
    require(digest(T74 / 'plan.json') == PLAN_SHA, 'Exact unchanged prospective scientific plan required')
    require(not (T74 / 'await-source-v2-freeze.json').exists()
            and not (T74 / 'input-resource-admission.json').exists()
            and not (PROJECT / '.scratch/rsna-leaderboard/resource-approvals/ticket74-fixed-analysis-20261002.json').exists()
            and not any(T74.glob('*/launch-intent.json')), 'Only unused waiting lifecycle may be replaced')
    old = read(T74 / 'await-source-freeze.json')
    require(digest(old['frozen_source_path']) == digest(old['source_path']) == OLD_SHA, 'Preserve exact frozen old source')
    old_start = read(T74 / 'await-inputs-started.json')
    pid = old_start['pid']
    os.kill(pid, 0)
    raw = subprocess.check_output(['systemctl', '--user', 'show', OLD_UNIT, '--property=ActiveState', '--property=MainPID'], text=True, timeout=15)
    require('ActiveState=active' in raw and f'MainPID={pid}\n' in raw, 'Exact live old waiter required')
    source = PROJECT / 'scripts/ticket74_analysis_lifecycle_v2.py'
    old_tree = ast.parse(Path(old['source_path']).read_text())
    new_tree = ast.parse(source.read_text())
    def functions(tree):
        return {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
    a, b = functions(old_tree), functions(new_tree)
    require(all(a[n] == b[n] for n in ('remaining_reserves', 'terminal', 'launch', 'watch')),
            'Scientific completion, reserve-credit and one-attempt launch rules must remain identical')
    snapshot = T74 / 'await-source-snapshot-v2'
    snapshot.mkdir(mode=0o700)
    target = snapshot / source.name
    shutil.copyfile(source, target)
    write(T74 / 'await-source-v2-freeze.json', dict(plan_sha256=PLAN_SHA, protocol_retuned=False,
        prior_pid=pid, prior_unit=OLD_UNIT, prior_source_sha256=OLD_SHA, frozen_source_path=str(target),
        source_bindings=[dict(path=str(p), sha256=digest(p)) for p in (source, target)],
        changes='Joint pilot calendar/storage reservation and shared resource-admission lock only',
        final_inputs_admitted=False, resource_decision_admitted=False, launched=False, competition_submission=False))
    subprocess.run(['systemctl', '--user', 'stop', OLD_UNIT], check=True, timeout=30)
    require(not Path(f'/proc/{pid}').exists(), 'Old actual PID must stop before new owner starts')
    write(T74 / 'await-inputs-v1-stopped-for-v2.json', dict(prior_pid=pid, prior_unit=OLD_UNIT,
        prior_source_sha256=OLD_SHA, source_bytes_preserved=True, actual_process_missing=True,
        reason='Close future outstanding pilot resource reservation ownership; no science change'))
    subprocess.run(['systemd-run', '--user', '--collect', f'--unit={UNIT.removesuffix(".service")}',
        str(PROJECT / '.venv/bin/python'), str(target), 'await-inputs'], check=True, timeout=30)
    for _ in range(20):
        if (T74 / 'await-inputs-v2-started.json').exists():
            from prepare_ticket75_joint_resource_envelope import live_waiter
            _, actual_pid, _ = live_waiter()
            print(dict(status='UNCHANGED_SCIENCE_JOINT_RESERVE_WAITER_LIVE', pid=actual_pid, source_sha256=digest(target)))
            return
        time.sleep(0.5)
    raise TimeoutError('Re-poll the same new unit; observation expiry does not authorize restart')


if __name__ == '__main__':
    install()
