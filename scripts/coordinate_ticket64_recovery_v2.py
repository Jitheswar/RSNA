"""Advance only admitted v2 recovery stages, preserving the live v1 first stage."""

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from coordinate_ticket64_full_backbone_comparison import stage_state, next_stage, launch_environment

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'artifacts/private/ticket64-full-backbone-comparison-v2'
OLD = PROJECT / 'artifacts/private/ticket64-full-backbone-comparison-v1'
CHECKSUM = '41028974908d8ea8c4fe336f86de0263782100c76b9375dc7693bf60ac213eaf'
SCIENCE_SHA = '557230fec672f6db9252b1895a4d174652fe217c0388a8c9598bd749183aa985'


def read(path):
    return json.loads(path.read_text())


def stage_location(fold, stage):
    if fold == stage == 0:
        return OLD / 'full-backbone/fold-0/stage-0', SCIENCE_SHA
    return ROOT / f'full-backbone/fold-{fold}/stage-{stage}', CHECKSUM


def run(checksum):
    if checksum != CHECKSUM:
        raise ValueError('Only the exact admitted recovery can advance')
    with (ROOT / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (ROOT / 'plan.json').open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != checksum:
                raise ValueError('Recovery plan changed')
        plan = read(ROOT / 'plan.json'); missing_since = {}
        while True:
            control = stage_state(ROOT / 'control', checksum, None, '02:00:00')
            folds = [[stage_state(*[stage_location(f, s)[0], stage_location(f, s)[1]],
                                  min(7500, (s+1)*1000), '05:30:00')
                      for s in range(8)] for f in range(3)]
            if control == 'complete' and all(all(s == 'complete' for s in f) for f in folds):
                print('All six fits completed; independent original-MRI replay and final comparison remain required.', flush=True)
                return
            rows = subprocess.check_output(['squeue', '-u', os.environ['USER'], '-h', '-o', '%i %P %t'], text=True).splitlines()
            jobs = [r.split()[0] for r in rows if r.split()[1] == 'cluster1-cpu']
            active = [(ROOT / 'control', control)] + [(stage_location(f, s)[0], state)
                for f, states in enumerate(folds) for s, state in enumerate(states)]
            for directory, state in active:
                if state != 'active':
                    continue
                job = str(read(directory / 'accepted-launch.json')['job_id'])
                if job in jobs:
                    missing_since.pop(job, None)
                elif time.monotonic()-missing_since.setdefault(job, time.monotonic()) > 300:
                    raise ValueError(f'Accepted job {job} has neither live queue nor retained terminal; no retry')
            item = next_stage(control, folds, len(jobs))
            observation = dict(at_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=checksum,
                control=control, folds=folds, occupied_cpu_jobs=jobs, next_stage=item,
                inherited_stage0_job=8032, automatic_retry=False, candidate_promotion=False)
            temp = ROOT / 'coordinator-state.partial'; temp.write_text(json.dumps(observation, indent=2)+'\n')
            temp.replace(ROOT / 'coordinator-state.json')
            if item is not None:
                if datetime.now(timezone.utc) >= datetime.fromisoformat(plan['projection']['science_freeze_utc']):
                    raise ValueError('Science freeze reached; no further fitting launch')
                fold, stage = item
                if fold == stage == 0:
                    raise ValueError('Inherited first full stage cannot be relaunched')
                print(f'Advancing registered recovery fold {fold}, stage {stage}', flush=True)
                subprocess.run([str(PROJECT / '.venv/bin/python'),
                    str(Path(plan['source_snapshot']) / 'ticket64_full_backbone_recovery_v2.py'),
                    'launch', checksum, 'full', str(fold), str(stage)], env=launch_environment(plan), check=True)
                continue
            time.sleep(30)


if __name__ == '__main__':
    run(sys.argv[1])
