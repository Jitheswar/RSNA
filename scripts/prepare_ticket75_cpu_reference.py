"""Freeze one CPU reference request within the separately admitted whole route."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from ticket75_cpu_reference import digest, read, require, write, FIT_SHA, SELECTION_SHA, SCHEMA
from prepare_ticket75_joint_resource_envelope import PROJECT, ROUTE, live_waiter

ROOT = PROJECT / 'artifacts/private/ticket75-cpu-reference-v1'
SELECTION = PROJECT / 'artifacts/private/ticket75-real-mri-selection-v1/selection.json'


def prepare():
    require(not ROOT.exists() and digest(SELECTION) == SELECTION_SHA, 'Unused CPU preparation and original selection required')
    source, pid, _ = live_waiter()
    route = read(ROUTE)
    require(route['status'] == 'ADMITTED' and route['fit_plan_sha256'] == FIT_SHA
            and route['T74_lifecycle_source_sha256'] == digest(source)
            and route['accelerator_launch_admitted'] is False, 'Admitted joint route and sole CPU scope required')
    require(all(digest(r['path']) == r['sha256'] for r in route['frozen_files']), 'Joint admission input/source changed')
    from ticket73_configured_full_continuation import admission
    plan = admission(FIT_SHA)
    selection = read(SELECTION)
    require(all(digest(r['path']) == r['sha256'] for r in selection['frozen_metadata_sources']), 'Original selected sources changed')
    sys.path.insert(0, str(PROJECT / 'src'))
    from rsna_knee.folds import load_fold_manifest
    cohort = read(plan['cohort_parent_plan'])
    manifest_path = next(Path(r['path']) for r in cohort['frozen_files'] if Path(r['path']).name == 'fold-manifest.json')
    manifest = load_fold_manifest(manifest_path)
    require(manifest.manifest_checksum == selection['fold_manifest_checksum'], 'Original held-out exclusion contract required')
    ROOT.mkdir(mode=0o700)
    snapshot = ROOT / 'source-snapshot'
    shutil.copytree(plan['source_snapshot'], snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    scripts = ('ticket75_cpu_reference.py', 'prepare_ticket75_joint_resource_envelope.py',
               'ticket75_cpu_reference_lifecycle.py', 'prepare_ticket75_cpu_reference.py')
    originals = [PROJECT / 'scripts' / n for n in scripts]
    originals.append(PROJECT / 'src/rsna_knee/tpu_pilot_inputs.py')
    for path in originals:
        destination = snapshot / (Path('src/rsna_knee') / path.name if path.parent.name == 'rsna_knee' else Path(path.name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    pixels = read(Path(plan['pixel_root']) / 'plan.json')
    reader = next(r for r in pixels['frozen_files'] if Path(r['path']).name == 'ticket62_mri_pixel_materialize.py')
    require(digest(reader['path']) == reader['sha256'], 'Original native MRI reader changed')
    shutil.copyfile(reader['path'], snapshot / 'ticket62_mri_pixel_materialize.py')
    resources = dict(partition='cluster1-cpu', nodes=1, tasks=1, cpus_per_task=16, memory_gib=64,
        walltime_seconds=1800, code_watchdog_seconds=1740, maximum_launch_attempts=1, exclusive=False,
        requeue=False, gpu_count=0, tpu_count=0, external_spend_usd=0)
    files = [ROUTE, SELECTION, Path(plan['cohort_parent_plan']), manifest_path, Path(reader['path']), source, *originals]
    files.extend(Path(r['path']) for r in route['frozen_files'])
    files.extend(Path(r['path']) for r in selection['frozen_metadata_sources'])
    files.extend(p for p in snapshot.rglob('*') if p.is_file())
    request = dict(schema_version=SCHEMA, created_at_utc=datetime.now(timezone.utc).isoformat(), fit_plan_sha256=FIT_SHA,
        selection_sha256=SELECTION_SHA, selection_path=str(SELECTION), run_root=str(ROOT), source_snapshot=str(snapshot),
        joint_route_ownership_closed=True, joint_route_proof_path=str(ROUTE), joint_route_proof_sha256=digest(ROUTE),
        resources=resources, runtime_versions=plan['runtime_versions'], replay_training_study_indices=list(range(8)),
        original_held_out_study_uids=sorted(manifest.held_out_study_uids(0)),
        frozen_files=[dict(path=str(p), sha256=digest(p)) for p in sorted(set(files))],
        candidate_promotion=False, competition_submission=False, original_fit_updates_credited=False,
        accelerator_launch_admitted=False, actual_native_MRI_reference_verified=False)
    write(ROOT / 'plan.json', request)
    checksum = digest(ROOT / 'plan.json')
    from ticket75_cpu_reference import validate_request
    validate_request(request)
    launcher = ROOT / 'run.sbatch'
    launcher.write_text(f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket75-cpu-reference
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output={ROOT}/slurm-%j.out
#SBATCH --error={ROOT}/slurm-%j.err
#SBATCH --chdir={PROJECT}
set -euo pipefail
umask 077
export PYTHONPATH={snapshot}/src
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
{PROJECT}/.venv/bin/python {snapshot}/ticket75_cpu_reference.py {ROOT}/plan.json {checksum}
''')
    subprocess.run(['bash', '-n', str(launcher)], check=True)
    write(ROOT / 'prepared-validation.json', dict(plan_sha256=checksum, launcher_sha256=digest(launcher),
        lifecycle_source_path=str(snapshot / 'ticket75_cpu_reference_lifecycle.py'),
        lifecycle_source_sha256=digest(snapshot / 'ticket75_cpu_reference_lifecycle.py'),
        launched=False, accelerator_launch_admitted=False, competition_submission=False))
    print(json.dumps(dict(status='CPU_REFERENCE_PREPARED_ONLY', plan_sha256=checksum)))


if __name__ == '__main__':
    prepare()
