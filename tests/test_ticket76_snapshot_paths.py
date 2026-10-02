"""Actual copied-helper import regression; no MRI loading or optimizer updates."""
import os
from pathlib import Path
import subprocess

import pytest

PROJECT = Path(__file__).parents[1]
SNAPSHOT = PROJECT / 'artifacts/private/ticket75-cpu-reference-v1/source-snapshot'
pytestmark = pytest.mark.skipif(not SNAPSHOT.exists(),reason='Actual immutable private snapshot required for this integration repro')


def check(root_value):
    environment = dict(os.environ,PYTHONPATH=str(SNAPSHOT / 'src'))
    environment.pop('RSNA_PROJECT_ROOT',None)
    if root_value is not None: environment['RSNA_PROJECT_ROOT'] = root_value
    return subprocess.run([str(PROJECT / '.venv/bin/python'),str(PROJECT / 'scripts/check_ticket76_snapshot_paths.py'),str(SNAPSHOT)],
        cwd=PROJECT,env=environment,text=True,capture_output=True,timeout=30)


def test_omitted_root_reproduces_exact_original_fold_manifest_failure():
    original = (PROJECT / 'artifacts/private/ticket75-cpu-reference-v1/run.sbatch').read_text()
    assert 'export RSNA_PROJECT_ROOT=' not in original
    result = check(None)
    assert result.returncode != 0 and 'FoldManifestIntegrityError' in result.stderr
    assert str(SNAPSHOT / 'artifacts/private/ticket-04-real-v2/private/fold-manifest.json') in result.stderr


def test_only_correct_root_fixes_actual_original_helper_chain():
    result = check(str(PROJECT))
    assert result.returncode == 0,result.stderr
    assert 'ACTUAL_HELPER_PATHS_PASS' in result.stdout and '"MRI_loaded": false' in result.stdout and '"optimizer_updates": 0' in result.stdout
    wrong = check(str(SNAPSHOT))
    assert wrong.returncode != 0 and 'FoldManifestIntegrityError' in wrong.stderr
