#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket44-analysis
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/home/ch.jitheswar/RSNA/artifacts/private/ticket44-full-weak-exposure/analysis-%j.out
#SBATCH --error=/home/ch.jitheswar/RSNA/artifacts/private/ticket44-full-weak-exposure/analysis-%j.err
#SBATCH --chdir=/home/ch.jitheswar/RSNA
set -euo pipefail
umask 077
project_dir=/home/ch.jitheswar/RSNA
export RSNA_PROJECT_ROOT="$project_dir"
snapshot="$("$project_dir/.venv/bin/python" - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('artifacts/private/ticket44-full-weak-exposure/plan.json').read_text())['source_snapshot'])
PY
)"
export PYTHONPATH="$snapshot/src"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export SLURM_JOB_PARTITION=cluster1-cpu
"$project_dir/.venv/bin/python" - <<'PY'
import json, shutil, subprocess
from pathlib import Path
from rsna_knee import ticket44
plan=ticket44._load_plan()
snapshot=Path(plan['source_snapshot'])
try:
    result=subprocess.run(['git','rev-parse','HEAD'],cwd=snapshot,capture_output=True,text=True,timeout=5)
    probe={'git_executable':shutil.which('git'),'returncode':result.returncode,'stderr':result.stderr.strip()}
except OSError as exc:
    probe={'git_executable':shutil.which('git'),'error':str(exc)}
print('Training snapshot Git probe:',json.dumps(probe))
PY
analysis_script="${RSNA_TICKET44_ANALYSIS_SCRIPT:?immutable analysis script required}"
analysis_sha256="${RSNA_TICKET44_ANALYSIS_SHA256:?immutable analysis script digest required}"
"$project_dir/.venv/bin/python" - "$analysis_script" "$analysis_sha256" <<'PY'
import hashlib, sys
from pathlib import Path
if hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() != sys.argv[2]:
    raise SystemExit('analysis script changed')
PY
"$project_dir/.venv/bin/python" "$analysis_script"
