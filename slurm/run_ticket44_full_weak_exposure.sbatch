#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket44
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/ch.jitheswar/RSNA/artifacts/private/ticket44-full-weak-exposure/slurm-%j.out
#SBATCH --error=/home/ch.jitheswar/RSNA/artifacts/private/ticket44-full-weak-exposure/slurm-%j.err
#SBATCH --chdir=/home/ch.jitheswar/RSNA

set -euo pipefail
umask 077
project_dir=/home/ch.jitheswar/RSNA
python="$project_dir/.venv/bin/python"
plan="$project_dir/artifacts/private/ticket44-full-weak-exposure/plan.json"
snapshot="$("$python" - "$plan" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())['source_snapshot'])
PY
)"
export RSNA_PROJECT_ROOT="$project_dir"
export PYTHONPATH="$snapshot/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
# The live scontrol allocation is checked in ticket44.run before preflight.
# This cluster omits some of these standard Slurm environment fields.
export SLURM_JOB_PARTITION=cluster1-cpu
export SLURM_JOB_NUM_NODES=1
export SLURM_NTASKS=1
export SLURM_CPUS_PER_TASK=16
export SLURM_MEM_PER_NODE=64G
export SLURM_TIMELIMIT=02:00:00
export SLURM_JOB_EXCLUSIVE=0
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
"$python" -m rsna_knee.ticket44 run
