#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket24-v4
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/private/ticket-24-v4-validation/slurm-%j.out
#SBATCH --error=artifacts/private/ticket-24-v4-validation/slurm-%j.err

set -euo pipefail
cd /home/ch.jitheswar/RSNA

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export PYTHONHASHSEED=0

PYTHON=.venv/bin/python

env -u SLURM_JOB_ID -u SLURM_JOB_PARTITION "$PYTHON" -m pytest -q
"$PYTHON" -m mypy src/rsna_knee
"$PYTHON" -m compileall -q src/rsna_knee tests

for script in slurm/*.sh slurm/*.sbatch; do
    if [ -f "$script" ]; then
        bash -n "$script"
    fi
done

"$PYTHON" scripts/ticket24_v4_representative_validation.py
