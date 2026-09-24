#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket25-efficiency
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/private/ticket-25-efficiency/slurm-%j.out
#SBATCH --error=artifacts/private/ticket-25-efficiency/slurm-%j.err

set -euo pipefail
cd /home/ch.jitheswar/RSNA

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export PYTHONHASHSEED=0

OUTPUT_ROOT="artifacts/private/ticket-25-efficiency/run-${SLURM_JOB_ID}"
.venv/bin/python scripts/ticket25_efficiency_benchmark.py --output-root "$OUTPUT_ROOT"
