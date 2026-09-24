#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket24-v6
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/private/ticket-24-v6-validation/slurm-%j.out
#SBATCH --error=artifacts/private/ticket-24-v6-validation/slurm-%j.err

set -euo pipefail
cd /home/ch.jitheswar/RSNA

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export PYTHONHASHSEED=0

.venv/bin/python scripts/ticket24_v6_representative_validation.py
