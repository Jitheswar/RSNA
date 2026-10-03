#!/usr/bin/env bash
#SBATCH --job-name=v2-pack384
#SBATCH --partition=cluster1-cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --array=4-7
#SBATCH --output=artifacts/private/v2/vol384/logs/pack-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
~/rsna-gpu2/bin/python scripts/v2/pack.py --split train --out artifacts/private/v2/vol384 \
  --shard "${SLURM_ARRAY_TASK_ID}" --nshards 8 --workers 23
