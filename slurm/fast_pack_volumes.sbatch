#!/usr/bin/env bash
#SBATCH --job-name=rsna-fast-pack
#SBATCH --partition=cluster1-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=artifacts/private/fast-v1/logs/pack-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
.venv/bin/python scripts/fast/pack_volumes.py --shard "${SLURM_ARRAY_TASK_ID}" --nshards 4 --workers 16
