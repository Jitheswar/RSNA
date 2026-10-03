#!/usr/bin/env bash
#SBATCH --job-name=rsna-pack320
#SBATCH --partition=cluster2-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=artifacts/private/fast-v2-320/logs/pack-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
.venv/bin/python scripts/fast/pack_dicom.py --size 320 --depth 32 --out artifacts/private/fast-v2-320 \
  --shard "${SLURM_ARRAY_TASK_ID}" --nshards 5 --workers 14
