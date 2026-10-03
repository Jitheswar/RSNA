#!/usr/bin/env bash
#SBATCH --job-name=rsna-fast-cpufold
#SBATCH --partition=cluster1-cpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --array=0-4
#SBATCH --output=artifacts/private/fast-v1/logs/cpufold-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
~/rsna-gpu2/bin/python -u scripts/fast/train.py --fold "${SLURM_ARRAY_TASK_ID}" --epochs 10 --workers 6 \
  --train-csv artifacts/private/fast-v1/train.csv --series-csv artifacts/private/fast-v1/train_series.csv \
  --out artifacts/private/fast-v1/run3-cpu
