#!/usr/bin/env bash
#SBATCH --job-name=rsna-fast-train
#SBATCH --partition=cluster1-gpu-add
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/private/fast-v1/logs/train-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
nvidia-smi -L
export HF_HUB_OFFLINE=1
~/rsna-gpu2/bin/python scripts/fast/train.py \
  --train-csv artifacts/private/fast-v1/train.csv \
  --series-csv artifacts/private/fast-v1/train_series.csv "$@"
