#!/usr/bin/env bash
#SBATCH --job-name=rsna-fast-shard
#SBATCH --partition=cluster1-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/private/fast-v1/logs/shard-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
.venv/bin/python scripts/fast/shard_for_kaggle.py
