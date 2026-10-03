#!/usr/bin/env bash
#SBATCH --job-name=v2-gpu
#SBATCH --partition=cluster1-gpu,cluster1-gpu-add
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=artifacts/private/v2/queue/logs/worker-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
exec bash scripts/v2/worker.sh
