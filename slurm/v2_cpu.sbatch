#!/usr/bin/env bash
#SBATCH --job-name=v2-cpu
#SBATCH --partition=cluster1-cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/private/v2/logs/cpu-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
eval "$@"
