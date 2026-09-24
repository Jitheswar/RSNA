#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket24-v4-preflight
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/private/ticket-24-v4-validation/preflight-%j.out
#SBATCH --error=artifacts/private/ticket-24-v4-validation/preflight-%j.err

set -euo pipefail
cd /home/ch.jitheswar/RSNA

".venv/bin/python" -m compileall -q src/rsna_knee tests scripts

for script in slurm/*.sh slurm/*.sbatch; do
    if [ -f "$script" ]; then
        bash -n "$script"
    fi
done

".venv/bin/python" scripts/ticket24_v4_artifact_preflight.py
