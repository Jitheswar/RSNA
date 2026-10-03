#!/usr/bin/env bash
#SBATCH --job-name=rsna-llm-label
#SBATCH --partition=cluster1-cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/private/fast-v1/logs/llm-%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=32
~/rsna-gpu2/bin/python scripts/fast/llm_label.py --model ~/models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218 \
  --train-csv artifacts/private/fast-v1/train.csv --threads 32 "$@"
