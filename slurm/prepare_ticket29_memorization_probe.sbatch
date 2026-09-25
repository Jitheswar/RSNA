#!/usr/bin/env bash
#SBATCH --job-name=rsna-t29-probe-cache
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/private/ticket29-memorization-debug/prepare-%j.out
#SBATCH --error=artifacts/private/ticket29-memorization-debug/prepare-%j.err

set -euo pipefail

project_dir="$SLURM_SUBMIT_DIR"
canonical_repo="/home/ch.jitheswar/RSNA"
ticket29_root="/home/ch.jitheswar/RSNA-ticket-29"
python="$canonical_repo/.venv/bin/python"
private_root="$project_dir/artifacts/private/ticket29-memorization-debug"
test -x "$python"
test -d "$private_root"
export PYTHONPATH="$project_dir/src"
export OMP_NUM_THREADS=8
exec "$python" "$project_dir/scripts/ticket29_memorization_probe.py" prepare \
  --plan "$ticket29_root/artifacts/private/ticket-29-real-diagnostic/diagnostic-plan.json" \
  --integration-manifest "$canonical_repo/artifacts/private/ticket-05-real-training-input/private/integration-manifest.json" \
  --fold-manifest "$canonical_repo/artifacts/private/ticket-04-real-v2/private/fold-manifest.json" \
  --selection-report "$canonical_repo/artifacts/private/ticket-03-inventory-run/private/selection.json" \
  --source-archive "$canonical_repo/rsna-knee-abnormality-detection.zip" \
  --output "$private_root/cache-$SLURM_JOB_ID"
