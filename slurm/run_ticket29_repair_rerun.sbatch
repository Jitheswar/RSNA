#!/usr/bin/env bash
#SBATCH --job-name=rsna-t29-repair-real
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/private/ticket29-memorization-debug/repair-%j.out
#SBATCH --error=artifacts/private/ticket29-memorization-debug/repair-%j.err

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: sbatch run_ticket29_repair_rerun.sbatch PASSING_PROBE_RESULT" >&2
  exit 2
fi

project_dir="${SLURM_SUBMIT_DIR:?}"
canonical_repo="/home/ch.jitheswar/RSNA"
python="$canonical_repo/.venv/bin/python"
private_root="$project_dir/artifacts/private/ticket29-memorization-debug"
original_root="/home/ch.jitheswar/RSNA-ticket-29/artifacts/private/ticket-29-real-diagnostic"
test -x "$python"
test -f "$1"
export PYTHONPATH="$project_dir/src:$project_dir/scripts"
export OMP_NUM_THREADS=8
exec "$python" "$project_dir/scripts/ticket29_repair_rerun.py" \
  --plan "$original_root/diagnostic-plan.json" \
  --integration-manifest "$canonical_repo/artifacts/private/ticket-05-real-training-input/private/integration-manifest.json" \
  --fold-manifest "$canonical_repo/artifacts/private/ticket-04-real-v2/private/fold-manifest.json" \
  --selection-report "$canonical_repo/artifacts/private/ticket-03-inventory-run/private/selection.json" \
  --source-archive "$canonical_repo/rsna-knee-abnormality-detection.zip" \
  --cache "$private_root/cache-7162" \
  --probe-result "$1" \
  --repair-plan "$private_root/real-repair-plan.json" \
  --output "$private_root/real-repair-$SLURM_JOB_ID"
