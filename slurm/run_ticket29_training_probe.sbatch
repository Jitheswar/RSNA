#!/usr/bin/env bash
#SBATCH --job-name=rsna-t29-train-probe
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/private/ticket29-memorization-debug/probe-%j.out
#SBATCH --error=artifacts/private/ticket29-memorization-debug/probe-%j.err

set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: sbatch run_ticket29_training_probe.sbatch HYPOTHESIS LR STEPS C1 C2 C3" >&2
  exit 2
fi

project_dir="$SLURM_SUBMIT_DIR"
canonical_repo="/home/ch.jitheswar/RSNA"
ticket29_root="/home/ch.jitheswar/RSNA-ticket-29"
python="$canonical_repo/.venv/bin/python"
private_root="$project_dir/artifacts/private/ticket29-memorization-debug"
test -x "$python"
test -f "$private_root/cache-7162/metadata.json"
export PYTHONPATH="$project_dir/src:$project_dir/scripts"
export OMP_NUM_THREADS=8
exec "$python" "$project_dir/scripts/ticket29_training_probe.py" \
  --cache "$private_root/cache-7162" \
  --integration-manifest "$canonical_repo/artifacts/private/ticket-05-real-training-input/private/integration-manifest.json" \
  --parent-checkpoint "$ticket29_root/artifacts/private/ticket-29-real-diagnostic/run-7160/training/checkpoints/final.pt" \
  --output "$private_root/$1-$SLURM_JOB_ID" \
  --hypothesis "$1" \
  --learning-rate "$2" \
  --steps "$3" \
  --channels "$4" "$5" "$6"
