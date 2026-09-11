#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket02
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/ticket-02-slurm-%j.out
#SBATCH --error=artifacts/ticket-02-slurm-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
cd "${project_dir}"
mkdir -p artifacts

if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "missing ${project_dir}/.venv/bin/python; provision the pinned user-space environment first" >&2
  exit 1
fi

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${project_dir}/.venv/bin/python" -m rsna_knee.demo \
  --require-slurm \
  --output "${project_dir}/artifacts/submission.csv" \
  --evidence "${project_dir}/artifacts/ticket-02-inference-${SLURM_JOB_ID}.json"
