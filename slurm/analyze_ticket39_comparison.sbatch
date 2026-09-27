#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket39-analysis
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/private/ticket39-resnet18/analysis-%j.out
#SBATCH --error=artifacts/private/ticket39-resnet18/analysis-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
cd "${project_dir}"

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" || "${SLURM_JOB_NUM_NODES:-}" != "1" || "${SLURM_NTASKS:-}" != "1" || "${SLURM_CPUS_PER_TASK:-}" != "16" ]]; then
  echo "Ticket 39 analysis requires one cluster1-cpu node, one task and 16 CPUs" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "missing pinned project Python environment" >&2
  exit 1
fi

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
exec "${project_dir}/.venv/bin/python" "${project_dir}/artifacts/private/ticket39-resnet18/analyze_ticket39.py"
