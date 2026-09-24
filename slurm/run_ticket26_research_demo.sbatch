#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket26-demo
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/ticket-26-demo-%j.out
#SBATCH --error=artifacts/ticket-26-demo-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
report_path="${RSNA_TICKET26_REPORT:?set RSNA_TICKET26_REPORT to a new .md path outside the repository}"
cd "${project_dir}"

if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "missing ${project_dir}/.venv/bin/python; provision the pinned user-space environment first" >&2
  exit 1
fi

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${project_dir}/.venv/bin/python" -m rsna_knee.research_demo \
  --require-slurm \
  --report "${report_path}"
