#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket06
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/ticket-06-slurm-%j.out
#SBATCH --error=artifacts/ticket-06-slurm-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
manifest_path="${RSNA_TICKET06_MANIFEST:?set RSNA_TICKET06_MANIFEST to the private ticket-05 integration manifest}"
source_path="${RSNA_TICKET06_SOURCE:?set RSNA_TICKET06_SOURCE to the private DICOM archive or fixture directory}"
cache_root="${RSNA_TICKET06_CACHE_ROOT:?set RSNA_TICKET06_CACHE_ROOT to a private disposable cache directory}"
output_path="${RSNA_TICKET06_OUTPUT:?set RSNA_TICKET06_OUTPUT to the aggregate ticket-06 evidence JSON}"
cd "${project_dir}"
mkdir -p artifacts "$(dirname "${output_path}")"

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "ticket-06 requires the cluster1-cpu partition" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "missing ${project_dir}/.venv/bin/python; provision the pinned user-space environment first" >&2
  exit 1
fi
if [[ ! -f "${manifest_path}" || ( ! -f "${source_path}" && ! -d "${source_path}" ) ]]; then
  echo "ticket-06 manifest or real-DICOM source does not exist" >&2
  exit 1
fi

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${project_dir}/.venv/bin/python" -m rsna_knee.cache_benchmark_cli \
  --manifest "${manifest_path}" \
  --source "${source_path}" \
  --cache-root "${cache_root}" \
  --output "${output_path}" \
  --workers "${SLURM_CPUS_PER_TASK:-1}" \
  --require-slurm
