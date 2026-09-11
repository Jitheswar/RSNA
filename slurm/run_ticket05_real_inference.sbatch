#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket05
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/ticket-05-slurm-%j.out
#SBATCH --error=artifacts/ticket-05-slurm-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
source_path="${RSNA_TICKET05_SOURCE:?set RSNA_TICKET05_SOURCE to the private DICOM archive or extracted root}"
inventory_path="${RSNA_TICKET05_INVENTORY:?set RSNA_TICKET05_INVENTORY to the private ticket-03 inventory.json}"
selection_path="${RSNA_TICKET05_SELECTION:?set RSNA_TICKET05_SELECTION to the private ticket-03 selection.json}"
output_dir="${RSNA_TICKET05_OUTPUT:?set RSNA_TICKET05_OUTPUT to a private ticket-05 output directory}"
manifest_path="${RSNA_TICKET05_MANIFEST:-${output_dir}/private/integration-manifest.json}"
cd "${project_dir}"
mkdir -p artifacts "${output_dir}/private" "${output_dir}/public"

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "ticket-05 requires the cluster1-cpu partition" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "missing ${project_dir}/.venv/bin/python; provision the pinned user-space environment first" >&2
  exit 1
fi
if [[ ! -f "${source_path}" && ! -d "${source_path}" ]]; then
  echo "real-DICOM source does not exist: ${source_path}" >&2
  exit 1
fi
if [[ ! -f "${inventory_path}" || ! -f "${selection_path}" ]]; then
  echo "ticket-03 private inventory and selection are required" >&2
  exit 1
fi

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
real_args=(
  --manifest "${manifest_path}"
  --source "${source_path}"
  --output "${output_dir}/private/submission.csv"
  --evidence "${output_dir}/public/evidence.json"
  --private-evidence "${output_dir}/private/evidence.json"
  --inventory "${inventory_path}"
  --selection "${selection_path}"
  --require-slurm
)

exec "${project_dir}/.venv/bin/python" -m rsna_knee.real_inference_cli "${real_args[@]}"
