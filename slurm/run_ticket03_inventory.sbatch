#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket03
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/ticket-03-slurm-%j.out
#SBATCH --error=artifacts/ticket-03-slurm-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
archive_path="${RSNA_TICKET03_ARCHIVE:?set RSNA_TICKET03_ARCHIVE to the private source archive}"
output_dir="${RSNA_TICKET03_OUTPUT:?set RSNA_TICKET03_OUTPUT to a private report directory}"
source_version="${RSNA_TICKET03_SOURCE_VERSION:?set RSNA_TICKET03_SOURCE_VERSION to the recorded source version}"
cd "${project_dir}"
mkdir -p artifacts

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "ticket-03 requires the cluster1-cpu partition" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "missing ${project_dir}/.venv/bin/python; provision the pinned user-space environment first" >&2
  exit 1
fi
if [[ ! -f "${archive_path}" ]]; then
  echo "source archive does not exist: ${archive_path}" >&2
  exit 1
fi

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
inventory_args=(
  --archive "${archive_path}"
  --output-dir "${output_dir}"
  --source-version "${source_version}"
  --report-only-count "${RSNA_TICKET03_REPORT_ONLY_COUNT:-100}"
  --seed "${RSNA_TICKET03_SEED:-20260911}"
  --current-project-bytes "${RSNA_TICKET03_CURRENT_PROJECT_BYTES:-0}"
  --require-slurm
)
if [[ -n "${RSNA_TICKET03_EXTRACT_TO:-}" ]]; then
  inventory_args+=(--extract-to "${RSNA_TICKET03_EXTRACT_TO}")
fi
if [[ -n "${RSNA_TICKET03_SUPPORTED_TRANSFER_SYNTAXES:-}" ]]; then
  IFS=',' read -r -a supported_syntaxes <<< "${RSNA_TICKET03_SUPPORTED_TRANSFER_SYNTAXES}"
  for syntax in "${supported_syntaxes[@]}"; do
    inventory_args+=(--supported-transfer-syntax "${syntax}")
  done
fi

exec "${project_dir}/.venv/bin/python" -m rsna_knee.inventory_cli "${inventory_args[@]}"
