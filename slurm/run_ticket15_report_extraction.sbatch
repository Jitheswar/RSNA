#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket15
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/ticket-15-slurm-%j.out
#SBATCH --error=artifacts/ticket-15-slurm-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
fold_manifest="${RSNA_TICKET15_FOLD_MANIFEST:?set RSNA_TICKET15_FOLD_MANIFEST to the private ticket-04 manifest}"
reports="${RSNA_TICKET15_REPORTS:?set RSNA_TICKET15_REPORTS to the private report corpus JSON}"
output_dir="${RSNA_TICKET15_OUTPUT:?set RSNA_TICKET15_OUTPUT to a private ticket-15 output directory}"
cd "${project_dir}"
mkdir -p artifacts

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "ticket-15 requires the cluster1-cpu partition" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "missing ${project_dir}/.venv/bin/python; provision the pinned user-space environment first" >&2
  exit 1
fi
if [[ ! -f "${fold_manifest}" || ! -f "${reports}" ]]; then
  echo "ticket-15 private fold manifest and report corpus are required" >&2
  exit 1
fi

public_output="${RSNA_TICKET15_PUBLIC_OUTPUT:-${output_dir}/public/report-evidence.json}"
private_output="${RSNA_TICKET15_PRIVATE_OUTPUT:-${output_dir}/private/report-labels.json}"
export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${project_dir}/.venv/bin/python" -m rsna_knee.report_extraction_cli \
  --fold-manifest "${fold_manifest}" \
  --reports "${reports}" \
  --public-output "${public_output}" \
  --private-output "${private_output}" \
  --require-slurm
