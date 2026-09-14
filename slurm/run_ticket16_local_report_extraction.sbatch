#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket16
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/ticket-16-slurm-%j.out
#SBATCH --error=artifacts/ticket-16-slurm-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
fold_manifest="${RSNA_TICKET16_FOLD_MANIFEST:?set RSNA_TICKET16_FOLD_MANIFEST to the private ticket-04 manifest}"
reports="${RSNA_TICKET16_REPORTS:?set RSNA_TICKET16_REPORTS to the private report corpus JSON}"
approval="${RSNA_TICKET16_MODEL_APPROVAL:?set RSNA_TICKET16_MODEL_APPROVAL to the owner-approved model JSON}"
model_dir="${RSNA_TICKET16_MODEL_DIR:?set RSNA_TICKET16_MODEL_DIR to the verified local model directory}"
output_dir="${RSNA_TICKET16_OUTPUT:?set RSNA_TICKET16_OUTPUT to a private ticket-16 output directory}"

cd "${project_dir}"
mkdir -p artifacts

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "ticket-16 requires the cluster1-cpu partition" >&2
  exit 1
fi
if [[ "${SLURM_CPUS_PER_TASK:-0}" -gt 16 ]]; then
  echo "ticket-16 cannot use more than 16 CPUs" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "missing ${project_dir}/.venv/bin/python; provision the pinned user-space environment first" >&2
  exit 1
fi
if [[ ! -f "${fold_manifest}" || ! -f "${reports}" || ! -f "${approval}" ]]; then
  echo "ticket-16 private fold manifest, report corpus and approval JSON are required" >&2
  exit 1
fi
if [[ ! -d "${model_dir}" ]]; then
  echo "ticket-16 approved local model directory is required" >&2
  exit 1
fi

public_output="${RSNA_TICKET16_PUBLIC_OUTPUT:-${output_dir}/public/comparison.json}"
private_output="${RSNA_TICKET16_PRIVATE_OUTPUT:-${output_dir}/private/comparison.json}"
export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Both the loader and the environment reject hub/network resolution. Reports
# are passed only to the in-process local model.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec "${project_dir}/.venv/bin/python" -m rsna_knee.local_report_extraction_cli \
  --fold-manifest "${fold_manifest}" \
  --reports "${reports}" \
  --model-approval "${approval}" \
  --model-dir "${model_dir}" \
  --public-output "${public_output}" \
  --private-output "${private_output}" \
  --require-slurm
