#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket29-real
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/private/ticket-29-real-diagnostic/slurm-%j.out
#SBATCH --error=artifacts/private/ticket-29-real-diagnostic/slurm-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
canonical_repo="/home/ch.jitheswar/RSNA"
python="${canonical_repo}/.venv/bin/python"
source_archive="${canonical_repo}/rsna-knee-abnormality-detection.zip"
integration_manifest="${canonical_repo}/artifacts/private/ticket-05-real-training-input/private/integration-manifest.json"
fold_manifest="${canonical_repo}/artifacts/private/ticket-04-real-v2/private/fold-manifest.json"
selection_report="${canonical_repo}/artifacts/private/ticket-03-inventory-run/private/selection.json"
private_root="${project_dir}/artifacts/private/ticket-29-real-diagnostic"
plan="${private_root}/diagnostic-plan.json"
run_directory="${private_root}/run-${SLURM_JOB_ID:?Slurm job ID is required}"
public_evidence="${project_dir}/docs/evidence/ticket-29-image-diagnostic.md"

if [[ ! -x "${python}" ]]; then
  echo "missing pinned project Python 3.11 environment" >&2
  exit 1
fi
if [[ ! -f "${source_archive}" || ! -f "${integration_manifest}" || ! -f "${fold_manifest}" || ! -f "${selection_report}" || ! -f "${plan}" ]]; then
  echo "one or more frozen private source inputs are unavailable" >&2
  exit 1
fi
if [[ -e "${run_directory}" && -n "$(find "${run_directory}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite an existing Ticket 29 run directory" >&2
  exit 1
fi

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${python}" -m rsna_knee.image_diagnostic run \
  --plan "${plan}" \
  --integration-manifest "${integration_manifest}" \
  --fold-manifest "${fold_manifest}" \
  --selection-report "${selection_report}" \
  --source-archive "${source_archive}" \
  --run-directory "${run_directory}" \
  --project-root "${canonical_repo}" \
  --project-root "${project_dir}" \
  --public-evidence "${public_evidence}" \
  --submit-epoch "${RSNA_TICKET29_SUBMIT_EPOCH:?submit time must be exported by the wrapper}"
