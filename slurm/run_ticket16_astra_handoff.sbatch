#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket16-handoff
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/ticket-16-handoff-%j.out
#SBATCH --error=artifacts/ticket-16-handoff-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
fold_manifest="${RSNA_TICKET16_FOLD_MANIFEST:?set RSNA_TICKET16_FOLD_MANIFEST to the private ticket-04 manifest}"
reports="${RSNA_TICKET16_REPORTS:?set RSNA_TICKET16_REPORTS to the private report corpus JSON}"
approval="${RSNA_TICKET16_ASTRA_APPROVAL:?set RSNA_TICKET16_ASTRA_APPROVAL to the Astra identity approval JSON}"
prompt="${RSNA_TICKET16_ASTRA_PROMPT:?set RSNA_TICKET16_ASTRA_PROMPT to the owner-approved prompt JSON}"
public_plan="${RSNA_TICKET16_ASTRA_PUBLIC_PLAN:?set RSNA_TICKET16_ASTRA_PUBLIC_PLAN to a public plan path}"
private_dir="${RSNA_TICKET16_ASTRA_PRIVATE_DIR:?set RSNA_TICKET16_ASTRA_PRIVATE_DIR to a private handoff directory}"

cd "${project_dir}"
if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "Ticket 16 handoff preparation requires cluster1-cpu" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "missing ${project_dir}/.venv/bin/python" >&2
  exit 1
fi

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
# This job only writes an immutable private handoff plan. It never switches a
# model, calls Astra, sends a report to a provider, or validates a response.
exec "${project_dir}/.venv/bin/python" -m rsna_knee.astra_handoff_cli prepare \
  --fold-manifest "${fold_manifest}" \
  --reports "${reports}" \
  --approval "${approval}" \
  --prompt "${prompt}" \
  --public-plan "${public_plan}" \
  --private-dir "${private_dir}"
