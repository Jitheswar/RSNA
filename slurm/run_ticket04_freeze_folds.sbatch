#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket04
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/ticket-04-slurm-%j.out
#SBATCH --error=artifacts/ticket-04-slurm-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
inventory_path="${RSNA_TICKET04_INVENTORY:?set RSNA_TICKET04_INVENTORY to the private ticket-03 inventory.json}"
output_dir="${RSNA_TICKET04_OUTPUT:?set RSNA_TICKET04_OUTPUT to a private fold-output directory}"
cd "${project_dir}"
mkdir -p artifacts

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "ticket-04 requires the cluster1-cpu partition" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "missing ${project_dir}/.venv/bin/python; provision the pinned user-space environment first" >&2
  exit 1
fi
if [[ ! -f "${inventory_path}" ]]; then
  echo "private ticket-03 inventory does not exist: ${inventory_path}" >&2
  exit 1
fi

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
freeze_args=(
  --inventory "${inventory_path}"
  --output-dir "${output_dir}"
  --seed-name "${RSNA_TICKET04_SEED_NAME:-ticket-04-fold-freeze-v1}"
  --seed "${RSNA_TICKET04_SEED:-20260911}"
  --repeat-name "${RSNA_TICKET04_REPEAT_NAME:-repeat-1}"
  --require-slurm
)
if [[ -n "${RSNA_TICKET04_REPEAT_JUSTIFICATION:-}" ]]; then
  freeze_args+=(--repeat-justification "${RSNA_TICKET04_REPEAT_JUSTIFICATION}")
fi
if [[ -n "${RSNA_TICKET04_GROUPING_EVIDENCE:-}" ]]; then
  freeze_args+=(--grouping-evidence "${RSNA_TICKET04_GROUPING_EVIDENCE}")
fi

exec "${project_dir}/.venv/bin/python" -m rsna_knee.folds_cli "${freeze_args[@]}"
