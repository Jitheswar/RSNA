#!/usr/bin/env bash
# Verify one frozen fold manifest against the V14 gate.  Read-only with respect
# to the manifest; writes one immutable identifier-free evidence file.
#SBATCH --job-name=rsna-ticket04-v14
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/ticket-04-v14-%j.out
#SBATCH --error=artifacts/ticket-04-v14-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
manifest_path="${RSNA_TICKET04_V14_MANIFEST:?set RSNA_TICKET04_V14_MANIFEST to the private fold manifest}"
inventory_path="${RSNA_TICKET04_INVENTORY:?set RSNA_TICKET04_INVENTORY to the private ticket-03 inventory.json}"
grouping_path="${RSNA_TICKET04_GROUPING_EVIDENCE:?set RSNA_TICKET04_GROUPING_EVIDENCE to the V02 grouping audit}"
output_path="${RSNA_TICKET04_V14_OUTPUT:?set RSNA_TICKET04_V14_OUTPUT to a new evidence path}"
cd "${project_dir}"
mkdir -p artifacts

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "V14 verification requires the cluster1-cpu partition" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "missing ${project_dir}/.venv/bin/python; provision the pinned user-space environment first" >&2
  exit 1
fi
for required in "${manifest_path}" "${inventory_path}" "${grouping_path}"; do
  if [[ ! -f "${required}" ]]; then
    echo "V14 input does not exist: ${required}" >&2
    exit 1
  fi
done

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
verify_args=(
  --manifest "${manifest_path}"
  --inventory "${inventory_path}"
  --grouping-evidence "${grouping_path}"
  --output "${output_path}"
  --require-slurm
)
if [[ -n "${RSNA_TICKET04_V14_PUBLIC_EVIDENCE:-}" ]]; then
  verify_args+=(--public-evidence "${RSNA_TICKET04_V14_PUBLIC_EVIDENCE}")
fi
if [[ -n "${RSNA_TICKET04_SUPERSEDES_MANIFEST:-}" ]]; then
  verify_args+=(--superseded-manifest "${RSNA_TICKET04_SUPERSEDES_MANIFEST}")
fi
if [[ -n "${RSNA_TICKET04_EXPECTED_GOLD_COUNT:-}" ]]; then
  verify_args+=(--expected-gold-count "${RSNA_TICKET04_EXPECTED_GOLD_COUNT}")
fi

exec "${project_dir}/.venv/bin/python" -m rsna_knee.fold_verification_cli "${verify_args[@]}"
