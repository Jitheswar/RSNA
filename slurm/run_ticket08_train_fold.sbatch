#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket08
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/ticket-08-slurm-%j.out
#SBATCH --error=artifacts/ticket-08-slurm-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
integration_manifest="${RSNA_TICKET08_INTEGRATION_MANIFEST:?set RSNA_TICKET08_INTEGRATION_MANIFEST to the private ticket-05 integration manifest}"
source_path="${RSNA_TICKET08_SOURCE:?set RSNA_TICKET08_SOURCE to the private DICOM archive or extracted root}"
fold_manifest="${RSNA_TICKET08_FOLD_MANIFEST:?set RSNA_TICKET08_FOLD_MANIFEST to the private ticket-04 fold manifest}"
fold_id="${RSNA_TICKET08_FOLD:?set RSNA_TICKET08_FOLD to one of 0, 1, or 2}"
output_dir="${RSNA_TICKET08_OUTPUT:?set RSNA_TICKET08_OUTPUT to a private ticket-08 experiment directory}"
source_version="${RSNA_TICKET08_SOURCE_VERSION:?set RSNA_TICKET08_SOURCE_VERSION to the admitted source version}"
source_fingerprint="${RSNA_TICKET08_SOURCE_FINGERPRINT:-}"
cache_root="${RSNA_TICKET08_CACHE_ROOT:-}"
preflight_path="${RSNA_TICKET08_PREFLIGHT_EVIDENCE:-${output_dir}.preflight.json}"
evidence_path="${RSNA_TICKET08_EVIDENCE:-${output_dir}.public.json}"

cd "${project_dir}"
mkdir -p artifacts

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "ticket-08 requires the cluster1-cpu partition" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "missing ${project_dir}/.venv/bin/python; provision the pinned user-space environment first" >&2
  exit 1
fi
if [[ ! -f "${integration_manifest}" || ! -f "${fold_manifest}" ]]; then
  echo "ticket-08 requires the private integration and fold manifests" >&2
  exit 1
fi
if [[ ! -f "${source_path}" && ! -d "${source_path}" ]]; then
  echo "real-DICOM source does not exist: ${source_path}" >&2
  exit 1
fi
if [[ ! "${fold_id}" =~ ^[012]$ ]]; then
  echo "ticket-08 fold must be 0, 1, or 2: ${fold_id}" >&2
  exit 1
fi

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
training_args=(
  --integration-manifest "${integration_manifest}"
  --source "${source_path}"
  --fold-manifest "${fold_manifest}"
  --fold "${fold_id}"
  --output "${output_dir}"
  --source-version "${source_version}"
  --preflight-evidence "${preflight_path}"
  --evidence "${evidence_path}"
  --require-slurm
)
if [[ -n "${source_fingerprint}" ]]; then
  training_args+=(--source-fingerprint "${source_fingerprint}")
fi
if [[ -n "${cache_root}" ]]; then
  training_args+=(--cache-root "${cache_root}")
fi

exec "${project_dir}/.venv/bin/python" -m rsna_knee.training_cli "${training_args[@]}"
