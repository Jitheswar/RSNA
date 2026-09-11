#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket07
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/ticket-07-slurm-%j.out
#SBATCH --error=artifacts/ticket-07-slurm-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
manifest_path="${RSNA_TICKET07_MANIFEST:?set RSNA_TICKET07_MANIFEST to the private ticket-05 integration manifest}"
source_path="${RSNA_TICKET07_SOURCE:?set RSNA_TICKET07_SOURCE to the private DICOM archive or extracted root}"
output_dir="${RSNA_TICKET07_OUTPUT:?set RSNA_TICKET07_OUTPUT to a private ticket-07 output directory}"
fallback_labels_path="${RSNA_TICKET07_FALLBACK_REFERENCE_LABELS:-}"
eligible_uids_path="${RSNA_TICKET07_FALLBACK_ELIGIBLE_STUDIES:-}"
held_out_uids_path="${RSNA_TICKET07_FALLBACK_HELD_OUT_STUDIES:-}"
held_out_fold="${RSNA_TICKET07_FALLBACK_HELD_OUT_FOLD:-}"
cache_root="${RSNA_TICKET07_CACHE_ROOT:-}"

cd "${project_dir}"
mkdir -p artifacts "${output_dir}/private" "${output_dir}/public"

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "ticket-07 requires the cluster1-cpu partition" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "missing ${project_dir}/.venv/bin/python; provision the pinned user-space environment first" >&2
  exit 1
fi
if [[ ! -f "${manifest_path}" ]]; then
  echo "ticket-05 private integration manifest does not exist: ${manifest_path}" >&2
  exit 1
fi
if [[ ! -f "${source_path}" && ! -d "${source_path}" ]]; then
  echo "real-DICOM source does not exist: ${source_path}" >&2
  exit 1
fi
if [[ -n "${fallback_labels_path}" && ! -f "${fallback_labels_path}" ]]; then
  echo "fallback reference-label file does not exist: ${fallback_labels_path}" >&2
  exit 1
fi
if [[ -n "${eligible_uids_path}" && ! -f "${eligible_uids_path}" ]]; then
  echo "fallback eligible-study file does not exist: ${eligible_uids_path}" >&2
  exit 1
fi
if [[ -n "${held_out_uids_path}" && ! -f "${held_out_uids_path}" ]]; then
  echo "fallback held-out-study file does not exist: ${held_out_uids_path}" >&2
  exit 1
fi
if [[ -n "${held_out_fold}" && ! "${held_out_fold}" =~ ^[0-9]+$ ]]; then
  echo "fallback held-out fold must be a non-negative integer: ${held_out_fold}" >&2
  exit 1
fi

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
recovery_args=(
  --manifest "${manifest_path}"
  --source "${source_path}"
  --output "${output_dir}/private/submission.csv"
  --evidence "${output_dir}/public/evidence.json"
  --private-evidence "${output_dir}/private/evidence.json"
  --evidence-ticket 07
  --recover-damaged-inputs
  --require-slurm
)
if [[ -n "${fallback_labels_path}" ]]; then
  recovery_args+=(--fallback-reference-labels "${fallback_labels_path}")
fi
if [[ -n "${eligible_uids_path}" ]]; then
  recovery_args+=(--fallback-eligible-study-uids "${eligible_uids_path}")
fi
if [[ -n "${held_out_uids_path}" ]]; then
  recovery_args+=(--fallback-held-out-study-uids "${held_out_uids_path}")
fi
if [[ -n "${held_out_fold}" ]]; then
  recovery_args+=(--fallback-held-out-fold "${held_out_fold}")
fi
if [[ -n "${cache_root}" ]]; then
  recovery_args+=(--cache-root "${cache_root}")
fi

exec "${project_dir}/.venv/bin/python" -m rsna_knee.real_inference_cli "${recovery_args[@]}"
