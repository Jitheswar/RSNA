#!/usr/bin/env bash
# Execute exactly one owner-approved ticket-13 GPU request.  The submitter
# supplies the exact resource flags; the Python runner verifies them again
# inside the allocation before loading competition data or starting training.
# A40 model and available memory are measured at runtime; neither is assumed.
#SBATCH --job-name=rsna-ticket13-gpu
#SBATCH --partition=cluster1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/ticket-13-gpu-%j.out
#SBATCH --error=artifacts/ticket-13-gpu-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
job_id="${SLURM_JOB_ID:?sbatch must provide SLURM_JOB_ID}"
gpu_python="${RSNA_TICKET13_GPU_PYTHON:-${project_dir}/.venv-gpu/bin/python}"
request="${RSNA_TICKET13_REQUEST:?set RSNA_TICKET13_REQUEST to the immutable GPU request directory or request.json}"
approval="${RSNA_TICKET13_APPROVAL:?set RSNA_TICKET13_APPROVAL to the owner approval JSON}"
preflight="${RSNA_TICKET13_PREFLIGHT:?set RSNA_TICKET13_PREFLIGHT to the passed CPU preflight JSON}"
integration_manifest="${RSNA_TICKET13_INTEGRATION_MANIFEST:?set RSNA_TICKET13_INTEGRATION_MANIFEST}"
source_path="${RSNA_TICKET13_SOURCE:?set RSNA_TICKET13_SOURCE}"
fold_manifest="${RSNA_TICKET13_FOLD_MANIFEST:?set RSNA_TICKET13_FOLD_MANIFEST}"
fold_id="${RSNA_TICKET13_FOLD:?set RSNA_TICKET13_FOLD to 0, 1, or 2}"
output_dir="${RSNA_TICKET13_OUTPUT:?set RSNA_TICKET13_OUTPUT to a private execution directory}"
evidence_path="${RSNA_TICKET13_EVIDENCE:?set RSNA_TICKET13_EVIDENCE to a private evidence JSON}"
source_version="${RSNA_TICKET13_SOURCE_VERSION:?set RSNA_TICKET13_SOURCE_VERSION}"
source_fingerprint="${RSNA_TICKET13_SOURCE_FINGERPRINT:-}"
cache_root="${RSNA_TICKET13_CACHE_ROOT:-}"

cd "${project_dir}"
mkdir -p artifacts

if [[ ! -x "${gpu_python}" ]]; then
  echo "missing ${gpu_python}; provision the separately pinned GPU environment first" >&2
  exit 1
fi
if [[ "${gpu_python}" == "${project_dir}/.venv/bin/python" ]]; then
  echo "refusing to use the CPU-only project environment for GPU training" >&2
  exit 1
fi
if [[ ! -f "${request}" && ! -d "${request}" ]]; then
  echo "GPU request does not exist: ${request}" >&2
  exit 1
fi
if [[ ! -f "${approval}" || ! -f "${preflight}" ]]; then
  echo "GPU execution requires the owner approval and passed CPU preflight files" >&2
  exit 1
fi
if [[ ! -f "${integration_manifest}" || ! -f "${fold_manifest}" ]]; then
  echo "ticket-13 GPU execution requires the private integration and fold manifests" >&2
  exit 1
fi
if [[ ! -f "${source_path}" && ! -d "${source_path}" ]]; then
  echo "real-DICOM source does not exist: ${source_path}" >&2
  exit 1
fi
if [[ ! "${fold_id}" =~ ^[012]$ ]]; then
  echo "ticket-13 fold must be 0, 1, or 2: ${fold_id}" >&2
  exit 1
fi

if [[ -z "${SLURM_JOB_SHARED:-${SLURM_SHARED:-}}" ]] && command -v scontrol >/dev/null 2>&1; then
  job_details="$(scontrol show job -o "${job_id}" 2>/dev/null || true)"
  if [[ "${job_details}" =~ Shared=([^[:space:]]+) ]]; then
    export SLURM_JOB_SHARED="${BASH_REMATCH[1]}"
  fi
fi
if [[ -z "${SLURM_JOB_SHARED:-${SLURM_SHARED:-}}" ]]; then
  echo "Slurm sharing behavior could not be verified; stop before private-data loading" >&2
  exit 1
fi

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
if ! "${gpu_python}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)'; then
  echo "the selected GPU environment must use Python 3.11; stop before loading private data" >&2
  exit 1
fi
if ! "${gpu_python}" -c 'import numpy, pydicom, torch; raise SystemExit(0 if (numpy.__version__ == "2.2.6" and pydicom.__version__ == "3.0.1" and torch.__version__.split("+", 1)[0] == "2.7.1" and torch.version.cuda == "12.6" and torch.cuda.is_available()) else 1)'; then
  echo "the selected GPU environment does not match the pinned CUDA runtime contract; stop before loading private data" >&2
  exit 1
fi
run_args=(
  run
  --request "${request}"
  --approval "${approval}"
  --preflight "${preflight}"
  --integration-manifest "${integration_manifest}"
  --source "${source_path}"
  --fold-manifest "${fold_manifest}"
  --fold "${fold_id}"
  --output "${output_dir}"
  --evidence "${evidence_path}"
  --source-version "${source_version}"
)
if [[ -n "${source_fingerprint}" ]]; then
  run_args+=(--source-fingerprint "${source_fingerprint}")
fi
if [[ -n "${cache_root}" ]]; then
  run_args+=(--cache-root "${cache_root}")
fi

exec "${gpu_python}" -m rsna_knee.gpu_cli "${run_args[@]}"
