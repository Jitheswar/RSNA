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
# shellcheck source=slurm/ticket13_gres.sh
source "${project_dir}/slurm/ticket13_gres.sh"

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

job_details=""
unset SLURM_JOB_GRES_DETAIL SLURM_NODE_GRES SLURM_GRES_CONFIG_ENTRY \
  SLURM_GRES_CONFIG_SCOPE SLURM_GRES_DEVICE_FILE SLURM_CONSTRAIN_DEVICES SLURM_GRES_TYPES \
  SLURM_SELECT_TYPE SLURM_SELECT_TYPE_PARAMETERS SLURM_GRES_SHARING_MODE
if command -v scontrol >/dev/null 2>&1; then
  # The detailed job record is the authoritative source for the concrete GRES
  # index.  OverSubscribe remains a separate node/CPU-sharing fact.
  job_details="$(scontrol show job -d "${job_id}" 2>/dev/null || true)"
  scheduler_gres=""
  for field in ${job_details}; do
    case "${field}" in
      JOB_GRES=*) export SLURM_JOB_GRES="${field#JOB_GRES=}"; scheduler_gres="${field#JOB_GRES=}" ;;
      TresPerNode=*)
        export SLURM_TRES_PER_NODE="${field#TresPerNode=}"
        if [[ -z "${scheduler_gres}" ]]; then
          scheduler_gres="${field#TresPerNode=}"
        fi
        ;;
      GRES=*) export SLURM_JOB_GRES_DETAIL="${field#GRES=}" ;;
      NodeList=*) export SLURM_JOB_NODELIST="${field#NodeList=}" ;;
      Shared=*)
        if [[ -z "${SLURM_JOB_SHARED:-${SLURM_SHARED:-}}" ]]; then
          export SLURM_JOB_SHARED="${field#Shared=}"
        fi
        ;;
      OverSubscribe=*)
        if [[ -z "${SLURM_JOB_SHARED:-${SLURM_SHARED:-}}" ]]; then
          # Slurm 20.11 on this cluster exposes the sharing signal as
          # OverSubscribe=... rather than Shared=....
          export SLURM_JOB_SHARED="${field#OverSubscribe=}"
        fi
        ;;
    esac
  done
  if [[ -z "${SLURM_JOB_GRES:-}" && -n "${scheduler_gres}" ]]; then
    export SLURM_JOB_GRES="${scheduler_gres}"
  fi
  if [[ -z "${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-}}" && "${scheduler_gres}" =~ ^gpu(:[A-Za-z0-9_.-]+)?:([0-9]+)$ ]]; then
    export SLURM_GPUS_ON_NODE="${BASH_REMATCH[2]}"
  fi
fi
if [[ -z "${SLURM_JOB_SHARED:-${SLURM_SHARED:-}}" ]]; then
  echo "Slurm sharing behavior could not be verified; stop before private-data loading" >&2
  exit 1
fi

if [[ -z "${SLURM_TIMELIMIT:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  if job_info="$(scontrol show job "${job_id}" 2>/dev/null)"; then
    for field in ${job_info}; do
      if [[ "${field}" == TimeLimit=* ]]; then
        export SLURM_TIMELIMIT="${field#TimeLimit=}"
        break
      fi
    done
  fi
fi
if [[ -z "${SLURM_TIMELIMIT:-}" ]]; then
  echo "Slurm time limit could not be verified; stop before private-data loading" >&2
  exit 1
fi

if command -v scontrol >/dev/null 2>&1; then
  node_name="${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-}}"
  node_details=""
  if [[ -n "${node_name}" ]]; then
    node_details="$(scontrol show node -o "${node_name}" 2>/dev/null || true)"
    if [[ "${node_details}" =~ Gres=([^[:space:]]+) ]]; then
      export SLURM_NODE_GRES="${BASH_REMATCH[1]}"
    fi
  fi

  cluster_config="$(scontrol show config 2>/dev/null || true)"
  if [[ "${cluster_config}" =~ GresTypes[[:space:]]*=[[:space:]]*([^[:space:]]+) ]]; then
    export SLURM_GRES_TYPES="${BASH_REMATCH[1]}"
  fi
  if [[ "${cluster_config}" =~ ConstrainDevices[[:space:]]*=[[:space:]]*([^[:space:]]+) ]]; then
    export SLURM_CONSTRAIN_DEVICES="${BASH_REMATCH[1]}"
  fi
  if [[ "${cluster_config}" =~ SelectType[[:space:]]*=[[:space:]]*([^[:space:]]+) ]]; then
    export SLURM_SELECT_TYPE="${BASH_REMATCH[1]}"
  fi
  if [[ "${cluster_config}" =~ SelectTypeParameters[[:space:]]*=[[:space:]]*([^[:space:]]+) ]]; then
    export SLURM_SELECT_TYPE_PARAMETERS="${BASH_REMATCH[1]}"
  fi

  gres_config_path="/etc/slurm/gres.conf"
  if [[ -n "${node_name}" && -r "${gres_config_path}" ]]; then
    if ticket13_collect_gres_config "${node_name}" "${gres_config_path}"; then
      if [[ -n "${TICKET13_GRES_CONFIG_ENTRY}" ]]; then
        export SLURM_GRES_CONFIG_ENTRY="${TICKET13_GRES_CONFIG_ENTRY}"
        export SLURM_GRES_CONFIG_SCOPE="${TICKET13_GRES_CONFIG_SCOPE}"
        export SLURM_GRES_DEVICE_FILE="${TICKET13_GRES_DEVICE_FILE}"
      fi
    fi
  fi

  mode_text="${SLURM_GRES_TYPES:-} ${SLURM_GRES_CONFIG_ENTRY:-}"
  mode_text="${mode_text,,}"
  if [[ -z "${SLURM_GRES_TYPES:-}" || -z "${SLURM_GRES_CONFIG_ENTRY:-}" || -z "${SLURM_CONSTRAIN_DEVICES:-}" ]]; then
    export SLURM_GRES_SHARING_MODE="unknown"
  elif [[ "${mode_text}" == *mps* || "${mode_text}" == *mig* || "${mode_text}" == *shard* || "${mode_text}" == *fraction* || "${mode_text}" == *slice* ]]; then
    export SLURM_GRES_SHARING_MODE="fractional_or_shared"
  else
    export SLURM_GRES_SHARING_MODE="none"
  fi
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
