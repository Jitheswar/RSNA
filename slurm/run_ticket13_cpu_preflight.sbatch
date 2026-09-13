#!/usr/bin/env bash
# Establish the passed cluster1-cpu environment evidence required before any
# ticket-13 GPU request is prepared or submitted.
#SBATCH --job-name=rsna-ticket13-preflight
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/ticket-13-preflight-%j.out
#SBATCH --error=artifacts/ticket-13-preflight-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
output_path="${RSNA_TICKET13_PREFLIGHT_OUTPUT:?set RSNA_TICKET13_PREFLIGHT_OUTPUT to a persistent preflight JSON}"

cd "${project_dir}"
mkdir -p artifacts

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "ticket-13 CPU preflight requires the cluster1-cpu partition" >&2
  exit 1
fi
if [[ "${SLURM_JOB_ID:-}" == "" || "${SLURM_JOB_NUM_NODES:-}" != "1" || "${SLURM_NTASKS:-}" != "1" || "${SLURM_CPUS_PER_TASK:-}" != "2" ]]; then
  echo "ticket-13 CPU preflight allocation does not match one node, one task and two CPUs" >&2
  exit 1
fi
case "${SLURM_MEM_PER_NODE:-}" in
  8G|8192) ;;
  *) echo "ticket-13 CPU preflight requires exactly 8 GB per node" >&2; exit 1 ;;
esac
timelimit="${SLURM_TIMELIMIT:-}"
if [[ -z "${timelimit}" ]] && command -v scontrol >/dev/null 2>&1; then
  if job_info="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null)"; then
    for field in ${job_info}; do
      if [[ "${field}" == TimeLimit=* ]]; then
        timelimit="${field#TimeLimit=}"
        break
      fi
    done
  fi
fi
timelimit_minutes=-1
if [[ "${timelimit}" =~ ^([0-9]+)-([0-9]{2}):([0-9]{2}):([0-9]{2})$ ]]; then
  timelimit_minutes=$((10#${BASH_REMATCH[1]} * 1440 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))
  if (( 10#${BASH_REMATCH[4]} != 0 )); then
    timelimit_minutes=-1
  fi
elif [[ "${timelimit}" =~ ^([0-9]+):([0-9]{2})(:([0-9]{2}))?$ ]]; then
  timelimit_minutes=$((10#${BASH_REMATCH[1]} * 60 + 10#${BASH_REMATCH[2]}))
  if [[ -n "${BASH_REMATCH[4]:-}" ]] && (( 10#${BASH_REMATCH[4]} != 0 )); then
    timelimit_minutes=-1
  fi
elif [[ "${timelimit}" =~ ^[0-9]+$ ]]; then
  timelimit_minutes=$((10#${timelimit}))
fi
if (( timelimit_minutes != 30 )); then
  echo "ticket-13 CPU preflight requires exactly 30 minutes (got ${timelimit:-unset})" >&2
  exit 1
fi
export SLURM_TIMELIMIT="${timelimit}"
case "${SLURM_JOB_EXCLUSIVE:-0}" in
  1|true|TRUE|yes|YES) echo "ticket-13 CPU preflight must not request an exclusive allocation" >&2; exit 1 ;;
esac
if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "missing ${project_dir}/.venv/bin/python; provision the pinned user-space environment first" >&2
  exit 1
fi

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${project_dir}/.venv/bin/python" -m rsna_knee.gpu_cli preflight --output "${output_path}"
