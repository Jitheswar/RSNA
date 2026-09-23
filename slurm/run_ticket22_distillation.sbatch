#!/usr/bin/env bash
# Three-fold Ticket 22 soft-target student comparison, CPU only.
# This single job stays within the project six-hour, 64 GB CPU envelope.
#SBATCH --job-name=rsna-t22distill
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/ticket-22-distillation-%j.out
#SBATCH --error=artifacts/ticket-22-distillation-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
integration_manifest="${RSNA_TICKET22_INTEGRATION_MANIFEST:?set the private integration manifest}"
source_path="${RSNA_TICKET22_SOURCE:?set the approved local source path}"
fold_manifest="${RSNA_TICKET22_FOLD_MANIFEST:?set the frozen version-2 fold manifest}"
teacher_run_root="${RSNA_TICKET22_TEACHER_RUN_ROOT:?set the immutable Ticket 19 teacher run root}"
output_root="${RSNA_TICKET22_OUTPUT_ROOT:?set a unique output root outside the repository}"
source_fingerprint="${RSNA_TICKET22_SOURCE_FINGERPRINT:-}"
cache_root="${RSNA_TICKET22_CACHE_ROOT:-}"
code_revision="${RSNA_CODE_REVISION:?set the committed full Ticket 22 revision}"

project_dir="$(realpath -m "${project_dir}")"
output_root="$(realpath -m "${output_root}")"
teacher_run_root="$(realpath -m "${teacher_run_root}")"
if [[ "${output_root}" == "${project_dir}" || "${output_root}" == "${project_dir}"/* ]]; then
  echo "Ticket 22 output must stay outside the repository" >&2
  exit 1
fi
if [[ "${teacher_run_root}" == "${project_dir}" || "${teacher_run_root}" == "${project_dir}"/* ]]; then
  echo "Ticket 22 teacher artifacts must stay outside the repository" >&2
  exit 1
fi
if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "Ticket 22 requires cluster1-cpu" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" || ! -x /usr/bin/time ]]; then
  echo "Ticket 22 requires the isolated Python 3.11 environment and /usr/bin/time" >&2
  exit 1
fi
for input_path in "${integration_manifest}" "${fold_manifest}"; do
  if [[ ! -f "${input_path}" ]]; then
    echo "Ticket 22 input manifest does not exist: ${input_path}" >&2
    exit 1
  fi
done
if [[ ! -f "${source_path}" && ! -d "${source_path}" ]]; then
  echo "Ticket 22 source does not exist" >&2
  exit 1
fi
for fold_id in 0 1 2; do
  if [[ ! -f "${teacher_run_root}/runs/fold-${fold_id}/run.json" ||
        ! -f "${teacher_run_root}/runs/fold-${fold_id}/configuration.json" ||
        ! -f "${teacher_run_root}/runs/fold-${fold_id}/checkpoints/final.pt" ]]; then
    echo "Ticket 22 requires complete Ticket 19 fold ${fold_id} teacher artifacts" >&2
    exit 1
  fi
done
if [[ -e "${output_root}" && -n "$(find "${output_root}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite existing Ticket 22 output: ${output_root}" >&2
  exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "Ticket 22 requires flock for its concurrency guard" >&2
  exit 1
fi
mkdir -p "${project_dir}/artifacts"
exec 9>"${project_dir}/artifacts/.ticket22-distillation-concurrency.lock"
flock -x 9
if ! active_jobs="$(squeue --array --noheader --user "${USER:-$(id -un)}" \
  --partition=cluster1-cpu --states=RUNNING,CONFIGURING,COMPLETING --format="%i")"; then
  echo "could not inspect active cluster1-cpu Slurm jobs for this user" >&2
  exit 1
fi
active_count="$(printf '%s\n' "${active_jobs}" | awk 'NF { count += 1 } END { print count + 0 }')"
if ! printf '%s\n' "${active_jobs}" | awk -v job="${SLURM_JOB_ID}" '$0 == job { found = 1 } END { exit !found }'; then
  active_count=$((active_count + 1))
fi
if (( active_count > 2 )); then
  echo "cluster1-cpu concurrency limit exceeded for this user: ${active_count} active jobs" >&2
  exit 1
fi
flock -u 9

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RSNA_CODE_REVISION="${code_revision}"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1

# Limit this Python process below the Slurm request. RLIMIT_AS caps every
# process mapping; 48 GiB leaves headroom for wrappers and the node OS.
address_space_limit_kib=$((48 * 1024 * 1024))
if ! ulimit -v "${address_space_limit_kib}"; then
  echo "could not apply the 48 GiB Ticket 22 process memory bound" >&2
  exit 1
fi

cli_args=(
  --integration-manifest "${integration_manifest}"
  --source "${source_path}"
  --fold-manifest "${fold_manifest}"
  --teacher-run-root "${teacher_run_root}"
  --output-root "${output_root}"
  --preflight-evidence "${output_root}/preflight.json"
  --compute-evidence "${output_root}/cli-cost.json"
  --code-revision "${code_revision}"
)
if [[ -n "${source_fingerprint}" ]]; then
  cli_args+=(--source-fingerprint "${source_fingerprint}")
fi
if [[ -n "${cache_root}" ]]; then
  cli_args+=(--cache-root "${cache_root}")
fi

/usr/bin/time \
  --format='{"schema_version":"rsna-knee-observed-compute-v1","elapsed_seconds":%e,"user_cpu_seconds":%U,"system_cpu_seconds":%S,"peak_rss_kb":%M}' \
  --output="${output_root}.observed-compute.json" \
  "${project_dir}/.venv/bin/python" -m rsna_knee.distillation_cli "${cli_args[@]}"
