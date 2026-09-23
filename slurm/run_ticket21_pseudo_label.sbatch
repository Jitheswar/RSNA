#!/usr/bin/env bash
# One fold-0 baseline teacher and additive pseudo-label student, CPU only.
# The script permits at most two concurrent Ticket 21 jobs.
#SBATCH --job-name=rsna-t21pseudo
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/ticket-21-pseudo-%j.out
#SBATCH --error=artifacts/ticket-21-pseudo-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
integration_manifest="${RSNA_TICKET21_INTEGRATION_MANIFEST:?set the private integration manifest}"
source_path="${RSNA_TICKET21_SOURCE:?set the approved local source path}"
fold_manifest="${RSNA_TICKET21_FOLD_MANIFEST:?set the frozen version-2 fold manifest}"
output_root="${RSNA_TICKET21_OUTPUT_ROOT:?set a unique output root outside the repository}"
code_revision="${RSNA_CODE_REVISION:?set the committed full Git revision}"
source_fingerprint="${RSNA_TICKET21_SOURCE_FINGERPRINT:-}"
cache_root="${RSNA_TICKET21_CACHE_ROOT:-}"

output_root="$(realpath -m "${output_root}")"
project_dir="$(realpath -m "${project_dir}")"
if [[ "${output_root}" == "${project_dir}" || "${output_root}" == "${project_dir}"/* ]]; then
  echo "Ticket 21 output must be outside the repository" >&2
  exit 1
fi
for input_path in "${integration_manifest}" "${fold_manifest}"; do
  if [[ ! -f "${input_path}" ]]; then
    echo "Ticket 21 input manifest does not exist: ${input_path}" >&2
    exit 1
  fi
done
if [[ ! -f "${source_path}" && ! -d "${source_path}" ]]; then
  echo "Ticket 21 source does not exist" >&2
  exit 1
fi
if [[ -e "${output_root}" && -n "$(find "${output_root}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite existing Ticket 21 output: ${output_root}" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" || ! -x /usr/bin/time ]]; then
  echo "Ticket 21 requires the isolated Python 3.11 environment and /usr/bin/time" >&2
  exit 1
fi

if ! command -v flock >/dev/null 2>&1; then
  echo "Ticket 21 requires flock for its concurrency guard" >&2
  exit 1
fi
exec 9>"${output_root}.concurrency.lock"
flock -x 9
if ! active_jobs="$(squeue --array --noheader --user "${USER:-$(id -un)}" --name "rsna-t21pseudo" \
  --states=RUNNING,CONFIGURING,COMPLETING --format="%A")"; then
  echo "could not inspect active Ticket 21 Slurm jobs" >&2
  exit 1
fi
active_count="$(printf '%s\n' "${active_jobs}" | awk 'NF { count += 1 } END { print count + 0 }')"
if (( active_count > 2 )); then
  echo "Ticket 21 concurrency limit exceeded: ${active_count} active jobs" >&2
  exit 1
fi
flock -u 9

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export RSNA_CODE_REVISION="${code_revision}"

cli_args=(
  --integration-manifest "${integration_manifest}"
  --source "${source_path}"
  --fold-manifest "${fold_manifest}"
  --fold 0
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
  "${project_dir}/.venv/bin/python" -m rsna_knee.pseudo_label_cli "${cli_args[@]}"
