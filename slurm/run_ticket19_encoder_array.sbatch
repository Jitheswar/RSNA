#!/usr/bin/env bash
# Gold-only Ticket 19 wider-encoder runs on the frozen Ticket 18 folds.
# The array cap keeps concurrent autonomous CPU jobs at two.
#SBATCH --job-name=rsna-t19wide
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%2
#SBATCH --output=artifacts/ticket-19-wide-%A_%a.out
#SBATCH --error=artifacts/ticket-19-wide-%A_%a.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
integration_manifest="${RSNA_TICKET19_INTEGRATION_MANIFEST:?set the private integration manifest}"
source_path="${RSNA_TICKET19_SOURCE:?set the approved local source path}"
fold_manifest="${RSNA_TICKET19_FOLD_MANIFEST:?set the frozen version-2 fold manifest}"
output_root="${RSNA_TICKET19_OUTPUT_ROOT:?set an immutable output root outside the repository}"
source_version="${RSNA_TICKET19_SOURCE_VERSION:?set the admitted source version}"
source_fingerprint="${RSNA_TICKET19_SOURCE_FINGERPRINT:?set the source manifest fingerprint}"
code_revision="${RSNA_CODE_REVISION:?set the committed Ticket 19 code revision}"

case "${SLURM_ARRAY_TASK_ID:?expected array task id}" in
  0|1|2) fold_id="${SLURM_ARRAY_TASK_ID}" ;;
  *) echo "unsupported Ticket 19 fold ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

run_dir="${output_root}/runs/fold-${fold_id}"
preflight_path="${output_root}/runs/fold-${fold_id}.preflight.json"
public_evidence="${output_root}/runs/fold-${fold_id}.public.json"
compute_evidence="${output_root}/runs/fold-${fold_id}.compute.json"

cd "${project_dir}"
mkdir -p artifacts "${output_root}/runs"
if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "Ticket 19 training requires cluster1-cpu" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" || ! -x /usr/bin/time ]]; then
  echo "Ticket 19 requires the isolated Python 3.11 environment and /usr/bin/time" >&2
  exit 1
fi
for input_path in "${integration_manifest}" "${fold_manifest}"; do
  if [[ ! -f "${input_path}" ]]; then
    echo "Ticket 19 input file does not exist: ${input_path}" >&2
    exit 1
  fi
done
if [[ ! -f "${source_path}" && ! -d "${source_path}" ]]; then
  echo "Ticket 19 source does not exist" >&2
  exit 1
fi
if [[ -e "${run_dir}" && -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite existing Ticket 19 run: ${run_dir}" >&2
  exit 1
fi

# Keep manual resubmissions under the same two-job limit as the array.
if ! command -v flock >/dev/null 2>&1; then
  echo "Ticket 19 requires flock for its concurrency guard" >&2
  exit 1
fi
exec 9>"${output_root}/fold-concurrency.lock"
flock -x 9
user_name="${USER:-$(id -un)}"
if ! active_jobs="$(squeue --array --noheader --user "${user_name}" --name "rsna-t19wide" \
  --states=RUNNING,CONFIGURING,COMPLETING --format="%A_%a")"; then
  echo "could not inspect active Ticket 19 Slurm tasks" >&2
  exit 1
fi
active_count="$(printf '%s\n' "${active_jobs}" | awk 'NF { count += 1 } END { print count + 0 }')"
if (( active_count > 2 )); then
  echo "Ticket 19 concurrency limit exceeded: ${active_count} active array tasks" >&2
  exit 1
fi
flock -u 9

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RSNA_CODE_REVISION="${code_revision}"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1

/usr/bin/time \
  --format='{"schema_version":"rsna-knee-observed-compute-v1","elapsed_seconds":%e,"user_cpu_seconds":%U,"system_cpu_seconds":%S,"peak_rss_kb":%M}' \
  --output="${compute_evidence}" \
  "${project_dir}/.venv/bin/python" -m rsna_knee.training_cli \
  --integration-manifest "${integration_manifest}" \
  --source "${source_path}" \
  --fold-manifest "${fold_manifest}" \
  --fold "${fold_id}" \
  --output "${run_dir}" \
  --source-version "${source_version}" \
  --source-fingerprint "${source_fingerprint}" \
  --preflight-evidence "${preflight_path}" \
  --evidence "${public_evidence}" \
  --image-variant ticket19-wider-slice-encoder-v1 \
  --require-slurm
