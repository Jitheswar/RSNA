#!/usr/bin/env bash
# One fold of the controlled Ticket 18 supervision comparison.
# Submit at most two fold jobs at a time; the start guard rejects overflow.
#SBATCH --job-name=rsna-ticket18
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/ticket-18-fold-%j.out
#SBATCH --error=artifacts/ticket-18-fold-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
arm="${RSNA_TICKET18_ARM:?set RSNA_TICKET18_ARM to gold or report}"
integration_manifest="${RSNA_TICKET18_INTEGRATION_MANIFEST:?set RSNA_TICKET18_INTEGRATION_MANIFEST}"
source_path="${RSNA_TICKET18_SOURCE:?set RSNA_TICKET18_SOURCE}"
fold_manifest="${RSNA_TICKET18_FOLD_MANIFEST:?set RSNA_TICKET18_FOLD_MANIFEST}"
fold_id="${RSNA_TICKET18_FOLD:?set RSNA_TICKET18_FOLD to 0, 1 or 2}"
output_dir="${RSNA_TICKET18_OUTPUT:?set RSNA_TICKET18_OUTPUT}"
source_version="${RSNA_TICKET18_SOURCE_VERSION:?set RSNA_TICKET18_SOURCE_VERSION}"
source_fingerprint="${RSNA_TICKET18_SOURCE_FINGERPRINT:?set RSNA_TICKET18_SOURCE_FINGERPRINT}"
reports="${RSNA_TICKET18_REPORTS:-}"
frozen_report_artifact="${RSNA_TICKET18_FROZEN_REPORT_ARTIFACT:-}"
preflight_path="${RSNA_TICKET18_PREFLIGHT_EVIDENCE:-${output_dir}.preflight.json}"
evidence_path="${RSNA_TICKET18_EVIDENCE:-${output_dir}.public.json}"
compute_evidence_path="${RSNA_TICKET18_COMPUTE_EVIDENCE:-${output_dir}.compute.json}"

cd "${project_dir}"
mkdir -p artifacts

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "ticket-18 training requires the cluster1-cpu partition" >&2
  exit 1
fi
if [[ "${arm}" != "gold" && "${arm}" != "report" ]]; then
  echo "ticket-18 arm must be gold or report: ${arm}" >&2
  exit 1
fi
if [[ ! "${fold_id}" =~ ^[012]$ ]]; then
  echo "ticket-18 fold must be 0, 1, or 2: ${fold_id}" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "missing ${project_dir}/.venv/bin/python" >&2
  exit 1
fi
if [[ ! -x /usr/bin/time ]]; then
  echo "ticket-18 requires /usr/bin/time to record observed process compute" >&2
  exit 1
fi
if [[ ! -f "${integration_manifest}" || ! -f "${fold_manifest}" ]]; then
  echo "ticket-18 requires the private integration and fold manifests" >&2
  exit 1
fi
if [[ ! -f "${source_path}" && ! -d "${source_path}" ]]; then
  echo "ticket-18 source does not exist" >&2
  exit 1
fi
if [[ "${arm}" == "report" && ( ! -f "${reports}" || ! -f "${frozen_report_artifact}" ) ]]; then
  echo "report supervision requires the frozen Ticket 15 corpus and labels" >&2
  exit 1
fi

# Keep this comparison within the owner's two concurrent CPU-job limit.
if ! command -v flock >/dev/null 2>&1; then
  echo "ticket-18 requires flock for its concurrency guard" >&2
  exit 1
fi
concurrency_lock="${RSNA_TICKET18_CONCURRENCY_LOCK:-${project_dir}/artifacts/ticket18-fold-concurrency.lock}"
exec 9>"${concurrency_lock}"
flock -x 9
user_name="${USER:-$(id -un)}"
if ! active_ticket18_jobs="$(
  squeue --noheader --user "${user_name}" --name "rsna-ticket18" \
    --states=RUNNING,CONFIGURING,COMPLETING --format="%A"
)"; then
  echo "could not inspect active Ticket 18 fold jobs" >&2
  exit 1
fi
active_ticket18_count="$(printf '%s\n' "${active_ticket18_jobs}" | awk 'NF { count += 1 } END { print count + 0 }')"
if (( active_ticket18_count > 2 )); then
  echo "ticket-18 concurrency limit exceeded: ${active_ticket18_count} active jobs" >&2
  exit 1
fi
flock -u 9

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RSNA_CODE_REVISION="${RSNA_CODE_REVISION:-$(git rev-parse HEAD)}"
mkdir -p "$(dirname "${compute_evidence_path}")"

run_training_with_compute_record() {
  /usr/bin/time \
    --format='{"schema_version":"rsna-knee-observed-compute-v1","elapsed_seconds":%e,"user_cpu_seconds":%U,"system_cpu_seconds":%S,"peak_rss_kb":%M}' \
    --output="${compute_evidence_path}" \
    "$@"
}

training_args=(
  --integration-manifest "${integration_manifest}"
  --source "${source_path}"
  --fold-manifest "${fold_manifest}"
  --fold "${fold_id}"
  --output "${output_dir}"
  --source-version "${source_version}"
  --source-fingerprint "${source_fingerprint}"
  --preflight-evidence "${preflight_path}"
  --require-slurm
)

if [[ "${arm}" == "gold" ]]; then
  training_args+=(--evidence "${evidence_path}")
  run_training_with_compute_record "${project_dir}/.venv/bin/python" -m rsna_knee.training_cli "${training_args[@]}"
  exit 0
fi

training_args+=(
  --reports "${reports}"
  --frozen-report-artifact "${frozen_report_artifact}"
  --evidence "${evidence_path}"
)
run_training_with_compute_record "${project_dir}/.venv/bin/python" -m rsna_knee.weak_supervision_cli "${training_args[@]}"
