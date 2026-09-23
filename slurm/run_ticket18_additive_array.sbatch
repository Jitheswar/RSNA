#!/usr/bin/env bash
# Frozen additive Ticket 18 v2 comparison: paired gold-only and gold-plus-weak folds.
# Submit as a six-task array with a maximum of two concurrent CPU jobs.
#SBATCH --job-name=rsna-t18add
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-5%2
#SBATCH --output=artifacts/ticket-18-additive-%A_%a.out
#SBATCH --error=artifacts/ticket-18-additive-%A_%a.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
integration_manifest="${RSNA_TICKET18_INTEGRATION_MANIFEST:?set RSNA_TICKET18_INTEGRATION_MANIFEST}"
source_path="${RSNA_TICKET18_SOURCE:?set RSNA_TICKET18_SOURCE}"
fold_manifest="${RSNA_TICKET18_FOLD_MANIFEST:?set RSNA_TICKET18_FOLD_MANIFEST}"
output_root="${RSNA_TICKET18_V2_OUTPUT_ROOT:?set RSNA_TICKET18_V2_OUTPUT_ROOT}"
source_version="${RSNA_TICKET18_SOURCE_VERSION:?set RSNA_TICKET18_SOURCE_VERSION}"
source_fingerprint="${RSNA_TICKET18_SOURCE_FINGERPRINT:?set RSNA_TICKET18_SOURCE_FINGERPRINT}"
reports="${RSNA_TICKET18_REPORTS:?set RSNA_TICKET18_REPORTS}"
frozen_report_artifact="${RSNA_TICKET18_FROZEN_REPORT_ARTIFACT:?set RSNA_TICKET18_FROZEN_REPORT_ARTIFACT}"

case "${SLURM_ARRAY_TASK_ID:?expected array task id}" in
  0) arm="gold"; fold_id=0 ;;
  1) arm="report"; fold_id=0 ;;
  2) arm="gold"; fold_id=1 ;;
  3) arm="report"; fold_id=1 ;;
  4) arm="gold"; fold_id=2 ;;
  5) arm="report"; fold_id=2 ;;
  *) echo "unsupported Ticket 18 array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

output_dir="${output_root}/runs/${arm}-fold-${fold_id}"
preflight_path="${output_root}/runs/${arm}-fold-${fold_id}.preflight.json"
evidence_path="${output_root}/runs/${arm}-fold-${fold_id}.public.json"
compute_evidence_path="${output_root}/runs/${arm}-fold-${fold_id}.compute.json"

cd "${project_dir}"
mkdir -p artifacts "${output_root}/runs"
if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "corrected Ticket 18 training requires cluster1-cpu" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" || ! -x /usr/bin/time ]]; then
  echo "corrected Ticket 18 requires the isolated Python 3.11 environment and /usr/bin/time" >&2
  exit 1
fi
for input_path in "${integration_manifest}" "${fold_manifest}" "${reports}" "${frozen_report_artifact}"; do
  if [[ ! -f "${input_path}" ]]; then
    echo "corrected Ticket 18 input file does not exist: ${input_path}" >&2
    exit 1
  fi
done
if [[ ! -f "${source_path}" && ! -d "${source_path}" ]]; then
  echo "corrected Ticket 18 source does not exist: ${source_path}" >&2
  exit 1
fi
if [[ -e "${output_dir}" && -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite existing corrected Ticket 18 run: ${output_dir}" >&2
  exit 1
fi

# The array cap is the primary guard; this check also protects manual resubmits.
if ! active_jobs="$(squeue --noheader --user "${USER:-$(id -un)}" --name "rsna-t18add" \
  --states=RUNNING,CONFIGURING,COMPLETING --format="%A")"; then
  echo "could not inspect active corrected Ticket 18 jobs" >&2
  exit 1
fi
active_count="$(printf '%s\n' "${active_jobs}" | awk 'NF { count += 1 } END { print count + 0 }')"
if (( active_count > 2 )); then
  echo "corrected Ticket 18 concurrency limit exceeded: ${active_count} active jobs" >&2
  exit 1
fi

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RSNA_CODE_REVISION="${RSNA_CODE_REVISION:?set a frozen code revision before submission}"

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
  --evidence "${evidence_path}"
  --require-slurm
  --ticket18-additive-design
)

if [[ "${arm}" == "gold" ]]; then
  run_training_with_compute_record \
    "${project_dir}/.venv/bin/python" -m rsna_knee.training_cli "${training_args[@]}"
else
  training_args+=(
    --reports "${reports}"
    --frozen-report-artifact "${frozen_report_artifact}"
  )
  run_training_with_compute_record \
    "${project_dir}/.venv/bin/python" -m rsna_knee.weak_supervision_cli "${training_args[@]}"
fi
