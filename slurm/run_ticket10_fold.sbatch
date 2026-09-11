#!/usr/bin/env bash
# Run one independent ticket-10 fold record.  The aggregation job reads these
# directories later; no fold job writes a shared comparison table.
#SBATCH --job-name=rsna-ticket10-fold
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/ticket-10-fold-%j.out
#SBATCH --error=artifacts/ticket-10-fold-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
integration_manifest="${RSNA_TICKET10_INTEGRATION_MANIFEST:?set RSNA_TICKET10_INTEGRATION_MANIFEST}"
source_path="${RSNA_TICKET10_SOURCE:?set RSNA_TICKET10_SOURCE}"
fold_manifest="${RSNA_TICKET10_FOLD_MANIFEST:?set RSNA_TICKET10_FOLD_MANIFEST}"
fold_id="${RSNA_TICKET10_FOLD:?set RSNA_TICKET10_FOLD to 0, 1, or 2}"
run_root="${RSNA_TICKET10_RUN_ROOT:?set RSNA_TICKET10_RUN_ROOT to a private run root}"
source_version="${RSNA_TICKET10_SOURCE_VERSION:?set RSNA_TICKET10_SOURCE_VERSION}"
source_fingerprint="${RSNA_TICKET10_SOURCE_FINGERPRINT:-}"
cache_root="${RSNA_TICKET10_CACHE_ROOT:-}"

cd "${project_dir}"
mkdir -p artifacts

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "ticket-10 fold requires the cluster1-cpu partition" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "missing ${project_dir}/.venv/bin/python" >&2
  exit 1
fi
if [[ ! -f "${integration_manifest}" || ! -f "${fold_manifest}" ]]; then
  echo "ticket-10 fold requires the private integration and fold manifests" >&2
  exit 1
fi
if [[ ! -f "${source_path}" && ! -d "${source_path}" ]]; then
  echo "real-DICOM source does not exist: ${source_path}" >&2
  exit 1
fi
if [[ ! "${fold_id}" =~ ^[012]$ ]]; then
  echo "ticket-10 fold must be 0, 1, or 2" >&2
  exit 1
fi

output_dir="${run_root}/fold-${fold_id}"
preflight_path="${run_root}/preflight-fold-${fold_id}.json"
evidence_path="${run_root}/fold-${fold_id}.public.json"

# Reserve one of the two autonomous CPU slots across independently submitted
# fold jobs. The lock is held through exec() so the guard covers training, not
# merely the preflight check.
if ! command -v flock >/dev/null 2>&1; then
  echo "ticket-10 fold requires flock for the two-job concurrency guard" >&2
  exit 1
fi
concurrency_lock="${RSNA_TICKET10_CONCURRENCY_LOCK:-${project_dir}/artifacts/ticket10-fold-concurrency.lock}"
exec 9>"${concurrency_lock}"
flock -x 9
user_name="${USER:-$(id -un)}"
if ! active_fold_jobs="$(
  squeue --noheader --user "${user_name}" --name "rsna-ticket10-fold" \
    --states=RUNNING,CONFIGURING,COMPLETING --format="%A"
)"; then
  echo "could not inspect active ticket-10 fold jobs" >&2
  exit 1
fi
active_fold_count="$(printf '%s\n' "${active_fold_jobs}" | awk 'NF { count += 1 } END { print count + 0 }')"
if (( active_fold_count > 2 )); then
  echo "ticket-10 fold concurrency limit exceeded: ${active_fold_count} active jobs" >&2
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
