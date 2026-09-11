#!/usr/bin/env bash
# Run the frozen candidate on the bounded representative real-DICOM workload.
# Candidate contents are read-only; all execution evidence is external.
#SBATCH --job-name=rsna-ticket11-candidate
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/ticket-11-candidate-%j.out
#SBATCH --error=artifacts/ticket-11-candidate-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
candidate="${RSNA_TICKET11_CANDIDATE:?set RSNA_TICKET11_CANDIDATE to the frozen candidate root}"
manifest="${RSNA_TICKET11_INTEGRATION_MANIFEST:?set RSNA_TICKET11_INTEGRATION_MANIFEST}"
source_path="${RSNA_TICKET11_SOURCE:?set RSNA_TICKET11_SOURCE to the verified source ZIP}"
output_path="${RSNA_TICKET11_OUTPUT:?set RSNA_TICKET11_OUTPUT to a new execution output root}/submission.csv"
evidence_path="${RSNA_TICKET11_EVIDENCE:-${output_path%/submission.csv}/public/candidate-evidence.json}"
private_evidence_path="${RSNA_TICKET11_PRIVATE_EVIDENCE:-${output_path%/submission.csv}/private/candidate-evidence.json}"
cache_root="${RSNA_TICKET11_CACHE_ROOT:-}"
expected_studies="${RSNA_TICKET11_EXPECTED_STUDIES:-1300}"
long_tail_studies="${RSNA_TICKET11_LONG_TAIL_STUDIES:-}"

cd "${project_dir}"
mkdir -p artifacts

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "ticket-11 candidate requires the cluster1-cpu partition" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "missing ${project_dir}/.venv/bin/python; provision the pinned user-space environment first" >&2
  exit 1
fi
if [[ ! -d "${candidate}" || ! -f "${candidate}/manifest.json" ]]; then
  echo "frozen candidate manifest does not exist: ${candidate}" >&2
  exit 1
fi
if [[ ! -f "${manifest}" || ! -f "${source_path}" ]]; then
  echo "ticket-11 candidate requires the private integration manifest and source ZIP" >&2
  exit 1
fi

benchmark_args=(
  benchmark
  --candidate "${candidate}"
  --manifest "${manifest}"
  --source "${source_path}"
  --output "${output_path}"
  --evidence "${evidence_path}"
  --private-evidence "${private_evidence_path}"
  --expected-study-count "${expected_studies}"
  --require-slurm
  --recover-damaged-inputs
)
if [[ -n "${cache_root}" ]]; then
  benchmark_args+=(--cache-root "${cache_root}")
fi
if [[ -n "${long_tail_studies}" ]]; then
  benchmark_args+=(--long-tail-studies "${long_tail_studies}")
fi

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${project_dir}/.venv/bin/python" -m rsna_knee.candidate_cli "${benchmark_args[@]}"
