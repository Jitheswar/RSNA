#!/usr/bin/env bash
# Proposed only. Requires a separately approved one-job GPU analysis amendment.
#SBATCH --job-name=rsna-ticket39-analysis-replay
#SBATCH --partition=cluster1-gpu
#SBATCH --nodelist=KLEF-G-NODE3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=artifacts/private/ticket39-resnet18/analysis-replay-%j.out
#SBATCH --error=artifacts/private/ticket39-resnet18/analysis-replay-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
request="${RSNA_TICKET39_ANALYSIS_REQUEST:?immutable analysis request required}"
request_sha256="${RSNA_TICKET39_ANALYSIS_REQUEST_SHA256:?analysis request checksum required}"
ledger="${RSNA_TICKET39_CAMPAIGN_LEDGER:?shared GPU campaign ledger required}"
submitted_epoch="${RSNA_TICKET39_SUBMITTED_EPOCH:?submission epoch required}"
allocation_started_epoch="$(date +%s)"
record_allocation() {
  local exit_code="$?"
  local allocation_ended_epoch
  trap - EXIT
  allocation_ended_epoch="$(date +%s)"
  if ! PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${project_dir}/.venv/bin/python" -m rsna_knee.gpu_cli record-allocation \
      --ledger "${ledger}" \
      --request-checksum "${request_sha256}" \
      --attempt-number 1 \
      --job-id "${SLURM_JOB_ID:?Slurm job ID required}" \
      --allocated-start-epoch "${allocation_started_epoch}" \
      --allocated-end-epoch "${allocation_ended_epoch}" \
      --exit-code "${exit_code}"; then
    echo "Ticket 39 proposed replay allocation accounting failed" >&2
    exit 1
  fi
  exit "${exit_code}"
}
trap record_allocation EXIT
if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-gpu" || "${SLURM_JOB_NUM_NODES:-}" != "1" || "${SLURM_NTASKS:-}" != "1" || "${SLURM_CPUS_PER_TASK:-}" != "16" ]]; then
  echo "Ticket 39 proposed replay requires one approved GPU node, one task and 16 CPUs" >&2
  exit 1
fi
if [[ ! -f "${request}" || ! -x "/home/ch.jitheswar/ticket13-gpu-env-20260913/bin/python" ]]; then
  echo "Ticket 39 proposed replay request or pinned CUDA environment is missing" >&2
  exit 1
fi
cd "${project_dir}"
export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export RSNA_TICKET39_ANALYSIS_REQUEST="${request}"
export RSNA_TICKET39_ANALYSIS_REQUEST_SHA256="${request_sha256}"
exec /home/ch.jitheswar/ticket13-gpu-env-20260913/bin/python "${project_dir}/artifacts/private/ticket39-resnet18/analyze_ticket39_gpu_replay_proposal.py"
