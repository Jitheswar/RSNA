#!/usr/bin/env bash
#SBATCH --job-name=rsna-t29-storage
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=artifacts/private/ticket-29-real-diagnostic/storage-inventory-%j.out
#SBATCH --error=artifacts/private/ticket-29-real-diagnostic/storage-inventory-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
canonical_repo="/home/ch.jitheswar/RSNA"
python="${canonical_repo}/.venv/bin/python"
archive="${canonical_repo}/rsna-knee-abnormality-detection.zip"
output="${project_dir}/artifacts/private/ticket-29-real-diagnostic/campaign-storage-inventory-${SLURM_JOB_ID:?Slurm job ID is required}.json"

if [[ ! -x "${python}" || ! -f "${archive}" ]]; then
  echo "pinned Python environment or retained source archive is unavailable" >&2
  exit 1
fi

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${python}" "${project_dir}/scripts/ticket29_campaign_storage_inventory.py" \
  --archive "${archive}" \
  --output "${output}" \
  --root canonical-repository=/home/ch.jitheswar/RSNA \
  --root rsna-ticket28=/home/ch.jitheswar/RSNA-ticket-28 \
  --root rsna-ticket29=/home/ch.jitheswar/RSNA-ticket-29 \
  --root rsna-ticket30=/home/ch.jitheswar/RSNA-ticket-30 \
  --root ticket15-primary=/home/ch.jitheswar/ticket-15-real-20260914 \
  --root ticket15-repeat=/home/ch.jitheswar/ticket-15-real-20260914-repeat \
  --root ticket16=/home/ch.jitheswar/ticket-16-real-20260914 \
  --root ticket17=/home/ch.jitheswar/ticket-17-real-20260921 \
  --root ticket18-additive=/home/ch.jitheswar/ticket-18-additive-v2 \
  --root ticket18-real=/home/ch.jitheswar/ticket-18-real-20260923 \
  --root ticket19=/home/ch.jitheswar/ticket-19-wider-encoder-v1 \
  --root ticket21-primary=/home/ch.jitheswar/ticket-21-pseudo-20260923 \
  --root ticket21-repeat=/home/ch.jitheswar/ticket-21-pseudo-20260923-attempt2 \
  --root ticket22-v1=/home/ch.jitheswar/ticket-22-distillation-v1 \
  --root ticket22-v2=/home/ch.jitheswar/ticket-22-distillation-v2 \
  --root ticket22-v3=/home/ch.jitheswar/ticket-22-distillation-v3 \
  --root ticket30-audit=/home/ch.jitheswar/ticket-30-report-audit-20260925 \
  --root ticket13-gpu-environment=/home/ch.jitheswar/ticket13-gpu-env-20260913 \
  --root ticket13-operational-environment=/home/ch.jitheswar/ticket13-operational-20260913
