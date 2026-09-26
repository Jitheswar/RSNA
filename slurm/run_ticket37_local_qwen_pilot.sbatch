#!/usr/bin/env bash
#SBATCH --job-name=ticket37-qwen-pilot
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/ch.jitheswar/ticket-37-qwen3-8b/slurm-%j.out
#SBATCH --error=/home/ch.jitheswar/ticket-37-qwen3-8b/slurm-%j.err

set -euo pipefail

repo_dir="/home/ch.jitheswar/RSNA"
run_dir="/home/ch.jitheswar/ticket-37-qwen3-8b"
model_dir="/home/ch.jitheswar/models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218"
report_corpus="/home/ch.jitheswar/ticket-15-real-20260914/private/report-corpus.json"
fold_manifest="${repo_dir}/artifacts/private/ticket-04-real-v2/private/fold-manifest.json"
approval="${run_dir}/private/model-and-report-use-approval.json"
frozen_policy="/home/ch.jitheswar/ticket-30-report-audit-20260925/private/frozen-policy.json"
ticket30_public_audit="${repo_dir}/docs/evidence/ticket-30-report-label-audit.json"
ticket30_private_admissions="/home/ch.jitheswar/ticket-30-report-audit-20260925/private/admitted-report-labels-v4.json"

export PYTHONPATH="${repo_dir}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM="false"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"

cd "${repo_dir}"
exec .venv/bin/python -m rsna_knee.ticket37_cli pilot \
  --approval "${approval}" \
  --model-dir "${model_dir}" \
  --fold-manifest "${fold_manifest}" \
  --reports "${report_corpus}" \
  --frozen-policy "${frozen_policy}" \
  --ticket30-public-audit "${ticket30_public_audit}" \
  --ticket30-private-admissions "${ticket30_private_admissions}" \
  --output-dir "${run_dir}" \
  --report-only-per-fold 2 \
  --max-elapsed-seconds 19800
