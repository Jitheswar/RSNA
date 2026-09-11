#!/usr/bin/env bash
# Regenerate identifier-free OOF comparison views from completed fold records.
# This job is read-only with respect to each fold run directory.
#SBATCH --job-name=rsna-ticket10-oof
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/ticket-10-oof-%j.out
#SBATCH --error=artifacts/ticket-10-oof-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
fold_manifest="${RSNA_TICKET10_FOLD_MANIFEST:?set RSNA_TICKET10_FOLD_MANIFEST}"
inventory="${RSNA_TICKET10_INVENTORY:?set RSNA_TICKET10_INVENTORY to the private ticket-03 inventory}"
run_root="${RSNA_TICKET10_RUN_ROOT:?set RSNA_TICKET10_RUN_ROOT to the completed fold run root}"
fold0_run="${RSNA_TICKET10_FOLD0_RUN:-${run_root}/fold-0}"
output_root="${RSNA_TICKET10_OUTPUT:?set RSNA_TICKET10_OUTPUT to a private OOF output root}"
bootstrap_resamples="${RSNA_TICKET10_BOOTSTRAP_RESAMPLES:-2000}"
bootstrap_seed="${RSNA_TICKET10_BOOTSTRAP_SEED:-20260911}"
bootstrap_unit="${RSNA_TICKET10_BOOTSTRAP_UNIT:-auto}"
permutation_resamples="${RSNA_TICKET10_PERMUTATION_RESAMPLES:-10000}"
permutation_seed="${RSNA_TICKET10_PERMUTATION_SEED:-20260911}"

cd "${project_dir}"
mkdir -p artifacts "${output_root}/private" "${output_root}/public"

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "ticket-10 aggregation requires the cluster1-cpu partition" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "missing ${project_dir}/.venv/bin/python" >&2
  exit 1
fi
if [[ ! -f "${fold_manifest}" || ! -f "${inventory}" ]]; then
  echo "ticket-10 aggregation requires the private fold manifest and inventory" >&2
  exit 1
fi
if [[ ! -f "${fold0_run}/run.json" ]]; then
  echo "missing completed fold-0 run record: ${fold0_run}" >&2
  exit 1
fi
for fold_id in 1 2; do
  if [[ ! -f "${run_root}/fold-${fold_id}/run.json" ]]; then
    echo "missing completed fold-${fold_id} run record" >&2
    exit 1
  fi
done

history_args=()
# Historical replay of runs on a superseded version-1 fold manifest only.
if [[ "${RSNA_TICKET10_ALLOW_SUPERSEDED_FOLD_MANIFEST:-0}" == "1" ]]; then
  history_args+=(--allow-superseded-fold-manifest)
fi

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${project_dir}/.venv/bin/python" -m rsna_knee.oof_cli ${history_args[@]+"${history_args[@]}"} \
  --fold-manifest "${fold_manifest}" \
  --inventory "${inventory}" \
  --run "${fold0_run}" \
  --run "${run_root}/fold-1" \
  --run "${run_root}/fold-2" \
  --public-output "${output_root}/public/oof-evidence.json" \
  --private-output "${output_root}/private/oof-evidence.json" \
  --comparison-output "${output_root}/public/comparison.md" \
  --expected-gold-count 58 \
  --bootstrap-resamples "${bootstrap_resamples}" \
  --bootstrap-seed "${bootstrap_seed}" \
  --bootstrap-unit "${bootstrap_unit}" \
  --permutation-resamples "${permutation_resamples}" \
  --permutation-seed "${permutation_seed}"
