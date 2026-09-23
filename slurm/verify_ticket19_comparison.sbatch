#!/usr/bin/env bash
# Read-only validation and aggregation of Ticket 19 against immutable Ticket 18 runs.
# Submit after the Ticket 19 three-fold array completes.
#SBATCH --job-name=rsna-t19verify
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/ticket-19-verify-%j.out
#SBATCH --error=artifacts/ticket-19-verify-%j.err

set -euo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
fold_manifest="${RSNA_TICKET19_FOLD_MANIFEST:?set the frozen version-2 fold manifest}"
baseline_root="${RSNA_TICKET19_BASELINE_RUN_ROOT:?set the immutable Ticket 18 gold-only run root}"
output_root="${RSNA_TICKET19_OUTPUT_ROOT:?set the Ticket 19 output root}"
comparison_root="${output_root}/comparison"

cd "${project_dir}"
mkdir -p artifacts "${comparison_root}"
if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "Ticket 19 aggregation requires cluster1-cpu" >&2
  exit 1
fi
if [[ ! -x "${project_dir}/.venv/bin/python" || ! -f "${fold_manifest}" ]]; then
  echo "Ticket 19 aggregation requires Python 3.11 and the frozen fold manifest" >&2
  exit 1
fi

baseline_args=()
candidate_args=()
for fold_id in 0 1 2; do
  baseline_run="${baseline_root}/gold-fold-${fold_id}"
  candidate_run="${output_root}/runs/fold-${fold_id}"
  if [[ ! -f "${baseline_run}/run.json" || ! -f "${candidate_run}/run.json" ]]; then
    echo "Ticket 19 comparison is missing a completed run for fold ${fold_id}" >&2
    exit 1
  fi
  baseline_args+=(--baseline-run "${baseline_run}")
  candidate_args+=(--candidate-run "${candidate_run}")
done

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${project_dir}/.venv/bin/python" -m rsna_knee.image_model_comparison \
  --fold-manifest "${fold_manifest}" \
  "${baseline_args[@]}" \
  "${candidate_args[@]}" \
  --json-output "${comparison_root}/ticket-19-image-model.json" \
  --markdown-output "${comparison_root}/ticket-19-image-model.md" \
  --bootstrap-resamples "${RSNA_TICKET19_BOOTSTRAP_RESAMPLES:-2000}" \
  --bootstrap-seed "${RSNA_TICKET19_BOOTSTRAP_SEED:-20260919}"
