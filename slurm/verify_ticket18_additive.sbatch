#!/usr/bin/env bash
# Post-run verification and aggregation for the frozen additive Ticket 18 v2 array.
# Submit with an afterany dependency on the six-task training array.
#SBATCH --job-name=rsna-t18vfy
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/ticket-18-additive-verify-%j.out
#SBATCH --error=artifacts/ticket-18-additive-verify-%j.err

set -uo pipefail

project_dir="${SLURM_SUBMIT_DIR:?sbatch must provide SLURM_SUBMIT_DIR}"
output_root="${RSNA_TICKET18_V2_OUTPUT_ROOT:?set RSNA_TICKET18_V2_OUTPUT_ROOT}"
fold_manifest="${RSNA_TICKET18_FOLD_MANIFEST:?set RSNA_TICKET18_FOLD_MANIFEST}"
integration_manifest="${RSNA_TICKET18_INTEGRATION_MANIFEST:?set RSNA_TICKET18_INTEGRATION_MANIFEST}"

cd "${project_dir}"
export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RSNA_CODE_REVISION="${RSNA_CODE_REVISION:?set the frozen additive code revision}"
mkdir -p artifacts "${output_root}/verification" "${output_root}/comparison"

overall_status=0
run_check() {
  local name="$1"
  shift
  if "$@" >"${output_root}/verification/${name}.log" 2>&1; then
    printf '{"check":"%s","status":"passed"}\n' "${name}" \
      >"${output_root}/verification/${name}.status.json"
  else
    local result=$?
    printf '{"check":"%s","status":"failed","exit_code":%s}\n' \
      "${name}" "${result}" >"${output_root}/verification/${name}.status.json"
    overall_status=1
  fi
}

run_check full-pytest /usr/bin/env -u SLURM_JOB_ID -u SLURM_JOB_PARTITION \
  "${project_dir}/.venv/bin/python" -m pytest -q
run_check mypy "${project_dir}/.venv/bin/mypy" src/rsna_knee
run_check diff-check /usr/bin/git diff --check
run_check shell-syntax bash -n slurm/run_ticket18_additive_array.sbatch

if ! "${project_dir}/.venv/bin/python" - "${output_root}" "${fold_manifest}" "${integration_manifest}" \
  >"${output_root}/verification/checkpoint-and-pair.log" 2>&1 <<'PY'
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
from rsna_knee.folds import load_fold_manifest
from rsna_knee.model import HierarchicalStudyModel
from rsna_knee.real_inference import load_integration_manifest
from rsna_knee.training import TrainingConfig, load_training_checkpoint

root = Path(sys.argv[1])
manifest = load_fold_manifest(sys.argv[2])
integration = load_integration_manifest(sys.argv[3])
config = TrainingConfig(preprocessing_config=integration.preprocess_config)
design = json.loads((root / "design.json").read_text(encoding="utf-8"))
if design.get("design_id") != "ticket18-additive-report-supervision-v2":
    raise SystemExit("frozen design identity is missing")
records = []
for fold in range(3):
    paired = {}
    for arm in ("gold", "report"):
        run_dir = root / "runs" / f"{arm}-fold-{fold}"
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        saved = json.loads((run_dir / "configuration.json").read_text(encoding="utf-8"))
        if run.get("status") != "completed_fixed_budget_training":
            raise SystemExit(f"{arm} fold {fold} did not complete")
        if run.get("fold") != fold or run.get("config_fingerprint") != config.config_fingerprint:
            raise SystemExit(f"{arm} fold {fold} has a mismatched fold/configuration")
        if saved.get("environment", {}).get("code_revision") != design.get("code_revision"):
            raise SystemExit(f"{arm} fold {fold} used a different code revision")
        counts = run.get("counts", {})
        if (counts.get("successful_optimizer_steps"), counts.get("scheduler_steps"),
            counts.get("attempted_batches"), counts.get("skipped_all_masked_batches")) != (100, 100, 100, 0):
            raise SystemExit(f"{arm} fold {fold} does not have the frozen 100-step budget")
        checkpoint = run_dir / "checkpoints" / "final.pt"
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if run.get("checkpoint", {}).get("final_sha256") != checkpoint_sha:
            raise SystemExit(f"{arm} fold {fold} final checkpoint hash disagrees")
        model = HierarchicalStudyModel(config.model_config())
        load_training_checkpoint(
            checkpoint,
            model=model,
            expected_config=config,
            expected_fold=fold,
            expected_identity=saved.get("identity"),
            expected_checkpoint_kind="final_selected",
        )
        train_uids = set(run.get("training_study_uids", []))
        held_out_uids = set(run.get("held_out_study_uids", []))
        if not train_uids or train_uids.intersection(held_out_uids):
            raise SystemExit(f"{arm} fold {fold} violates training/held-out isolation")
        paired[arm] = (run, counts, checkpoint_sha, held_out_uids)
        records.append({
            "arm": arm,
            "fold": fold,
            "run_id": run.get("run_id"),
            "slurm_job_id": saved.get("resources", {}).get("slurm_attempt_id"),
            "checkpoint_sha256": checkpoint_sha,
            "config_fingerprint": run.get("config_fingerprint"),
            "code_revision": saved.get("environment", {}).get("code_revision"),
            "gold_exposure_sequence_sha256": counts.get("gold_exposure_sequence_sha256"),
            "weak_exposure_sequence_sha256": counts.get("weak_exposure_sequence_sha256"),
            "initial_model_state_sha256": counts.get("initial_model_state_sha256"),
        })
    gold_run, gold_counts, _, gold_held_out = paired["gold"]
    report_run, report_counts, _, report_held_out = paired["report"]
    if gold_held_out != report_held_out:
        raise SystemExit(f"fold {fold} held-out evaluation identities differ")
    if gold_counts.get("gold_exposure_sequence_sha256") != report_counts.get("gold_exposure_sequence_sha256"):
        raise SystemExit(f"fold {fold} ordered gold exposure sequence differs")
    if gold_counts.get("initial_model_state_sha256") != report_counts.get("initial_model_state_sha256"):
        raise SystemExit(f"fold {fold} initialized model states differ")
    gold_ref = gold_counts.get("exposures_by_source", {}).get("reference", {})
    report_ref = report_counts.get("exposures_by_source", {}).get("reference", {})
    if gold_ref != report_ref or gold_ref != {"sample_exposures": 200, "supervised_entry_exposures": 2400}:
        raise SystemExit(f"fold {fold} exact gold exposure counts differ")
    weak_exposure = report_counts.get("exposures_by_source", {}).get("weak", {})
    if weak_exposure.get("sample_exposures") != 200 or not report_counts.get("weak_exposure_sequence_sha256"):
        raise SystemExit(f"fold {fold} does not contain the frozen additional weak exposure")

result = {
    "schema_version": "rsna-knee-ticket18-additive-v2-checkpoint-verification",
    "status": "passed",
    "fold_manifest_checksum": manifest.manifest_checksum,
    "training_configuration_fingerprint": config.config_fingerprint,
    "runs": records,
}
path = root / "verification" / "checkpoint-and-pair.json"
path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
then
  overall_status=1
  printf '{"check":"checkpoint-and-pair","status":"failed"}\n' \
    >"${output_root}/verification/checkpoint-and-pair.status.json"
else
  printf '{"check":"checkpoint-and-pair","status":"passed"}\n' \
    >"${output_root}/verification/checkpoint-and-pair.status.json"
fi

run_check comparison-aggregate "${project_dir}/.venv/bin/python" -m rsna_knee.supervision_comparison \
  --fold-manifest "${fold_manifest}" \
  --gold-only-run "${output_root}/runs/gold-fold-0" \
  --gold-only-run "${output_root}/runs/gold-fold-1" \
  --gold-only-run "${output_root}/runs/gold-fold-2" \
  --report-supervised-run "${output_root}/runs/report-fold-0" \
  --report-supervised-run "${output_root}/runs/report-fold-1" \
  --report-supervised-run "${output_root}/runs/report-fold-2" \
  --public-json "${output_root}/comparison/ticket-18-additive-v2.json" \
  --markdown "${output_root}/comparison/ticket-18-additive-v2.md" \
  --bootstrap-resamples 2000 \
  --bootstrap-seed 20260918 \
  --bootstrap-confidence-level 0.95 \
  --require-additive-gold-exposure

exit "${overall_status}"
