"""Freeze and execute independent original-MRI replay after all Ticket 64 fits.

Preparation cannot admit unfinished checkpoints. No fitting, promotion,
submission or retry is performed here. The complete campaign reserves 90
CPU-node minutes for this separately frozen request.
"""

from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

PROJECT = Path(os.environ.get("RSNA_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
FIT = PROJECT / "artifacts/private/ticket64-full-backbone-comparison-v4"
PRIOR = PROJECT / "artifacts/private/ticket64-full-backbone-comparison-v2"
PRIOR_SHA = "41028974908d8ea8c4fe336f86de0263782100c76b9375dc7693bf60ac213eaf"
OLD = PROJECT / "artifacts/private/ticket64-full-backbone-comparison-v1"
ROOT = FIT / "independent-original-mri-replay-v1"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket73-independent-original-mri-replay-20261002.json"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
ARMS = ("head-continuation", "full-backbone")
FIT_SHA = hashlib.sha256((FIT / "plan.json").read_bytes()).hexdigest() if (FIT / "plan.json").exists() else None
SCIENCE_SHA = "557230fec672f6db9252b1895a4d174652fe217c0388a8c9598bd749183aa985"


def stage_location(fold, stage):
    from ticket73_configured_full_continuation import stage_location as locate
    return locate(fold, stage, FIT_SHA)


def final_checkpoint_path(arm, fold):
    require(arm in ARMS and fold in (0, 1, 2), "Unknown final fit")
    base = (OLD if fold == 0 else PRIOR) if arm == "head-continuation" else FIT
    return base / arm / f"fold-{fold}/training/checkpoints/final.pt"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def maximum_error(actual, expected):
    import numpy as np
    require(bool(actual) and set(actual) == set(expected), "Exact prediction membership required")
    require(all(np.asarray(actual[uid]).shape == np.asarray(expected[uid]).shape == (12,)
                and np.isfinite(actual[uid]).all() and np.isfinite(expected[uid]).all() for uid in actual),
            "Finite twelve-target predictions required")
    return max(float(np.max(np.abs(np.asarray(actual[uid]) - np.asarray(expected[uid])))) for uid in actual)


def final_fit_records(checksum):
    records = []; seen = {arm: set() for arm in ARMS}
    for fold in range(3):
        for arm in ARMS:
            directory = (PRIOR if arm == "head-continuation" else FIT) / arm / f"fold-{fold}"
            path = directory / ("run.json" if arm == ARMS[0] else "stage-7/stage-result.json")
            record = read(path)
            require(record["status"] == "COMPLETE" and record["plan_sha256"] == (PRIOR_SHA if arm == "head-continuation" else checksum)
                    and record["scientific_plan_sha256"] == SCIENCE_SHA
                    and record["progress"]["successful_optimizer_steps"] == record["progress"]["attempted_batches"] == 7500,
                    "Every registered final fit must complete")
            predictions = record["predictions"]
            require(len(predictions) == (20 if fold == 0 else 19) and not seen[arm] & set(predictions),
                    "Exact disjoint final gold prediction counts required")
            maximum_error(predictions, record["encoded_image_ablation"])
            seen[arm].update(predictions)
            checkpoint = final_checkpoint_path(arm, fold)
            require(Path(record["checkpoint_path"]) == checkpoint and digest(checkpoint) == record["checkpoint_sha256"],
                    "Exact final checkpoint required")
            records.append(dict(arm=arm, fold=fold, checkpoint_path=str(checkpoint),
                                checkpoint_sha256=record["checkpoint_sha256"]))
    return records


def ensure_final_analysis(request, generator):
    path = FIT / "analysis.json"
    if request["analysis_generated_in_replay_allocation"]:
        require(not path.exists(), "Unexpected analysis appeared after immutable replay admission")
        generator(request["fit_plan_sha256"])
    analysis = read(path)
    require(analysis["status"] == "COMPLETE" and analysis["plan_sha256"] == request["fit_plan_sha256"]
            and analysis["run_records"] == request["final_run_records"] and not analysis["candidate_promotion"],
            "Exact complete final comparison required")
    return analysis


def prepare():
    require(not ROOT.exists() and not DECISION.exists(), "Immutable verifier request already exists")
    # The coordinator's independent, tensor-free terminal/counter guards are
    # reused only to establish completed inputs before creating this request.
    plan = read(FIT / "plan.json")
    checksum = digest(FIT / "plan.json")
    require(checksum == FIT_SHA, "Exact registered fit required")
    require(plan["projection"]["independent_replay_wall_seconds"] == 5400,
            "Original campaign replay reserve differs")
    from ticket73_configured_full_continuation import audited_boundaries, successful_terminal, stage_record, AUDIT
    audited_boundaries()
    successful_terminal(PRIOR / "control", PRIOR_SHA, "02:00:00")
    inputs = [AUTH, FIT / "plan.json", AUDIT / "plan.json", AUDIT / "checkpoint-audit.json", FIT / "live-fold-1-input.json"]
    require((FIT / "live-fold-1-input.json").exists(), "Completed inherited live input required")
    inputs.extend(Path(r["path"]) for r in read(FIT / "live-fold-1-input.json")["frozen_files"])
    control = PRIOR / "control"
    job = read(control / "accepted-launch.json")["job_id"]
    inputs.extend([control / "accepted-launch.json", control / "stage-result.json",
                   control / f"controller-{job}/terminal-controller.txt",
                   control / f"controller-{job}/terminal-observation.json"])
    for fold in range(3):
        for stage in range(8):
            current, ancestry = stage_record(fold, stage, checksum)
            directory, _ = stage_location(fold, stage)
            job = read(directory / "accepted-launch.json")["job_id"]
            inputs.extend([directory / "accepted-launch.json", directory / "stage-result.json"])
            if ancestry == "actual_successful_terminal":
                inputs.extend([directory / f"controller-{job}/terminal-controller.txt",
                               directory / f"controller-{job}/terminal-observation.json"])
            else:
                require(ancestry == "independently_audited_checkpoint_boundary", "Explicit audit ancestry required")
    final_records = final_fit_records(checksum)
    generate_analysis = not (FIT / "analysis.json").exists()
    if not generate_analysis:
        analysis = read(FIT / "analysis.json")
        require(analysis["status"] == "COMPLETE" and analysis["plan_sha256"] == checksum
                and analysis["run_records"] == final_records and not analysis["candidate_promotion"],
                "Exact existing final comparison required")
        inputs.append(FIT / "analysis.json")
    for row in final_records:
        arm, fold = row["arm"], row["fold"]
        directory = (PRIOR if arm == "head-continuation" else FIT) / arm / f"fold-{fold}"
        path = final_checkpoint_path(arm, fold)
        require(Path(row["checkpoint_path"]) == path and digest(path) == row["checkpoint_sha256"], "Exact final checkpoint required")
        inputs.append(path)
        if arm == "head-continuation":
            inputs.append(directory / "run.json")
    usage = subprocess.check_output(["du", "--apparent-size", "-s", "-B1", str(PROJECT),
        str(PROJECT.parent / "models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218")], text=True)
    require(sum(int(r.split()[0]) for r in usage.splitlines()) + 1_000_000_000 < 2_000_000_000_000
            and shutil.disk_usage(PROJECT).free > 1_000_000_000, "Fresh verifier storage required")
    ROOT.mkdir(mode=0o700)
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(plan["source_snapshot"], snapshot,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(__file__, snapshot / Path(__file__).name)
    pixel_plan = read(Path(plan["pixel_root"]) / "plan.json")
    source_reader = next(Path(r["path"]) for r in pixel_plan["frozen_files"]
                         if Path(r["path"]).name == "ticket62_mri_pixel_materialize.py")
    require(digest(source_reader) == next(r["sha256"] for r in pixel_plan["frozen_files"]
                                        if Path(r["path"]) == source_reader), "Admitted original reader changed")
    shutil.copyfile(source_reader, snapshot / source_reader.name)
    inputs.extend(p for p in snapshot.rglob("*") if p.is_file())
    resources = dict(partition="cluster1-cpu", nodes=1, tasks=1, cpus_per_task=16, memory_gib=64,
                     walltime_seconds=5400, code_watchdog_seconds=5340, maximum_launch_attempts=1,
                     exclusive=False, requeue=False, gpu_count=0, external_spend_usd=0)
    request = dict(schema_version="rsna-ticket73-independent-original-mri-plan-v1",
        created_at_utc=datetime.now(timezone.utc).isoformat(), fit_plan_sha256=checksum,
        source_snapshot=str(snapshot), decision_path=str(DECISION), resources=resources,
        authorization_sha256=digest(AUTH), runtime_versions=plan["runtime_versions"],
        storage_observation=usage, maximum_prediction_error=1e-6,
        analysis_generated_in_replay_allocation=generate_analysis, final_run_records=final_records,
        frozen_files=[dict(path=str(p), sha256=digest(p)) for p in inputs],
        historical_backend_terminals_verified=False, original_strict_ancestry_passed=False,
        original_missing_allocation_charge_seconds=39600,
        fitting=False, candidate_promotion=False, competition_submission=False)
    write(ROOT / "plan.json", request)
    sha = digest(ROOT / "plan.json")
    write(DECISION, dict(decision="approved", plan_sha256=sha, resources=resources,
        authorization_sha256=digest(AUTH), decision_made_by="campaign agent under standing owner authorization",
        basis="Original-source replay, strict final states and independently recomputed fixed comparisons within the registered 5400-second reserve.",
        new_exact_owner_numeric_approval_claimed=False, gpu_ledger_changed=False,
        tpu_ledger_changed=False, candidate_promotion=False))
    launcher = ROOT / "run.sbatch"
    launcher.write_text(f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket64-original-replay
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output={ROOT}/slurm-%j.out
#SBATCH --error={ROOT}/slurm-%j.err
#SBATCH --chdir={PROJECT}
set -euo pipefail
umask 077
export RSNA_PROJECT_ROOT={PROJECT}
export PYTHONPATH={snapshot}/src
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
{PROJECT}/.venv/bin/python {snapshot / Path(__file__).name} run {sha}
''')
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    write(ROOT / "prepared-validation.json", dict(plan_sha256=sha, launcher_sha256=digest(launcher), launched=False))
    print(json.dumps(dict(plan_sha256=sha, launched=False)))


def launch(checksum):
    require(digest(ROOT / "plan.json") == checksum, "Exact verifier plan required")
    plan = read(ROOT / "plan.json"); decision = read(DECISION)
    prepared = read(ROOT / "prepared-validation.json")
    require(decision["decision"] == "approved" and decision["plan_sha256"] == checksum
            and decision["resources"] == plan["resources"] and digest(AUTH) == plan["authorization_sha256"]
            and prepared["plan_sha256"] == checksum
            and digest(ROOT / "run.sbatch") == prepared["launcher_sha256"], "Exact verifier admission required")
    require(all(digest(r["path"]) == r["sha256"] for r in plan["frozen_files"]), "Frozen verifier inputs changed")
    jobs = subprocess.check_output(["squeue", "-u", os.environ["USER"], "-h", "-o", "%i %P"], text=True)
    require(sum(r.split()[1] == "cluster1-cpu" for r in jobs.splitlines()) < 2, "Two CPU slots occupied")
    write(ROOT / "launch-intent.json", dict(plan_sha256=checksum, launcher_sha256=prepared["launcher_sha256"], launch_attempt=1))
    receipt = subprocess.check_output(["sbatch", "--parsable", str(ROOT / "run.sbatch")], text=True).strip()
    job = int(receipt.split(";")[0])
    write(ROOT / "accepted-launch.json", dict(job_id=job, plan_sha256=checksum, launch_attempt=1))
    subprocess.run(["systemd-run", "--user", f"--unit=rsna-ticket64-native-{job}-observer", "--collect",
                    str(PROJECT / ".venv/bin/python"), str(PROJECT / "scripts/monitor_ticket48_cpu_pilot.py"),
                    str(ROOT), str(job)], check=True)
    print(json.dumps(dict(job_id=job, plan_sha256=checksum)))


def run(checksum):
    started = time.monotonic()
    require(digest(ROOT / "plan.json") == checksum, "Frozen independent request changed")
    request = read(ROOT / "plan.json"); decision = read(request["decision_path"])
    require(decision["decision"] == "approved" and decision["plan_sha256"] == checksum
            and decision["resources"] == request["resources"] and digest(AUTH) == request["authorization_sha256"],
            "Independent runtime admission differs")
    allocation = subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True)
    fields = dict(t.split("=", 1) for t in allocation.split() if "=" in t)
    expected = dict(JobId=os.environ["SLURM_JOB_ID"], JobState="RUNNING", Partition="cluster1-cpu",
                    NumNodes="1", NumCPUs="16", NumTasks="1", MinMemoryNode="64G", TimeLimit="01:30:00",
                    OverSubscribe="OK", Requeue="0", Restarts="0")
    require(all(fields.get(k) == v for k, v in expected.items()) and "gpu" not in fields.get("TRES", "").lower(),
            "Actual independent CPU allocation differs")
    require(Path(__file__).resolve().parent == Path(request["source_snapshot"])
            and os.environ.get("PYTHONPATH") == str(Path(request["source_snapshot"]) / "src"),
            "Frozen independent runtime required")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Original-MRI replay watchdog")))
    signal.alarm(5340)
    plan = read(FIT / "plan.json")
    require(digest(FIT / "plan.json") == request["fit_plan_sha256"] == FIT_SHA, "Exact fitted route changed")
    fit_decision = read(plan["decision_path"])
    require(fit_decision["decision"] == "approved" and fit_decision["plan_sha256"] == FIT_SHA
            and fit_decision["resources"] == plan["resources"] and digest(AUTH) == plan["authorization_sha256"],
            "Original fitting admission changed")
    for row in request["frozen_files"] + plan["frozen_files"]:
        require(digest(row["path"]) == row["sha256"], "Frozen scientific source or verifier changed")
    require(all(importlib.metadata.version(n) == v for n, v in request["runtime_versions"].items()),
            "Pinned verifier versions differ")
    import numpy as np
    import torch
    from rsna_knee.folds import load_fold_manifest
    from rsna_knee.image_model_comparison import compare_image_model_predictions
    from rsna_knee.pixel_cache import load_pixel_entry
    from rsna_knee.ticket34 import admit_ticket30_fold_supervision
    from rsna_knee.training import _batch_indices, _model_state_sha256, load_training_checkpoint
    from ticket62_mri_pixel_materialize import iter_packed_sources
    from ticket73_configured_full_continuation import helpers, analyze, stage_record
    original, scientific_identity, _ = helpers()
    config_for_arm, initial_for_arm, terminal = original.config_for_arm, original.initial_for_arm, original.terminal
    from ticket63_full_backbone_training_cost import PARENT, t36
    torch.set_num_threads(16); torch.set_num_interop_threads(1)
    require(torch.version.cuda is None and not torch.cuda.is_available(), "CPU-only replay required")
    ensure_final_analysis(request, analyze)
    manifest = load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    gold = {m.study_instance_uid: m for m in manifest.gold_memberships}
    require(len(gold) == 58 and not manifest.superseded, "Complete corrected gold membership required")
    pixel_root = Path(plan["pixel_root"])
    pixel_plan = read(pixel_root / "plan.json")
    archive = Path(pixel_plan["source_archive_path"]); stat = archive.stat()
    require(stat.st_size == pixel_plan["source_archive_bytes"] and stat.st_mtime_ns == pixel_plan["source_archive_mtime_ns"],
            "Original archive stat identity changed")
    pixel_manifest = read(pixel_root / "pixel-manifest.json")
    rows = {r["study_instance_uid"]: r for r in pixel_manifest["pixels"]}
    packed = {}; source_audit = []
    for uid, value, source, prior, _ in iter_packed_sources(pixel_plan, sorted(gold)):
        row = rows[uid]
        entry = load_pixel_entry(pixel_root / "pixels" / row["pixel_file"], expected_sha256=row["pixel_file_sha256"],
            expected_study_uid=uid, expected_identity_sha256=pixel_plan["pixel_identity_sha256"],
            expected_source_identity_sha256=source, expected_plan_sha256=digest(pixel_root / "plan.json"))
        pixels = entry.packed_images()
        require(np.array_equal(value.images, pixels)
                and all(np.array_equal(getattr(value, k), getattr(entry, k))
                        for k in ("slice_mask", "series_mask", "series_descriptors"))
                and entry.metadata["sampled_source_indices"] == [list(v) for v in value.sampled_source_indices],
                "Original-source pixels or packing differ from the fitted inputs")
        packed[uid] = value
        source_audit.append(dict(study_instance_uid=uid, source_identity_sha256=source,
                                 pixel_file_sha256=row["pixel_file_sha256"]))
    require(set(packed) == set(gold), "All original gold MRI must replay")
    terminal(PRIOR / "control", PRIOR_SHA, "02:00:00")
    predicted = {a: {} for a in ARMS}; ablated = {a: {} for a in ARMS}
    native_all = {a: {} for a in ARMS}; native_zero = {a: {} for a in ARMS}; native_parent_all = {}
    parent_predictions = {}; audits = []
    folds = read(plan["cohort_parent_plan"])["folds"]

    def predict(model, uids, zero=False):
        output = {}; model.eval()
        with torch.inference_mode():
            for offset in range(0, len(uids), 2):
                batch = uids[offset:offset+2]
                masks = [torch.from_numpy(np.stack([getattr(packed[u], k) for u in batch]))
                         for k in ("slice_mask", "series_mask", "series_descriptors")]
                scores = (model.predict_proba_from_features(torch.zeros((len(batch), 16, 8, 512)), *masks)
                          if zero else model.predict_proba(torch.from_numpy(np.stack([packed[u].images for u in batch])), *masks))
                output.update({uid: value.tolist() for uid, value in zip(batch, scores.numpy(), strict=True)})
        return output

    for fold_row in folds:
        fold = fold_row["fold"]; held_all = set(manifest.held_out_study_uids(fold)); held = sorted(held_all & set(gold))
        train, reports = [fold_row[k] for k in ("gold_training_study_uids", "weak_training_study_uids")]
        require(set(train) == set(gold)-set(held) and not set(train+reports) & held_all,
                "Original training/held-out membership differs")
        labels = admit_ticket30_fold_supervision(manifest, fold, t36.TICKET30_ARTIFACT,
            expected_artifact_sha256=t36.TICKET30_AUDITED_ARTIFACT_SHA256).for_image_studies(list(rows))
        require(labels.to_public_dict() == fold_row["supervision_admission"], "Original audited supervision differs")
        exposures = {kind: dict(sample_exposures=0, supervised_entry_exposures=0)
                     for kind in ("reference", "weak", "mixed", "pseudo")}
        for attempt in range(7500):
            for kind, pool, seed in (("reference", train, 20260912), ("weak", reports, 20260912+34034034)):
                for index in _batch_indices(len(pool), 2, attempt, seed):
                    uid = pool[index]; mask = labels.by_study[uid].supervision_mask
                    require(any(mask), "Empty original training supervision")
                    exposures[kind]["sample_exposures"] += 1
                    exposures[kind]["supervised_entry_exposures"] += sum(mask)
        warm = initial_for_arm(fold, "full-backbone")
        initial = {k: v.clone() for k, v in warm.state_dict().items()}
        initial_sha = _model_state_sha256(warm)
        native_parent = predict(warm, held)
        parent = read(PARENT / f"report-supervision/fold-{fold}/run.json")["predictions"]
        parent_error = maximum_error(native_parent, parent)
        require(parent_error <= 1e-6, "Warm parent original-MRI prediction mismatch")
        parent_predictions.update(parent)
        native_parent_all.update(native_parent)
        del warm
        for stage in range(8):
            directory, stage_sha = stage_location(fold, stage)
            stage_record(fold, stage, request["fit_plan_sha256"])
            current = read(directory / "stage-result.json")
            if stage:
                previous_directory, _ = stage_location(fold, stage-1)
                previous = read(previous_directory / "stage-result.json")
                require(current["prior_checkpoint_sha256"] == previous["checkpoint_sha256"],
                        "Exact inherited/resumed checkpoint chain differs")
        for arm in ARMS:
            directory = (PRIOR if arm == "head-continuation" else FIT) / arm / f"fold-{fold}"
            record = read(directory / ("run.json" if arm == ARMS[0] else "stage-7/stage-result.json"))
            require(record["initial_model_state_sha256"] == initial_sha, "Actual matched initial tensors differ")
            checkpoint_path = final_checkpoint_path(arm, fold)
            require(Path(record["checkpoint_path"]) == checkpoint_path and digest(checkpoint_path) == record["checkpoint_sha256"],
                    "Final checkpoint identity differs")
            model = initial_for_arm(fold, arm)
            checkpoint = load_training_checkpoint(checkpoint_path, model=model, expected_config=config_for_arm(arm), expected_fold=fold,
                expected_identity=scientific_identity(plan, request["fit_plan_sha256"], fold, arm))
            progress = checkpoint["progress"]
            require(progress == record["progress"] and progress["successful_optimizer_steps"] == progress["attempted_batches"] == 7500
                    and progress["sample_exposures"] == 30000 and progress["exposures_by_source"] == exposures
                    and progress["skipped_all_masked_batches"] == 0
                    and progress["supervised_entry_exposures"] == sum(v["supervised_entry_exposures"] for v in exposures.values()),
                    "Exact original-source exposure budget differs")
            require(all(torch.isfinite(v).all() for v in model.state_dict().values()), "Nonfinite final model")
            optimizer = checkpoint["optimizer_state_dict"]
            rates = [group["lr"] for group in optimizer["param_groups"]]
            require(rates == ([1e-4] if arm == ARMS[0] else [1e-5, 1e-4])
                    and all(float(v["step"]) == 7500 and torch.isfinite(v["exp_avg"]).all()
                            and torch.isfinite(v["exp_avg_sq"]).all() for v in optimizer["state"].values()),
                    "Actual Adam state, rates or duration differ")
            state = model.state_dict()
            require(all(torch.equal(v, initial[k]) for k, v in state.items()
                        if k.endswith(("running_mean", "running_var", "num_batches_tracked"))), "Normalization buffers changed")
            if arm == ARMS[0]:
                require(all(torch.equal(v, initial[k]) for k, v in state.items() if k.startswith("encoder."))
                        and any(not torch.equal(v, initial[k]) for k, v in state.items() if k.startswith("head")),
                        "Control encoder changed or head did not change")
            else:
                require(all(any(not torch.equal(v, initial[k]) for k, v in state.items() if k.startswith(prefix))
                            for prefix in ("encoder.conv1", "encoder.layer1", "encoder.layer2", "encoder.layer3", "encoder.layer4", "head")),
                        "Every full-encoder stage must change")
            native = predict(model, held); zero = predict(model, held, True)
            errors = dict(native_prediction_error=maximum_error(native, record["predictions"]),
                          native_ablation_error=maximum_error(zero, record["encoded_image_ablation"]), parent_error=parent_error)
            require(max(errors.values()) <= 1e-6, "Original-MRI final prediction replay exceeds 1e-6")
            predicted[arm].update(record["predictions"]); ablated[arm].update(record["encoded_image_ablation"])
            native_all[arm].update(native); native_zero[arm].update(zero)
            audits.append(dict(arm=arm, fold=fold, checkpoint_sha256=digest(checkpoint_path), **errors))
            print(json.dumps(audits[-1]), flush=True)
            del model, checkpoint; gc.collect()
        del initial
    def compare(base, new):
        return compare_image_model_predictions(base, new, manifest, bootstrap_resamples=2000, bootstrap_seed=20261002)
    comparisons = dict(primary=compare(predicted[ARMS[0]], predicted[ARMS[1]]))
    for arm in ARMS:
        comparisons[f"{arm}-vs-parent"] = compare(parent_predictions, predicted[arm])
        comparisons[f"{arm}-image-signal"] = compare(ablated[arm], predicted[arm])
    analysis = read(FIT / "analysis.json")
    require(comparisons == analysis["comparisons"], "Independently recomputed registered comparisons differ")
    native_comparisons = dict(primary=compare(native_all[ARMS[0]], native_all[ARMS[1]]))
    for arm in ARMS:
        native_comparisons[f"{arm}-vs-parent"] = compare(native_parent_all, native_all[arm])
        native_comparisons[f"{arm}-image-signal"] = compare(native_zero[arm], native_all[arm])
    require(native_comparisons == comparisons, "Original-MRI prediction ranks change registered results")
    primary = comparisons["primary"]; lower = primary["paired_study_bootstrap"]["macro_auc"]["lower"]
    passed = (lower is not None and lower > 0
        and all(r["delta_macro_auc"] is not None and r["delta_macro_auc"] > 0 for r in primary["delta"]["folds"])
        and all(comparisons[f"{a}-image-signal"]["paired_study_bootstrap"]["macro_auc"]["lower"] is not None
                and comparisons[f"{a}-image-signal"]["paired_study_bootstrap"]["macro_auc"]["lower"] > 0 for a in ARMS))
    require(passed == analysis["provisional_review_gate_passed"], "Registered provisional gate differs")
    write(ROOT / "validation.json", dict(status="COMPLETE", job_id=os.environ["SLURM_JOB_ID"],
        plan_sha256=checksum, fit_plan_sha256=request["fit_plan_sha256"], original_mri_studies=58,
        final_runs_verified=6, final_predictions_verified=116, source_audit=source_audit, audits=audits,
        original_pixels_targets_masks_exposures_verified=True, all_registered_comparisons_recomputed=True,
        maximum_prediction_error=max(v[k] for v in audits for k in ("native_prediction_error", "native_ablation_error", "parent_error")),
        provisional_review_gate_passed=passed, candidate_promotion=False, first_place_established=False,
        historical_backend_terminals_verified=False, original_strict_ancestry_passed=False,
        original_missing_allocation_charge_seconds=39600,
        fitting=False, gpu_count=0, external_spend_usd=0,
        elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "prepare":
        prepare()
    elif action == "launch":
        launch(sys.argv[2])
    elif action == "run":
        run(sys.argv[2])
    else:
        raise ValueError("Unknown independent replay action")
