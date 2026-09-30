from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rsna_knee.folds import FoldManifest, freeze_folds
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.report_extraction import (
    REPORT_EXTRACTION_SCHEMA_VERSION,
    FrozenDeterministicExtractor,
)
from rsna_knee.supervision import (
    ADMISSION_SCHEMA_VERSION,
    MIXTURE_POLICY_VERSION,
    WEAK_LABEL_VERSION,
)
from rsna_knee.supervision_comparison import (
    SupervisionComparisonError,
    compare_supervision_oof,
    render_supervision_comparison_markdown,
    write_supervision_comparison,
)
from rsna_knee.targets import TARGET_COLUMNS


SOURCE_FINGERPRINT = "a" * 64
INTEGRATION_CHECKSUM = "b" * 64
REPORT_DIGEST = "c" * 64


def _manifest() -> FoldManifest:
    studies = tuple(
        [
            SourceStudy(
                f"gold-{index}",
                True,
                tuple((index + target_index) % 2 for target_index in range(len(TARGET_COLUMNS))),
            )
            for index in range(18)
        ]
        + [
            SourceStudy(f"report-{index}", True, (None,) * len(TARGET_COLUMNS))
            for index in range(18)
        ]
    )
    return freeze_folds(
        SourceManifest("train", studies, ()), source_version="comparison-fixture-v1", seed=31
    )


def test_ticket44_analysis_view_recovers_source_identity_without_changing_training(tmp_path: Path) -> None:
    import runpy
    from rsna_knee.oof import load_oof_run_record

    analysis_view = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/ticket44_completed_analysis.py"))["analysis_view"]
    source = tmp_path / "original"
    _write_run(source, _manifest(), 0, arm="gold")
    configuration = json.loads((source / "configuration.json").read_text())
    configuration["environment"].update(code_revision=None, slurm_job_id="7830")
    (source / "configuration.json").write_text(json.dumps(configuration))
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    derived = tmp_path / "analysis"
    supplement = analysis_view(source, derived, snapshot_sha256="a" * 64, job_id="7830")
    record = load_oof_run_record(derived)
    assert record.environment["code_revision"] == "verified-source-snapshot-sha256:" + "a" * 64
    assert record.predictions == load_oof_run_record(source).predictions
    assert supplement["git_revision_recovered"] is False
    assert {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()} == before


def test_ticket44_analysis_view_rejects_a_different_training_allocation(tmp_path: Path) -> None:
    import runpy

    analysis_view = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/ticket44_completed_analysis.py"))["analysis_view"]
    source = tmp_path / "original"
    _write_run(source, _manifest(), 0, arm="gold")
    configuration = json.loads((source / "configuration.json").read_text())
    configuration["environment"].update(code_revision=None, slurm_job_id="different-job")
    (source / "configuration.json").write_text(json.dumps(configuration))
    with pytest.raises(ValueError, match="missing-revision allocation"):
        analysis_view(source, tmp_path / "analysis", snapshot_sha256="a" * 64, job_id="7830")
    assert not (tmp_path / "analysis").exists()


def _training_config() -> dict[str, object]:
    return {
        "schema_version": "rsna-knee-training-v1",
        "model": {
            "image_size": [128, 128],
            "channels": [8, 16, 32],
            "input_channels": 1,
            "num_targets": len(TARGET_COLUMNS),
        },
        "packing": {"max_slices": 8, "max_series": 16},
        "preprocessing": {"schema_version": "rsna-knee-preprocess-v1"},
        "optimization": {
            "batch_size": 2,
            "successful_optimizer_steps": 100,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "schedule": "constant",
        },
        "seeds": {"initialization": 20260911, "batch_order": 20260912},
        "loss": "masked_bce",
        "augmentation": "none",
        "selection_policy": "final_checkpoint",
        "device": "cpu",
    }


def _reference_labels(manifest: FoldManifest) -> dict[str, tuple[int, ...]]:
    return {
        item.study_instance_uid: tuple(int(value) for value in item.reference_labels)
        for item in manifest.gold_memberships
    }


def _source_exposures(arm: str) -> dict[str, dict[str, int]]:
    if arm == "gold":
        return {
            "reference": {"sample_exposures": 200, "supervised_entry_exposures": 2400},
            "weak": {"sample_exposures": 0, "supervised_entry_exposures": 0},
            "mixed": {"sample_exposures": 0, "supervised_entry_exposures": 0},
        }
    return {
        "reference": {"sample_exposures": 100, "supervised_entry_exposures": 1200},
        "weak": {"sample_exposures": 100, "supervised_entry_exposures": 300},
        "mixed": {"sample_exposures": 0, "supervised_entry_exposures": 0},
    }


def _weak_supervision(manifest: FoldManifest, fold: int, train_uids: tuple[str, ...]) -> dict[str, object]:
    held_out = tuple(sorted(item.study_instance_uid for item in manifest.memberships if item.fold == fold))
    gold_labels = _reference_labels(manifest)
    report_uids = tuple(sorted(set(train_uids) - set(gold_labels)))
    weak_extractor = FrozenDeterministicExtractor().to_public_dict()
    resolved = []
    for uid in train_uids:
        if uid in gold_labels:
            targets: list[int | None] = list(gold_labels[uid])
            mask = [1] * len(TARGET_COLUMNS)
            source_kind = "reference"
            source_report_ref = None
            source_report_sha256 = None
        else:
            targets = [1, 0, 1] + [None] * (len(TARGET_COLUMNS) - 3)
            mask = [int(value is not None) for value in targets]
            source_kind = "weak"
            source_report_ref = f"private://synthetic-report/{uid}"
            source_report_sha256 = "d" * 64
        resolved.append(
            {
                "study_instance_uid": uid,
                "targets": targets,
                "supervision_mask": mask,
                "source_kind": source_kind,
                "source_report_ref": source_report_ref,
                "source_report_sha256": source_report_sha256,
                "supervised_entry_count": sum(mask),
                "disagreements": [],
            }
        )
    return {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "source_schema_version": REPORT_EXTRACTION_SCHEMA_VERSION,
        "fold": fold,
        "fold_manifest_checksum": manifest.manifest_checksum,
        "source_artifact_digest": REPORT_DIGEST,
        "weak_label_version": WEAK_LABEL_VERSION,
        "mixture_policy": MIXTURE_POLICY_VERSION,
        "extractor": {
            "version": weak_extractor["extractor_version"],
            "configuration_fingerprint": weak_extractor["configuration_fingerprint"],
            "ruleset_fingerprint": weak_extractor["ruleset_fingerprint"],
        },
        "training_study_uids": list(train_uids),
        "held_out_study_uids": list(held_out),
        "eligible_report_only_study_uids": list(report_uids),
        "admitted_report_only_study_uids": list(report_uids),
        "missing_report_only_study_uids": [],
        "excluded_image_report_only_study_uids": [],
        "resolved_studies": resolved,
        "source_disagreements": [],
        "counts": {
            "training_studies": len(train_uids),
            "reference_studies": len(train_uids) - len(report_uids),
            "weak_studies": len(report_uids),
            "eligible_report_only_studies": len(report_uids),
            "admitted_report_only_studies": len(report_uids),
            "missing_report_only_studies": 0,
            "image_excluded_report_only_studies": 0,
            "supervised_entries": sum(sum(item["supervision_mask"]) for item in resolved),
            "source_disagreements": 0,
        },
        "boundaries": {
            "held_out_inputs_excluded": True,
            "held_out_report_labels_evaluation_only": True,
            "reference_labels_take_precedence": True,
            "unknown_uncertain_failed_masked": True,
            "fallback_source": "trusted_reference_labels_only",
            "image_scope_restricted": False,
        },
    }


def _write_run(
    path: Path,
    manifest: FoldManifest,
    fold: int,
    *,
    arm: str,
    first_target_inverted: bool = False,
    config: dict[str, object] | None = None,
) -> None:
    path.mkdir(parents=True)
    training_uids = tuple(
        sorted(
            item.study_instance_uid
            for item in manifest.memberships
            if item.fold != fold and (arm == "report" or item.study_instance_uid in _reference_labels(manifest))
        )
    )
    held_out = tuple(
        sorted(item.study_instance_uid for item in manifest.gold_memberships if item.fold == fold)
    )
    labels = _reference_labels(manifest)
    predictions: dict[str, list[float]] = {}
    for uid in held_out:
        vector = list(float(value) for value in labels[uid])
        if arm == "report" and first_target_inverted:
            vector[0] = 1.0 - vector[0]
        predictions[uid] = vector

    run_id = f"{arm}-run-fold-{fold}"
    run_config = config or _training_config()
    final_checkpoint = f"final-checkpoint-{arm}-{fold}".encode()
    final_sha256 = hashlib.sha256(final_checkpoint).hexdigest()
    (path / "checkpoints").mkdir()
    (path / "checkpoints" / "final.pt").write_bytes(final_checkpoint)
    (path / "checkpoints" / "latest.pt").write_bytes(b"latest-checkpoint")

    gold_uids = set(_reference_labels(manifest))
    gold_training_uids = tuple(uid for uid in training_uids if uid in gold_uids)
    report_training_uids = tuple(uid for uid in training_uids if uid not in gold_uids)
    supervision = (
        _weak_supervision(manifest, fold, training_uids)
        if arm == "report"
        else {
            "kind": "reference",
            "policy": "reference-only-v1",
            "fold_manifest_checksum": manifest.manifest_checksum,
            "counts": {
                "training_studies": len(training_uids),
                "reference_studies": len(training_uids),
                "weak_studies": 0,
                "supervised_entries": len(training_uids) * len(TARGET_COLUMNS),
                "source_disagreements": 0,
            },
            "held_out_inputs_excluded": True,
            "reports_used": False,
        }
    )
    configuration_supervision = (
        {
            "kind": "reference",
            "fold_manifest_checksum": manifest.manifest_checksum,
            "training_studies": list(training_uids),
            "held_out_studies": list(held_out),
            "held_out_inputs_excluded": True,
            "reports_used": False,
        }
        if arm == "gold"
        else supervision
    )
    exposure_counts = _source_exposures(arm)
    counts = {
        "training_studies": len(training_uids),
        "held_out_studies": len(held_out),
        "successful_optimizer_steps": 100,
        "scheduler_steps": 100,
        "attempted_batches": 100,
        "skipped_all_masked_batches": 0,
        "sample_exposures": sum(item["sample_exposures"] for item in exposure_counts.values()),
        "supervised_entry_exposures": sum(
            item["supervised_entry_exposures"] for item in exposure_counts.values()
        ),
        "exposures_by_source": exposure_counts,
    }
    identity = {
        "source_version": manifest.source_version,
        "source_fingerprint": SOURCE_FINGERPRINT,
        "cache_fingerprint": "not-used",
        "fold_manifest_checksum": manifest.manifest_checksum,
        "integration_manifest_checksum": INTEGRATION_CHECKSUM,
        "data_fingerprint": f"synthetic-{arm}-data-{fold}",
        "label_source": "reference-only:frozen-fold-manifest",
        "model_source": "local:synthetic",
        "training_schema_version": "rsna-knee-training-v1",
    }
    (path / "configuration.json").write_text(
        json.dumps(
            {
                "schema_version": "rsna-knee-training-v1",
                "run_id": run_id,
                "fold": fold,
                "config": run_config,
                "scientific_configuration_fingerprint": "same-training-config-v1",
                "identity": identity,
                "environment": {
                    "code_revision": "synthetic-revision",
                    "environment_fingerprint": "synthetic-environment",
                    "python_executable": ".venv/bin/python",
                    "python_version": "3.11",
                },
                "resources": {
                    "requested_partition": "cluster1-cpu",
                    "requested_nodes": 1,
                    "requested_cpus": 2,
                    "requested_memory_gb": 8,
                    "requested_wall_time_hours": 1,
                    "exclusive": False,
                    "slurm_attempt_id": str(7100 + fold * 2 + (arm == "report")),
                },
                "inference_configuration": {"checkpoint_selection": "final_checkpoint"},
                "supervision": configuration_supervision,
                "preflight": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    (path / "run.json").write_text(
        json.dumps(
            {
                "schema_version": "rsna-knee-training-v1",
                "status": "completed_fixed_budget_training",
                "run_id": run_id,
                "fold": fold,
                "config_fingerprint": "same-training-config-v1",
                "config": run_config,
                "reference_supervision_only": arm == "gold",
                "held_out_inputs_excluded_from_fitting": True,
                "training_study_uids": list(training_uids),
                "held_out_study_uids": list(held_out),
                "counts": counts,
                "checkpoint": {
                    "final_present": True,
                    "final_sha256": final_sha256,
                    "latest_present": True,
                },
                "predictions": predictions,
                "supervision": supervision,
            }
        ),
        encoding="utf-8",
    )
    (path / "metrics.json").write_text(
        json.dumps({"fold": fold, "prediction_count": len(predictions)}), encoding="utf-8"
    )
    path.with_name(f"{path.name}.compute.json").write_text(
        json.dumps(
            {
                "schema_version": "rsna-knee-observed-compute-v1",
                "elapsed_seconds": 12.5,
                "user_cpu_seconds": 10.0,
                "system_cpu_seconds": 1.5,
                "peak_rss_kb": 4096,
            }
        ),
        encoding="utf-8",
    )
    if arm == "report":
        public_admission = {field: supervision[field] for field in (
            "schema_version",
            "fold",
            "fold_manifest_checksum",
            "source_artifact_digest",
            "source_schema_version",
            "weak_label_version",
            "mixture_policy",
            "extractor",
            "counts",
            "boundaries",
        )}
        (path.parent / f"{path.name}.public.json").write_text(
            json.dumps(
                {
                    "ticket": "17",
                    "fold": fold,
                    "fold_manifest_checksum": manifest.manifest_checksum,
                    "preflight_passed": True,
                    "report_extraction_semantic_digest": REPORT_DIGEST,
                    "frozen_report_artifact_sha256": "e" * 64,
                    "supervision_admission": public_admission,
                    "full_fold_supervision_admission": public_admission,
                }
            ),
            encoding="utf-8",
        )


def _write_arms(
    tmp_path: Path, manifest: FoldManifest, *, first_target_inverted: bool = True
) -> tuple[list[Path], list[Path]]:
    gold_paths: list[Path] = []
    report_paths: list[Path] = []
    for fold in (0, 1, 2):
        gold_path = tmp_path / "gold" / f"fold-{fold}"
        report_path = tmp_path / "report" / f"fold-{fold}"
        gold_path.parent.mkdir(exist_ok=True)
        report_path.parent.mkdir(exist_ok=True)
        _write_run(gold_path, manifest, fold, arm="gold")
        _write_run(
            report_path,
            manifest,
            fold,
            arm="report",
            first_target_inverted=first_target_inverted,
        )
        gold_paths.append(gold_path)
        report_paths.append(report_path)
    return gold_paths, report_paths


def test_comparison_reports_paired_oof_delta_uncertainty_and_exposure(tmp_path: Path) -> None:
    manifest = _manifest()
    gold_paths, report_paths = _write_arms(tmp_path, manifest)

    result = compare_supervision_oof(
        gold_paths,
        report_paths,
        manifest,
        bootstrap_resamples=64,
        bootstrap_seed=7,
    )

    assert result.gold_only.prediction_count == 18
    assert result.report_supervised.prediction_count == 18
    assert result.pooled_macro_auc_delta == pytest.approx(-1 / 12)
    assert result.pooled_targets[0].delta_auc == pytest.approx(-1.0)
    assert result.pooled_targets[1].delta_auc == pytest.approx(0.0)
    assert result.macro_auc_delta_interval.lower == pytest.approx(-1 / 12)
    assert result.macro_auc_delta_interval.upper == pytest.approx(-1 / 12)
    assert len(result.folds) == 3
    assert all(item.macro_auc_delta == pytest.approx(-1 / 12) for item in result.folds)
    assert result.folds[0].report_supervised_exposure.exposures_by_source["weak"]["sample_exposures"] == 100
    assert result.folds[0].gold_only_exposure.exposures_by_source["reference"]["supervised_entry_exposures"] == 2400
    assert result.report_semantic_digest == REPORT_DIGEST
    assert "negative" in result.interpretation

    public = json.dumps(result.to_public_dict(), sort_keys=True)
    markdown = render_supervision_comparison_markdown(result)
    assert all(uid not in public and uid not in markdown for uid in ("gold-0", "report-0"))
    assert "paired study bootstrap" in markdown.lower()
    assert "development cv" in markdown.lower()
    assert result.weak_label_version in markdown
    assert result.mixture_policy in markdown
    assert "Slurm compute requested" in markdown
    assert "cluster1-cpu" in markdown
    assert "8 GB" in markdown
    assert "100 / 100 / 0" in markdown
    assert "12.5 s / 10.0 s / 1.5 s" in markdown
    assert "4096 KB" in markdown
    assert "Reference sample / entry exposures" in markdown
    assert "Weak sample / entry exposures" in markdown
    assert "| 0 | Gold-only | 12 (12 + 0) | 200 / 2400 | 0 / 0 | 0 / 0 |" in markdown
    assert "| 0 | Gold + clean reports | 24 (12 + 12) | 100 / 1200 | 100 / 300 | 0 / 0 |" in markdown


def test_additive_comparison_requires_exact_gold_exposure_and_initialization(tmp_path: Path) -> None:
    manifest = _manifest()
    gold_paths, report_paths = _write_arms(tmp_path, manifest)

    for fold, gold_path, report_path in zip(range(3), gold_paths, report_paths):
        gold_run_path = gold_path / "run.json"
        report_run_path = report_path / "run.json"
        gold_run = json.loads(gold_run_path.read_text(encoding="utf-8"))
        report_run = json.loads(report_run_path.read_text(encoding="utf-8"))
        gold_reference = gold_run["counts"]["exposures_by_source"]["reference"]
        report_exposures = report_run["counts"]["exposures_by_source"]
        report_exposures["reference"] = dict(gold_reference)
        report_exposures["weak"]["sample_exposures"] = 200
        report_run["counts"]["sample_exposures"] = 400
        report_run["counts"]["supervised_entry_exposures"] = (
            report_exposures["reference"]["supervised_entry_exposures"]
            + report_exposures["weak"]["supervised_entry_exposures"]
        )
        initial_hash = hashlib.sha256(f"initial-{fold}".encode()).hexdigest()
        gold_sequence = hashlib.sha256(f"gold-sequence-{fold}".encode()).hexdigest()
        weak_sequence = hashlib.sha256(f"weak-sequence-{fold}".encode()).hexdigest()
        for run, weak_hash in ((gold_run, None), (report_run, weak_sequence)):
            run["counts"]["training_design"] = "ticket18-additive-report-supervision-v2"
            run["counts"]["gold_exposure_sequence_sha256"] = gold_sequence
            run["counts"]["weak_exposure_sequence_sha256"] = weak_hash
            run["counts"]["initial_model_state_sha256"] = initial_hash
        report_run["counts"]["weak_sampling_policy"] = "ticket18-report-only-cyclic-shuffle-v1"
        report_run["counts"]["weak_sampling_seed"] = 38_440_930
        gold_run_path.write_text(json.dumps(gold_run), encoding="utf-8")
        report_run_path.write_text(json.dumps(report_run), encoding="utf-8")
        report_public_path = report_path.parent / f"{report_path.name}.public.json"
        report_public = json.loads(report_public_path.read_text(encoding="utf-8"))
        report_public["ticket"] = "18"
        report_public_path.write_text(json.dumps(report_public), encoding="utf-8")

    result = compare_supervision_oof(
        gold_paths,
        report_paths,
        manifest,
        bootstrap_resamples=16,
        bootstrap_seed=19,
        require_additive_gold_exposure=True,
    )

    assert result.additive_design is True
    assert result.folds[0].gold_only_exposure.gold_exposure_sequence_sha256 == (
        result.folds[0].report_supervised_exposure.gold_exposure_sequence_sha256
    )
    assert "adds one deterministic two-study weak microbatch" in render_supervision_comparison_markdown(
        result
    )

    report_run_path = report_paths[0] / "run.json"
    report_run = json.loads(report_run_path.read_text(encoding="utf-8"))
    report_run["counts"]["gold_exposure_sequence_sha256"] = "f" * 64
    report_run_path.write_text(json.dumps(report_run), encoding="utf-8")
    with pytest.raises(ValueError, match="ordered gold exposure sequence differs"):
        compare_supervision_oof(
            gold_paths,
            report_paths,
            manifest,
            bootstrap_resamples=16,
            bootstrap_seed=19,
            require_additive_gold_exposure=True,
        )


def test_comparison_counts_folds_with_zero_macro_delta(tmp_path: Path) -> None:
    manifest = _manifest()
    gold_paths, report_paths = _write_arms(
        tmp_path, manifest, first_target_inverted=False
    )

    result = compare_supervision_oof(
        gold_paths, report_paths, manifest, bootstrap_resamples=16
    )

    assert result.positive_folds == 0
    assert result.negative_folds == 0
    assert result.unchanged_folds == 3
    assert result.undefined_folds == 0
    assert result.to_public_dict()["paired_delta"]["unchanged_folds"] == 3
    assert "Fold deltas: 0 positive, 0 negative, 3 unchanged, 0 undefined." in (
        render_supervision_comparison_markdown(result)
    )


@pytest.mark.parametrize(
    ("confidence_level", "label"),
    [
        (0.90, "Paired 90% interval"),
        (0.9999999999999999, "Paired 99.999999999999986% interval"),
    ],
)
def test_comparison_markdown_uses_configured_confidence_level(
    tmp_path: Path, confidence_level: float, label: str
) -> None:
    manifest = _manifest()
    gold_paths, report_paths = _write_arms(tmp_path, manifest)

    result = compare_supervision_oof(
        gold_paths,
        report_paths,
        manifest,
        bootstrap_resamples=16,
        bootstrap_confidence_level=confidence_level,
    )

    markdown = render_supervision_comparison_markdown(result)
    assert label in markdown
    assert "Paired 100% interval" not in markdown


def test_comparison_reports_mean_fold_macro_auc_and_variation(tmp_path: Path) -> None:
    manifest = _manifest()
    gold_paths, report_paths = _write_arms(tmp_path, manifest)
    result = compare_supervision_oof(gold_paths, report_paths, manifest, bootstrap_resamples=16)

    report = result.to_public_dict()
    assert report["gold_only"]["fold_macro_auc"] == {
        "mean": result.gold_only.mean_fold_macro_auc,
        "standard_deviation": result.gold_only.fold_macro_auc_std,
        "minimum": result.gold_only.fold_macro_auc_min,
        "maximum": result.gold_only.fold_macro_auc_max,
    }
    assert report["report_supervised"]["fold_macro_auc"]["standard_deviation"] == (
        result.report_supervised.fold_macro_auc_std
    )
    markdown = render_supervision_comparison_markdown(result)
    assert "Mean fold macro AUC" in markdown
    assert "Fold macro AUC standard deviation" in markdown
    assert "Minimum fold macro AUC" in markdown
    assert "Maximum fold macro AUC" in markdown
    assert (
        f"| Gold-only | {result.gold_only.pooled_macro_auc:.4f} | "
        f"{result.gold_only.mean_fold_macro_auc:.4f} | "
        f"{result.gold_only.fold_macro_auc_std:.4f} | "
        f"{result.gold_only.fold_macro_auc_min:.4f} | "
        f"{result.gold_only.fold_macro_auc_max:.4f} |"
    ) in markdown


def test_comparison_rejects_reference_configuration_count_disagreement(tmp_path: Path) -> None:
    manifest = _manifest()
    gold_paths, report_paths = _write_arms(tmp_path, manifest)
    configuration_path = gold_paths[0] / "configuration.json"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    training_count = len(configuration["supervision"]["training_studies"])
    configuration["supervision"]["counts"] = {
        "training_studies": training_count,
        "reference_studies": training_count,
        "weak_studies": 1,
        "supervised_entries": training_count * len(TARGET_COLUMNS),
        "source_disagreements": 0,
    }
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")

    with pytest.raises(SupervisionComparisonError, match="reference supervision counts"):
        compare_supervision_oof(gold_paths, report_paths, manifest, bootstrap_resamples=16)


def test_comparison_rejects_training_config_mismatch_across_arms(tmp_path: Path) -> None:
    manifest = _manifest()
    gold_paths, report_paths = _write_arms(tmp_path, manifest)
    run_path = report_paths[1] / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["config"]["optimization"]["learning_rate"] = 0.002
    run_path.write_text(json.dumps(run), encoding="utf-8")
    configuration_path = report_paths[1] / "configuration.json"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    configuration["config"]["optimization"]["learning_rate"] = 0.002
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")

    with pytest.raises(SupervisionComparisonError, match="training configuration"):
        compare_supervision_oof(gold_paths, report_paths, manifest, bootstrap_resamples=16)


def test_comparison_rejects_weak_lineage_that_includes_held_out_studies(tmp_path: Path) -> None:
    manifest = _manifest()
    gold_paths, report_paths = _write_arms(tmp_path, manifest)
    config_path = report_paths[0] / "configuration.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["supervision"]["boundaries"]["held_out_inputs_excluded"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(SupervisionComparisonError, match="held-out"):
        compare_supervision_oof(gold_paths, report_paths, manifest, bootstrap_resamples=16)


def test_comparison_rejects_reference_partition_disagreement(tmp_path: Path) -> None:
    manifest = _manifest()
    gold_paths, report_paths = _write_arms(tmp_path, manifest)
    configuration_path = gold_paths[0] / "configuration.json"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    configuration["supervision"]["training_studies"].pop()
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")

    with pytest.raises(SupervisionComparisonError, match="partitions disagree"):
        compare_supervision_oof(gold_paths, report_paths, manifest, bootstrap_resamples=16)


@pytest.mark.parametrize("arm", ["gold", "report"])
def test_comparison_rejects_supervision_manifest_checksum_disagreement(
    tmp_path: Path, arm: str
) -> None:
    manifest = _manifest()
    gold_paths, report_paths = _write_arms(tmp_path, manifest)
    run_path = (gold_paths if arm == "gold" else report_paths)[0] / "run.json"
    configuration_path = run_path.parent / "configuration.json"
    incorrect_checksum = "0" * 64

    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    run_payload["supervision"]["fold_manifest_checksum"] = incorrect_checksum
    run_path.write_text(json.dumps(run_payload), encoding="utf-8")

    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    configuration["supervision"]["fold_manifest_checksum"] = incorrect_checksum
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")

    with pytest.raises(SupervisionComparisonError, match="supervision uses a different fold manifest"):
        compare_supervision_oof(gold_paths, report_paths, manifest, bootstrap_resamples=16)


def test_comparison_rejects_weak_exposure_counts_that_do_not_reconcile(tmp_path: Path) -> None:
    manifest = _manifest()
    gold_paths, report_paths = _write_arms(tmp_path, manifest)
    run_path = report_paths[0] / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["counts"]["exposures_by_source"]["weak"]["sample_exposures"] += 1
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(SupervisionComparisonError, match="exposure"):
        compare_supervision_oof(gold_paths, report_paths, manifest, bootstrap_resamples=16)


def test_comparison_rejects_report_arm_without_report_label_exposure(tmp_path: Path) -> None:
    manifest = _manifest()
    gold_paths, report_paths = _write_arms(tmp_path, manifest)
    run_path = report_paths[0] / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["counts"]["exposures_by_source"]["weak"]["supervised_entry_exposures"] = 0
    run["counts"]["supervised_entry_exposures"] -= 300
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(SupervisionComparisonError, match="no report-derived supervised-entry exposure"):
        compare_supervision_oof(gold_paths, report_paths, manifest, bootstrap_resamples=16)


def test_comparison_requires_observed_compute_evidence(tmp_path: Path) -> None:
    manifest = _manifest()
    gold_paths, report_paths = _write_arms(tmp_path, manifest)
    gold_paths[0].with_name(f"{gold_paths[0].name}.compute.json").unlink()

    with pytest.raises(SupervisionComparisonError, match="observed compute evidence"):
        compare_supervision_oof(gold_paths, report_paths, manifest, bootstrap_resamples=16)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("elapsed_seconds", 3601, "observed runtime exceeds"),
        ("peak_rss_kb", 8 * 1024 * 1024 + 1, "observed peak RSS exceeds"),
        ("user_cpu_seconds", 100, "observed CPU time exceeds"),
    ],
)
def test_comparison_rejects_observed_compute_over_requested_limits(
    tmp_path: Path,
    field: str,
    value: int,
    message: str,
) -> None:
    manifest = _manifest()
    gold_paths, report_paths = _write_arms(tmp_path, manifest)
    compute_path = report_paths[0].with_name(f"{report_paths[0].name}.compute.json")
    compute = json.loads(compute_path.read_text(encoding="utf-8"))
    compute[field] = value
    compute_path.write_text(json.dumps(compute), encoding="utf-8")

    with pytest.raises(SupervisionComparisonError, match=message):
        compare_supervision_oof(gold_paths, report_paths, manifest, bootstrap_resamples=16)


def test_comparison_writer_keeps_source_runs_immutable_and_rejects_in_run_outputs(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    gold_paths, report_paths = _write_arms(tmp_path, manifest)
    result = compare_supervision_oof(gold_paths, report_paths, manifest, bootstrap_resamples=16)
    before = [(path / "run.json").read_bytes() for path in gold_paths + report_paths]

    public_path = tmp_path / "public" / "comparison.json"
    markdown_path = tmp_path / "public" / "comparison.md"
    write_supervision_comparison(
        result,
        public_path=public_path,
        markdown_path=markdown_path,
        source_run_paths=gold_paths + report_paths,
    )

    assert public_path.exists()
    assert markdown_path.exists()
    assert before == [(path / "run.json").read_bytes() for path in gold_paths + report_paths]
    assert all(uid not in public_path.read_text() for uid in ("gold-0", "report-0"))
    with pytest.raises(SupervisionComparisonError, match="inside a source run"):
        write_supervision_comparison(
            result,
            public_path=gold_paths[0] / "comparison.json",
            markdown_path=tmp_path / "public" / "other.md",
            source_run_paths=gold_paths + report_paths,
        )
