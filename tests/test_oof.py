from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rsna_knee.folds import FoldManifest, freeze_folds
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.oof import (
    OOFArtifactError,
    aggregate_oof_run_records,
    render_oof_comparison,
    write_oof_reports,
)
from rsna_knee.targets import TARGET_COLUMNS


def _manifest() -> FoldManifest:
    source = SourceManifest(
        "train",
        tuple(
            SourceStudy(
                f"gold-{index}",
                True,
                tuple((index + target_index) % 2 for target_index in range(len(TARGET_COLUMNS))),
            )
            for index in range(12)
        ),
        (),
    )
    return freeze_folds(source, source_version="oof-artifact-v1", seed=31)


def _write_run(path: Path, manifest: FoldManifest, fold: int, *, config: str = "config-v1") -> None:
    held_out = tuple(sorted(manifest.held_out_study_uids(fold)))
    predictions = {
        uid: [float(label) for label in next(
            item.reference_labels
            for item in manifest.gold_memberships
            if item.study_instance_uid == uid
        )]
        for uid in held_out
    }
    path.mkdir()
    run_id = f"run-fold-{fold}"
    run_config = {
        "model": {"name": "synthetic"},
        "preprocessing": {"name": "synthetic"},
        "packing": {"name": "synthetic"},
        "optimization": {"name": "synthetic"},
        "loss": "masked_bce",
        "seeds": {"initialization": 1, "batch_order": 2},
    }
    final_checkpoint = b"synthetic-final-checkpoint"
    (path / "checkpoints").mkdir()
    (path / "checkpoints" / "final.pt").write_bytes(final_checkpoint)
    (path / "checkpoints" / "latest.pt").write_bytes(b"synthetic-latest-checkpoint")
    identity = {
        "source_version": manifest.source_version,
        "source_fingerprint": "synthetic-source-fingerprint",
        "cache_fingerprint": "synthetic-cache-fingerprint",
        "fold_manifest_checksum": manifest.manifest_checksum,
        "integration_manifest_checksum": "synthetic-integration-manifest",
        "data_fingerprint": f"synthetic-data-{fold}",
        "label_source": "reference-only:frozen-fold-manifest",
        "model_source": "local:synthetic",
        "training_schema_version": "rsna-knee-training-v1",
    }
    supervision = {
        "kind": "reference",
        "fold_manifest_checksum": manifest.manifest_checksum,
        "held_out_inputs_excluded": True,
        "reports_used": False,
    }
    (path / "configuration.json").write_text(
        json.dumps(
            {
                "schema_version": "rsna-knee-training-v1",
                "run_id": run_id,
                "fold": fold,
                "config": run_config,
                "scientific_configuration_fingerprint": config,
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
                    "requested_cpus": 16,
                    "requested_memory_gb": 64,
                    "requested_wall_time_hours": 6,
                    "exclusive": False,
                },
                "inference_configuration": {"checkpoint_selection": "final_checkpoint"},
                "supervision": supervision,
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
                "config_fingerprint": config,
                "config": run_config,
                "reference_supervision_only": True,
                "held_out_inputs_excluded_from_fitting": True,
                "training_study_uids": list(manifest.training_study_uids(fold)),
                "held_out_study_uids": list(held_out),
                "counts": {
                    "training_studies": len(manifest.training_study_uids(fold)),
                    "held_out_studies": len(held_out),
                    "successful_optimizer_steps": 100,
                    "scheduler_steps": 100,
                    "attempted_batches": 100,
                    "skipped_all_masked_batches": 0,
                    "sample_exposures": 200,
                    "supervised_entry_exposures": 2400,
                    "exposures_by_source": {
                        "reference": {
                            "sample_exposures": 200,
                            "supervised_entry_exposures": 2400,
                        },
                        "weak": {"sample_exposures": 0, "supervised_entry_exposures": 0},
                        "mixed": {"sample_exposures": 0, "supervised_entry_exposures": 0},
                    },
                },
                "supervision": supervision,
                "checkpoint": {
                    "final_present": True,
                    "final_sha256": hashlib.sha256(final_checkpoint).hexdigest(),
                    "latest_present": True,
                },
                "predictions": predictions,
            }
        ),
        encoding="utf-8",
    )
    (path / "metrics.json").write_text(
        json.dumps({"fold": fold, "prediction_count": len(predictions)}),
        encoding="utf-8",
    )


def test_aggregate_oof_run_records_recomputes_complete_three_fold_view(tmp_path: Path) -> None:
    manifest = _manifest()
    paths = []
    for fold in (0, 1, 2):
        path = tmp_path / f"fold-{fold}"
        _write_run(path, manifest, fold)
        paths.append(path)

    result = aggregate_oof_run_records(paths, manifest, bootstrap_resamples=32, bootstrap_seed=7)

    assert result.prediction_count == 12
    assert result.pooled_macro_auc == 1.0
    assert result.repeat_name == "repeat-1"
    assert result.run_ids == ("run-fold-0", "run-fold-1", "run-fold-2")
    assert result.bootstrap is not None


def test_aggregate_oof_run_records_rejects_semantic_config_mismatch(tmp_path: Path) -> None:
    manifest = _manifest()
    paths = []
    for fold in (0, 1, 2):
        path = tmp_path / f"fold-{fold}"
        _write_run(path, manifest, fold, config="config-a" if fold != 2 else "config-b")
        paths.append(path)

    with pytest.raises(OOFArtifactError, match="configuration fingerprint mismatch"):
        aggregate_oof_run_records(paths, manifest)


def test_load_oof_run_record_requires_complete_artifact_contract(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "fold-0"
    _write_run(path, manifest, 0)
    (path / "configuration.json").unlink()

    with pytest.raises(OOFArtifactError, match="missing configuration.json"):
        aggregate_oof_run_records([path], manifest)


def test_aggregate_oof_run_records_rejects_prediction_outside_held_out_fold(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    paths = []
    for fold in (0, 1, 2):
        path = tmp_path / f"fold-{fold}"
        _write_run(path, manifest, fold)
        paths.append(path)
    fold_zero_path = paths[0] / "run.json"
    payload = json.loads(fold_zero_path.read_text(encoding="utf-8"))
    wrong_uid = next(iter(manifest.held_out_study_uids(1)))
    payload["predictions"] = {wrong_uid: [0.5] * len(TARGET_COLUMNS)}
    fold_zero_path.write_text(json.dumps(payload), encoding="utf-8")
    metrics_path = paths[0] / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["prediction_count"] = 1
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(OOFArtifactError, match="predictions disagree with held-out membership"):
        aggregate_oof_run_records(paths, manifest)


def test_write_oof_reports_does_not_modify_source_run_records(tmp_path: Path) -> None:
    manifest = _manifest()
    paths = []
    before = []
    for fold in (0, 1, 2):
        path = tmp_path / f"fold-{fold}"
        _write_run(path, manifest, fold)
        run_path = path / "run.json"
        before.append(run_path.read_bytes())
        paths.append(path)
    result = aggregate_oof_run_records(paths, manifest, bootstrap_resamples=16, bootstrap_seed=3)

    public_path = tmp_path / "public" / "oof-evidence.json"
    comparison_path = tmp_path / "public" / "comparison.md"
    write_oof_reports(
        result,
        public_path=public_path,
        comparison_path=comparison_path,
        source_run_paths=paths,
    )

    assert public_path.exists()
    assert comparison_path.exists()
    assert all((path / "run.json").read_bytes() == original for path, original in zip(paths, before))
    assert "gold-" not in public_path.read_text(encoding="utf-8")


def test_write_oof_reports_rejects_source_run_destination(tmp_path: Path) -> None:
    manifest = _manifest()
    paths = []
    for fold in (0, 1, 2):
        path = tmp_path / f"fold-{fold}"
        _write_run(path, manifest, fold)
        paths.append(path)
    result = aggregate_oof_run_records(paths, manifest)

    with pytest.raises(OOFArtifactError, match="inside a source run directory"):
        write_oof_reports(
            result,
            public_path=paths[0] / "run.json",
            source_run_paths=paths,
        )


def test_aggregation_refuses_superseded_v1_manifest_unless_replaying_history(tmp_path: Path) -> None:
    source = SourceManifest(
        "train",
        tuple(
            SourceStudy(
                f"gold-{index}",
                True,
                tuple((index + target_index) % 2 for target_index in range(len(TARGET_COLUMNS))),
            )
            for index in range(12)
        ),
        (),
    )
    legacy = freeze_folds(source, source_version="oof-artifact-v1", seed=31, generation_version=1)
    paths = []
    for fold in (0, 1, 2):
        path = tmp_path / f"fold-{fold}"
        _write_run(path, legacy, fold)
        paths.append(path)

    with pytest.raises(OOFArtifactError, match="superseded version-1 generator"):
        aggregate_oof_run_records(paths, legacy)
    historical = aggregate_oof_run_records(paths, legacy, allow_superseded_fold_manifest=True)

    assert historical.fold_manifest_superseded is True
    assert historical.to_public_dict()["fold_manifest_superseded"] is True


def test_comparison_view_reports_within_fold_baseline_and_score_scale(tmp_path: Path) -> None:
    manifest = _manifest()
    paths = []
    for fold in (0, 1, 2):
        path = tmp_path / f"fold-{fold}"
        _write_run(path, manifest, fold)
        paths.append(path)

    result = aggregate_oof_run_records(
        paths,
        manifest,
        bootstrap_resamples=16,
        bootstrap_seed=3,
        permutation_resamples=200,
        permutation_seed=4,
    )
    comparison = render_oof_comparison(result)
    public = result.to_public_dict()

    assert result.diagnostics is not None
    assert result.diagnostics.within_fold_macro_auc == 1.0
    assert public["fold_manifest_superseded"] is False
    assert public["diagnostics"]["within_fold_permutation_test"]["resamples"] == 200
    assert "Pair-weighted within-fold macro ROC-AUC: `1.000000`" in comparison
    assert "Training-prevalence-only pooled macro ROC-AUC" in comparison
    assert "## Fold score scale" in comparison
    assert "measurable image signal: `true`" in comparison
    assert "gold-" not in comparison
