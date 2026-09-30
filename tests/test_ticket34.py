from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rsna_knee.folds import FOLD_IDS, FoldManifest, freeze_folds
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.pretrained import (
    RESNET18_ARTIFACT_BYTES,
    RESNET18_DOWNLOAD_URL,
    RESNET18_SHA256,
    RESNET18_SOURCE_REVISION,
)
from rsna_knee.targets import NUM_TARGETS, TARGET_COLUMNS
from rsna_knee.ticket34 import (
    TICKET30_ADMISSION_POLICY_SHA256,
    TICKET30_ADMISSION_POLICY_VERSION,
    TICKET30_AUDIT_SCHEMA_VERSION,
    TICKET30_SOURCE_ID,
    TICKET34_ADDITIVE_DESIGN,
    TICKET34_MIXTURE_POLICY,
    TICKET34_RESOURCE_SCOPE,
    TICKET34_WEAK_BATCH_SEED_OFFSET,
    TICKET34_WEAK_LABEL_VERSION,
    Ticket34Error,
    admit_ticket30_fold_supervision,
    compare_supervision_pilot,
)


def _manifest() -> FoldManifest:
    gold = tuple(
        SourceStudy(
            f"gold-{index}",
            True,
            tuple((index + target) % 2 for target in range(NUM_TARGETS)),
        )
        for index in range(9)
    )
    report_only = tuple(
        SourceStudy(f"report-{index}", True, (None,) * NUM_TARGETS)
        for index in range(6)
    )
    return freeze_folds(
        SourceManifest("train", gold + report_only, ()),
        source_version="ticket34-admission-fixture-v1",
        seed=17,
    )


def _ticket30_payload(manifest: FoldManifest) -> dict[str, object]:
    studies_by_fold: list[dict[str, object]] = []
    for fold in FOLD_IDS:
        eligible = [
            item for item in manifest.report_only_memberships if item.fold != fold
        ]
        rows: list[dict[str, object]] = []
        admitted_studies = 0
        admitted_calls = 0
        for index, item in enumerate(eligible):
            labels: list[dict[str, object]] = []
            for target_index, target in enumerate(TARGET_COLUMNS):
                if target_index == 0:
                    call = {
                        "target": target,
                        "state": "positive",
                        "value": 1,
                        "supervision_mask": int(index == 0),
                        "admitted": index == 0,
                        "rejection_reason": None if index == 0 else "policy_threshold_not_met",
                    }
                    if index == 0:
                        admitted_calls += 1
                elif target_index == 1:
                    call = {
                        "target": target,
                        "state": "uncertain",
                        "value": None,
                        "supervision_mask": 0,
                        "admitted": False,
                        "rejection_reason": "state_is_always_masked",
                    }
                else:
                    call = {
                        "target": target,
                        "state": "not-mentioned",
                        "value": None,
                        "supervision_mask": 0,
                        "admitted": False,
                        "rejection_reason": "state_is_always_masked",
                    }
                labels.append(call)
            any_admitted = any(bool(item["admitted"]) for item in labels)
            admitted_studies += int(any_admitted)
            rows.append(
                {
                    "study_instance_uid": item.study_instance_uid,
                    "assigned_fold": item.fold,
                    "consuming_fold": fold,
                    "report_sha256": hashlib.sha256(item.study_instance_uid.encode()).hexdigest(),
                    "targets": labels,
                }
            )
        studies_by_fold.append(
            {
                "fold": fold,
                "eligible_report_only_study_count": len(eligible),
                "admitted_studies_with_any_call": admitted_studies,
                "admitted_target_call_count": admitted_calls,
                "calibration_folds": [value for value in FOLD_IDS if value != fold],
                "studies": rows,
            }
        )
    return {
        "schema_version": TICKET30_AUDIT_SCHEMA_VERSION,
        "status": "completed_existing_artifact_audit",
        "policy": {
            "schema_version": "rsna-knee-report-admission-policy-v1",
            "policy_version": TICKET30_ADMISSION_POLICY_VERSION,
            "policy_sha256": TICKET30_ADMISSION_POLICY_SHA256,
            "target_order": list(TARGET_COLUMNS),
            "held_out_evaluation_used_to_fit_policy": False,
        },
        "lineage": {
            "fold_identity": {"fold_manifest_checksum": manifest.manifest_checksum},
            "held_out_outputs_used_in_their_own_fold_policy_fit": False,
            "ticket16": {
                "fold_manifest_checksum": manifest.manifest_checksum,
                "model_name": "Astra Medium",
                "model_revision": "subscription-selected-astra-medium",
                "model_identity_fingerprint": "1" * 64,
                "prompt_fingerprint": "2" * 64,
                "experiment_id": "ticket16-astra-medium-fixture-v1",
                "private_report_execution_approved": True,
                "prompt_approved": True,
                "prompt_frozen": True,
                "prompt_tuning_or_adaptation": False,
                "validation_failure_count": 0,
            },
        },
        "fold_specific_training_admissions": [
            {"source_id": TICKET30_SOURCE_ID, "folds": studies_by_fold}
        ],
    }


def _write_audit(path: Path, manifest: FoldManifest) -> str:
    _write_audit_payload(path, _ticket30_payload(manifest))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_audit_payload(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def test_ticket30_admission_keeps_only_fold_safe_binary_calls(tmp_path: Path) -> None:
    manifest = _manifest()
    audit_path = tmp_path / "admitted-report-labels-v4.json"
    audit_sha256 = _write_audit(audit_path, manifest)

    admitted = admit_ticket30_fold_supervision(
        manifest,
        0,
        audit_path,
        expected_artifact_sha256=audit_sha256,
    )

    expected_gold = {
        item.study_instance_uid
        for item in manifest.gold_memberships
        if item.fold != 0
    }
    expected_report_only = {
        item.study_instance_uid
        for item in manifest.report_only_memberships
        if item.fold != 0
    }
    weak_uids = set(admitted.admitted_report_only_study_uids)
    assert admitted.source_id == TICKET30_SOURCE_ID
    assert admitted.fold == 0
    assert admitted.source_artifact_digest == audit_sha256
    assert set(admitted.held_out_study_uids) == set(manifest.held_out_study_uids(0))
    assert not set(admitted.held_out_study_uids).intersection(admitted.training_study_uids)
    assert set(admitted.reference_study_uids) == expected_gold
    assert set(admitted.eligible_report_only_study_uids) == expected_report_only
    assert weak_uids == {min(expected_report_only)}
    assert set(admitted.missing_report_only_study_uids) == expected_report_only - weak_uids

    weak = admitted.by_study[min(weak_uids)]
    assert weak.targets[0] == 1
    assert weak.supervision_mask[0] == 1
    assert weak.targets[1] is None
    assert weak.supervision_mask[1] == 0
    assert weak.supervised_entry_count == 1


def test_ticket30_admission_rejects_changed_artifact_digest(tmp_path: Path) -> None:
    manifest = _manifest()
    audit_path = tmp_path / "admitted-report-labels-v4.json"
    _write_audit(audit_path, manifest)

    with pytest.raises(Ticket34Error, match="SHA-256"):
        admit_ticket30_fold_supervision(
            manifest,
            0,
            audit_path,
            expected_artifact_sha256="0" * 64,
        )


def test_ticket30_image_scope_retains_no_call_report_studies_as_masked(tmp_path: Path) -> None:
    manifest = _manifest()
    audit_path = tmp_path / "admitted-report-labels-v4.json"
    audit_sha256 = _write_audit(audit_path, manifest)

    admitted = admit_ticket30_fold_supervision(
        manifest,
        0,
        audit_path,
        expected_artifact_sha256=audit_sha256,
    )

    scoped = admitted.for_image_studies(
        tuple(item.study_instance_uid for item in manifest.memberships)
    )

    expected_report_only = {
        item.study_instance_uid
        for item in manifest.report_only_memberships
        if item.fold != 0
    }
    assert set(scoped.training_study_uids) == set(admitted.training_study_uids)
    assert set(scoped.eligible_report_only_study_uids) == expected_report_only
    assert set(scoped.admitted_report_only_study_uids) == {min(expected_report_only)}
    assert set(scoped.missing_report_only_study_uids) == expected_report_only - set(
        scoped.admitted_report_only_study_uids
    )
    assert scoped.excluded_image_report_only_study_uids == ()


def test_ticket30_admission_rejects_a_masked_state_marked_admitted(tmp_path: Path) -> None:
    manifest = _manifest()
    payload = _ticket30_payload(manifest)
    source = payload["fold_specific_training_admissions"][0]
    selected = next(item for item in source["folds"] if item["fold"] == 0)
    call = selected["studies"][0]["targets"][0]
    call.update(
        {
            "state": "uncertain",
            "value": None,
            "supervision_mask": 1,
            "admitted": True,
            "rejection_reason": None,
        }
    )
    audit_path = tmp_path / "admitted-report-labels-v4.json"
    _write_audit_payload(audit_path, payload)

    with pytest.raises(Ticket34Error, match="only admitted positive or negative calls"):
        admit_ticket30_fold_supervision(
            manifest,
            0,
            audit_path,
            expected_artifact_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        )


def test_ticket30_admission_rejects_heldout_report_rows(tmp_path: Path) -> None:
    manifest = _manifest()
    payload = _ticket30_payload(manifest)
    heldout = next(item for item in manifest.report_only_memberships if item.fold == 0)
    source = payload["fold_specific_training_admissions"][0]
    selected = next(item for item in source["folds"] if item["fold"] == 0)
    selected["studies"].append(
        {
            "study_instance_uid": heldout.study_instance_uid,
            "assigned_fold": 0,
            "consuming_fold": 0,
            "report_sha256": hashlib.sha256(heldout.study_instance_uid.encode()).hexdigest(),
            "targets": [
                {
                    "target": target,
                    "state": "not-mentioned",
                    "value": None,
                    "supervision_mask": 0,
                    "admitted": False,
                    "rejection_reason": "state_is_always_masked",
                }
                for target in TARGET_COLUMNS
            ],
        }
    )
    audit_path = tmp_path / "admitted-report-labels-v4.json"
    _write_audit_payload(audit_path, payload)

    with pytest.raises(Ticket34Error, match="held out or outside"):
        admit_ticket30_fold_supervision(
            manifest,
            0,
            audit_path,
            expected_artifact_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        )


def _comparison_manifest() -> FoldManifest:
    gold = tuple(
        SourceStudy(
            f"comparison-gold-{index}",
            True,
            tuple(index % 2 for _ in TARGET_COLUMNS),
        )
        for index in range(18)
    )
    report_only = tuple(
        SourceStudy(f"comparison-report-{index}", True, (None,) * NUM_TARGETS)
        for index in range(6)
    )
    return freeze_folds(
        SourceManifest("train", gold + report_only, ()),
        source_version="ticket34-comparison-fixture-v1",
        seed=31,
    )


def _write_comparison_arm(
    root: Path,
    *,
    manifest: FoldManifest,
    report_arm: bool,
    predictions: dict[str, list[float]],
    artifact_sha256: str,
    initial_state_sha256: str = "a" * 64,
    gold_exposure_sha256: str = "b" * 64,
) -> Path:
    root.mkdir(parents=True)
    gold_training = tuple(
        sorted(
            item.study_instance_uid
            for item in manifest.gold_memberships
            if item.fold != 0
        )
    )
    heldout = tuple(
        sorted(
            item.study_instance_uid
            for item in manifest.gold_memberships
            if item.fold == 0
        )
    )
    membership_by_uid = {item.study_instance_uid: item for item in manifest.memberships}
    report_training = tuple(
        sorted(
            item.study_instance_uid
            for item in manifest.report_only_memberships
            if item.fold != 0
        )
    ) if report_arm else ()
    training = tuple(sorted((*gold_training, *report_training)))
    config_fingerprint = "c" * 64
    integration_checksum = "d" * 64
    source_fingerprint = "e" * 64
    config = {
        "schema_version": "rsna-knee-training-v1",
        "model": {
            "model_id": "resnet18-frozen-slice-features-v1",
            "encoder_id": "torchvision-resnet18-imagenet1k-v1",
            "encoder_weights_sha256": RESNET18_SHA256,
        },
        "optimization": {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "schedule": "constant",
            "batch_size": 2,
            "successful_optimizer_steps": 5,
        },
        "seeds": {"initialization": 20260911, "batch_order": 20260912},
        "loss": "masked_bce",
        "augmentation": "none",
    }
    reference_exposure = 10
    weak_exposure = 10 if report_arm else 0
    source_exposures = {
        "reference": {
            "sample_exposures": reference_exposure,
            "supervised_entry_exposures": reference_exposure * NUM_TARGETS,
        },
        "weak": {
            "sample_exposures": weak_exposure,
            "supervised_entry_exposures": weak_exposure,
        },
        "mixed": {"sample_exposures": 0, "supervised_entry_exposures": 0},
    }
    counts = {
        "training_studies": len(training),
        "held_out_studies": len(heldout),
        "successful_optimizer_steps": 5,
        "scheduler_steps": 5,
        "attempted_batches": 5,
        "skipped_all_masked_batches": 0,
        "sample_exposures": reference_exposure + weak_exposure,
        "supervised_entry_exposures": reference_exposure * NUM_TARGETS + weak_exposure,
        "exposures_by_source": source_exposures,
        "gold_exposure_sequence_sha256": gold_exposure_sha256,
        "weak_exposure_sequence_sha256": "9" * 64 if report_arm else None,
        "initial_model_state_sha256": initial_state_sha256,
        "training_design": TICKET34_ADDITIVE_DESIGN,
        "weak_sampling_policy": "ticket34-report-only-cyclic-shuffle-v1" if report_arm else None,
        "weak_sampling_seed": 20260912 + TICKET34_WEAK_BATCH_SEED_OFFSET if report_arm else None,
    }
    supervision: dict[str, object]
    if report_arm:
        rows = [
            {
                "study_instance_uid": uid,
                "targets": [1] + [None] * (NUM_TARGETS - 1),
                "supervision_mask": [1] + [0] * (NUM_TARGETS - 1),
                "source_kind": "weak",
            }
            for uid in report_training
        ]
        rows.extend(
            {
                "study_instance_uid": uid,
                "targets": [int(value) for value in membership_by_uid[uid].reference_labels],
                "supervision_mask": [1] * NUM_TARGETS,
                "source_kind": "reference",
            }
            for uid in gold_training
        )
        supervision = {
            "kind": "weak",
            "schema_version": "rsna-knee-supervision-admission-v1",
            "fold": 0,
            "fold_manifest_checksum": manifest.manifest_checksum,
            "source_artifact_digest": artifact_sha256,
            "source_id": TICKET30_SOURCE_ID,
            "source_schema_version": TICKET30_AUDIT_SCHEMA_VERSION,
            "weak_label_version": TICKET34_WEAK_LABEL_VERSION,
            "mixture_policy": TICKET34_MIXTURE_POLICY,
            "training_study_uids": list(training),
            "held_out_study_uids": list(manifest.held_out_study_uids(0)),
            "eligible_report_only_study_uids": list(report_training),
            "admitted_report_only_study_uids": list(report_training),
            "missing_report_only_study_uids": [],
            "excluded_image_report_only_study_uids": [],
            "resolved_studies": rows,
            "counts": {
                "training_studies": len(training),
                "reference_studies": len(gold_training),
                "weak_studies": len(report_training),
                "eligible_report_only_studies": len(report_training),
                "admitted_report_only_studies": len(report_training),
                "missing_report_only_studies": 0,
                "image_excluded_report_only_studies": 0,
                "supervised_entries": len(gold_training) * NUM_TARGETS + len(report_training),
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
    else:
        supervision = {
            "kind": "reference",
            "fold_manifest_checksum": manifest.manifest_checksum,
            "held_out_inputs_excluded": True,
            "reports_used": False,
            "training_studies": list(gold_training),
            "held_out_studies": list(heldout),
        }
    checkpoint_bytes = f"{root.name}-checkpoint".encode()
    (root / "checkpoints").mkdir()
    (root / "checkpoints" / "final.pt").write_bytes(checkpoint_bytes)
    (root / "checkpoints" / "latest.pt").write_bytes(checkpoint_bytes)
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    run_id = f"ticket34-{root.name}"
    semantic_fingerprint = "1" * 64 if report_arm else "2" * 64
    approval = {
        "artifact_id": "torchvision.ResNet18_Weights.IMAGENET1K_V1",
        "source_revision": RESNET18_SOURCE_REVISION,
        "download_url": RESNET18_DOWNLOAD_URL,
        "expected_bytes": RESNET18_ARTIFACT_BYTES,
        "sha256": RESNET18_SHA256,
        "scope": TICKET34_RESOURCE_SCOPE,
        "decision": "approved",
        "rights_status": "eligible",
        "external_spend": 0,
    }
    identity = {
        "source_version": manifest.source_version,
        "source_fingerprint": source_fingerprint,
        "cache_fingerprint": "8" * 64,
        "fold_manifest_checksum": manifest.manifest_checksum,
        "integration_manifest_checksum": integration_checksum,
        "configuration_fingerprint": config_fingerprint,
        "training_schema_version": "rsna-knee-training-v1",
        "model_source": "external:torchvision-resnet18-imagenet1k-v1;frozen",
        "architecture": config["model"],
        "initialization": {"kind": "frozen-external-weights", "sha256": RESNET18_SHA256},
        "preprocessing": {"image_size": [224, 224]},
        "packing": {"max_series": 16, "max_slices": 8},
        "optimization": config["optimization"],
        "loss": config["loss"],
        "augmentation": config["augmentation"],
        "budget": {"successful_optimizer_steps": 5, "batch_size": 2},
        "seeds": {"initialization": 20260911, "batch_order": 20260912},
        "pretrained_resource_approval": approval,
        "supervision": {"kind": "weak" if report_arm else "reference"},
    }
    configuration = {
        "run_id": run_id,
        "fold": 0,
        "scientific_configuration_fingerprint": config_fingerprint,
        "semantic_fingerprint": semantic_fingerprint,
        "config": config,
        "identity": identity,
        "supervision": supervision,
        "inference_configuration": {"checkpoint_selection": "final_checkpoint"},
        "preflight": {"passed": True},
        "resources": {"requested_partition": "cluster1-cpu", "requested_nodes": 1,
                      "requested_cpus": 16, "requested_memory_gb": 64,
                      "requested_wall_time_hours": 6, "exclusive": False},
        "environment": {"code_revision": "fixture", "slurm_job_id": "123"},
    }
    run = {
        "schema_version": "rsna-knee-training-v1",
        "status": "completed_fixed_budget_training",
        "run_id": run_id,
        "fold": 0,
        "config_fingerprint": config_fingerprint,
        "semantic_fingerprint": semantic_fingerprint,
        "config": config,
        "counts": counts,
        "checkpoint": {"final_present": True, "latest_present": True,
                       "final_sha256": checkpoint_sha256},
        "predictions": predictions,
        "reference_supervision_only": not report_arm,
        "held_out_inputs_excluded_from_fitting": True,
        "supervision": supervision,
        "training_study_uids": list(training if report_arm else gold_training),
        "held_out_study_uids": list(heldout),
    }
    (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (root / "configuration.json").write_text(json.dumps(configuration), encoding="utf-8")
    return root


def test_ticket34_comparison_reports_paired_one_fold_delta_and_uncertainty(
    tmp_path: Path,
) -> None:
    manifest = _comparison_manifest()
    heldout = [
        item for item in manifest.gold_memberships if item.fold == 0
    ]
    baseline_predictions = {
        item.study_instance_uid: [float(1 - label) for label in item.reference_labels]
        for item in heldout
    }
    report_predictions = {
        item.study_instance_uid: [float(label) for label in item.reference_labels]
        for item in heldout
    }
    audit_sha256 = "7" * 64
    reference_run = _write_comparison_arm(
        tmp_path / "reference",
        manifest=manifest,
        report_arm=False,
        predictions=baseline_predictions,
        artifact_sha256=audit_sha256,
    )
    report_run = _write_comparison_arm(
        tmp_path / "report",
        manifest=manifest,
        report_arm=True,
        predictions=report_predictions,
        artifact_sha256=audit_sha256,
    )

    comparison = compare_supervision_pilot(
        reference_run,
        report_run,
        manifest,
        expected_ticket30_artifact_sha256=audit_sha256,
        image_study_uids=tuple(item.study_instance_uid for item in manifest.memberships),
        bootstrap_resamples=40,
        bootstrap_seed=19,
    )

    assert comparison["fold"] == 0
    assert comparison["cohort"]["held_out_studies"] == len(heldout)
    assert comparison["cohort"]["image_membership"]["image_backed_report_only_studies_without_admitted_calls"] == 0
    assert comparison["one_fold_limit"] is True
    assert comparison["reference_macro_auc"] == 0.0
    assert comparison["report_macro_auc"] == 1.0
    assert comparison["delta_macro_auc_report_minus_reference"] == 1.0
    assert all(item["delta_auc_report_minus_reference"] == 1.0 for item in comparison["per_target"])
    interval = comparison["paired_study_bootstrap"]["macro_auc_delta"]
    assert interval["defined_resamples"] + interval["undefined_resamples"] == 40
    assert interval["lower"] == interval["upper"] == 1.0


def test_ticket34_comparison_rejects_changed_reference_exposure(tmp_path: Path) -> None:
    manifest = _comparison_manifest()
    heldout = [item for item in manifest.gold_memberships if item.fold == 0]
    predictions = {
        item.study_instance_uid: [float(label) for label in item.reference_labels]
        for item in heldout
    }
    audit_sha256 = "7" * 64
    reference_run = _write_comparison_arm(
        tmp_path / "reference",
        manifest=manifest,
        report_arm=False,
        predictions=predictions,
        artifact_sha256=audit_sha256,
    )
    report_run = _write_comparison_arm(
        tmp_path / "report",
        manifest=manifest,
        report_arm=True,
        predictions=predictions,
        artifact_sha256=audit_sha256,
        gold_exposure_sha256="0" * 64,
    )

    with pytest.raises(Ticket34Error, match="reference exposure"):
        compare_supervision_pilot(
            reference_run,
            report_run,
            manifest,
            expected_ticket30_artifact_sha256=audit_sha256,
            image_study_uids=tuple(item.study_instance_uid for item in manifest.memberships),
            bootstrap_resamples=20,
        )


def test_ticket34_comparison_rejects_incomplete_heldout_predictions(tmp_path: Path) -> None:
    manifest = _comparison_manifest()
    heldout = [item for item in manifest.gold_memberships if item.fold == 0]
    predictions = {
        item.study_instance_uid: [float(label) for label in item.reference_labels]
        for item in heldout
    }
    audit_sha256 = "7" * 64
    reference_run = _write_comparison_arm(
        tmp_path / "reference",
        manifest=manifest,
        report_arm=False,
        predictions=predictions,
        artifact_sha256=audit_sha256,
    )
    report_run = _write_comparison_arm(
        tmp_path / "report",
        manifest=manifest,
        report_arm=True,
        predictions=predictions,
        artifact_sha256=audit_sha256,
    )
    run_path = report_run / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["predictions"].pop(heldout[0].study_instance_uid)
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(Ticket34Error, match="complete held-out prediction cohort"):
        compare_supervision_pilot(
            reference_run,
            report_run,
            manifest,
            expected_ticket30_artifact_sha256=audit_sha256,
            image_study_uids=tuple(item.study_instance_uid for item in manifest.memberships),
            bootstrap_resamples=20,
        )


def test_ticket34_comparison_rejects_mismatched_initial_state(tmp_path: Path) -> None:
    manifest = _comparison_manifest()
    heldout = [item for item in manifest.gold_memberships if item.fold == 0]
    predictions = {
        item.study_instance_uid: [float(label) for label in item.reference_labels]
        for item in heldout
    }
    audit_sha256 = "7" * 64
    image_uids = tuple(item.study_instance_uid for item in manifest.memberships)
    reference_run = _write_comparison_arm(
        tmp_path / "reference",
        manifest=manifest,
        report_arm=False,
        predictions=predictions,
        artifact_sha256=audit_sha256,
    )
    report_run = _write_comparison_arm(
        tmp_path / "report",
        manifest=manifest,
        report_arm=True,
        predictions=predictions,
        artifact_sha256=audit_sha256,
        initial_state_sha256="0" * 64,
    )

    with pytest.raises(Ticket34Error, match="initial model state differs"):
        compare_supervision_pilot(
            reference_run,
            report_run,
            manifest,
            expected_ticket30_artifact_sha256=audit_sha256,
            image_study_uids=image_uids,
            bootstrap_resamples=20,
        )


def test_ticket34_comparison_rejects_heldout_supervision_lineage_change(
    tmp_path: Path,
) -> None:
    manifest = _comparison_manifest()
    heldout = [item for item in manifest.gold_memberships if item.fold == 0]
    predictions = {
        item.study_instance_uid: [float(label) for label in item.reference_labels]
        for item in heldout
    }
    audit_sha256 = "7" * 64
    image_uids = tuple(item.study_instance_uid for item in manifest.memberships)
    reference_run = _write_comparison_arm(
        tmp_path / "reference",
        manifest=manifest,
        report_arm=False,
        predictions=predictions,
        artifact_sha256=audit_sha256,
    )
    report_run = _write_comparison_arm(
        tmp_path / "report",
        manifest=manifest,
        report_arm=True,
        predictions=predictions,
        artifact_sha256=audit_sha256,
    )
    run_path = report_run / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["supervision"]["held_out_study_uids"][0] = "outside-the-fold"
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(Ticket34Error, match="report supervision held-out lineage"):
        compare_supervision_pilot(
            reference_run,
            report_run,
            manifest,
            expected_ticket30_artifact_sha256=audit_sha256,
            image_study_uids=image_uids,
            bootstrap_resamples=20,
        )
