from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from rsna_knee.astra_handoff import (
    ASTRA_EFFORT,
    ASTRA_REASONING_EFFORT,
    ASTRA_SEMANTIC_STATES,
    AstraApprovalError,
    AstraContextMismatchError,
    AstraPromptError,
    astra_medium_experiment_approval,
    astra_output_example,
    build_astra_batch_plan,
    completed_astra_batch_ids,
    expected_astra_batch_count,
    load_astra_batch_plan,
    make_astra_prompt_artifact,
    pending_astra_batches,
    record_astra_output,
    validate_astra_output,
    write_astra_batch_plan,
    write_astra_public_evidence,
)
from rsna_knee.folds import FoldManifest, freeze_folds
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.report_extraction import ReportCorpus, ReportRecord
from rsna_knee.targets import TARGET_COLUMNS


def _manifest() -> FoldManifest:
    studies = [
        SourceStudy(
            f"gold-{index:02d}",
            True,
            tuple((index + target_index) % 2 for target_index in range(len(TARGET_COLUMNS))),
        )
        for index in range(12)
    ]
    studies.extend(
        SourceStudy(f"report-{index:02d}", True, (None,) * len(TARGET_COLUMNS))
        for index in range(6)
    )
    return freeze_folds(
        SourceManifest("train", tuple(studies), ()),
        source_version="astra-handoff-fixture-v1",
        seed=17,
    )


def _corpus(manifest: FoldManifest) -> ReportCorpus:
    records = tuple(
        ReportRecord(item.study_instance_uid, f"private://{item.study_instance_uid}", f"report {item.study_instance_uid}")
        for item in manifest.memberships
    )
    return ReportCorpus("astra-handoff-reports-v1", records)


def _approved_prompt():
    return make_astra_prompt_artifact(
        "Owner-supplied frozen medical extraction policy. Emit only the exact schema."
    ).approve(
        approval_reference="synthetic-owner-approval",
        approved_by="test-owner",
        approved_on="2026-09-14",
    )


def _plan(tmp_path: Path, *, batch_size: int = 4):
    manifest = _manifest()
    corpus = _corpus(manifest)
    return build_astra_batch_plan(
        manifest,
        corpus,
        astra_medium_experiment_approval(),
        _approved_prompt(),
        batch_size=batch_size,
    )


def test_identity_fixes_astra_medium_and_medium_effort_without_an_api() -> None:
    approval = astra_medium_experiment_approval()

    assert approval.model_name == "Astra Medium"
    assert approval.reasoning_effort == ASTRA_REASONING_EFFORT == "medium"
    assert approval.effort == ASTRA_EFFORT == "medium"
    assert approval.identity_approved
    assert approval.no_api_integration
    assert approval.no_api_key_handling
    assert approval.no_provider_sdk
    assert approval.no_external_model_wrapper


def test_unapproved_prompt_blocks_batch_planning(tmp_path: Path) -> None:
    manifest = _manifest()
    corpus = _corpus(manifest)
    prompt = make_astra_prompt_artifact("not approved yet")

    with pytest.raises(AstraPromptError, match="approved"):
        build_astra_batch_plan(manifest, corpus, astra_medium_experiment_approval(), prompt)


def test_batch_ids_and_cohorts_are_stable_and_fold_clean() -> None:
    first = _plan(Path("/tmp"), batch_size=4)
    second = _plan(Path("/tmp"), batch_size=4)

    assert first.plan_fingerprint == second.plan_fingerprint
    assert [item.batch_id for item in first.batches] == [item.batch_id for item in second.batches]
    assert first.batch_count == expected_astra_batch_count(18, batch_size=4)
    assert first.cohorts.available_report_uids == frozenset(
        item.study_instance_uid for item in _manifest().memberships
    )
    for batch in first.batches:
        for record in batch.records:
            if record.cohort == "gold":
                assert record.consuming_folds == (record.assigned_fold,)
                assert all(partition == "held_out_evaluation" for _, partition in record.partitions_by_fold)
            else:
                assert record.assigned_fold not in record.consuming_folds
                assert all(partition == "admitted_training_labels" for _, partition in record.partitions_by_fold)


def test_strict_output_maps_only_positive_negative_to_mask_one() -> None:
    plan = _plan(Path("/tmp"), batch_size=4)
    batch = plan.batches[0]
    payload = astra_output_example(batch)
    states = ("positive", "negative") + ("unknown",) * (len(TARGET_COLUMNS) - 2)
    for index, target in enumerate(payload["studies"][0]["targets"]):  # type: ignore[index]
        target["state"] = states[index]  # type: ignore[index]
    result = validate_astra_output(json.dumps(payload), batch)

    assert result.status == "validated"
    assert result.studies[0].supervision_mask == (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert ASTRA_SEMANTIC_STATES == ("positive", "negative", "not-mentioned", "uncertain", "unknown")
    assert "extraction-failed" not in ASTRA_SEMANTIC_STATES


def test_malformed_or_incomplete_output_is_all_masked_failure() -> None:
    plan = _plan(Path("/tmp"), batch_size=4)
    batch = plan.batches[0]

    malformed = validate_astra_output("not json", batch)
    assert malformed.status == "extraction-failed"
    assert malformed.failure_code == "invalid-json"
    assert all(mask == 0 for study in malformed.studies for mask in study.supervision_mask)

    payload = astra_output_example(batch)
    payload["studies"] = payload["studies"][:-1]  # type: ignore[index]
    incomplete = validate_astra_output(json.dumps(payload), batch)
    assert incomplete.status == "extraction-failed"
    assert incomplete.failure_code in {"missing-study-or-target", "invalid-output"}
    assert all(
        item.state.value == "extraction-failed"
        for study in incomplete.studies
        for item in study.labels
    )


def test_context_mismatch_cannot_become_a_failure_for_this_batch() -> None:
    plan = _plan(Path("/tmp"), batch_size=4)
    batch = plan.batches[0]
    payload = astra_output_example(batch)
    payload["effort"] = "high"

    with pytest.raises(AstraContextMismatchError, match="effort"):
        validate_astra_output(json.dumps(payload), batch)


def test_immutable_plan_resume_and_text_free_public_evidence(tmp_path: Path) -> None:
    plan = _plan(tmp_path, batch_size=4)
    public_plan = tmp_path / "public" / "plan.json"
    private_dir = tmp_path / "private"
    write_astra_batch_plan(plan, public_path=public_plan, private_dir=private_dir)
    loaded_plan = load_astra_batch_plan(private_dir / "plan.json")
    assert loaded_plan.plan_fingerprint == plan.plan_fingerprint

    public_text = public_plan.read_text(encoding="utf-8")
    assert "report gold-00" not in public_text
    assert "gold-00" not in public_text
    assert pending_astra_batches(plan, private_dir)[0].batch_id == plan.batches[0].batch_id

    raw = json.dumps(astra_output_example(plan.batches[0]))
    audit = record_astra_output(plan, plan.batches[0].batch_id, raw, private_dir)
    assert audit.status == "validated"
    assert plan.batches[0].batch_id in completed_astra_batch_ids(plan, private_dir)
    assert plan.batches[0].batch_id not in {
        item.batch_id for item in pending_astra_batches(plan, private_dir)
    }
    with pytest.raises(Exception):
        record_astra_output(plan, plan.batches[0].batch_id, raw, private_dir)

    public_status = tmp_path / "public" / "status.json"
    write_astra_public_evidence(plan, private_dir, public_status)
    status_text = public_status.read_text(encoding="utf-8")
    assert "gold-00" not in status_text
    assert "report gold-00" not in status_text
    assert "report_only_accuracy_claimed" in status_text


def test_duplicate_and_extra_study_are_detected_without_guessing() -> None:
    plan = _plan(Path("/tmp"), batch_size=4)
    batch = plan.batches[0]
    payload = astra_output_example(batch)
    payload["studies"][1]["study_instance_uid"] = payload["studies"][0]["study_instance_uid"]  # type: ignore[index]
    duplicate = validate_astra_output(json.dumps(payload), batch)

    assert duplicate.status == "extraction-failed"
    assert duplicate.failure_code == "duplicate-study-or-target"
    assert all(mask == 0 for study in duplicate.studies for mask in study.supervision_mask)
