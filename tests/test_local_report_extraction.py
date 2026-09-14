from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from rsna_knee.local_report_extraction import (
    FLAN_T5_SMALL_MODEL_ID,
    FrozenLocalPrompt,
    LocalModelApproval,
    LocalModelGeneration,
    LocalLanguageModelExtractor,
    LocalModelApprovalError,
    LocalReportComparisonError,
    compare_report_extractors,
    freeze_local_prompt,
    proposed_flan_t5_small_approval,
    run_local_report_extraction,
    verify_local_model_artifact,
    write_local_model_comparison_reports,
)
from rsna_knee.folds import FoldManifest, freeze_folds
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.report_extraction import ReportRecord
from rsna_knee.report_labels import LabelState
from rsna_knee.targets import TARGET_COLUMNS


def _approved_proposal(*, artifact_sha256: str, artifact_relative_path: str) -> LocalModelApproval:
    proposal = proposed_flan_t5_small_approval()
    return replace(
        proposal,
        artifact_sha256=artifact_sha256,
        artifact_relative_path=artifact_relative_path,
        supporting_artifact_hashes=(),
        approved=True,
        approval_reference="owner-approval-ticket-16-01",
        approved_by="project-owner",
        approved_on="2026-09-14",
    )


def test_model_proposal_is_specific_but_cannot_be_used_without_approval(tmp_path: Path) -> None:
    proposal = proposed_flan_t5_small_approval()

    assert proposal.model_id == FLAN_T5_SMALL_MODEL_ID
    assert proposal.model_revision == "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
    assert proposal.approved is False

    with pytest.raises(LocalModelApprovalError, match="approved"):
        verify_local_model_artifact(proposal, tmp_path)


def test_approved_model_artifact_requires_the_declared_local_hash(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"local test weights")
    digest = sha256(weights.read_bytes()).hexdigest()
    approval = _approved_proposal(
        artifact_sha256=digest,
        artifact_relative_path="model.safetensors",
    )

    verified = verify_local_model_artifact(approval, tmp_path)

    assert verified.sha256 == digest
    assert verified.size_bytes == weights.stat().st_size

    weights.write_bytes(b"changed weights")
    with pytest.raises(LocalModelApprovalError, match="SHA-256"):
        verify_local_model_artifact(approval, tmp_path)


def test_approved_model_artifact_verifies_supporting_loader_files(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    config = tmp_path / "config.json"
    weights.write_bytes(b"local test weights")
    config.write_bytes(b'{"model_type":"t5"}')
    approval = _approved_proposal(
        artifact_sha256=sha256(weights.read_bytes()).hexdigest(),
        artifact_relative_path="model.safetensors",
    )
    approval = replace(
        approval,
        supporting_artifact_hashes=(("config.json", sha256(config.read_bytes()).hexdigest()),),
    )

    verify_local_model_artifact(approval, tmp_path)

    config.write_bytes(b'{"model_type":"changed"}')
    with pytest.raises(LocalModelApprovalError, match="supporting"):
        verify_local_model_artifact(approval, tmp_path)


def test_frozen_prompt_has_stable_identity_and_never_serializes_report_text() -> None:
    prompt = freeze_local_prompt()
    rendered = prompt.render("The ACL is intact.")
    payload = prompt.to_public_dict()

    assert prompt.is_frozen
    assert prompt.fingerprint == freeze_local_prompt().fingerprint
    assert "The ACL is intact." in rendered
    assert "The ACL is intact." not in json.dumps(payload)
    assert payload["target_order"]


class _StaticGenerator:
    def __init__(self, output: str) -> None:
        self.output = output

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> LocalModelGeneration:
        assert prompt
        assert max_new_tokens == 256
        assert do_sample is False
        assert temperature == 0.0
        return LocalModelGeneration(self.output, input_token_count=12, output_token_count=34)


def _model_output() -> str:
    states = (
        ("ACL", "positive", 0.91),
        ("MCL", "negative", 0.88),
        ("Medial Meniscus", "not-mentioned", None),
        ("Lateral Meniscus", "uncertain", None),
        ("Medial OA", "unknown", None),
        ("Lateral OA", "extraction-failed", None),
        ("PF OA", "not-mentioned", None),
        ("Effusion", "negative", 0.77),
        ("Synovitis", "positive", 0.66),
        ("Baker's", "not-mentioned", None),
        ("Contusion", "unknown", None),
        ("Fracture", "negative", 0.99),
    )
    return json.dumps(
        {
            "schema_version": "rsna-knee-local-model-output-v1",
            "targets": [
                {
                    "target": target,
                    "state": state,
                    "confidence": confidence,
                    "evidence": f"model evidence for {target}",
                }
                for target, state, confidence in states
            ],
        }
    )


def _approved_for_generator() -> LocalModelApproval:
    return _approved_proposal(
        artifact_sha256="a" * 64,
        artifact_relative_path="model.safetensors",
    )


def test_local_model_output_preserves_states_masks_and_private_audit_evidence() -> None:
    approval = _approved_for_generator()
    prompt: FrozenLocalPrompt = freeze_local_prompt()
    extractor = LocalLanguageModelExtractor.from_verified_generator(
        approval,
        prompt,
        _StaticGenerator(_model_output()),
        artifact_sha256=approval.artifact_sha256,
    )
    report = ReportRecord("study-1", "private://study-1", "The ACL is intact.")

    interpretation = extractor.extract(report, fold=1)

    assert interpretation.model_fingerprint == extractor.fingerprint
    assert [item.state for item in interpretation.labels] == [
        LabelState.POSITIVE,
        LabelState.NEGATIVE,
        LabelState.NOT_MENTIONED,
        LabelState.UNCERTAIN,
        LabelState.UNKNOWN,
        LabelState.EXTRACTION_FAILED,
        LabelState.NOT_MENTIONED,
        LabelState.NEGATIVE,
        LabelState.POSITIVE,
        LabelState.NOT_MENTIONED,
        LabelState.UNKNOWN,
        LabelState.NEGATIVE,
    ]
    assert interpretation.supervision_mask == (1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1)
    assert "The ACL is intact." not in json.dumps(interpretation.to_public_dict())
    assert "private://study-1" in json.dumps(interpretation.to_private_dict())


def test_invalid_local_model_json_is_recorded_as_extraction_failure() -> None:
    approval = _approved_for_generator()
    extractor = LocalLanguageModelExtractor.from_verified_generator(
        approval,
        freeze_local_prompt(),
        _StaticGenerator("not json"),
        artifact_sha256=approval.artifact_sha256,
    )

    interpretation = extractor.extract(
        ReportRecord("study-1", "private://study-1", "A report"),
        fold=0,
    )

    assert all(item.state is LabelState.EXTRACTION_FAILED for item in interpretation.labels)
    assert all(value == 0 for value in interpretation.supervision_mask)
    assert interpretation.failure_code == "invalid-json"


def _fold_manifest() -> FoldManifest:
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
        source_version="local-model-fixture-v1",
        seed=17,
    )


def _report_corpus(manifest: FoldManifest) -> object:
    from rsna_knee.local_report_extraction import ReportCorpus

    records = [
        ReportRecord(item.study_instance_uid, f"private://{item.study_instance_uid}", "A report")
        for item in manifest.gold_memberships
    ]
    records.extend(
        ReportRecord(item.study_instance_uid, f"private://{item.study_instance_uid}", "A report")
        for item in manifest.report_only_memberships
    )
    return ReportCorpus("local-model-reports-v1", tuple(records))


def test_local_model_run_matches_deterministic_cohorts_and_reports_cost() -> None:
    approval = _approved_for_generator()
    extractor = LocalLanguageModelExtractor.from_verified_generator(
        approval,
        freeze_local_prompt(),
        _StaticGenerator(_model_output()),
        artifact_sha256=approval.artifact_sha256,
    )
    manifest = _fold_manifest()
    corpus = _report_corpus(manifest)
    from rsna_knee.report_extraction import run_report_extraction

    baseline = run_report_extraction(manifest, corpus, verify_reproducibility=True)
    local = run_local_report_extraction(manifest, corpus, extractor, verify_reproducibility=True)

    comparison = compare_report_extractors(baseline, local)

    assert comparison.identical_evaluation_cohorts
    assert comparison.identical_training_eligibility
    assert local.evaluation.reproducibility_verified
    assert local.lineage_verified
    assert local.evaluation.pooled_report_count == baseline.evaluation.pooled_report_count
    assert local.evaluation.cost.report_count == local.evaluation.pooled_report_count + sum(
        len(item.training_labels) for item in local.folds
    )
    assert comparison.to_public_dict()["ticket"] == "16"
    assert comparison.to_public_dict()["gates"]["V15_lineage_and_fold_isolation"] is True


def test_comparison_rejects_different_report_cohorts() -> None:
    approval = _approved_for_generator()
    extractor = LocalLanguageModelExtractor.from_verified_generator(
        approval,
        freeze_local_prompt(),
        _StaticGenerator(_model_output()),
        artifact_sha256=approval.artifact_sha256,
    )
    manifest = _fold_manifest()
    corpus = _report_corpus(manifest)
    from rsna_knee.report_extraction import run_report_extraction

    baseline = run_report_extraction(manifest, corpus, verify_reproducibility=False)
    local = run_local_report_extraction(
        manifest,
        replace(corpus, source_version="different-source"),
        extractor,
        verify_reproducibility=False,
    )

    with pytest.raises(LocalReportComparisonError, match="source version"):
        compare_report_extractors(baseline, local)


def test_comparison_writer_keeps_identifiers_and_evidence_private(tmp_path: Path) -> None:
    approval = _approved_for_generator()
    extractor = LocalLanguageModelExtractor.from_verified_generator(
        approval,
        freeze_local_prompt(),
        _StaticGenerator(_model_output()),
        artifact_sha256=approval.artifact_sha256,
    )
    manifest = _fold_manifest()
    corpus = _report_corpus(manifest)
    from rsna_knee.report_extraction import run_report_extraction

    baseline = run_report_extraction(manifest, corpus, verify_reproducibility=False)
    local = run_local_report_extraction(manifest, corpus, extractor, verify_reproducibility=False)
    comparison = compare_report_extractors(baseline, local)
    public_path = tmp_path / "public" / "comparison.json"
    private_path = tmp_path / "private" / "comparison.json"

    assert write_local_model_comparison_reports(
        comparison,
        public_path=public_path,
        private_path=private_path,
    ) == (public_path, private_path)
    public_text = public_path.read_text(encoding="utf-8")
    private_text = private_path.read_text(encoding="utf-8")
    assert "gold-00" not in public_text
    assert "The ACL is intact." not in public_text
    assert "gold-00" in private_text
    assert "private://gold-00" in private_text
    assert "model evidence for ACL" in private_text

    with pytest.raises(FileExistsError):
        write_local_model_comparison_reports(
            comparison,
            public_path=public_path,
            private_path=tmp_path / "private" / "comparison-2.json",
        )
