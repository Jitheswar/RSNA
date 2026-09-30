"""An upstream COMPLETE receipt must not disguise a dropped model arm."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "public_frontier_validation", Path(__file__).parents[1] / "scripts/public_frontier_validation.py"
)
validation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validation)


def full_run():
    phases = {name: {"shape": [3, 12], "sha256": "a" * 64}
              for name in validation.REQUIRED_PHASES | {f"dino_{i}" for i in range(20)}}
    audit = dict(status="COMPLETE", cohort=3, dino_members=20, a5_folds=5, raptor_views=4,
                 family_reduction="rank_of_member_probability_mean", coat_family_members=4,
                 inner_rerank=False, events=[], phases=phases)
    blend = dict(members=sorted(validation.COAT_MEMBERS), within_coat={n: .25 for n in validation.COAT_MEMBERS},
                 family_reduction="rank_of_member_probability_mean", private_alpha=.4,
                 public_raptor_alpha=.6, inner_rerank=False, study_count=3)
    receipts = {n: dict(status=validation.COAT_STATUSES[n], fallback_studies=0, failures=[],
                       models=1 if n == "d4_swa3" else 3,
                       checkpoints=[{"epoch": e} for e in validation.COAT_EPOCHS.get(n, [])])
                for n in validation.COAT_MEMBERS}
    return audit, blend, receipts


def test_complete_four_reader_run_is_admissible():
    assert validation.validate_public_frontier(*full_run(), 3)["prediction_artifacts"] == 27


def test_upstream_complete_status_cannot_hide_missing_reader():
    audit, blend, receipts = full_run()
    blend["members"].remove("d4_swa3")
    with pytest.raises(ValueError, match="all four CoAt"):
        validation.validate_public_frontier(audit, blend, receipts, 3)


@pytest.mark.parametrize("event", ["submission_neutral_fill", "coat_d4_child_failed",
                                  "coat_repairv1_output_rejected", "raptor_study_failed"])
def test_complete_csv_cannot_hide_failed_predictions(event):
    audit, blend, receipts = full_run()
    audit["events"] = [{"kind": event}]
    with pytest.raises(ValueError, match="failed/fallback events"):
        validation.validate_public_frontier(audit, blend, receipts, 3)


def test_changed_probability_reduction_is_rejected():
    audit, blend, receipts = full_run()
    blend["family_reduction"] = "rank_sum_fallback_probabilities_unavailable"
    with pytest.raises(ValueError, match="probability reduction"):
        validation.validate_public_frontier(audit, blend, receipts, 3)


def test_checkpoint_substitution_is_rejected():
    audit, blend, receipts = full_run()
    receipts["global96_top3"]["checkpoints"][0]["epoch"] = 24
    with pytest.raises(ValueError, match="checkpoint epochs"):
        validation.validate_public_frontier(audit, blend, receipts, 3)


def test_receipt_cannot_hide_fallback_study():
    audit, blend, receipts = full_run()
    receipts["resgated_top3"]["fallback_studies"] = 1
    with pytest.raises(ValueError, match="fallback study count"):
        validation.validate_public_frontier(audit, blend, receipts, 3)


def test_failed_child_status_is_rejected_even_when_parent_claims_complete():
    audit, blend, receipts = full_run()
    receipts["resgated_top3"]["status"] = "FAILED"
    with pytest.raises(ValueError, match="child status"):
        validation.validate_public_frontier(audit, blend, receipts, 3)


def test_incomplete_dino_artifacts_are_rejected():
    audit, blend, receipts = full_run()
    del audit["phases"]["dino_19"]
    with pytest.raises(ValueError, match="DINO prediction artifacts"):
        validation.validate_public_frontier(audit, blend, receipts, 3)


def test_successful_numeric_retry_is_retained_without_dropped_output():
    audit, blend, receipts = full_run()
    audit["events"] = [{"kind": "dino_fp16_nonfinite_retry_fp32"}]
    original = deepcopy((audit, blend, receipts))
    validation.validate_public_frontier(audit, blend, receipts, 3)
    assert (audit, blend, receipts) == original
