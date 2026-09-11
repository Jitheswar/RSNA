from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.recovery import ReferenceFallbackPrior, compute_laplace_smoothed_reference_priors
from rsna_knee.targets import NUM_TARGETS


def _labels(*, acl: int | None = None, mcl: int | None = None) -> tuple[int | None, ...]:
    values: list[int | None] = [None] * NUM_TARGETS
    values[0] = acl
    values[1] = mcl
    return tuple(values)


def test_reference_priors_use_per_target_laplace_smoothing_and_midpoint_when_unavailable() -> None:
    priors = compute_laplace_smoothed_reference_priors(
        {
            "gold-positive": _labels(acl=1, mcl=1),
            "gold-negative": _labels(acl=0, mcl=0),
        },
        eligible_study_uids=("gold-positive", "gold-negative"),
    )

    assert priors.scores[0] == pytest.approx(0.5)
    assert priors.scores[1] == pytest.approx(0.5)
    np.testing.assert_array_equal(priors.scores[2:], 0.5)
    assert priors.positive_counts[:2] == (1, 1)
    assert priors.negative_counts[:2] == (1, 1)
    assert priors.eligible_study_count == 2


def test_reference_priors_ignore_noneligible_and_held_out_rows() -> None:
    priors = compute_laplace_smoothed_reference_priors(
        {
            "gold-positive": _labels(acl=1),
            "gold-negative": _labels(acl=0),
            "weak-report": _labels(acl=1),
            "held-out-gold": _labels(acl=1),
        },
        eligible_study_uids=("gold-positive", "gold-negative"),
        held_out_study_uids=("held-out-gold",),
    )

    assert priors.scores[0] == pytest.approx(0.5)
    assert priors.positive_counts[0] == 1
    assert priors.negative_counts[0] == 1
    assert priors.eligible_study_uids == ("gold-negative", "gold-positive")


def test_reference_priors_reject_held_out_eligibility_and_invalid_labels() -> None:
    with pytest.raises(ValueError, match="held-out"):
        compute_laplace_smoothed_reference_priors(
            {"study": _labels(acl=1)},
            eligible_study_uids=("study",),
            held_out_study_uids=("study",),
        )

    with pytest.raises(ValueError, match="0, 1 or missing"):
        compute_laplace_smoothed_reference_priors(
            {"study": (2,) + (None,) * (NUM_TARGETS - 1)},
            eligible_study_uids=("study",),
        )


def test_reference_priors_require_an_explicit_eligible_set_for_nonempty_inputs() -> None:
    with pytest.raises(ValueError, match="eligible_study_uids is required"):
        compute_laplace_smoothed_reference_priors({"study": _labels(acl=1)})


def test_precomputed_reference_prior_must_match_its_laplace_counts() -> None:
    with pytest.raises(ValueError, match="Laplace-smoothed counts"):
        ReferenceFallbackPrior(
            scores=np.full(NUM_TARGETS, 0.75),
            positive_counts=(0,) * NUM_TARGETS,
            negative_counts=(0,) * NUM_TARGETS,
            eligible_study_count=0,
        )
