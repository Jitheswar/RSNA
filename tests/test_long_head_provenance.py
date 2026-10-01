"""Regression for historical absent-report provenance, without relaxing draws."""

import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ticket55_resnet_long_report_head import weak_sequence_matches
from rsna_knee.frozen_head_experiment import sequence_digest


def counts(digest, draws=0, entries=0):
    return {"weak_exposure_sequence_sha256": digest,
            "exposures_by_source": {"weak": {"sample_exposures": draws,
                                             "supervised_entry_exposures": entries}}}


def test_historical_null_matches_canonical_absence():
    assert weak_sequence_matches(counts(sequence_digest([])), counts(None))


def test_null_does_not_hide_actual_report_exposure():
    assert not weak_sequence_matches(counts(sequence_digest([]), 2, 4), counts(None))
    assert not weak_sequence_matches(counts(sequence_digest([])), counts(None, 2, 4))


def test_absent_sequence_requires_the_exact_empty_digest():
    assert not weak_sequence_matches(counts(sequence_digest(["report-a"])), counts(None))
    assert not weak_sequence_matches(counts(None), counts(None))


def test_report_sequence_identity_remains_strict():
    original = counts(sequence_digest(["report-a", "report-b"]), 2, 8)
    assert weak_sequence_matches(copy.deepcopy(original), original)
    assert not weak_sequence_matches(counts(sequence_digest(["report-b", "report-a"]), 2, 8), original)
    assert not weak_sequence_matches(counts(sequence_digest([])), original)
