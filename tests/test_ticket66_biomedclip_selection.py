"""Pilot membership cannot depend on input ordering or abnormality labels."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location("ticket66_pilot",Path(__file__).resolve().parents[1]/"scripts/ticket66_biomedclip_mri_pilot.py")
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)


def test_selection_is_label_blind_and_order_independent():
    rows=[dict(study_instance_uid=str(i),source_kind="ticket35-prepared-series-cache-read-only",label=i%2) for i in range(20)]
    rows.append(dict(study_instance_uid="gold",source_kind="frozen-rsna-archive"))
    selected=pilot.select_studies(rows)
    expected=sorted([str(i) for i in range(20)],key=lambda uid:hashlib.sha256(uid.encode()).hexdigest())[:8]
    assert [row["study_instance_uid"] for row in selected]==expected
    for row in rows:row["label"]="changed"
    assert [row["study_instance_uid"] for row in pilot.select_studies(rows[::-1])]==expected


@pytest.mark.parametrize("uids",[["one"]*8,[str(i) for i in range(7)]])
def test_insufficient_or_duplicate_membership_rejected(uids):
    with pytest.raises(ValueError):
        pilot.select_studies([dict(study_instance_uid=uid,source_kind="ticket35-prepared-series-cache-read-only") for uid in uids])
