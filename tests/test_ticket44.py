from pathlib import Path
from types import SimpleNamespace

import pytest

from rsna_knee import ticket44
from rsna_knee.training import TrainingConfig


def test_completed_training_requires_a_final_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ticket44, "RUN_ROOT", tmp_path / "runs")
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-job")
    monkeypatch.setattr(
        ticket44,
        "train_one_fold",
        lambda *args, **kwargs: SimpleNamespace(final_checkpoint_path=None, model=None),
    )
    integration = SimpleNamespace(
        source_version="fixture-source",
        archive_fingerprint="a" * 64,
        manifest_checksum="b" * 64,
    )

    with pytest.raises(ticket44.Ticket44Error, match="final checkpoint"):
        ticket44._run_fold_arm(
            "reference-only", 0, {}, None, integration, TrainingConfig(), (), None, None,
            {"feature_manifest_sha256": "c" * 64},
            {"feature_identity_fingerprint": "d" * 64},
        )
