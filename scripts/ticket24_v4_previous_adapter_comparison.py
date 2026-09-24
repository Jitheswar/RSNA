"""Run the archived v3 adapter on the representative set for same-host comparison."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
V3_BUNDLE = REPO / "artifacts/private/ticket-24-kaggle-delivery-20260923-v3"
V3_CODE = V3_BUNDLE / "code"
CANDIDATE = V3_BUNDLE / "candidate"
INPUT = REPO / "artifacts/private/ticket-12-representative-input-20260912-v12"
OUTPUT_ROOT = REPO / "artifacts/private/ticket-24-v4-validation/representative-158-previous-v3-adapter"
ADAPTER = V3_CODE / "rsna_knee/kaggle.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite archived-adapter comparison: {OUTPUT_ROOT}")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    sys.path.insert(0, str(V3_CODE))
    from rsna_knee.kaggle import run_kaggle_baseline

    OUTPUT_ROOT.mkdir(parents=True)
    output = OUTPUT_ROOT / "submission.csv"
    evidence = OUTPUT_ROOT / "kaggle-execution.json"
    private_evidence = OUTPUT_ROOT / "private-execution.json"
    result = run_kaggle_baseline(
        CANDIDATE,
        INPUT,
        output,
        evidence_path=evidence,
        private_evidence_path=private_evidence,
        allow_local_smoke=True,
        device="cpu",
    )
    print(json.dumps({
        "status": "completed_previous_adapter_comparison",
        "archived_v3_adapter_sha256": sha256(ADAPTER),
        "study_count": result.evidence["workload"]["study_count"],
        "series_count": result.evidence["workload"]["series_count"],
        "slice_count": result.evidence["workload"]["slice_count"],
        "decode_failure_count": result.evidence["output"]["decode_failure_count"],
        "fallback_count": result.evidence["fallback"]["numerator"],
        "peak_rss_bytes": result.evidence["resources"]["peak_ram_bytes"],
        "csv_sha256": sha256(output),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
