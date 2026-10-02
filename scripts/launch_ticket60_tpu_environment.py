"""Launch one frozen TPU environment admission; never retry an ambiguous push."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "artifacts/private/ticket60-tpu-environment-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main():
    request = json.loads((ROOT / "request.json").read_text())
    decision = json.loads((PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket60-tpu-environment-20261002.json").read_text())
    if (ROOT / "launch-intent.json").exists():
        raise FileExistsError("Consumed launch intent: observe exact handle rather than retry")
    assert decision["decision"] == "approved"
    assert decision["request_sha256"] == digest(ROOT / "request.json")
    assert decision["resources"] == request["resources"]
    for item in request["frozen_files"]:
        assert digest(item["path"]) == item["sha256"], item["path"]
    assert digest(ROOT / "environment.ipynb") == request["notebook_sha256"]
    assert digest(ROOT / "kernel-metadata.json") == request["metadata_sha256"]
    metadata = json.loads((ROOT / "kernel-metadata.json").read_text())
    assert metadata["id"] == request["kernel_handle"]
    assert metadata["is_private"] is True and metadata["enable_internet"] is False
    assert metadata["enable_gpu"] is False and metadata["enable_tpu"] is True
    assert metadata["machine_shape"] == "TpuV5E8"
    assert all(metadata[key] == [] for key in ("dataset_sources", "competition_sources", "kernel_sources", "model_sources"))
    assert request["resources"]["maximum_session_seconds"] == 600
    assert request["resources"]["maximum_launch_attempts"] == 1
    api = KaggleApi()
    api.authenticate()
    quota = json.loads(api.quota_view().to_json(ignore_defaults=False))
    assert quota["tpuQuota"]["timeUsed"] == "0s"
    assert quota["tpuQuota"]["timeReserved"] == "0s"
    assert quota["tpuQuota"]["totalTimeAllowed"] == "72000s"
    write(ROOT / "quota-before.json", quota)
    write(ROOT / "launch-intent.json", {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_sha256": digest(ROOT / "request.json"),
        "kernel_handle": request["kernel_handle"],
        "quota_before_sha256": digest(ROOT / "quota-before.json"),
        "gpu_ledger_sha256": digest(PROJECT / ".scratch/rsna-leaderboard/gpu-campaign-ledger.json"),
        "resources": request["resources"],
    })
    try:
        response = json.loads(api.kernels_push(str(ROOT), timeout="600", acc="TpuV5E8").to_json(ignore_defaults=False))
    except BaseException as exc:
        write(ROOT / "launch-exception.json", {"type": type(exc).__name__, "message": str(exc), "retry_allowed": False})
        raise
    write(ROOT / "push-response.json", response)
    canonical = response["ref"].removeprefix("/code/")
    if response.get("error") or canonical != request["kernel_handle"] or response["versionNumber"] != 1:
        raise RuntimeError("Unexpected accepted response: preserve exact response; do not retry")
    write(ROOT / "accepted-launch.json", {
        "kernel_id": canonical, "kernel_version": response["versionNumber"],
        "kernel_numeric_id": response["kernelId"], "request_sha256": digest(ROOT / "request.json"),
        "response_sha256": digest(ROOT / "push-response.json"),
        "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
