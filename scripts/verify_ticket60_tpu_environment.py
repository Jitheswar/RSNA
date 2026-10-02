"""Independently reconcile a terminal environment probe, without launching work."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "artifacts/private/ticket60-tpu-environment-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def cell_sources(notebook):
    return [(cell["cell_type"], "".join(cell["source"])) for cell in notebook["cells"]]


def main():
    accepted = json.loads((ROOT / "accepted-launch.json").read_text())
    request = json.loads((ROOT / "request.json").read_text())
    intent = json.loads((ROOT / "launch-intent.json").read_text())
    assert accepted["request_sha256"] == digest(ROOT / "request.json") == intent["request_sha256"]
    assert accepted["response_sha256"] == digest(ROOT / "push-response.json")
    assert intent["gpu_ledger_sha256"] == digest(PROJECT / ".scratch/rsna-leaderboard/gpu-campaign-ledger.json")
    for item in request["frozen_files"]:
        assert digest(item["path"]) == item["sha256"], item["path"]
    api = KaggleApi()
    api.authenticate()
    status = json.loads(api.kernels_status(accepted["kernel_id"]).to_json(ignore_defaults=False))
    if status["status"] not in {"COMPLETE", "ERROR", "CANCELLED"}:
        raise RuntimeError("Probe still live: keep its exact observer; do not relaunch")
    query = ApiGetKernelRequest()
    query.user_name, query.kernel_slug = accepted["kernel_id"].split("/")
    with api.build_kaggle_client() as client:
        remote = json.loads(client.kernels.kernels_api_client.get_kernel(query).to_json(ignore_defaults=False))
    metadata = remote["metadata"]
    assert metadata["ref"] == accepted["kernel_id"]
    assert metadata["id"] == accepted["kernel_numeric_id"]
    assert metadata["currentVersionNumber"] == accepted["kernel_version"] == 1
    assert metadata["isPrivate"] and metadata["enableTpu"]
    assert not metadata["enableGpu"] and not metadata["enableInternet"]
    assert metadata["machineShape"] == "TpuV5E8"
    assert all(not metadata[key] for key in ("datasetDataSources", "kernelDataSources", "competitionDataSources", "modelDataSources"))
    assert cell_sources(json.loads((ROOT / "environment.ipynb").read_text())) == cell_sources(json.loads(remote["blob"]["source"]))
    output = ROOT / "terminal-output"
    if output.exists():
        raise FileExistsError("Retained output exists: inspect it before another retrieval")
    api.kernels_output(accepted["kernel_id"], str(output), quiet=True)
    receipts = list(output.rglob("tpu_environment_receipt.json"))
    if len(receipts) != 1:
        raise RuntimeError("No unique actual runtime receipt; preserve outputs and unknown allocation")
    receipt = json.loads(receipts[0].read_text())
    assert receipt["schema_version"] == "rsna-ticket60-tpu-environment-v1"
    assert receipt["gpu_requested"] is False and receipt["external_spend_usd"] == 0
    assert all(receipt[key] is False for key in ("model_inference", "fitting", "mri_inputs", "competition_submission"))
    verified = receipt["status"] == "ACTUAL_TPU_VERIFIED"
    if verified:
        assert status["status"] == "COMPLETE"
        assert receipt["actual_device_type"] == "TPU"
        assert receipt["global_runtime_devices"] == 8 and receipt["addressable_runtime_devices"] >= 1
        assert not receipt["cuda_available"] and receipt["cuda_device_count"] == 0
        calculation = receipt["calculation"]
        assert calculation["shape"] == [16, 16] and calculation["dtype"] == "torch.float32"
        assert calculation["minimum"] == calculation["maximum"] == 16.0
        assert calculation["sum"] == 4096.0
    else:
        assert receipt["status"] in {"ENVIRONMENT_ADMISSION_FAILED", "WATCHDOG_EXPIRED"}
        assert status["status"] == "ERROR"
    quota = json.loads(api.quota_view().to_json(ignore_defaults=False))
    write(ROOT / "quota-after.json", quota)
    write(ROOT / "remote-terminal.json", remote)
    result = {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
        "kernel_id": accepted["kernel_id"], "kernel_version": 1, "terminal_status": status,
        "exact_remote_source_and_settings": True, "actual_tpu_verified": verified,
        "actual_receipt": receipt, "receipt_sha256": digest(receipts[0]),
        "quota_after_sha256": digest(ROOT / "quota-after.json"),
        "quota_before_sha256": digest(ROOT / "quota-before.json"),
        "conservative_tpu_charge_seconds": 600, "gpu_ledger_unchanged": True,
        "predictive_gain_established": False, "training_campaign_authorized_by_result": False,
    }
    write(ROOT / "independent-verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
