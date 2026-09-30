"""Observe one exact saved notebook version without launching or retrying."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest


def main():
    root = Path(sys.argv[1]).resolve()
    accepted = json.loads((root / "accepted-launch.json").read_text())
    handle = accepted["kernel_id"]
    api = KaggleApi()
    api.authenticate()
    while True:
        observation = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "kernel_id": handle, "kernel_version": accepted["kernel_version"]}
        try:
            raw = json.loads(api.kernels_status(handle).to_json(ignore_defaults=False))
            request = ApiGetKernelRequest()
            request.user_name, request.kernel_slug = handle.split("/")
            with api.build_kaggle_client() as client:
                value = client.kernels.kernels_api_client.get_kernel(request)
            metadata = json.loads(value.to_json(ignore_defaults=False))["metadata"]
            if metadata["ref"] != handle or metadata["currentVersionNumber"] != accepted["kernel_version"]:
                raise ValueError("canonical kernel version changed")
            observation.update(status=raw["status"], failure_message=raw.get("failureMessage"),
                               kernel_numeric_id=metadata["id"], status_response=raw)
            temporary = root / "kernel-status.json.partial"
            temporary.write_text(json.dumps(observation, indent=2) + "\n")
            temporary.replace(root / "kernel-status.json")
            if observation["status"] in {"COMPLETE", "ERROR", "CANCELLED"}:
                with (root / "kernel-terminal.json").open("x") as handle_file:
                    json.dump(observation, handle_file, indent=2)
                    handle_file.write("\n")
                with (root / "kernel-observations.jsonl").open("a") as handle_file:
                    handle_file.write(json.dumps(observation) + "\n")
                return
        except Exception as exc:
            observation["observation_error"] = f"{type(exc).__name__}: {exc}"
        with (root / "kernel-observations.jsonl").open("a") as handle_file:
            handle_file.write(json.dumps(observation) + "\n")
        print(json.dumps(observation), flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
