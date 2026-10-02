"""Acquire one pinned BiomedCLIP asset; never construct a model or read MRI data."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time
from urllib.request import urlopen


PROJECT = Path(os.environ.get("RSNA_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
ROOT = PROJECT / "artifacts/private/ticket65-biomedclip-asset-admission-v1"
MODEL = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
REVISION = "9f341de24bfb00180f1b847274256e9b65a3a32e"
WEIGHT_SHA = "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be"
WEIGHT_BYTES = 783705670
APPROVAL = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket65-biomedclip-assets-20261002.json"
AUTHORIZATION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
RESEARCH = PROJECT / "artifacts/private/medical-image-encoder-options-20261002"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def checked_identity(path: Path, expected_bytes: int, expected_sha: str) -> None:
    if path.stat().st_size != expected_bytes or digest(path) != expected_sha:
        raise ValueError("Asset byte count or complete digest differs")


def tensor_state(value: object) -> tuple[dict, str, bool]:
    import torch

    wrapper = "flat"
    if isinstance(value, Mapping) and "state_dict" in value:
        value = value["state_dict"]
        wrapper = "state_dict"
    if not isinstance(value, Mapping) or not value:
        raise ValueError("Checkpoint has no supported tensor state mapping")
    if any(not isinstance(key, str) or not isinstance(tensor, torch.Tensor)
           for key, tensor in value.items()):
        raise ValueError("State contains a non-string key or non-tensor value")
    prefixes = [key.startswith("module.") for key in value]
    if any(prefixes) and not all(prefixes):
        raise ValueError("Mixed module prefixes are unsupported")
    prefixed = all(prefixes)
    state = {key.removeprefix("module.") if prefixed else key: tensor
             for key, tensor in value.items()}
    if not any(key.startswith("visual.") for key in state):
        raise ValueError("Checkpoint contains no vision namespace")
    return state, wrapper, prefixed


def prepare() -> None:
    if (ROOT / "plan.json").exists() or APPROVAL.exists():
        raise FileExistsError("Asset admission already frozen")
    rules = ROOT / "rules/current-rules.txt"
    body = rules.read_text()
    if "The use of external data and models is acceptable" not in body:
        raise ValueError("Fresh official external-model rules unavailable")
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=False)
    base = f"https://huggingface.co/{MODEL}/resolve/{REVISION}"
    with urlopen(f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true", timeout=60) as response:
        metadata_bytes = response.read(2_000_001)
    if len(metadata_bytes) > 2_000_000:
        raise ValueError("Metadata byte limit exceeded")
    (evidence / "model-api.json").write_bytes(metadata_bytes)
    metadata = json.loads(metadata_bytes)
    weight = next(row for row in metadata["siblings"] if row["rfilename"] == "open_clip_pytorch_model.bin")
    if (metadata["sha"] != REVISION or metadata["private"] or metadata["gated"] is not False
            or weight["size"] != WEIGHT_BYTES or weight["lfs"]["sha256"] != WEIGHT_SHA):
        raise ValueError("Pinned public artifact metadata differs")
    for name in ("LICENSE.md", "README.md", "open_clip_config.json"):
        with urlopen(f"{base}/{name}", timeout=60) as response:
            content = response.read(1_000_001)
        if len(content) > 1_000_000 or hashlib.sha256(content).hexdigest() != digest(RESEARCH / f"microsoft--{name}"):
            raise ValueError("Exact source evidence differs from reviewed revision")
        (evidence / name).write_bytes(content)
    usage = subprocess.check_output(["du", "--apparent-size", "-s", "-B1", str(PROJECT),
        str(PROJECT.parent / "models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218")], text=True)
    existing = sum(int(row.split()[0]) for row in usage.splitlines())
    recovery = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket64-full-backbone-recovery-20261002.json"
    reserve = json.loads(recovery.read_text())["projection"]["checkpoint_storage_allowance_bytes"] + 5_000_000_000
    allowance = WEIGHT_BYTES + 16_000_000
    if existing + reserve + allowance >= 2_000_000_000_000 or shutil.disk_usage(PROJECT).free < allowance:
        raise ValueError("Asset acquisition does not fit reserved project storage")
    snapshot = ROOT / "asset-admission.py"
    shutil.copyfile(Path(__file__), snapshot)
    fixed = [AUTHORIZATION, recovery, rules, ROOT / "rules/capture.json", snapshot, *sorted(evidence.iterdir())]
    plan = dict(schema_version="rsna-ticket65-asset-admission-v1", created_at_utc=timestamp(),
        model=MODEL, revision=REVISION, checkpoint_bytes=WEIGHT_BYTES, checkpoint_sha256=WEIGHT_SHA,
        download_url=f"{base}/open_clip_pytorch_model.bin", checkpoint_path=str(ROOT / "weights/open_clip_pytorch_model.bin"),
        source_sha256=digest(snapshot), authorization_sha256=digest(AUTHORIZATION),
        frozen_files=[dict(path=str(path), sha256=digest(path)) for path in fixed],
        storage=dict(du_output=usage, existing_apparent_bytes=existing,
            retained_ticket64_allowance_bytes=reserve, asset_allowance_bytes=allowance,
            projected_apparent_bytes=existing + reserve + allowance, soft_limit_bytes=2_000_000_000_000),
        resources=dict(maximum_attempts=1, watchdog_seconds=900, socket_timeout_seconds=60,
            cpu_threads=1, maximum_peak_rss_bytes=4 * 1024**3, virtual_address_limit_bytes=12 * 1024**3,
            scheduler_jobs=0, gpu_count=0, tpu_count=0, external_spend_usd=0),
        scope="Pinned asset acquisition, full hash, restricted CPU tensor inventory only",
        model_forward=False, mri_data_read=False, labels_read=False, installs=False)
    save(ROOT / "plan.json", plan)
    save(APPROVAL, dict(schema_version="rsna-ticket65-resource-decision-v1", decision="approved",
        decision_made_by="campaign agent under standing owner authorization",
        authorization_sha256=digest(AUTHORIZATION), new_exact_owner_numeric_approval_claimed=False,
        plan_sha256=digest(ROOT / "plan.json"), checkpoint_sha256=WEIGHT_SHA,
        source_revision=REVISION, license="MIT", rights_status="eligible_for_private_asset_investigation",
        rights_basis="Ungated no-cost public checkpoint; exact Microsoft weight license permits research use. Retain license/copyright. Model-card deployment is out of scope. This private investigation makes no third-party rights grant or final-delivery eligibility claim.",
        resources=plan["resources"], storage=plan["storage"],
        pilot_execution_authorized=False, full_extraction_authorized=False, fitting_authorized=False,
        publication_authorized=False, final_candidate_selection=False,
        gpu_ledger_changed=False, tpu_ledger_changed=False))
    print(json.dumps(dict(plan_sha256=digest(ROOT / "plan.json"), prepared=True)))


def acquire(plan_sha: str) -> None:
    plan_path = ROOT / "plan.json"
    approval = json.loads(APPROVAL.read_text())
    if (digest(plan_path) != plan_sha or approval["plan_sha256"] != plan_sha
            or approval["decision"] != "approved"
            or approval["authorization_sha256"] != digest(AUTHORIZATION)):
        raise ValueError("Exact admission identity required")
    plan = json.loads(plan_path.read_text())
    for row in plan["frozen_files"]:
        if digest(Path(row["path"])) != row["sha256"]:
            raise ValueError("Frozen admission evidence changed")
    if digest(Path(__file__)) != plan["source_sha256"]:
        raise ValueError("Frozen asset worker required")
    resources = plan["resources"]
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Asset watchdog expired")))
    signal.alarm(resources["watchdog_seconds"])
    resource.setrlimit(resource.RLIMIT_AS, (resources["virtual_address_limit_bytes"], resources["virtual_address_limit_bytes"]))
    save(ROOT / "attempt.json", dict(plan_sha256=plan_sha, started_at_utc=timestamp(), maximum_attempts=1))
    started = time.monotonic()
    path = Path(plan["checkpoint_path"])
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix(".part")
    try:
        count = 0
        with urlopen(plan["download_url"], timeout=resources["socket_timeout_seconds"]) as response, temporary.open("xb") as output:
            while chunk := response.read(4 * 1024**2):
                count += len(chunk)
                if count > WEIGHT_BYTES:
                    raise ValueError("Checkpoint exceeds registered byte limit")
                output.write(chunk)
        checked_identity(temporary, WEIGHT_BYTES, WEIGHT_SHA)
        if path.exists():
            raise FileExistsError("Checkpoint already published")
        temporary.rename(path)
        import torch

        torch.set_num_threads(1)
        state, wrapper, prefixed = tensor_state(torch.load(path, map_location="cpu", weights_only=True))
        rows = []
        for key, tensor in sorted(state.items()):
            if tensor.device.type != "cpu" or not bool(torch.isfinite(tensor).all()):
                raise ValueError("Nonfinite or non-CPU checkpoint tensor")
            rows.append(dict(key=key, shape=list(tensor.shape), dtype=str(tensor.dtype),
                bytes=tensor.numel() * tensor.element_size(), visual=key.startswith("visual.")))
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if peak > resources["maximum_peak_rss_bytes"]:
            raise ValueError("Asset inspection peak memory exceeds admission")
        result = dict(status="COMPLETE", plan_sha256=plan_sha, completed_at_utc=timestamp(),
            checkpoint_sha256=WEIGHT_SHA, checkpoint_bytes=path.stat().st_size,
            restricted_load=True, unrestricted_pickle_fallback=False, wrapper=wrapper, module_prefix=prefixed,
            torch_version=torch.__version__, tensor_count=len(rows),
            vision_tensor_count=sum(row["visual"] for row in rows),
            vision_tensor_bytes=sum(row["bytes"] for row in rows if row["visual"]),
            tensors=rows, elapsed_seconds=time.monotonic()-started, peak_rss_bytes=peak,
            strict_model_coverage_verified=False, model_forward=False, mri_data_read=False,
            labels_read=False, gpu_seconds=0, tpu_seconds=0, external_spend_usd=0)
        save(ROOT / "result.json", result)
        print(json.dumps({key:value for key,value in result.items() if key != "tensors"}))
    except Exception as error:
        save(ROOT / "failed-attempt.json", dict(status="FAILED", plan_sha256=plan_sha,
            failure_type=type(error).__name__, message=str(error), elapsed_seconds=time.monotonic()-started,
            automatic_retry=False))
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "acquire"))
    parser.add_argument("plan_sha256", nargs="?")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.plan_sha256:
        acquire(args.plan_sha256)
    else:
        parser.error("acquire requires the exact frozen plan SHA-256")
