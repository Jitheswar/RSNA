"""Pin and acquire one isolated runtime; no model forward or scheduler launch."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import urlopen

PROJECT = Path(os.environ.get("RSNA_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
ROOT = PROJECT / "artifacts/private/ticket66-biomedclip-mri-pilot-v1"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket66-biomedclip-runtime-20261002.json"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
VERSIONS = {"open_clip_torch": "2.23.0", "transformers": "4.35.2", "tokenizers": "0.15.2",
            "ftfy": "6.3.1", "sentencepiece": "0.2.1", "protobuf": "4.25.8"}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def prepare():
    if ROOT.exists() or DECISION.exists():
        raise FileExistsError("Isolated runtime preparation already exists")
    ROOT.mkdir(mode=0o700)
    evidence = ROOT / "runtime-evidence"; evidence.mkdir()
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
    supported = set(sys_tags())
    wheels = []
    for name, version in VERSIONS.items():
        with urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60) as response:
            content = response.read(2_000_001)
        if len(content) > 2_000_000:
            raise ValueError("Package metadata cap exceeded")
        path = evidence / f"{name}-{version}.json"; path.write_bytes(content)
        metadata = json.loads(content)
        choices = [row for row in metadata["urls"] if row["packagetype"] == "bdist_wheel"
            and not row["yanked"] and supported.intersection(parse_wheel_filename(row["filename"])[3])]
        if not choices:
            raise ValueError(f"No exact compatible wheel for {name}")
        row = sorted(choices, key=lambda item: item["filename"])[0]
        wheels.append(dict(name=name, version=version, filename=row["filename"],
            url=row["url"], bytes=row["size"], sha256=row["digests"]["sha256"],
            metadata_path=str(path), metadata_sha256=digest(path),
            license=metadata["info"].get("license"), classifiers=metadata["info"].get("classifiers")))
    if sum(row["bytes"] for row in wheels) > 64_000_000:
        raise ValueError("Exact runtime wheels exceed download cap")
    usage = subprocess.check_output(["du", "--apparent-size", "-s", "-B1", str(PROJECT),
        str(PROJECT.parent / "models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218")], text=True)
    existing = sum(int(row.split()[0]) for row in usage.splitlines())
    recovery = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket64-full-backbone-recovery-20261002.json"
    reserve = json.loads(recovery.read_text())["projection"]["checkpoint_storage_allowance_bytes"] + 5_000_000_000
    allowance = 2_000_000_000
    if existing + reserve + allowance >= 2_000_000_000_000 or shutil.disk_usage(PROJECT).free < allowance:
        raise ValueError("Runtime and pilot payload cannot fit protected storage")
    snapshot = ROOT / "runtime-admission.py"; shutil.copyfile(__file__, snapshot)
    plan = dict(schema_version="rsna-ticket66-runtime-admission-v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
        wheels=wheels, source_sha256=digest(snapshot), authorization_sha256=digest(AUTH),
        dependency_sites=[str(PROJECT / "artifacts/private/ticket48-convnext-cpu/runtime/lib/python3.11/site-packages"),
                          str(PROJECT / ".venv/lib/python3.11/site-packages")],
        resources=dict(wheel_download_cap_bytes=64_000_000, local_allowance_bytes=allowance,
            maximum_attempts=1, external_spend_usd=0, scheduler_jobs=0, model_forward=False),
        storage=dict(existing_apparent_bytes=existing, du_output=usage, retained_ticket64_allowance_bytes=reserve,
            additional_allowance_bytes=allowance, projected_apparent_bytes=existing+reserve+allowance),
        pilot_execution_authorized=False, fitting_authorized=False)
    write(ROOT / "runtime-plan.json", plan)
    write(DECISION, dict(schema_version="rsna-campaign-autonomous-decision-v1", decision="approved",
        decision_made_by="campaign agent under standing owner authorization", authorization_sha256=digest(AUTH),
        plan_sha256=digest(ROOT / "runtime-plan.json"), resources=plan["resources"], storage=plan["storage"],
        basis="Isolated exact official library runtime after independent pinned-weight admission; no current environment change or model execution.",
        new_exact_owner_numeric_approval_claimed=False, gpu_ledger_changed=False, tpu_ledger_changed=False,
        pilot_execution_authorized=False, full_extraction_authorized=False, fitting_authorized=False))
    print(json.dumps(dict(plan_sha256=digest(ROOT / "runtime-plan.json"), wheel_bytes=sum(row["bytes"] for row in wheels))))


def acquire(checksum):
    plan_path = ROOT / "runtime-plan.json"; plan = json.loads(plan_path.read_text()); approval=json.loads(DECISION.read_text())
    if (digest(plan_path) != checksum or approval["plan_sha256"] != checksum or approval["decision"] != "approved"
            or digest(__file__) != plan["source_sha256"] or digest(AUTH) != plan["authorization_sha256"]):
        raise ValueError("Frozen runtime admission required")
    write(ROOT / "runtime-attempt.json", dict(plan_sha256=checksum, started_at_utc=datetime.now(timezone.utc).isoformat()))
    directory = ROOT / "wheels"; directory.mkdir()
    for row in plan["wheels"]:
        if digest(row["metadata_path"]) != row["metadata_sha256"]:
            raise ValueError("Pinned wheel metadata changed")
        path = directory / row["filename"]
        count = 0
        with urlopen(row["url"], timeout=60) as response, path.open("xb") as output:
            while chunk := response.read(1024**2):
                count += len(chunk)
                if count > row["bytes"]:
                    raise ValueError("Wheel byte cap exceeded")
                output.write(chunk)
        if count != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Wheel full publisher identity differs")
    runtime = ROOT / "runtime"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(runtime)], check=True, timeout=60)
    site = runtime / "lib/python3.11/site-packages"
    (site / "rsna-admitted-cpu-dependencies.pth").write_text("\n".join(plan["dependency_sites"])+"\n")
    subprocess.run([str(PROJECT / ".venv/bin/python"), "-m", "pip", "--python", str(runtime / "bin/python"),
        "install", "--no-index", "--no-deps", *[str(directory / row["filename"]) for row in plan["wheels"]]],
        check=True, timeout=120)
    write(ROOT / "runtime-acquisition.json", dict(status="COMPLETE", plan_sha256=checksum,
        completed_at_utc=datetime.now(timezone.utc).isoformat(), wheel_count=len(plan["wheels"]),
        wheel_bytes=sum(row["bytes"] for row in plan["wheels"]), model_forward=False,
        existing_environments_changed=False, gpu_seconds=0, tpu_seconds=0, external_spend_usd=0))
    print(json.dumps(dict(status="COMPLETE", runtime=str(runtime))))


if __name__ == "__main__":
    if sys.argv[1] == "prepare": prepare()
    elif sys.argv[1] == "acquire": acquire(sys.argv[2])
    else: raise ValueError("prepare or acquire required")
