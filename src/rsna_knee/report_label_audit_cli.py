"""CLI for freezing and applying the Ticket 30 report admission policy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence, cast

from .folds import FoldManifest
from .report_label_audit import (
    ReportLabelAuditError,
    _json_file,
    build_audit_artifacts,
    file_sha256,
    load_existing_artifacts,
    make_policy_payload,
    utc_now,
    validate_frozen_policy,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-policy", help="write the prospective policy before evaluation")
    freeze.add_argument("--output", type=Path, required=True)

    audit = subparsers.add_parser("audit-existing", help="audit only immutable stored Ticket 15/16/17 outputs")
    audit.add_argument("--policy", type=Path, required=True)
    audit.add_argument("--ticket04-manifest", type=Path, required=True)
    audit.add_argument("--ticket15-labels", type=Path, required=True)
    audit.add_argument("--ticket15-repeat-labels", type=Path)
    audit.add_argument("--ticket16-plan", type=Path, required=True)
    audit.add_argument("--ticket16-private-dir", type=Path, required=True)
    audit.add_argument("--ticket16-private-aggregation", type=Path, required=True)
    audit.add_argument("--ticket16-public-aggregation", type=Path, required=True)
    audit.add_argument("--ticket16-responses", type=Path, required=True)
    audit.add_argument("--ticket17-private-supervision", type=Path, required=True)
    audit.add_argument("--ticket17-public-supervision", type=Path, required=True)
    audit.add_argument("--private-output", type=Path, required=True)
    audit.add_argument("--public-output", type=Path, required=True)
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _audit_tree_sha256(root: Path) -> tuple[str, int]:
    files = sorted(root.glob("batches/*/attempt-*.audit.json"))
    if not files:
        raise ReportLabelAuditError("Ticket 16 immutable batch audit directory is empty")
    rows = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in files
    ]
    digest = hashlib.sha256(
        json.dumps(rows, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest, len(rows)


def _input_path_manifest(args: argparse.Namespace, loaded: Mapping[str, object]) -> dict[str, object]:
    paths = {
        "ticket04_fold_manifest": args.ticket04_manifest,
        "ticket15_labels": args.ticket15_labels,
        "ticket15_repeat_labels": args.ticket15_repeat_labels,
        "ticket16_plan": args.ticket16_plan,
        "ticket16_private_aggregation": args.ticket16_private_aggregation,
        "ticket16_public_aggregation": args.ticket16_public_aggregation,
        "ticket16_approved_prompt": args.ticket16_private_dir / "approved-prompt.json",
        "ticket16_response_directory": args.ticket16_responses,
        "ticket16_private_batch_directory": args.ticket16_private_dir,
        "ticket17_private_supervision": args.ticket17_private_supervision,
        "ticket17_public_supervision": args.ticket17_public_supervision,
    }
    file_keys = {
        "ticket04_fold_manifest": "manifest_sha256",
        "ticket15_labels": "ticket15_labels_sha256",
        "ticket15_repeat_labels": "ticket15_repeat_labels_sha256",
        "ticket16_plan": "ticket16_plan_sha256",
        "ticket16_private_aggregation": "ticket16_private_aggregation_sha256",
        "ticket16_public_aggregation": "ticket16_public_aggregation_sha256",
        "ticket16_approved_prompt": "ticket16_prompt_sha256",
        "ticket17_private_supervision": "ticket17_private_sha256",
        "ticket17_public_supervision": "ticket17_public_sha256",
    }
    paths_output: dict[str, object] = {}
    for name, path in paths.items():
        if path is None:
            continue
        resolved = Path(path).resolve()
        entry: dict[str, object] = {"path": str(resolved)}
        digest_key = file_keys.get(name)
        if digest_key is not None:
            entry["sha256"] = loaded.get(digest_key)
        elif name == "ticket16_response_directory":
            entry["response_manifest_sha256"] = hashlib.sha256(
                json.dumps(
                    loaded["ticket16_response_manifest"],
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        elif name == "ticket16_private_batch_directory":
            tree_hash, file_count = _audit_tree_sha256(args.ticket16_private_dir)
            entry.update({"audit_tree_sha256": tree_hash, "audit_file_count": file_count})
        paths_output[name] = entry
    private_aggregation = cast(Mapping[str, object], loaded["ticket16_private_aggregation"])
    private_inputs = cast(Mapping[str, object], private_aggregation["private_inputs"])
    report_corpus_path = Path(cast(str, private_inputs["report_corpus_path"])).resolve()
    paths_output["ticket16_source_report_corpus"] = {
        "path": str(report_corpus_path),
        "sha256": loaded["ticket16_report_corpus_sha256"],
        "purpose": "byte identity only; report text was not decoded or interpreted",
    }
    return paths_output


def _freeze(args: argparse.Namespace) -> int:
    payload = make_policy_payload()
    destination = write_immutable_json(args.output, payload, mode=0o600)
    print(json.dumps({"path": str(destination.resolve()), "sha256": file_sha256(destination), "policy_sha256": payload["policy_sha256"]}))
    return 0


def _audit(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    if _inside(args.private_output, repo_root):
        raise ReportLabelAuditError("private admitted labels must be written outside the repository")
    if args.private_output.resolve() == args.public_output.resolve():
        raise ReportLabelAuditError("public and private outputs must use separate paths")
    policy_payload = _json_file(args.policy, "frozen policy")
    policy = validate_frozen_policy(policy_payload)
    frozen_at = datetime.fromisoformat(cast(str, policy["frozen_at_utc"]).replace("Z", "+00:00"))
    audit_started_at = datetime.now(timezone.utc)
    if frozen_at > audit_started_at:
        raise ReportLabelAuditError("policy timestamp is later than held-out evaluation start")
    loaded = load_existing_artifacts(
        ticket15_labels_path=args.ticket15_labels,
        ticket15_repeat_labels_path=args.ticket15_repeat_labels,
        ticket16_plan_path=args.ticket16_plan,
        ticket16_private_dir=args.ticket16_private_dir,
        ticket16_private_aggregation_path=args.ticket16_private_aggregation,
        ticket16_public_aggregation_path=args.ticket16_public_aggregation,
        ticket16_response_dir=args.ticket16_responses,
        ticket17_private_supervision_path=args.ticket17_private_supervision,
        ticket17_public_supervision_path=args.ticket17_public_supervision,
        fold_manifest_path=args.ticket04_manifest,
    )
    private, public = build_audit_artifacts(
        frozen_policy=policy,
        manifest=cast(FoldManifest, loaded["manifest"]),
        ticket15_labels=cast(Mapping[str, object], loaded["ticket15_labels"]),
        ticket15_labels_sha256=cast(str, loaded["ticket15_labels_sha256"]),
        ticket15_repeat_labels=cast(Mapping[str, object] | None, loaded["ticket15_repeat_labels"]),
        ticket15_repeat_labels_sha256=cast(str | None, loaded["ticket15_repeat_labels_sha256"]),
        ticket16_plan=loaded["ticket16_plan"],
        ticket16_plan_metadata=cast(Mapping[str, object], loaded["ticket16_plan_metadata"]),
        ticket16_audits=cast(Sequence[Any], loaded["ticket16_audits"]),
        ticket16_private_aggregation=cast(Mapping[str, object], loaded["ticket16_private_aggregation"]),
        ticket16_public_aggregation=cast(Mapping[str, object], loaded["ticket16_public_aggregation"]),
        ticket16_private_aggregation_sha256=cast(str, loaded["ticket16_private_aggregation_sha256"]),
        ticket16_public_aggregation_sha256=cast(str, loaded["ticket16_public_aggregation_sha256"]),
        ticket16_plan_sha256=cast(str, loaded["ticket16_plan_sha256"]),
        ticket16_prompt_sha256=cast(str, loaded["ticket16_prompt_sha256"]),
        ticket16_response_manifest=cast(Mapping[str, object], loaded["ticket16_response_manifest"]),
        ticket17_private_supervision=cast(Mapping[str, object], loaded["ticket17_private_supervision"]),
        ticket17_public_supervision=cast(Mapping[str, object], loaded["ticket17_public_supervision"]),
        ticket17_private_sha256=cast(str, loaded["ticket17_private_sha256"]),
        ticket17_public_sha256=cast(str, loaded["ticket17_public_sha256"]),
        ticket04_manifest_sha256=cast(str, loaded["manifest_sha256"]),
        ticket16_report_corpus_sha256=cast(str, loaded["ticket16_report_corpus_sha256"]),
    )
    input_paths = _input_path_manifest(args, loaded)
    audit_meta = {
        "audit_started_at_utc": audit_started_at.isoformat().replace("+00:00", "Z"),
        "frozen_policy_artifact_path": str(args.policy.resolve()),
        "frozen_policy_artifact_sha256": file_sha256(args.policy),
        "input_artifacts": input_paths,
    }
    private["audit"] = audit_meta
    public["audit"] = {
        "audit_started_at_utc": audit_meta["audit_started_at_utc"],
        "frozen_policy_artifact_sha256": audit_meta["frozen_policy_artifact_sha256"],
        "ticket16_batch_audit_file_count": cast(
            Mapping[str, object], input_paths["ticket16_private_batch_directory"]
        )["audit_file_count"],
        "ticket16_batch_audit_tree_sha256": cast(
            Mapping[str, object], input_paths["ticket16_private_batch_directory"]
        )["audit_tree_sha256"],
    }
    private_path = write_immutable_json(args.private_output, private, mode=0o600)
    public_path = write_immutable_json(args.public_output, public, mode=0o644)
    print(
        json.dumps(
            {
                "private_path": str(private_path.resolve()),
                "private_sha256": file_sha256(private_path),
                "public_path": str(public_path.resolve()),
                "public_sha256": file_sha256(public_path),
                "policy_sha256": policy["policy_sha256"],
                "folds": 3,
                "report_text_decoded_or_interpreted": False,
                "new_remote_report_processing": False,
                "image_training_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze-policy":
            return _freeze(args)
        if args.command == "audit-existing":
            return _audit(args)
        raise ReportLabelAuditError("unknown audit command")
    except (ReportLabelAuditError, OSError) as exc:
        print(f"report-label-audit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
