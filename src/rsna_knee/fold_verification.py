"""V14 verification of a frozen fold manifest against its ticket-03 source.

The verifier is read-only with respect to the manifest.  It reloads the
checksum-covered private manifest, joins it to the private inventory,
regenerates it independently from the recorded generation parameters, and
recomputes every balance and support gate.  Its evidence is identifier-free.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from .archive import atomic_json_write
from .folds import (
    FOLD_GENERATION_METHOD,
    FOLD_IDS,
    FoldFreezeError,
    FoldManifest,
    evaluate_fold_validation_gates,
    freeze_folds,
    load_fold_manifest,
    load_grouping_assessment,
    load_inventory_source,
    source_manifest_checksum,
    validate_fold_manifest_against_source,
)
from .targets import TARGET_COLUMNS


V14_SCHEMA_VERSION = "rsna-knee-v14-verification-v1"


@dataclass(frozen=True)
class V14Check:
    """One named V14 criterion with an explicit outcome."""

    name: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        if self.status not in {"passed", "failed", "not_applicable"}:
            raise ValueError(f"unknown V14 check status {self.status!r}")

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def _check(name: str, passed: bool, detail: str) -> V14Check:
    return V14Check(name=name, status="passed" if passed else "failed", detail=detail)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_counts_by_fold(manifest: FoldManifest) -> dict[str, list[dict[str, object]]]:
    counts: dict[str, list[dict[str, object]]] = {}
    for index, target in enumerate(TARGET_COLUMNS):
        rows: list[dict[str, object]] = []
        for fold in FOLD_IDS:
            gold = [item for item in manifest.gold_memberships if item.fold == fold]
            rows.append(
                {
                    "fold": fold,
                    "gold_studies": len(gold),
                    "positive": sum(item.reference_labels[index] == 1 for item in gold),
                    "negative": sum(item.reference_labels[index] == 0 for item in gold),
                }
            )
        counts[target] = rows
    return counts


def _manifest_summary(manifest: FoldManifest) -> dict[str, object]:
    gates = evaluate_fold_validation_gates(
        manifest.memberships,
        manifest.grouping,
        deterministic_regeneration=True,
    )
    return {
        "manifest_checksum": manifest.manifest_checksum,
        "schema_version": manifest.schema_version,
        "generation_method": manifest.generation_method,
        "generation_version": manifest.generation_version,
        "superseded": manifest.superseded,
        "seed_name": manifest.seed_name,
        "seed": manifest.seed,
        "repeat_name": manifest.repeat_name,
        "fold_study_counts": [
            sum(item.fold == fold for item in manifest.memberships) for fold in FOLD_IDS
        ],
        "fold_gold_sizes": list(manifest.fold_gold_sizes),
        "fold_report_only_sizes": [
            sum(item.fold == fold for item in manifest.report_only_memberships) for fold in FOLD_IDS
        ],
        "gold_class_counts_by_fold": _class_counts_by_fold(manifest),
        "balance_gates_recomputed_from_memberships": {
            "gold_fold_size_balance": gates.size_balance_passed,
            "per_target_class_balance": gates.class_balance_passed,
            "both_class_support": gates.both_class_support_passed,
            "max_abs_class_deviation": gates.max_class_deviation,
            "failed_class_checks": sum(not item.passed for item in gates.class_balance),
            "failed_gates": [
                name for name in gates.failed_gates if name != "deterministic_generation"
            ],
        },
    }


def verify_v14(
    manifest_path: str | Path,
    inventory_path: str | Path,
    grouping_path: str | Path,
    *,
    public_evidence_path: str | Path | None = None,
    superseded_manifest_path: str | Path | None = None,
    expected_gold_count: int | None = None,
) -> dict[str, object]:
    """Return identifier-free V14 evidence; ``status`` is ``passed`` only if every check passes."""

    manifest_file = Path(manifest_path)
    checks: list[V14Check] = []
    manifest = load_fold_manifest(manifest_file)
    checks.append(
        _check(
            "manifest_checksum_and_self_consistency",
            True,
            "private manifest reloaded; checksum, fold summaries and stored gates re-derived",
        )
    )
    checks.append(
        _check(
            "current_generation_method",
            not manifest.superseded and manifest.generation_method == FOLD_GENERATION_METHOD,
            f"method {manifest.generation_method}",
        )
    )
    public_file = Path(public_evidence_path) if public_evidence_path is not None else None
    if public_file is not None:
        public = json.loads(public_file.read_text(encoding="utf-8"))
        checks.append(
            _check(
                "public_evidence_matches_manifest",
                public == manifest.to_public_dict(),
                "public evidence equals the evidence regenerated from the private manifest",
            )
        )

    source = load_inventory_source(inventory_path)
    checks.append(
        _check(
            "source_inventory_reconciled",
            source.full_corpus_reconciled,
            f"{source.source_manifest.study_count} source studies",
        )
    )
    try:
        validate_fold_manifest_against_source(manifest, source.source_manifest)
        join_ok, join_detail = True, "exact study set, cohort and all twelve reference labels agree"
    except (FoldFreezeError, ValueError) as exc:
        join_ok, join_detail = False, str(exc)
    checks.append(_check("membership_cohort_and_label_joins", join_ok, join_detail))
    uids = [item.study_instance_uid for item in manifest.memberships]
    checks.append(
        _check(
            "exactly_one_membership_per_eligible_study",
            len(uids) == len(set(uids)) == source.source_manifest.study_count,
            f"{len(uids)} memberships for {source.source_manifest.study_count} source studies",
        )
    )
    checks.append(
        _check(
            "valid_declared_fold_ids",
            manifest.fold_ids == FOLD_IDS and {item.fold for item in manifest.memberships} <= set(FOLD_IDS),
            f"declared {list(manifest.fold_ids)}",
        )
    )
    if expected_gold_count is not None:
        checks.append(
            _check(
                "expected_gold_count",
                len(manifest.gold_memberships) == expected_gold_count,
                f"{len(manifest.gold_memberships)} gold studies; expected {expected_gold_count}",
            )
        )

    grouping = load_grouping_assessment(grouping_path)
    checks.append(
        _check(
            "grouping_evidence_matches_manifest",
            grouping.to_private_dict() == manifest.grouping.to_private_dict(),
            f"grouping status {grouping.status}",
        )
    )
    if manifest.grouping.reliable:
        folds_by_group: dict[str, set[int]] = {}
        for item in manifest.memberships:
            folds_by_group.setdefault(item.group_key or "", set()).add(item.fold)
        split = sum(len(folds) > 1 for folds in folds_by_group.values())
        checks.append(
            _check("zero_reliable_group_overlap", split == 0, f"{split} groups span more than one fold")
        )
    else:
        checks.append(
            V14Check(
                name="zero_reliable_group_overlap",
                status="not_applicable",
                detail=(
                    "no reliable patient key; study-level separation is used and repeated-patient "
                    "leakage cannot be ruled out"
                ),
            )
        )

    try:
        regenerated = freeze_folds(
            source.source_manifest,
            source_version=source.source_version,
            source_fingerprint=source.source_fingerprint,
            input_inventory_reconciled=source.full_corpus_reconciled,
            grouping=grouping,
            seed_name=manifest.seed_name,
            seed=manifest.seed,
            repeat_name=manifest.repeat_name,
            repeat_justification=manifest.repeat_justification,
            generation_version=manifest.generation_version,
            supersedes=manifest.supersedes,
        )
        regeneration_ok = regenerated.manifest_checksum == manifest.manifest_checksum
        regeneration_detail = (
            "independent regeneration from the inventory reproduced the manifest checksum"
            if regeneration_ok
            else f"regenerated checksum {regenerated.manifest_checksum} differs"
        )
    except (FoldFreezeError, ValueError) as exc:
        regeneration_ok, regeneration_detail = False, f"regeneration failed: {exc}"
    checks.append(_check("deterministic_generation", regeneration_ok, regeneration_detail))

    summary = _manifest_summary(manifest)
    recomputed = cast(Mapping[str, object], summary["balance_gates_recomputed_from_memberships"])
    checks.append(
        _check(
            "gold_fold_size_balance",
            bool(recomputed["gold_fold_size_balance"]),
            f"gold fold sizes {summary['fold_gold_sizes']}",
        )
    )
    checks.append(
        _check(
            "per_target_class_balance",
            bool(recomputed["per_target_class_balance"]),
            f"max |count - expected| = {float(cast(float, recomputed['max_abs_class_deviation'])):.3f}",
        )
    )
    checks.append(
        _check(
            "both_class_support_where_feasible",
            bool(recomputed["both_class_support"]),
            "every feasible target has positive and negative gold studies in every fold",
        )
    )

    superseded_summary: dict[str, object] | None = None
    if superseded_manifest_path is not None:
        previous = load_fold_manifest(superseded_manifest_path)
        superseded_summary = _manifest_summary(previous)
        reference = manifest.supersedes
        provenance_ok = (
            reference is not None
            and reference.manifest_checksum == previous.manifest_checksum
            and (previous.seed_name, previous.seed, previous.repeat_name)
            == (manifest.seed_name, manifest.seed, manifest.repeat_name)
            and previous.source_manifest_checksum == manifest.source_manifest_checksum
        )
        checks.append(
            _check(
                "no_seed_search_single_correction",
                provenance_ok,
                "the manifest records the superseded checksum and reuses its source, named seed "
                "and repeat; only the generation method changed",
            )
        )

    gates = manifest.validation_gates
    status = "passed" if all(item.status != "failed" for item in checks) else "failed"
    return {
        "schema_version": V14_SCHEMA_VERSION,
        "ticket": "04",
        "gate": "V14",
        "status": status,
        "checks": [item.to_dict() for item in checks],
        "manifest": summary,
        "manifest_files": {
            "private_manifest_sha256": _sha256_file(manifest_file),
            "public_evidence_sha256": _sha256_file(public_file) if public_file is not None else None,
        },
        "source": {
            "source_version": manifest.source_version,
            "source_manifest_checksum": manifest.source_manifest_checksum,
            "inventory_source_manifest_checksum": source_manifest_checksum(source.source_manifest),
            "source_fingerprint": manifest.source_fingerprint,
        },
        "grouping": manifest.grouping.to_public_dict(),
        "counts": {
            "eligible_training_studies": len(manifest.memberships),
            "gold_studies": len(manifest.gold_memberships),
            "report_only_studies": len(manifest.report_only_memberships),
        },
        "pooled_gold_target_counts": manifest.pooled_gold_target_counts,
        "stored_validation_gates": gates.to_dict() if gates is not None else None,
        "supersedes": manifest.supersedes.to_dict() if manifest.supersedes is not None else None,
        "superseded_manifest": superseded_summary,
        "environment": {
            "python_version": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
        "limitations": [
            "Study-level separation cannot rule out repeated-patient leakage."
            if not manifest.grouping.reliable
            else "Grouping follows the attested reliable key.",
            "V14 verifies fold membership and support; it makes no model-performance claim.",
        ],
    }


def _parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--grouping-evidence", type=Path, required=True)
    parser.add_argument("--public-evidence", type=Path, default=None)
    parser.add_argument("--superseded-manifest", type=Path, default=None)
    parser.add_argument("--expected-gold-count", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--allow-local-smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        job_id = os.environ.get("SLURM_JOB_ID")
        if args.require_slurm and (not job_id or os.environ.get("SLURM_JOB_PARTITION") != "cluster1-cpu"):
            raise RuntimeError("V14 verification must run in a cluster1-cpu Slurm allocation")
        if not job_id and not args.allow_local_smoke:
            raise RuntimeError("refusing V14 verification outside Slurm without --allow-local-smoke")
        evidence = verify_v14(
            args.manifest,
            args.inventory,
            args.grouping_evidence,
            public_evidence_path=args.public_evidence,
            superseded_manifest_path=args.superseded_manifest,
            expected_gold_count=args.expected_gold_count,
        )
        atomic_json_write(args.output, evidence)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"V14 verification failed to run: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["status"] == "passed" else 1


__all__ = ["V14Check", "V14_SCHEMA_VERSION", "main", "verify_v14"]
