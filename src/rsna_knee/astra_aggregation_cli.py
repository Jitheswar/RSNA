"""Aggregate a completed Ticket 16 Astra Medium run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .astra_aggregation import (
    AstraAggregationError,
    aggregate_astra_run,
    write_astra_aggregation_reports,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-plan", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--ticket15-labels", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = aggregate_astra_run(
            private_plan_path=args.private_plan,
            private_dir=args.private_dir,
            response_dir=args.responses,
            ticket15_labels_path=args.ticket15_labels,
            fold_manifest_path=args.fold_manifest,
            report_corpus_path=args.reports,
        )
        public, private, markdown = write_astra_aggregation_reports(
            result,
            public_path=args.public_output,
            private_path=args.private_output,
            markdown_path=args.markdown_output,
        )
        print(
            {
                "public": str(public),
                "private": str(private),
                "markdown": str(markdown) if markdown is not None else None,
                "semantic_output_digest": result.public["provenance"]["semantic_output_digest"],  # type: ignore[index]
            }
        )
        return 0
    except (AstraAggregationError, OSError, RuntimeError, ValueError) as exc:
        print(f"ticket-16 Astra aggregation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
