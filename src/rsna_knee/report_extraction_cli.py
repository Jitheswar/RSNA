"""Command-line entrypoint for ticket-15 report extraction."""

from __future__ import annotations

from .report_extraction import main


if __name__ == "__main__":
    raise SystemExit(main())
