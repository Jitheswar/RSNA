"""Command-line entrypoint for ticket-10 OOF aggregation."""

from __future__ import annotations

from .oof import main


if __name__ == "__main__":
    raise SystemExit(main())
