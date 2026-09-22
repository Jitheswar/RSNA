"""Command-line entrypoint for the controlled Ticket 18 OOF comparison."""

from __future__ import annotations

from .supervision_comparison import main


if __name__ == "__main__":
    raise SystemExit(main())
