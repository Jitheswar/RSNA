"""Command-line entrypoint for ticket-05 real-DICOM integration."""

from __future__ import annotations

from .real_inference import main


if __name__ == "__main__":
    raise SystemExit(main())
