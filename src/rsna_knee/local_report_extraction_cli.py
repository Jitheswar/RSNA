"""Deprecated Ticket 16 local-checkpoint command.

Ticket 16 no longer loads FLAN-T5 or any other local model.  Use
``rsna_knee.astra_handoff_cli`` for the manual Astra Medium Codex workflow.
This shim remains only so an old command fails loudly instead of silently
running the superseded learned-extractor path.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    del argv
    print(
        "ticket-16 local-checkpoint extraction is deprecated; "
        "use python -m rsna_knee.astra_handoff_cli",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
