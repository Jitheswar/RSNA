"""Shared naming policy for reliably identifying non-diagnostic series."""

from __future__ import annotations

import re
from typing import Iterable


NON_DIAGNOSTIC_TOKENS = (
    "scout",
    "localizer",
    "localiser",
    "survey",
    "pilot",
    "calibration",
)


def has_non_diagnostic_token(values: Iterable[str]) -> bool:
    """Return true only when a complete policy token appears in metadata."""

    return any(
        re.search(rf"(?<![\w]){re.escape(token)}(?![\w])", value.casefold())
        for value in values
        for token in NON_DIAGNOSTIC_TOKENS
    )


__all__ = ["NON_DIAGNOSTIC_TOKENS", "has_non_diagnostic_token"]
