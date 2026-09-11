"""The competition's immutable target vector and output order."""

from __future__ import annotations


TARGET_COLUMNS: tuple[str, ...] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
NUM_TARGETS = len(TARGET_COLUMNS)
SUBMISSION_COLUMNS: tuple[str, ...] = ("StudyInstanceUID",) + TARGET_COLUMNS
