# Establish reliable supervision options without clinical review

Parent: ../map.md
Type: research
Labels: wayfinder:research
Status: resolved
Assignee: supervision_research
Blocked by: none

Research branch: research/rsna-supervision-options
Research worktree: /tmp/rsna-wayfinder-supervision-options

## Question

Given 58 reference-labelled and 4,349 report-only training studies, no clinical reviewer, zero external spending, and report-free inference, which supervision approaches deserve controlled evaluation? Use primary research and exact public model/resource documentation. Address knee-label semantics, missing mention versus negatives, extraction reliability, fold isolation, multilingual uncertainty, and local resource needs. Inspect existing inconclusive extraction/weak-supervision evidence to avoid proposing the same experiment as a proven advance. Produce a bounded shortlist with uncertainties and evaluation requirements, not invented clinical labels or claimed performance.

## Answer

Resolved 2026-09-25 by the research agent. Preserve reference-only and deterministic comparators; consider existing learned outputs only after provenance and prospective admission review, or evaluate one local multilingual extractor with abstention. `Qwen/Qwen3-8B` is a concrete research candidate, not a selected or approved resource. Evaluate new coverage separately from agreement, keep all report adaptation isolated by fold, and report target-level support and uncertainty. No clinical reviewer is available to adjudicate report/reference disagreement. Image–report pretraining should remain a later hypothesis until useful image learning is established.

Asset: [Reliable supervision options without clinical review](../../../docs/research/leaderboard-supervision-options.md).

Context pointer: branch `research/rsna-supervision-options`, commit `9aae53d`, report `docs/research/leaderboard-supervision-options.md`. Eligibility, exact revisions, clinical-definition freshness, language coverage, grouping, and actual extraction costs remain unresolved. No reports were transmitted and no weights or experiments were run.
