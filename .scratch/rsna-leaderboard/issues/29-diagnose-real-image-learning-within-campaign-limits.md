# 29: Diagnose real-image learning within campaign limits

**What to build:** A bounded real-data diagnostic determines whether the current input and optimization paths can learn training examples, while producing the storage and resource accounting needed to plan subsequent experiments.

**Blocked by:** None (can start immediately).

**Parent:** [RSNA leaderboard campaign with zero external spend](../spec.md)

**Status:** resolved

**Execution constraints:** Follow the approved spec: zero external spend; significant work through Slurm; existing CPU ceilings; one active or pending GPU request and 20 allocated GPU-hours across the first campaign, including failures and retries; 2 TB soft storage budget; exact resource and delivery approvals where required. Questions go in visible chat. Synthetic checks establish software behavior only. No project-imposed candidate or submission-count cap.

## Acceptance criteria

- [x] Reconcile the selected image/source/cache identity, corrected version-2 folds and twelve-target order; distinguish the 158 prepared studies from the 4,407-study header inventory.
- [x] Inspect real training-side pixels, geometry, anatomical coverage, masks and label alignment without publishing images, reports or identifiers.
- [x] Freeze a tiny training-only cohort, numerical learning checks and bounded step/time budgets before execution; record loss, gradient/parameter change and short-versus-longer learning curves.
- [x] Record whether optimization competence is demonstrated and the evidence for any failure; do not interpret memorization as held-out generalization. Repair a diagnosed path within this slice or leave its dependent experiment explicitly blocked.
- [x] Establish retained and projected storage including originals, derivatives and temporary writes; shared free space is not project headroom and protected artifacts are not deleted.
- [x] Provide campaign accounting that records allocated GPU attempts, queue wait separately, remaining hours, and the single-request constraint; exercise budget rejection through a mocked scheduler boundary.
- [x] Use the existing CPU diagnostic route and resource ceilings; show aggregate real-run evidence separately from fixture tests and state missing access honestly.
- [x] Record the diagnostic disposition and evidence for final-freeze accounting when it affects a selected experiment.

## Completion evidence

Record the demonstration, focused boundary checks, exact artifact identities, measured scope, unresolved limitations and any required branch disposition. A ready-for-agent label authorizes preparation within the spec; it does not establish resource approval or successful scientific performance.

## Comments

### 2026-09-25 — Resolved after real DICOM memorization repair

The original 500-step real run (Slurm `7160`) was retained as a failed gate: training macro AUC `0.6871527777777778`, BCE `0.641625851392746`. A frozen eight-study, reference-only cache exactly reproduced it. Changing only Adam learning rate to `0.01` for 500 steps improved but still failed; extending that configuration to 2,000 steps passed. Full source-DICOM rerun `7166` reverified the archive, corrected folds, all nine real-input checks and exact packed tensor equality, then independently passed the frozen training-only thresholds on all 12 targets: AUC `0.9625`, BCE `0.2128522638231516` versus maximum `0.3345430965224902`, finite initial gradient L2 `0.14292845332737203`, relative parameter change `5.296873969186302`. It read zero held-out studies, used zero GPUs, and exited `0:0`. No held-out generalization is claimed. The optimization block is removed; Ticket 31 has not started.

The 19-root inventory plus new worktree measurements remains far below 2 TB. Ticket 29 GPU attempts and allocated GPU-hours are exactly zero. Campaign-wide remaining GPU hours are recorded as unknown because `sacct` still cannot connect to its accounting database; this is an access limit, not a claim of remaining capacity. See `docs/evidence/ticket-29-image-diagnostic.md`, private real-rerun evidence SHA-256 `d395d3922072ed037c0538b41c6d1c4c3d5be6216da26c0e4b413b9254fb10b6`, and `/home/ch.jitheswar/RSNA-tickets-28-29-30-report.md`.
