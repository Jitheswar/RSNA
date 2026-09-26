# 35: Expand eligible image preparation in restartable batches

**What to build:** Additional eligible report-only studies are admitted in bounded batches, converted to the selected representation and demonstrated through a real study prediction path.

**Blocked by:** 34: [Measure admitted report supervision on the prepared images](34-measure-report-supervision-on-prepared-images.md)

**Parent:** [RSNA leaderboard campaign with zero external spend](../spec.md)

**Status:** resolved

**Disposition:** scope-limited — implementation, synthetic recovery checks and a checksum-bound expansion plan are complete. Real DICOM preparation and model inference remain unrun pending exact resource approvals and a fresh campaign-wide storage ledger.

**Execution constraints:** Follow the approved spec: zero external spend; significant work through Slurm; existing CPU ceilings; one active or pending GPU request and 20 allocated GPU-hours across the first campaign, including failures and retries; 2 TB soft storage budget; exact resource and delivery approvals where required. Questions go in visible chat. Synthetic checks establish software behavior only. No project-imposed candidate or submission-count cap.

**Execution trigger:** Expand only when the prepared-image experiment and label coverage justify a declared larger comparison within storage, compute and time limits.

## Acceptance criteria

- [x] Freeze the selected eligible membership, fold exclusions, source identity, label identity and target coverage required by the larger comparison.
- [ ] Use the storage ledger to estimate retained and temporary peak growth before each real batch; stop at the 2 TB soft budget without deleting protected artifacts. The runner now measures the declared roots in the same Slurm job and gates before DICOM access.
- [ ] Decode and cache admitted studies with the established geometry, failure and recovery policies; retain checksum-covered provenance and explicit exclusions. Only synthetic studies were decoded in this scope.
- [ ] Run newly prepared studies through the selected preprocessing, feature extraction and prediction path, checking study identity, masks and output completeness. Synthetic inference passed; no real prediction was produced.
- [x] Demonstrate interrupted-batch recovery, valid-cache reuse and rejection of changed or incomplete cache entries through the public materialization/inference boundaries using a tiny synthetic DICOM fixture.
- [ ] Record actual studies, series, slices, supervised target entries, elapsed time and storage. Do not claim all 4,407 studies are image-ready unless verified.
- [ ] Use bounded Slurm batches and measured CPU/GPU throughput, with all GPU attempts consuming the campaign ledger.
- [x] Deliver a reproducible expanded input manifest and explicit scope-limiting disposition. Record the branch's terminal decision for final-freeze readiness.

## Completion evidence

Record the demonstration, focused boundary checks, exact artifact identities, measured scope, unresolved limitations and any required branch disposition. A ready-for-agent label authorizes preparation within the spec; it does not establish resource approval or successful scientific performance.

## Comments

- 2026-09-26 — Froze private membership v2 at `artifacts/private/ticket35-expanded-image-preparation/membership-v2.json` (canonical manifest checksum `6cacc1f4fad294d843b6a1f42cd8a3a38b9dd57513d2e69ca6faaa8345aa4f2d`; file SHA-256 `e3b339ec5543a0081ccbc43e71d9cd8c38bde492f7ab5f12886fa2b986de0506`). It binds fold manifest `8460234d0fcbfd2eb9ca3a3e4165929f50a86acac37d2cd03be1164c2f287fb8`, Ticket 05 integration manifest file digest, full inventory file digest, source archive fingerprint `25a70674e5af8808c4cf6d7e0da6c8b8f894e1856a11e064470ea0bcdab25648`, and Ticket 30 artifact `33ae5e71499cf91fcc825d903e07d8519557e2bb2e1a40e64891f4c0fe00f6df`. The earlier v1 draft remains immutable; v2 makes per-fold `target_coverage` image-ready-only and preserves admitted pre-exclusion coverage separately. This source cohort has zero metadata-excluded admitted studies, so both coverage totals are 39,971.
- The fold-safe union contains 4,178 admitted report-only studies with complete inventory series; 171 of 4,349 eligible reports have no admitted target call. The plan covers 23,107 series, 788,807 inventory frames, 39,971 supervised target entries across the three consuming folds, 547,333,907,118 uncompressed source bytes and a 158,316,720,128-byte 224-pixel float32 cache projection. It identifies 4,084 studies outside the prior 158-study integration manifest. Inventory completeness is not decoded-image readiness.
- The plan divides the image-ready union into 131 deterministic batches of 32 studies (last batch: 18). A metadata-only integration-manifest build verified batch 0 as 32 studies, 182 series and 6,248 source entries at 224 pixels. No source archive DICOM pixels were read. The builder check used the archive size alone as its measured baseline and is not a project-wide storage approval.
- Added a same-job campaign-root storage scan and a hard 2 TB decimal peak check to the batch runner. The first 32-study batch reserves 10,066,651,736 bytes of temporary-plus-retained growth beyond the measured baseline. A local `du` of the repository was 304,977,321,984 bytes including the 265,018,885,676-byte archive; the other campaign roots were not measured in this turn, so total headroom is unknown. The current proposal authorizes no run: `.scratch/rsna-leaderboard/resource-approvals/ticket35-image-preparation-cpu.json` is pending owner decision for batch 0 only (16 CPUs, 64 GiB, six hours, one CPU job, no model use).
- Synthetic boundary checks passed for fold exclusion, checksum rejection, the 2 TB estimate, real-DICOM fixture preprocessing/inference, resume/cache reuse, incomplete-cache rebuild and changed-source cache identity. No real batch, Slurm job, throughput measurement, pretrained checkpoint load, feature extraction, or scientific prediction was run. External spend and GPU use were zero.
- **Terminal branch decision:** stop at the frozen preparation plan. Do not start later image batches, expanded feature extraction/prediction, or Ticket 36 training/comparison until the batch-0 CPU scope is approved, the same-job ledger fits, its throughput and cache growth are measured, and a separate exact model-use scope is approved. Candidate promotion and final-freeze readiness remain blocked. The outside-repository implementation report is `/home/ch.jitheswar/ticket-35-implementation-report.md`.
