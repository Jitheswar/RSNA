# 35: Expand eligible image preparation in restartable batches

**What to build:** Additional eligible report-only studies are admitted in bounded batches, converted to the selected representation and demonstrated through a real study prediction path.

**Blocked by:** 34: [Measure admitted report supervision on the prepared images](34-measure-report-supervision-on-prepared-images.md)

**Parent:** [RSNA leaderboard campaign with zero external spend](../spec.md)

**Status:** resolved

**Disposition:** scope-limited — implementation, synthetic recovery checks, a checksum-bound expansion plan and approved real preparation of batch 0 are complete. Later image batches and real model inference remain unrun and require separate exact approvals.

**Execution constraints:** Follow the approved spec: zero external spend; significant work through Slurm; existing CPU ceilings; one active or pending GPU request and 20 allocated GPU-hours across the first campaign, including failures and retries; 2 TB soft storage budget; exact resource and delivery approvals where required. Questions go in visible chat. Synthetic checks establish software behavior only. No project-imposed candidate or submission-count cap.

**Execution trigger:** Expand only when the prepared-image experiment and label coverage justify a declared larger comparison within storage, compute and time limits.

## Acceptance criteria

- [x] Freeze the selected eligible membership, fold exclusions, source identity, label identity and target coverage required by the larger comparison.
- [x] Use the storage ledger to estimate retained and temporary peak growth before each real batch; stop at the 2 TB soft budget without deleting protected artifacts. Approved batch 0 passed a fresh all-root ledger at a 321,466,595,419-byte projected peak. Each later batch must repeat the gate.
- [ ] Decode and cache admitted studies with the established geometry, failure and recovery policies; retain checksum-covered provenance and explicit exclusions. The approved real pilot decoded and cached batch 0 (32 of 4,178 planned studies); later batches remain unrun.
- [ ] Run newly prepared studies through the selected preprocessing, feature extraction and prediction path, checking study identity, masks and output completeness. Synthetic inference passed; no real prediction was produced.
- [x] Demonstrate interrupted-batch recovery, valid-cache reuse and rejection of changed or incomplete cache entries through the public materialization/inference boundaries using a tiny synthetic DICOM fixture.
- [x] Record actual studies, series, slices, supervised target entries, elapsed time and storage. Do not claim all 4,407 studies are image-ready unless verified. Batch 0 results are recorded below.
- [x] Use bounded Slurm batches and measured CPU/GPU throughput, with all GPU attempts consuming the campaign ledger. The approved CPU pilot was one batch at a time; it measured 53.50 decoded source slices per second. No GPU attempts were made.
- [x] Deliver a reproducible expanded input manifest and explicit scope-limiting disposition. Record the branch's terminal decision for final-freeze readiness.

## Completion evidence

Record the demonstration, focused boundary checks, exact artifact identities, measured scope, unresolved limitations and any required branch disposition. A ready-for-agent label authorizes preparation within the spec; it does not establish resource approval or successful scientific performance.

## Comments

- 2026-09-26 — Froze private membership v2 at `artifacts/private/ticket35-expanded-image-preparation/membership-v2.json` (canonical manifest checksum `6cacc1f4fad294d843b6a1f42cd8a3a38b9dd57513d2e69ca6faaa8345aa4f2d`; file SHA-256 `e3b339ec5543a0081ccbc43e71d9cd8c38bde492f7ab5f12886fa2b986de0506`). It binds fold manifest `8460234d0fcbfd2eb9ca3a3e4165929f50a86acac37d2cd03be1164c2f287fb8`, Ticket 05 integration manifest file digest, full inventory file digest, source archive fingerprint `25a70674e5af8808c4cf6d7e0da6c8b8f894e1856a11e064470ea0bcdab25648`, and Ticket 30 artifact `33ae5e71499cf91fcc825d903e07d8519557e2bb2e1a40e64891f4c0fe00f6df`. The earlier v1 draft remains immutable; v2 makes per-fold `target_coverage` image-ready-only and preserves admitted pre-exclusion coverage separately. This source cohort has zero metadata-excluded admitted studies, so both coverage totals are 39,971.
- The fold-safe union contains 4,178 admitted report-only studies with complete inventory series; 171 of 4,349 eligible reports have no admitted target call. The plan covers 23,107 series, 788,807 inventory frames, 39,971 supervised target entries across the three consuming folds, 547,333,907,118 uncompressed source bytes and a 158,316,720,128-byte 224-pixel float32 cache projection. It identifies 4,084 studies outside the prior 158-study integration manifest. Inventory completeness is not decoded-image readiness.
- The plan divides the image-ready union into 131 deterministic batches of 32 studies (last batch: 18). Before the approved run, a metadata-only integration-manifest check verified batch 0 as 32 studies, 182 series and 6,248 source entries at 224 pixels. That pre-run check read no source archive DICOM pixels and used the archive size alone as its baseline; it was not a storage approval.
- Added a same-job campaign-root storage scan and a hard 2 TB decimal peak check to the batch runner. The initial 32-study batch reserves 10,066,651,736 bytes of temporary-plus-retained growth beyond its measured baseline. In approved job `7192`, the fresh all-root ledger measured 311,399,943,683 bytes including the 265,018,885,676-byte archive. The projected peak was 321,466,595,419 bytes, leaving 1,678,533,404,581 bytes under the soft cap at that snapshot.
- The owner approved CPU-only batch 0 under the exact 16-CPU, 64-GiB, six-hour request. Job `7191` failed at launch because the declared log directory did not yet exist; the Slurm step failed before the runner, ledger or DICOM reads. The private log directory was created with mode 700, then job `7192` completed successfully in 153.41 seconds. Its materialization evidence records 32 studies, 182/182 series prepared, 6,248/6,248 slices decoded, 308 supervised target entries across folds (131 / 88 / 89), no exclusions and no incomplete inputs. Image preparation took 116.78 seconds, or 53.50 decoded source slices per second.
- The batch wrote 1,257,600,038 cache bytes (1,257,600,125 bytes in the cache directory including its index), versus the 1,253,998,592-byte pixel-cache projection. Post-run verification found 182 complete cache entries with valid metadata and array checksums. The job used 118.26 user CPU seconds, 11.95 system CPU seconds and peaked at 5,238,444 KiB RSS. Its evidence is `artifacts/private/ticket35-expanded-image-preparation/batches/batch-0/job-7192-evidence.json`; the approved resource record is `.scratch/rsna-leaderboard/resource-approvals/ticket35-image-preparation-cpu.json`.
- Synthetic boundary checks passed for fold exclusion, checksum rejection, the 2 TB estimate, DICOM-fixture preprocessing/inference, resume/cache reuse, incomplete-cache rebuild and changed-source cache identity. Real batch 0 produced prepared pixels but no model predictions: the approval excluded model use, feature extraction, prediction and training. Later batches, pretrained checkpoint loads, GPU use and external spend remain zero.
- **Terminal branch decision:** stop after the approved batch-0 CPU pilot. Do not start batches 1–130, real feature extraction/prediction, or Ticket 36 training/comparison without a new exact approval and a passing fresh same-job ledger for each batch. Review the observed cache growth and throughput before proposing follow-up work. Candidate promotion and final-freeze readiness remain blocked. The outside-repository implementation report is `/home/ch.jitheswar/ticket-35-implementation-report.md`.
