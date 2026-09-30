# 34: Measure admitted report supervision on the prepared images

**What to build:** A bounded paired experiment measures whether admitted report-derived labels add useful training signal within the existing prepared image cohort.

**Blocked by:** 30: [Audit existing report-derived labels](30-audit-existing-report-derived-labels.md); 32: [Complete the three-fold ResNet-18 comparison](32-complete-three-fold-resnet18-comparison.md)

**Parent:** [RSNA leaderboard campaign with zero external spend](../spec.md)

**Status:** resolved

**Disposition:** completed — positive one-fold pilot, inconclusive; not promoted. A separate three-fold confirmation is justified for planning, not authorized by this Ticket 34 scope.

**Execution constraints:** Follow the approved spec: zero external spend; significant work through Slurm; existing CPU ceilings; one active or pending GPU request and 20 allocated GPU-hours across the first campaign, including failures and retries; 2 TB soft storage budget; exact resource and delivery approvals where required. Questions go in visible chat. Synthetic checks establish software behavior only. No project-imposed candidate or submission-count cap.

## Acceptance criteria

- [x] Start only with admissible label artifacts and a justified image-learning route; a rejected label source is not made eligible because the audit ticket has completed.
- [x] Freeze one bounded pilot fold and prospective comparison policy. Use matching initial weights, reference exposure sequence, seed, checkpoint policy and held-out cohort across reference-only and report-supervision arms.
- [x] Record weak-label exposures, masked targets, actual image membership and eligible report-only exclusions; do not equate a large report inventory with the smaller image cohort.
- [x] Use the existing supervision and paired-comparison boundaries; reject held-out lineage, mismatched initialization, changed reference exposure and incomplete prediction cohorts.
- [x] Report paired per-target and macro evidence with uncertainty, actual cost and the one-fold limitation. Retain the deterministic comparator and prior inconclusive results.
- [x] Record whether expansion is justified within remaining budget/time; a pilot may motivate a full comparison but cannot establish a three-fold gain.
- [x] The existing admitted Ticket 16 artifact sufficed; no local-extractor replacement or rerun was needed.
- [x] Record the selected pilot's promotion/rejection disposition and any triggered expansion before final freeze. Missing reliable supervision blocks this experiment rather than manufacturing labels.

## Completion evidence

Record the demonstration, focused boundary checks, exact artifact identities, measured scope, unresolved limitations and any required branch disposition. A ready-for-agent label authorizes preparation within the spec; it does not establish resource approval or successful scientific performance.

## Comments

- 2026-09-25: Prepared the fold-0 Ticket 30 v4 adapter, paired-run validator, approval-gated CPU runner and Slurm job. The frozen plan binds fold manifest `8460234d0fcbfd2eb9ca3a3e4165929f50a86acac37d2cd03be1164c2f287fb8`, image integration manifest `c0b11e22cd2575af7d964fbe3b8b12603ad0c6291a6a1705e0fb53fcfd957b94`, and Ticket 30 artifact `33ae5e71499cf91fcc825d903e07d8519557e2bb2e1a40e64891f4c0fe00f6df`.
- Read-only cohort reconciliation found 38 gold training studies, 20 held-out gold studies, 65 image-backed report-only training studies, 61 with 290 admitted target calls, and 4 image-backed reports with no admitted calls. Of 2,847 eligible report-only training studies, 2,737 had at least one admitted call and 110 remained without calls; 2,676 admitted studies have no image in this integration cohort. Only the 61 image-backed admitted studies enter the treatment arm.
- Focused checks passed: 34 tests across Ticket 34 admission/comparison and existing supervision boundaries; `mypy` on the new Ticket 34 modules; Python compilation and Slurm shell syntax checks. The pending resource approval gate rejected execution before loading weights, reading DICOM pixels, or creating run outputs. No model training or scientific performance result has been produced.
- Waiting for the owner to approve the exact Ticket 34 ImageNet use scope in `../resource-approvals/ticket34-resnet18-imagenet1k-v1.json`. Ticket 31 and Ticket 32 approvals do not cover this experiment. Until approval, actual cost, paired metrics, expansion disposition and promotion disposition remain outstanding.
- 2026-09-25: The owner approved the exact Ticket 34 scope in `/home/ch.jitheswar/tsk.txt`, including read-only Ticket 32 cache reuse in both arms. The separate approval record documents the bounded experimental-use basis and retains all winner-release gates as blocked. The already acquired checkpoint was rechecked after approval: 46,830,571 bytes and SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
- Slurm job `7180` stopped at the allocation environment gate before model loading or image reads. The actual request matched the approved 16-CPU, 64-GiB, six-hour CPU envelope; the launcher now attests allocation fields with `scontrol` and supplies the two resource variables omitted by this cluster. Job `7181` completed with exit `0:0` on `cluster1-cpu` in 793.65 seconds. CPU preflight passed; external spend and GPU use were zero.
- The paired fold-0 reference macro AUC was `0.497681`; report supervision was `0.581215`, delta `+0.083534`. The paired 2,000-resample study-bootstrap 95% interval was `-0.038625` to `+0.209216`, with 131 undefined resamples. Lateral Meniscus was the only target whose paired interval excluded zero. Per-target AUCs, supports and intervals are in `artifacts/private/ticket34-resnet18-supervision/comparison.json` and the readable report is `artifacts/private/ticket34-resnet18-supervision/comparison.md`.
- The image-backed treatment had 61 report-only studies and 290 admitted target calls; 1,439 sampled weak targets stayed masked. Held-out studies and labels were evaluation-only. Both arms completed 100 optimizer steps and matched the same initialization and 2,400 reference-target exposures.
- Both arms reused all 58 matching Ticket 32 features read-only. The treatment imported those 58 features and computed 61 report-only features in its separate Ticket 34 cache. A post-run digest audit verified all source entries; no Ticket 32 cache files were written. The original partition counters overcounted later lookups; corrected accounting is in `artifacts/private/ticket34-resnet18-supervision/feature-cache-accounting-correction.json`, and the training code now snapshots each partition before proceeding.
- **Disposition:** positive but inconclusive one-fold pilot; not promoted to final-candidate consideration. Planning a complete three-fold confirmation is justified by the point estimate and target evidence, but the overall interval crosses zero, only 20 studies were held out, and reliable patient grouping is unavailable. Ticket 35 image preparation and a separate exact Ticket 36 scope/resource approval are required before any broader run. No such run was started under this Ticket 34 approval. Final-candidate freeze, publication, redistribution and release remain blocked.
- Run IDs: reference-only `23c0e3b455973df76ffed1eb4e09cf2a70791390101d289518d0b31c64e13b21`; report-supervision `463313db4d44d4b5ecf39f19c21b0099bdf6be388af05a99e413fd82c7432088`. Detailed scope, costs and attempt history are in `artifacts/private/ticket34-resnet18-supervision/ticket34-run-plan.json`.
- The compute node did not expose a Git revision. `artifacts/private/ticket34-resnet18-supervision/code-source-manifest.json` records launch-time hashes for all 68 project Python modules and the Slurm launcher (source snapshot SHA-256 `2fa92ffc4e9bf413ec504dd26ab63f261b72994b5824b695c02eba79c3ffa7f0`). It reverses only the post-run cache-counter fix in `training.py`; scientific run logic is unchanged.

### 2026-09-25 — Closed by owner request

The owner requested closure after completion of the fold-0 pilot. The result remains positive but inconclusive and was not promoted. A three-fold follow-up still requires Ticket 35 preparation and separate Ticket 36 approval; the final-candidate, publication, redistribution, and release gate remains blocked.
