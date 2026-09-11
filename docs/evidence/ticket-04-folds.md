# Ticket 04 immutable fold-freeze evidence

Recorded 2026-09-11 (Asia/Kolkata). This is public-safe implementation
evidence; it contains no competition identifiers, reports, raw paths, patient
keys, or private manifest contents.

> **Superseded generator.** The version-1 generator described in the first
> half of this document produced the real manifest `ae81f327…d51ad3e7` with gold
> folds of 26, 4 and 28 studies. That manifest is superseded for comparison
> purposes by the version-2 correction recorded below. It is retained unchanged
> as historical evidence, and nothing downstream should be compared against it.

## Demonstrated scope

The synthetic source-manifest seam demonstrates:

- exact one-fold membership for every gold and report-only training study;
- deterministic multilabel-aware gold allocation over all twelve targets;
- stable identifier-hash allocation for report-only studies;
- whole-group allocation across gold and report-only cohorts when a grouping
  assessment is explicitly reliable;
- explicit study-level splitting and the repeated-patient leakage limitation
  when no reliable key is available;
- per-fold gold positive/negative counts for all twelve targets, including a
  documented mathematically unavoidable one-class case;
- checksum-covered private membership and generation metadata;
- public evidence without study, series, report, or group identifiers;
- immutable, atomic publication and checksum-verified reload; and
- a fold-level admission check that rejects held-out studies from adaptation
  inputs and declares held-out images, metadata, and reports as excluded.

The bounded adapter is `slurm/run_ticket04_freeze_folds.sbatch`. It uses one
`cluster1-cpu` node, two CPUs, 8 GB, and one hour without exclusive allocation.
The core itself does not inspect report text or DICOM pixels.

## Tests

The ticket-specific synthetic fold tests pass in the pinned Python 3.11
environment. They cover deterministic regeneration, target support, reliable
group joins, partial-label rejection, privacy, immutable publication,
checksum failure, and held-out adaptation rejection.

## Unresolved gates and limitations

- No private competition fold manifest was generated in this evidence pass.
  The owner must run the adapter against the private ticket-03 inventory before
  treating the resulting artifact as the canonical competition manifest.
- V02 remains unresolved until legitimate grouping fields are inspected across
  the indexed corpus. The default study-only path permits the baseline but does
  not support patient-independent generalization claims.
- V14 remains pending real-corpus membership and label-join verification. This
  implementation does not claim that a synthetic demonstration verifies the
  private fold-freeze gate.
- A later reliable patient key, changed source version, label join, or fold
  generation parameter requires a new immutable manifest and invalidates
  affected comparisons.
- No report adaptation, training, upload, submission, or protected deletion
  was performed.

## Version-2 generation correction (2026-09-11)

The owner approved the fold-generation correction proposed by the fold audit.
The version-1 phase-2 greedy placed each remaining gold study in the fold with
the lowest *absolute post-add* imbalance. A fold far below its target always
looked worse than a fold near its target, so the real allocation never chose
fold 1 in any of its 46 phase-2 steps. Fold 1 kept only the four studies that
phase 1 placed there to seed both-class support, giving gold folds of 26, 4 and
28 studies.

### Corrected method

`FOLD_GENERATION_METHOD` is now
`gold:iterative-multilabel-greedy-v2(support-seeded;phase2=marginal-total-imbalance;fold-size-cap);report-only:stable-study-or-reliable-group-identifier-hash-v1;groups:unsplit-v1`,
written with schema `rsna-knee-fold-manifest-v2`.

- **Phase-2 scoring.** A candidate fold is scored by the change in total
  imbalance, i.e. its imbalance after the placement minus its imbalance before
  it. Only the receiving fold's term changes, so this is the change in the sum
  over all folds. The imbalance function, study ordering, support bonus and
  tie-breaks are otherwise unchanged.
- **Fold-size cap.** When every gold group is one study (study-level
  separation), each fold holds `floor(N/3)` or `ceil(N/3)` gold studies, and at
  most `N mod 3` folds may exceed the floor. Phase 1 and phase 2 both respect
  this. For 58 studies the cap is 20 and the floor is 19. If an attested key
  produces multi-study groups, the cap widens to `ceil(N/3) + largest group − 1`,
  under which greedy whole-group placement always has room.
- **Validation gates.** Generation refuses to produce a manifest unless all of
  these pass:
  - gold fold-size balance;
  - per-target class balance: every fold/target/class count is within
    `max(3 studies, 0.15 × expected)` of `class total × fold gold size / N`;
  - both-class support for every feasible target;
  - deterministic generation: an independent in-process regeneration must be
    identical.

  Loading a v2 manifest recomputes every membership-derived gate. The error
  message states that no other seed may be substituted.
- **Tolerance calibration.** The tolerance is part of the v2 definition. It
  was fixed before the canonical run from synthetic data only:
  - 3,000 synthetic 58-study cohorts with the public pooled positive counts:
    worst v2 deviation 2.53 studies; zero size or support failures.
  - A 2-study tolerance would have failed 6.3% of those cohorts.
  - The v1 generator's median worst deviation on the same kind of cohorts was
    4.6–6.3 studies.
  - The 3,000-cohort calibration ran as synthetic-only cluster1-cpu job 6938.
    Its script and output are kept under the ignored `artifacts/dev-checks/`.
- **Versioning and replay.** Version-1 manifests still load with their
  original checksum and are marked `superseded`. `generation_version=1`
  reproduces the v1 generator for audit replay only; it matched the pre-change
  code's checksums on 147 synthetic cohorts and the frozen real gold
  assignment. `run_fold_freeze` publishes only v2.

  `--supersedes-manifest` records the superseded manifest's checksum and
  refuses the correction unless source, grouping evidence, named seed and
  repeat are identical, so the method is the only change and seed search is
  impossible.
  The training and OOF CLIs refuse a superseded manifest unless
  `--allow-superseded-fold-manifest` is given for historical replay.

### Regression tests

`tests/test_folds.py` now contains:

- A 58-study cohort with the reference cohort's positive counts. The v1
  generator reproduces the seed-only fold (sizes 27, 27, 4). The starved fold
  holds only studies that alone give both classes for every target, and the v1
  gates fail. The v2 manifest on the same cohort is 20/19/19 and passes every
  gate.
- Random multilabel cohorts at n = 36, 58, 90 and 300. Each passes every v2
  gate and regenerates identically, while v1 fails the size gate at every size.
- Reference-margin cohorts at the same sizes.
- Size-policy, grouped-cap and gate-refusal cases.
- Load-time gate recomputation, and unchanged v1 round trips.
- Supersession provenance, including rejection of a changed seed and of an
  existing output path.

`tests/test_fold_verification.py` covers V14. The complete suite passes 145
tests in the pinned Python 3.11 environment, and mypy passes.

### Canonical corrected manifest

The corrected manifest was generated exactly once, through
`slurm/run_ticket04_freeze_folds.sbatch` in cluster1-cpu job **6941**. It used:

- the reconciled ticket-03 inventory;
- the same V02 grouping evidence (`unavailable`: study-level separation);
- the superseded manifest as provenance input.

The named seed `ticket-04-real-fold-freeze-v1` / `20260911` and `repeat-1` were
inherited unchanged from that manifest. No other seed was tried and no model
score was consulted.

| Item | Version 1 (superseded) | Version 2 (canonical) |
| --- | --- | --- |
| Manifest checksum | `ae81f32753a65a58c2e27776d852820e4010c5fedb8d59ce05d73ae5d51ad3e7` | `8460234d0fcbfd2eb9ca3a3e4165929f50a86acac37d2cd03be1164c2f287fb8` |
| Gold studies per fold | 26 / 4 / 28 | 20 / 19 / 19 |
| Report-only studies per fold | 1,502 / 1,384 / 1,463 | 1,502 / 1,384 / 1,463 (unchanged hash allocation) |
| All studies per fold | 1,528 / 1,388 / 1,491 | 1,522 / 1,403 / 1,482 |
| Worst class-count deviation | 4.38 studies (13 checks over tolerance) | 1.53 studies (0 over tolerance) |
| Size / class-balance gates | fail / fail | pass / pass |

Gold positive/negative counts by fold:

| Target | v2 fold 0 (+/−) | v2 fold 1 (+/−) | v2 fold 2 (+/−) | v1 fold 0 (+/−) | v1 fold 1 (+/−) | v1 fold 2 (+/−) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 7/13 | 8/11 | 9/10 | 9/17 | 1/3 | 14/14 |
| MCL | 3/17 | 3/16 | 3/16 | 1/25 | 1/3 | 7/21 |
| Medial Meniscus | 9/11 | 8/11 | 9/10 | 10/16 | 1/3 | 15/13 |
| Lateral Meniscus | 9/11 | 8/11 | 6/13 | 7/19 | 2/2 | 14/14 |
| Medial OA | 5/15 | 5/14 | 5/14 | 4/22 | 1/3 | 10/18 |
| Lateral OA | 4/16 | 3/16 | 4/15 | 1/25 | 1/3 | 9/19 |
| PF OA | 7/13 | 7/12 | 7/12 | 8/18 | 2/2 | 11/17 |
| Effusion | 12/8 | 12/7 | 11/8 | 15/11 | 3/1 | 17/11 |
| Synovitis | 9/11 | 10/9 | 8/11 | 9/17 | 3/1 | 15/13 |
| Baker's | 4/16 | 4/15 | 4/15 | 1/25 | 1/3 | 10/18 |
| Contusion | 7/13 | 6/13 | 6/13 | 7/19 | 1/3 | 11/17 |
| Fracture | 6/14 | 6/13 | 6/13 | 6/20 | 1/3 | 11/17 |

All twelve targets are feasible and have both classes in every v2 fold. The
largest v2 deviation is Lateral Meniscus negatives in fold 2: 13 against an
expected 11.47.

### V14 re-run

`slurm/run_ticket04_verify_v14.sbatch` (cluster1-cpu job **6942**) wrote
identifier-free evidence below the private `ticket-04-real-v2-v14/public/` path.
V14 status: **passed**.

- **Checksum and consistency.** The manifest checksum, stored gates and fold
  summaries re-derive, and the public evidence equals its regeneration from
  the private manifest.
- **Membership and joins.** The source inventory reconciles 4,407 studies. The
  manifest has exactly one membership per study, valid fold IDs 0–2 and 58
  gold studies, and cohort and all twelve reference-label joins agree. The
  grouping evidence matches.
- **Reproducibility.** An independent regeneration from the inventory, in a
  separate process, reproduced the manifest checksum.
- **Gates.** Gold size balance, per-target class balance and both-class
  support all pass.
- **Seed provenance.** The manifest records the superseded checksum and reuses
  its source, named seed and repeat.

Reliable-group overlap is `not_applicable`: no reliable patient key exists, so
repeated-patient leakage still cannot be ruled out. The private v2 manifest
file has SHA-256 `6db019d0…cb5a6bb9` and its public evidence `f45fce11…f95b2228`.

### Preservation of superseded evidence

The v1 manifest, its public evidence and its grouping file are unchanged
(SHA-256 `035dcc2c…4787a841`, `579c0e0f…4f36fbd5` and `6bf4b9d0…fe68a5dc`). The
Ticket 08–10 runs on that manifest also remain in place, unmodified. They are
historical evidence only and are not comparable with any v2 result.
