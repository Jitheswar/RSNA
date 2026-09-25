# Ticket 29 real-image learning diagnostic

Recorded: 2026-09-25T10:01:13+05:30 (Asia/Kolkata).
This report contains aggregate evidence only. Study, series, and source-entry identifiers and pixels remain in the private run directory.

## Source and fold reconciliation

- Source version: `rsna-2026-08-04`; archive fingerprint: `25a70674e5af8808c4cf6d7e0da6c8b8f894e1856a11e064470ea0bcdab25648`.
- Integration identity: `c0b11e22cd2575af7d964fbe3b8b12603ad0c6291a6a1705e0fb53fcfd957b94`; it covers 158 prepared studies (58 reference and 100 report-only).
- Corrected folds: `8460234d0fcbfd2eb9ca3a3e4165929f50a86acac37d2cd03be1164c2f287fb8`; version 2, three folds, 4407 source studies (58 reference and 4349 report-only).
- Canonical target order: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.
- Frozen cohort: 8 fold-0 reference-training studies; held-out studies read: 0. It was selected from labels alone, before pixel reads.
- Input/cache identity: existing `rsna-knee-preprocess-v1` 128-pixel preparation policy; no prepared-image cache was supplied or written. Raw archive membership and manifest checksums bind this run.

## Real image and label checks

- Real DICOM studies loaded: 8/8; accepted series: 42, explicitly excluded non-diagnostic series: 0, declared series: 42; decoded slices in accepted series: 1306/1306.
- Finite slices: 1306; nonconstant finite slices: 1306 (1.000). Source dimensions: `{'1024x1024': 36, '256x256': 40, '320x300': 50, '320x320': 261, '384x348': 36, '384x384': 201, '400x400': 40, '416x416': 25, '512x384': 23, '512x512': 408, '640x512': 20, '640x540': 60, '640x640': 38, '768x768': 68}`.
- Anatomical-plane coverage by series: `{'Axial': 10, 'Coronal': 15, 'Sagittal': 17}`; fluid-sensitive states: `{'false': 17, 'true': 25}`; fat-suppression states: `{'false': 17, 'true': 25}`.
- Packed valid masks: 8 studies with input, 42 valid series, 336 valid slices; minimum per study: 32. All packed pixels finite: `True`.
- Supervision: 96/96 reference target entries aligned across the frozen cohort. Per-target positive/negative support: `[('ACL', 4, 4), ('MCL', 3, 5), ('Medial Meniscus', 3, 5), ('Lateral Meniscus', 4, 4), ('Medial OA', 3, 5), ('Lateral OA', 4, 4), ('PF OA', 4, 4), ('Effusion', 6, 2), ('Synovitis', 4, 4), ("Baker's", 3, 5), ('Contusion', 4, 4), ('Fracture', 3, 5)]`.
- Input checks: `{'series_study_identities_align_with_loaded_studies': True, 'all_frozen_studies_have_accepted_series': True, 'all_expected_series_accounted_for': True, 'all_packed_pixels_finite': True, 'all_frozen_studies_have_valid_series_and_slices': True, 'all_frozen_studies_meet_minimum_valid_slice_count': True, 'source_pixels_meet_nonconstant_fraction': True, 'all_twelve_targets_have_two_class_support': True, 'all_expected_reference_entries_supervised': True}`.

## Training-only optimization evidence

- Model path: existing mechanics baseline; Adam 0.001, constant schedule, no augmentation, seed 20260925, 500 bounded steps. The control is the selected cohort's constant training-prevalence loss.
- Prevalence-control masked BCE: 0.6690861930449804; initial loss: 0.6933632344007492; initial gradient L2 norm: 0.14292845332737203; nonzero gradient tensors: 8/8.
- Relative parameter L2 change: 0.9559077230143251; final training-cohort loss: 0.641625851392746; final supported-target training macro AUC: 0.6871527777777778 across 12 targets.
- Full training-cohort curves (0, 100, 250, and 500 optimizer steps):

| Steps | Training-only masked BCE | Supported-target training macro AUC |
| ---: | ---: | ---: |
| 0 | 0.6933632344007492 | 0.42638888888888893 |
| 100 | 0.6695694774389267 | 0.4774305555555555 |
| 250 | 0.6647218614816666 | 0.5579861111111111 |
| 500 | 0.641625851392746 | 0.6871527777777778 |

- Optimization competence demonstrated: **False**. This is memorization evidence on the selected training examples; it is not held-out generalization or a campaign score.
- Frozen success criteria: finite/nonzero gradient norm ≥1e-8; relative parameter change ≥1e-8; final masked BCE ≤50% of the training-prevalence control; supported-target training macro AUC ≥0.85 across all 12 targets. No held-out outcomes were used.

## Resource accounting and storage

- CPU Slurm job: `7160` on `KLEF-C-NODE2`, partition `cluster1-cpu`, 8 CPUs, 32 GB, nonexclusive, six-hour limit. Actual elapsed time: 147.9646155834198 s; CPU queue wait: 7.612216234207153 s.
- GPU: no request or allocation for Ticket 29; Ticket 29 allocated GPU attempts/hours: 0/0.0. Current owner active/pending GPU requests at pre-submit snapshot: 0. Campaign cap: 20.0 GPU-hours; total remaining is `None` because historical Slurm accounting is `unavailable: sacct persistent database connection refused`. Queue wait is reported separately from allocated time. The mocked scheduler gate rejected over-budget and concurrent requests.
- Retained archive: 265018885676 bytes; no raw or prepared-image copy was created. Added durable Ticket 29 outputs: 29844680 bytes; peak temporary write reserve: 64000000 bytes.
- Measured declared project roots before/after: 296360825644 / 296390670324 logical bytes; projected peak with 64 MB reserve: 296424825644 / 2,000,000,000,000 bytes. Shared free space: 24472598872064 bytes and is not project headroom.
- Storage scan scope: `canonical repository plus isolated Ticket 29 worktree, including their private artifact trees and local environment; no unrelated home or shared-filesystem trees`. historical external home-directory experiment folders are outside the measured roots; the private root aggregate is not claimed as whole-home usage

## Disposition and limitations

- Diagnostic disposition: `optimization_competence_not_demonstrated`. The next representation experiment is `blocked_pending_diagnosis_or_repair` based on this optimization result.
- No report text, images, study identifiers, or per-study predictions are published here. The private plan, checksums, profile, detailed training log, and checkpoints are retained under the ignored private artifact root.
- No GPU run was authorized or required; this evidence does not establish GPU throughput. Project storage beyond the measured repository and Ticket 29 worktree roots, and total campaign GPU remaining hours, remain unverified because Slurm accounting is unavailable.

## Follow-up campaign storage inventory

The preceding storage measurement covered only the canonical repository and the Ticket 29 worktree. Slurm CPU job `7161` completed a separate, read-only metadata scan of all 19 explicitly frozen, non-overlapping project roots, including the Ticket 28 and 30 worktrees, Ticket 30 report-audit/v4 artifacts, Ticket 15–22 worktrees present at the time of the scan, and both Ticket 13 environments. It ran on `cluster1-cpu` with 1 CPU, 8 GB, nonexclusive, and a 90-minute limit; `scontrol` records `COMPLETED`, exit `0:0`, 36 seconds.

- Root manifest SHA-256: `8eb4d25182a9f7df5bba0e749fcb4841c29bc845ed545440db4f5f21366911b0`. Each root was limited to 250,000 directory entries and 120 seconds; all 19 completed without truncation or error.
- Measured retained project roots: `302623766144` logical bytes. The source archive is `265018885676` bytes and is already included once in the canonical repository total. Non-archive files across all measured roots total `37604880468` logical bytes. The scan uses logical file sizes and includes the listed root contents; it does not follow symlinks or read/hash file contents. Symlink targets and home-directory paths outside the frozen list remain excluded.
- Per-root logical bytes / file entries:

| Root label | Logical bytes | File entries |
| --- | ---: | ---: |
| canonical-repository | 296340695698 | 77561 |
| rsna-ticket28 | 6116304 | 336 |
| rsna-ticket29 | 50021549 | 747 |
| rsna-ticket30 | 6124549 | 331 |
| ticket15-primary | 154030415 | 4 |
| ticket15-repeat | 145765540 | 2 |
| ticket16 | 33250884 | 1111 |
| ticket17 | 11250721 | 22 |
| ticket18-additive | 41119882 | 172 |
| ticket18-real | 49610286 | 186 |
| ticket19 | 12403342 | 79 |
| ticket21-primary | 1860 | 1 |
| ticket21-repeat | 2056909 | 25 |
| ticket22-v1 | 1861 | 1 |
| ticket22-v2 | 1859 | 1 |
| ticket22-v3 | 6723155 | 119 |
| ticket30-audit | 268285999 | 8 |
| ticket13-gpu-environment | 5495825182 | 22373 |
| ticket13-operational-environment | 480149 | 48 |

- Applying the diagnostic's `64000000`-byte temporary-write reserve to this measured total gives `302687766144` projected bytes against the `2000000000000`-byte soft budget, leaving `1697312233856` bytes of scoped projected headroom. Shared filesystem available bytes were `24472214044672`; that figure is not project headroom.
- The old selection projection of `321549983220` bytes remains a different, hypothetical preparation scenario with `current_project_bytes=0`: archive plus 26,202,082,276 extracted-original bytes, 2,063,466,496 pixel-cache bytes, and 28,265,548,772 temporary duplicate bytes. It is not added to this measured total. Ticket 29 created no extracted-original copy or prepared pixel cache. The inventory JSON (3,144 bytes) and Slurm stdout (360 bytes) were created after the snapshot and are covered by the 64 MB reserve; later report/code changes are also within that reserve.
- This follow-up supersedes the earlier narrow-root storage limitation for the declared 19-root scope. Other home-directory paths and any data reachable only through symlink targets remain outside the measured scope. The private scan record is `artifacts/private/ticket-29-real-diagnostic/campaign-storage-inventory-7161.json` (mode `0600`, SHA-256 `68b39784d2e5d7c418facaf1f2a4378110375b2e9cc34281ef7172b897a2a49d`).

## Slurm terminal timing and final-freeze accounting

The Slurm controller's terminal record for diagnostic job `7160` reports `COMPLETED`, exit `0:0`, and allocated runtime `00:02:55` (175 seconds). The `147.9646`-second value in the run evidence is the Python diagnostic application interval; it excludes Slurm startup/teardown. The separately reported CPU queue wait was `7.6122` seconds. `sacct` remains unavailable because the local Slurm accounting database refuses connections.

Final-freeze disposition: Ticket 29 remains diagnostic-only; no candidate identity was advanced from this run. The frozen optimization gate failed, so the dependent representation experiment stays blocked pending diagnosis or repair. The result is training-only memorization evidence and contains no held-out evaluation.
