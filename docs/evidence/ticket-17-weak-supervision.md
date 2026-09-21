# Ticket 17 — One fold with admissible weak labels

Status: one real fold completed with the pre-existing admissible Ticket 15
weak-label artifact. No benefit claim is made here; Ticket 18 owns the
controlled gold-only comparison.

## Implemented seam

`rsna_knee.supervision.admit_fold_supervision()` is separate from the imaging
records. It consumes a reproducibility-verified Ticket 15
`ReportExtractionRun`, checks the canonical fold manifest checksum and repeat,
the complete report-set digest, frozen deterministic extractor identity,
adaptation exclusions, and every fold's eligible/evaluation study set. It
returns an immutable `AdmittedFoldSupervision` artifact. `train_one_fold()`
accepts that artifact optionally; its default remains the Ticket 08/09
reference-only path.

The admission policy is deliberately narrow:

- `WEAK_LABEL_VERSION` is the frozen Ticket 15
  `report-only-confident-binary-v1` policy.
- `MIXTURE_POLICY_VERSION` is
  `reference-precedence-plus-report-only-v1`.
- Only confidence-one positive/negative labels with an authoritative official
  definition origin are admissible.
- Not-mentioned, uncertain, unknown and extraction-failed labels remain
  unknown with a zero mask.
- Reference labels override weak values; disagreements retain target, source
  report reference and digest in the private artifact.
- Reports from held-out studies and missing report-only rows cannot become
  training examples. Fallback provenance remains explicitly
  `trusted_reference_labels_only`.

The image architecture, preprocessing, masked BCE, optimizer, schedule and
final-checkpoint evaluation are unchanged. Training now also records actual
sample and supervised-entry exposures across attempted batches, including
all-masked batches, in private run/checkpoint evidence.

## Verification

The synthetic Ticket 17 tests cover:

1. clean fold admission and zero-masked unknown weak entries;
2. reference precedence and private disagreement recording;
3. rejection of incomplete upstream lineage; and
4. one fixed-budget fold training/evaluation pass using the unchanged image
   baseline and admitted report-only samples.

The bounded operational adapter is
`slurm/run_ticket17_train_weak_fold.sbatch`, which runs on one non-exclusive
`cluster1-cpu` node with 16 CPUs, 64 GB and a six-hour limit. It re-derives
the frozen deterministic labels from the private report corpus, writes
identifier-free public admission evidence and identifier-bearing private
provenance, then invokes the shared fold trainer.

## Real fold-0 execution

The single real execution used Slurm job `7056` on `cluster1-cpu`. The
allocation was one node, 16 CPUs, 64 GB, non-exclusive, with a six-hour limit;
the scheduler record reports `COMPLETED`, exit code `0:0`, and elapsed time
`00:11:30`. The run was submitted from implementation commit `5a2bb19` and
used the existing Ticket 05 representative integration and the canonical
Ticket 04 version-2 fold manifest. The required fold checksum is
`8460234d0fcbfd2eb9ca3a3e4165929f50a86acac37d2cd03be1164c2f287fb8`.

The full fold admission contained 2,885 training studies, of which 2,847 were
eligible and admitted report-only studies and 38 were gold/reference studies.
The bounded representative image integration covered 103 fold-0 training
studies: 38 reference studies and 65 admitted report-only studies. The other
2,782 eligible report-only studies were not relabeled as missing; they were
recorded as image-scope exclusions. The fold held out 1,522 studies for
report/evaluation isolation; the available representative image evaluation
artifact contains 20 gold studies.

The admitted weak rows contributed 115 supervised target entries and 665
masked entries. Gold contributed 456 supervised entries. There were zero
gold-vs-weak disagreements in this image-scoped run. Per-target weak counts
(`positive / negative / masked`) were:

| Target | Counts |
| --- | ---: |
| ACL | 1 / 13 / 51 |
| MCL | 0 / 17 / 48 |
| Medial Meniscus | 7 / 4 / 54 |
| Lateral Meniscus | 2 / 13 / 50 |
| Medial OA | 0 / 0 / 65 |
| Lateral OA | 0 / 2 / 63 |
| PF OA | 0 / 2 / 63 |
| Effusion | 0 / 17 / 48 |
| Synovitis | 7 / 0 / 58 |
| Baker's | 0 / 15 / 50 |
| Contusion | 6 / 3 / 56 |
| Fracture | 0 / 6 / 59 |

The frozen Ticket 15 report extraction semantic digest was
`031e38507e474855e9762c31be018ff87699d4cf88490b4fe70ac9d2ef11d341`; the
validated frozen artifact SHA-256 was
`9d896b50363cdbc047f474de7b48464363360c58198df30723cedc9fd09b5e59`. The
extractor identity remained `deterministic-synthetic-english-v1` with
configuration fingerprint
`a48facf07bb23a1b2138ac74030ddfb3693130f165b8db58840b9778fef7e1aa` and
ruleset fingerprint
`2a8fe3339aacb59fe2703bd0cd8e32df1468a4d76543e7ca9a7567cb50c4a6e1`.

The image-training configuration fingerprint was
`aa83200904398e7ffb07ba9f1f0c2f1fbb5be4d2b0dfcd885d1008cf9b5c49ea`.
Architecture, 128×128 preprocessing, 8/16/32 channels, max 8 slices and 16
series, Adam at `0.001`, constant schedule, masked BCE, no augmentation, the
fixed seeds, final-checkpoint selection and the 100 successful-optimizer-step
budget matched the existing mechanics configuration. Training attempted 109
batches, skipped 9 all-masked batches, completed 100 optimizer steps, and
reported finite losses and gradients. It did not resume or replay a prior
checkpoint.

The final checkpoint SHA-256 is
`977a3b876cf750d8fb4f9362066e2f3ab229c0c38817feb3c2ce318fd994497c`.
The trainer's final-checkpoint reload and an independent reload of both final
and latest checkpoints passed. Held-out evaluation produced 20 predictions,
with all 12 target AUCs defined; these metrics are recorded as execution
artifacts only and are not interpreted as evidence of benefit.

Independent post-run checks replayed the deterministic extractor, revalidated
the frozen artifact, compared public/private supervision serialization,
verified canonical target order and masks, checked held-out separation,
verified finite checkpoint histories, and confirmed the expected training
configuration. The compute-node environment record left `code_revision` null;
the external submission state was clean at commit `5a2bb19`, and this is
recorded as a metadata limitation rather than silently filled in.

## Scope and unresolved evidence

The real execution establishes one correctly isolated fold with the admitted
weak supervision. It does not claim a report-supervision AUC gain or a
complete controlled comparison. Astra outputs remain excluded from training
because Ticket 16 did not predeclare an admission rule. Ticket 18 owns the
gold-only versus gold-plus-clean-report comparison.
