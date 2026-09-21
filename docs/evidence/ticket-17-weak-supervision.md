# Ticket 17 — One fold with admissible weak labels

Status: implemented for the synthetic, locally testable scope. No private
competition training allocation was submitted by this implementation pass.

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

## Scope and unresolved evidence

This implementation proves the admission and synthetic training mechanics. It
does not claim a private real-corpus training run, a report-supervision AUC
gain, or a complete controlled comparison. Astra outputs remain excluded from
training because Ticket 16 did not predeclare an admission rule. Ticket 18
owns the gold-only versus gold-plus-clean-report comparison.
