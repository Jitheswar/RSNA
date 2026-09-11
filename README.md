# RSNA knee mechanics baseline

This repository contains the ticket 2 image-only inference core, the ticket 5
private real-DICOM integration path, and the ticket 7 input-recovery path. The
core turns synthetic or admitted MRI studies into the exact twelve-target
competition schema; the real-data runner consumes a checksum-covered private
manifest and uses the same decoder, geometry, preprocessing, hierarchical
model, and submission interfaces.

## Reproducible CPU environment

The project is pinned to Python 3.11.16 and a CPU PyTorch wheel. `mise` keeps
the interpreter in the user account; the project virtual environment must be
created from that interpreter:

```bash
mise install
mise exec -- python -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.lock
```

The lock contains the verified runtime and test dependencies. The source tree
can be used without installation by setting `PYTHONPATH=src`.

## Checks

```bash
PYTHONPATH=src .venv/bin/python -m pytest
PYTHONPATH=src .venv/bin/python -m mypy src/rsna_knee
```

## Synthetic CPU demonstration

Significant computation belongs in Slurm, not on `KLEF-MASTER`. After the
user-space environment is prepared, submit the bounded adapter:

```bash
sbatch --wait slurm/run_cpu_inference.sbatch
```

The adapter requests one `cluster1-cpu` node, four CPUs, 8 GB, and 30 minutes;
it has no exclusive allocation. It writes the ignored demonstration output
and JSON evidence below `artifacts/`. The CLI refuses to run outside Slurm
unless `--allow-local-smoke` is explicitly supplied for a tiny development
check.

The public interfaces are:

- `decode_dataset` and `read_dicom_slice`: map a pydicom dataset or file to a
  decoder-independent core record. Synthetic coverage is intentionally narrow;
  real-corpus compatibility remains a later gate.
- `order_slices`: validates geometry and orders by spatial position, never by
  filename or instance number.
- `prepare_series` and `pack_study`: apply fixed 128×128 preprocessing,
  retain all accepted slices, sample eight only at packing time, and expose
  slice/series masks.
- `HierarchicalStudyModel`: shared 2D encoding, masked slice-to-series and
  series-to-study mean pooling, and a twelve-logit head.
- `InferenceEngine` and `write_submission`: MRI-only inference, explicit
  all-invalid study fallback, finite bounded scores, exact UID order, and the
  thirteen-column output contract.

## Ticket 03 private inventory and admission

`rsna_knee.inventory` reads the source CSVs and every training DICOM member's
header without touching pixel data. It records transfer syntax, frame and
dimension counts, geometry status, slice-count buckets, and the source
anatomical-plane, fluid-sensitivity, and fat-suppression descriptors as
separate fields. Unreadable headers, malformed archive members, missing or
extra series, UID conflicts, and partial reference labels remain explicit
failures. Series whose DICOM `ImageType` or description reliably contains
`scout`, `localizer`, `survey`, `pilot`, or `calibration` are excluded as
non-diagnostic; absent or ambiguous descriptions remain explicitly unknown.

The inventory also selects all reference-labelled studies plus a deterministic
50–200 report-only sample, estimates the retained-archive/extraction/cache
peak, and refuses an over-2 TB batch without an explicit approval flag.
Selected members are extracted into a fresh destination only after each ZIP
member is read to EOF, its size/CRC is checked by the archive reader, and its
SHA-256 content checksum is recorded. Extraction rechecks the recorded archive
identity, and completion reports are atomic and distinguish verified batches
from failed attempts.

The full archive scan is intentionally Slurm-gated:

```bash
export RSNA_TICKET03_ARCHIVE=/private/path/rsna-knee-abnormality-detection.zip
export RSNA_TICKET03_OUTPUT=/private/path/ticket-03-inventory
export RSNA_TICKET03_SOURCE_VERSION=rsna-2026-08-04
sbatch --wait slurm/run_ticket03_inventory.sbatch
```

The adapter requests one `cluster1-cpu` node, eight CPUs, 32 GB, and six
hours, with no exclusive allocation. Set
`RSNA_TICKET03_SUPPORTED_TRANSFER_SYNTAXES` to the comma-separated syntax UIDs
that have passed the decoder gate, then set `RSNA_TICKET03_EXTRACT_TO` only
after reviewing the private selection and storage estimate. If other retained
project data already occupies the budget, pass its byte count through
`RSNA_TICKET03_CURRENT_PROJECT_BYTES` (excluding the source archive, which the
estimate adds separately). Public aggregate JSON is written below `public/`;
UID- and
path-bearing manifests belong below `private/`, which is ignored by Git.

This ticket's implementation and synthetic archive tests do not claim that
the private competition corpus has completed its full header scan, pixel
decoder verification, or cache admission. Those claims require the bounded
Slurm execution and its preserved evidence.

## Ticket 04 immutable fold freeze

`rsna_knee.folds` freezes one three-fold development manifest from the private
ticket-03 inventory. Reference-labelled (gold) studies use a deterministic
multilabel-aware allocation; report-only studies use a stable StudyInstanceUID
hash, or a reliable group-identifier hash when grouping is required. An
explicitly attested reliable grouping key is respected across both cohorts and
is never split. Without such an assessment, the manifest records study-level
splitting and the limitation that repeated-patient leakage cannot be ruled out.

The canonical generator is version 2 (`rsna-knee-fold-manifest-v2`):

- **Phase-2 scoring.** Each remaining gold study goes to the fold with the
  smallest change in total imbalance.
- **Size cap.** When every gold group is one study, each fold holds
  `floor(N/3)` or `ceil(N/3)` gold studies.
- **Gates.** A manifest is produced only when gold fold-size balance,
  per-target class balance, both-class support and deterministic regeneration
  all pass.

Version-1 manifests, whose phase-2 greedy could starve a fold down to its
support-seed studies, still load with their original checksum. They are
marked superseded, and the training and OOF CLIs refuse them unless
`--allow-superseded-fold-manifest` is given for historical replay. A
correction is published with `RSNA_TICKET04_SUPERSEDES_MANIFEST`. It records
the superseded checksum and is refused unless source, grouping evidence, named
seed and repeat are unchanged.

Every source study receives exactly one fold membership. The private manifest
contains identifiers and reference labels, while the public evidence contains
only generation metadata, checksums, cohort totals, and all twelve positive /
negative counts for each fold. Feasible targets must have both classes in every
fold; mathematically unavoidable one-class targets are recorded as undefined
support rather than filled with an invented value. Held-out images, metadata,
and reports are the declared exclusion set for every fold's adaptation.

After the private ticket-03 inventory is available, run the bounded adapter:

```bash
export RSNA_TICKET04_INVENTORY=/private/path/ticket-03-inventory/private/inventory.json
export RSNA_TICKET04_OUTPUT=/private/path/ticket-04-folds
export RSNA_TICKET04_GROUPING_EVIDENCE=/private/path/ticket-04-grouping.json
sbatch --wait slurm/run_ticket04_freeze_folds.sbatch
```

The runner requires a complete ticket-03 reconciliation and an explicit V02
JSON audit. If V02 establishes a reliable patient key, the file must include
the audit status, examined fields, scope/provenance, missingness and
consistency counts, and a `study_to_group` object covering every training
study. If no reliable key exists, use `status: unavailable` with the examined
fields and documented study-only limitation; the runner never infers a key or
silently chooses that path. The adapter requests one
`cluster1-cpu` node, two CPUs, 8 GB, and one hour without exclusive allocation.
Both output paths are immutable: rerunning with an existing manifest refuses
to overwrite it. Identifier-bearing output belongs below a private ignored
path; `public/fold-evidence.json` is safe for aggregate evidence only.

V14 is verified by a separate read-only job. It reloads the manifest, joins it
to the inventory, regenerates it independently from the recorded parameters,
recomputes every gate, and writes one immutable identifier-free evidence file:

```bash
export RSNA_TICKET04_V14_MANIFEST=/private/path/ticket-04-real-v2/private/fold-manifest.json
export RSNA_TICKET04_V14_PUBLIC_EVIDENCE=/private/path/ticket-04-real-v2/public/fold-evidence.json
export RSNA_TICKET04_INVENTORY=/private/path/ticket-03-inventory/private/inventory.json
export RSNA_TICKET04_GROUPING_EVIDENCE=/private/path/ticket-04-grouping.json
export RSNA_TICKET04_SUPERSEDES_MANIFEST=/private/path/ticket-04-real/private/fold-manifest.json
export RSNA_TICKET04_EXPECTED_GOLD_COUNT=58
export RSNA_TICKET04_V14_OUTPUT=/private/path/ticket-04-real-v2-v14/public/v14-evidence.json
sbatch --wait slurm/run_ticket04_verify_v14.sbatch
```

The version-2 correction, V14 result and superseded-manifest comparison are
recorded in [`docs/evidence/ticket-04-folds.md`](docs/evidence/ticket-04-folds.md).

## Ticket 05 real-DICOM integration

`rsna_knee.real_inference` builds a private integration manifest from the
ticket-03 inventory and selection, verifies the source archive identity, and
streams each admitted DICOM entry through the pinned decoder. It then uses the
shared geometry, preprocessing, masked eight-slice/series packing, hierarchical
model, and exact submission writer. The manifest freezes the geometry,
decoder, preprocessing, and packing policies; sampling is post-materialization
and model architecture is not part of the preprocessing cache identity.

The default real-input policy admits explicit VR little endian through pydicom
3.0.1, requires MR modality and single-frame grayscale pixels, rejects uncertain
geometry, requires at least three usable slices, excludes only reliably named
scout/localizer-style series, and records typed failures and warnings. Real
competition data and identifier-bearing artifacts stay outside Git:

```bash
export RSNA_TICKET05_SOURCE=/private/path/rsna-knee-abnormality-detection.zip
export RSNA_TICKET05_INVENTORY=/private/path/ticket-03-inventory/private/inventory.json
export RSNA_TICKET05_SELECTION=/private/path/ticket-03-inventory/private/selection.json
export RSNA_TICKET05_OUTPUT=/private/path/ticket-05-real-inference
sbatch --wait slurm/run_ticket05_real_inference.sbatch
```

The adapter requests one `cluster1-cpu` node, 16 CPUs, 64 GB, and six hours
without exclusive allocation. It writes the private manifest, submission, and
diagnostics below `private/`, plus identifier-free aggregate evidence below
`public/`. The bounded real run and its limitations are recorded in
[`docs/evidence/ticket-05-real-inference.md`](docs/evidence/ticket-05-real-inference.md).

## Ticket 06 incremental series cache

`rsna_knee.series_cache.SeriesCache` stores every accepted, deterministically
preprocessed slice in a private disposable cache. Each entry records the
source identity and content hash for every slice, preprocessing policy and
schema versions, geometry order, shape, warnings, NumPy checksum, and metadata
checksum. Sampling remains a packing concern, so changing from eight to
sixteen slices reuses the same materialized entry. A changed source slice or
materialization policy gets a new key; unrelated series are unaffected.

Cache publication uses a per-key lock, a same-directory staging directory, and
an atomic directory rename. Readers validate completion state, both checksums,
source identity, policy, dtype, shape, and slice membership before accepting a
hit. `InferenceEngine` and the real runner accept an optional `series_cache`
or `--cache-root`; the cache root must be a private disposable path, separate
from raw data, reports, folds, checkpoints, and candidate bundles. The cache
creates a private mode-0700 root marker and refuses to initialize a populated
or non-private unmarked directory.

The mechanics benchmark is available as `SeriesCache.benchmark(...,
workers=N)`. It reports cold/warm hits, bytes read and written, and the
temporary atomic-write footprint. Run it on the authorized Slurm CPU
allocation before increasing worker counts or cache scope; local tests do not
constitute the ticket's NFS throughput or peak-storage evidence.

For an admitted private integration manifest, the bounded adapter runs both
sampling variants and writes aggregate-only evidence:

```bash
export RSNA_TICKET06_MANIFEST=/private/path/ticket-05-real-inference/private/integration-manifest.json
export RSNA_TICKET06_SOURCE=/private/path/rsna-knee-abnormality-detection.zip
export RSNA_TICKET06_CACHE_ROOT=/private/path/ticket-06-cache
export RSNA_TICKET06_OUTPUT=/private/path/ticket-06/public/cache-evidence.json
sbatch --wait slurm/run_ticket06_cache_benchmark.sbatch
```

## Ticket 07 damaged-input recovery

`rsna_knee.recovery.compute_laplace_smoothed_reference_priors` computes the
emergency study fallback vector from an explicitly eligible set of trusted
reference-label rows. Each target uses `(positive + alpha) / (known + 2 *
alpha)`; a target with no known eligible reference label receives `0.5`.
Report-derived or weak rows must not be placed in the eligible set, and
held-out rows are rejected if they overlap it. Public prior evidence contains
counts and scores only; the private record retains the eligible identifiers.

`run_real_inference` admits a series when enough valid slices survive the
bounded decoder policy and geometry/minimum-slice checks. Failed slices are
skipped with their source entry, decoder attempt, typed reason, and recovery
action recorded. If no usable series remains, the study still receives one
submission row using the reference prior. Unexpected decoder infrastructure,
model, cache, or configuration failures remain fatal and are not converted to
universal fallback output. The recovery mode is selected explicitly by
`--recover-damaged-inputs` (and is implied for the ticket-07 evidence command);
the existing ticket-05 runner remains strict by default.

The recovery adapter is bounded to one `cluster1-cpu` node, 16 CPUs, 64 GB,
six hours, and no exclusive allocation. It requires the private ticket-05
integration manifest and source. A private inventory or fold manifest can
provide gold reference rows through `RSNA_TICKET07_FALLBACK_REFERENCE_LABELS`;
when the label file has no fold assignments, an explicit
`RSNA_TICKET07_FALLBACK_ELIGIBLE_STUDIES` list is required. For a fold-shaped
manifest, select one held-out fold with
`RSNA_TICKET07_FALLBACK_HELD_OUT_FOLD` (or provide explicit eligible and
held-out lists). For final-refit inference, use the explicit
`RSNA_TICKET07_FALLBACK_FINAL_REFIT=1` opt-in instead:

```bash
export RSNA_TICKET07_MANIFEST=/private/path/ticket-05-real-inference/private/integration-manifest.json
export RSNA_TICKET07_SOURCE=/private/path/rsna-knee-abnormality-detection.zip
export RSNA_TICKET07_OUTPUT=/private/path/ticket-07-recovery
export RSNA_TICKET07_FALLBACK_REFERENCE_LABELS=/private/path/ticket-04-folds/private/fold-manifest.json
export RSNA_TICKET07_FALLBACK_HELD_OUT_FOLD=1
sbatch --wait slurm/run_ticket07_recovery.sbatch
```

The adapter writes the exact submission and identifier-bearing recovery
diagnostics below `private/`, and aggregate fallback numerator/denominator,
slice loss, series loss, failure reasons, and prior counts below `public/`.
Fallback status follows the promotion thresholds: zero is preferred, below
0.5% requires investigation of every case, 0.5–1% normally blocks promotion,
and above 1% blocks promotion. Any nonzero run records fallback cases as
pending operator investigation and blocks promotion until each case has an
investigated/unavoidable disposition. A prepared adapter or synthetic fixture
does not constitute a private full-corpus or competition verification run.

## Ticket 08 fixed-budget gold-only training

`rsna_knee.training` owns the first reference-only scientific run. Its
`TrainingConfig` freezes the shared 8/16/32-channel model, 128×128 inputs,
hierarchical masks, batch size two, Adam at 0.001, constant learning rate, and
100 successful optimizer steps. `masked_bce_loss` averages only supervised
target entries; an all-masked batch advances neither optimizer nor scheduler,
and a persistent no-progress condition fails explicitly.

`build_fold_training_examples` admits only complete reference rows from the
immutable ticket-04 fold manifest. It does not fit normalization statistics,
does not inspect reports, and rejects held-out input from the fitting
partition. `train_one_fold` writes an immutable configuration record, complete
atomic latest/final checkpoints, JSON training logs, private metrics and
reloaded-final-checkpoint held-out metrics. `rsna_knee.evaluation` uses a
dependency-free rank AUC with average ties and preserves undefined one-class
targets instead of inventing a score.

The bounded CPU adapter is `slurm/run_ticket08_train_fold.sbatch`; it requests
one `cluster1-cpu` node, 16 CPUs, 64 GB, six hours, and no exclusive
allocation. It requires the private ticket-04 fold manifest, ticket-05
integration manifest/source, an admitted source version, and a passed runtime
preflight:

```bash
export RSNA_TICKET08_INTEGRATION_MANIFEST=/private/path/ticket-05-real-inference/private/integration-manifest.json
export RSNA_TICKET08_SOURCE=/private/path/rsna-knee-abnormality-detection.zip
export RSNA_TICKET08_FOLD_MANIFEST=/private/path/ticket-04-folds/private/fold-manifest.json
export RSNA_TICKET08_FOLD=0
export RSNA_TICKET08_OUTPUT=/private/path/ticket-08-fold-0
export RSNA_TICKET08_SOURCE_VERSION=rsna-2026-08-04
sbatch --wait slurm/run_ticket08_train_fold.sbatch
```

The public synthetic tests demonstrate finite gradients, learning, masked
progress, checkpoint compatibility, fold isolation, AUC behavior, and the
CPU preflight. No private ticket-08 training allocation or real-corpus AUC is
claimed by the source implementation; those require the authorized Slurm
run and unresolved V07/V09/V14 evidence.

## Ticket 09 interrupted-training recovery

`rsna_knee.training` resumes a fixed-budget run only from a complete
`latest_resumable` checkpoint whose semantic fingerprint still matches the
data, fold, labels, preprocessing, architecture, initialization, seeds,
optimizer, schedule, loss, augmentation, and budget. Operational changes such
as a new Slurm allocation or checkpoint cadence do not change the scientific
run ID. A mismatch is rejected; `fork_training_run` starts a fresh run with
explicit parent-checkpoint provenance.

Each checkpoint contains model, optimizer, scheduler, optional AMP scaler,
progress, best metric, RNG states, configuration, fold, semantic identity,
sampler continuation, and checkpoint version. Latest checkpoints are written
atomically and copied into an immutable history. Epoch boundaries and the
15-minute recovery interval are both checkpoint triggers. `final_selected` is
kept separate from `latest_resumable`, and every execution attempt owns its
own log plus immutable start and completion/failure records.
The loader also validates and migrates complete ticket-08 v1 checkpoints in
memory, preserving their original scientific run identity before the next
checkpoint is published in the v2 format.

After an interruption, rerun the bounded adapter with the same output,
manifests, source version, and configuration. It defaults to
`<output>/checkpoints/latest.pt` and creates a new attempt without overwriting
earlier evidence:

```bash
export RSNA_TICKET09_INTEGRATION_MANIFEST=/private/path/ticket-05-real-inference/private/integration-manifest.json
export RSNA_TICKET09_SOURCE=/private/path/rsna-knee-abnormality-detection.zip
export RSNA_TICKET09_FOLD_MANIFEST=/private/path/ticket-04-folds/private/fold-manifest.json
export RSNA_TICKET09_FOLD=0
export RSNA_TICKET09_OUTPUT=/private/path/ticket-08-fold-0
export RSNA_TICKET09_SOURCE_VERSION=rsna-2026-08-04
sbatch --wait slurm/run_ticket09_resume_training.sbatch
```

The public recovery evidence is synthetic and does not claim a private
Slurm execution, real-corpus rollback duration, GPU compatibility, or a
competition result.

## Ticket 10 complete three-fold OOF evidence

`rsna_knee.evaluation.evaluate_oof_predictions` requires exactly one finite
twelve-score vector for every gold study in the frozen manifest. It recomputes
pooled per-target ROC-AUC, the primary twelve-target macro AUC, each fold's
metrics and class counts, and explicit fold variation. One-class AUCs remain
undefined; `defined_target_macro_auc` is reported separately and is never used
as a silent replacement for the primary macro.

`bootstrap_oof_predictions` performs deterministic study bootstrap, or
reliable-group bootstrap when the fold audit attests a grouping key. It reports
percentile intervals over defined resamples and counts undefined target and
macro resamples explicitly. An OOF result records its frozen repeat name and
is labeled development cross-validation for model selection, not an untouched
internal test or clinical-generalization proof.

Every OOF evaluation also carries fold-structure diagnostics:

- **Pair decomposition.** Each target's pooled pairs are split exactly into
  within-fold and cross-fold pairs, reported with the pair-weighted within-fold
  macro and its bootstrap interval.
- **Prevalence baseline.** A training-prevalence-only predictor that uses no
  images.
- **Fold score scale.** Per-fold means and SDs, their correlation with fold
  training prevalence, and a fold-standardized diagnostic macro.
- **Signal test.** A seeded within-fold label-permutation test of the
  within-fold macro. Its pre-specified rule claims measurable image signal
  only when the one-sided p-value is below 0.05.

`rsna_knee.oof` loads one atomic `run.json` per fold plus its required
configuration, metrics and checkpoint evidence, independently checks
fold/configuration/membership/source-label joins, and refuses incomplete or
duplicate three-fold coverage. Aggregation only reads fold records and writes
regenerated identifier-free JSON/Markdown views; it rejects destinations inside
source run directories. The bounded adapters are
`slurm/run_ticket10_fold.sbatch` and
`slurm/run_ticket10_aggregate.sbatch`:

```bash
export RSNA_TICKET10_INTEGRATION_MANIFEST=/private/path/ticket-05-real-training-input/private/integration-manifest.json
export RSNA_TICKET10_SOURCE=/private/path/rsna-knee-abnormality-detection.zip
export RSNA_TICKET10_FOLD_MANIFEST=/private/path/ticket-04-real-v2/private/fold-manifest.json
export RSNA_TICKET10_RUN_ROOT=/private/path/ticket-10-real-v2
export RSNA_TICKET10_SOURCE_VERSION=rsna-2026-08-04
export RSNA_TICKET10_FOLD=0
sbatch --wait slurm/run_ticket10_fold.sbatch

export RSNA_TICKET10_INVENTORY=/private/path/ticket-03-inventory-run/private/inventory.json
export RSNA_TICKET10_OUTPUT=/private/path/ticket-10-real-v2-oof
sbatch --wait slurm/run_ticket10_aggregate.sbatch
```

The fold adapter is bounded to one non-exclusive `cluster1-cpu` node, 16
CPUs, 64 GB and six hours, and its shared lock/Slurm-state guard keeps no more
than two ticket-10 fold jobs active. The aggregation adapter uses one such
node, two CPUs, 8 GB and one hour. See
[`docs/evidence/ticket-10-oof.md`](docs/evidence/ticket-10-oof.md) for the
synthetic seam and real-corpus execution evidence.

## Ticket 11 locally validated offline candidate

`rsna_knee.candidate` packages the exact MRI-only inference path, three
selected fold checkpoints, the inference configuration, separate cluster and
offline runtime locks, a network-free notebook, corrected OOF evidence, and a
checksum-covered immutable manifest. Candidate validation excludes training
orchestration and report dependencies, verifies the exact thirteen-column
submission contract, and preserves candidate state outside execution output.

The bounded real runner is
`slurm/run_ticket11_candidate.sbatch`. It requests one non-exclusive
`cluster1-cpu` node, 16 CPUs, 64 GB and six hours. Its final representative
execution processed 158 studies, 965 series and 31,486 slices with zero
fallbacks, losses, decode failures or warnings. It produced 158 exact rows
with finite bounded scores and projected 4,973.767 seconds at 1,300 studies;
the projection is below the engineering target but is not hidden-scale proof.
The candidate remains locally validated until an approved Kaggle execution
and owner selection. See
[`docs/evidence/ticket-11-candidate.md`](docs/evidence/ticket-11-candidate.md)
for the full scope, checksums, runtime evidence and limitations.
