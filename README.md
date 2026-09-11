# RSNA knee mechanics baseline

This repository contains the ticket 2 tracer bullet: a small, image-only
inference core that turns in-memory synthetic pydicom studies into the exact
twelve-target competition schema. It is intentionally not a real-DICOM
compatibility or training-preflight claim.

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
