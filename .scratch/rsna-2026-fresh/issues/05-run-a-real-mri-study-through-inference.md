# 05: Run a real MRI study through inference

**What to build:** Demonstrate representative real DICOM series through the same tested preprocessing, hierarchical model and study-score output used for synthetic inputs.

**Blocked by:** 03: Inventory and safely admit competition data

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Verify complete representative real series on-cluster using private integration manifests; real competition data never becomes a Git fixture.
- [ ] Freeze numeric geometry, minimum-slice, decoder and intensity policies before admitting outputs; reject inconsistent geometry rather than guessing from filenames or InstanceNumber.
- [ ] Use one grayscale channel, aspect-preserving 128 by 128 resize/pad and eight evenly spaced accepted slices per valid series; shorter series have explicit padding masks.
- [ ] Exclude only reliably identified non-diagnostic series under the documented policy; preserve source ordering/mapping and diagnostic warnings.
- [ ] Keep cache/preprocessing independent of model architecture and inference independent of report/training orchestration.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V04 — Numerical geometry, series selection and minimum slices

- [ ] Freeze a versioned policy specifying orientation/orthogonality and position tolerances, duplicate/conflicting-position handling, spacing/dimension consistency, minimum usable-slice count, and reliable scout/localizer exclusions. Synthetic cases must prove invariance to filename/enumeration order and rejection of missing/inconsistent geometry. Representative complete real series must preserve source mapping and satisfy the configured geometry rules. Record rejection counts and reasons; classifying a series as rejected is an explicit outcome, not a hidden ordering guess.

**Blocking consequence:** Do not generate an admitted baseline cache or train on real series until the policy and tests pass. Reject each failing series and use the agreed study-recovery path. Do not silently enable filename/InstanceNumber ordering, relax tolerances, or lower minimum slices to rescue a score. If resulting study fallbacks breach V17, candidate promotion is blocked.

### V05 — Intensity, padding and resize policy

- [ ] Freeze the order and parameters for modality transforms, VOI/window use or deliberate omission, photometric polarity, padding masks, percentile/normalization domain, output dtype, interpolation and deterministic padding. Tests must match independently specified synthetic outcomes for signed values, stored padding/ranges, constant intensity, all-padding input and supported polarity without double transformation. Repeated real-series evaluation under the same configuration must agree within a declared numeric tolerance; output must be finite with the documented shape/range. Constant valid images produce the documented warning/output; all-padding and unsupported cases take explicit rejection paths. No validation-fitted statistics are allowed.

**Blocking consequence:** Block affected preprocessing caches, real-data training and candidate inference until semantics and tests pass. Treat unsupported combinations as rejected inputs rather than applying library defaults. A semantic change invalidates affected caches and creates a new scientific configuration; it cannot be an unrecorded repair to an existing run.

### V06 — Decoder compatibility and recovery

- [ ] Pin decoder versions and deterministic attempt order. For every syntax admitted by the inventory, decode representative real instances through the intended backend and validate declared dimensions, bit/sign representation and pixel semantics against known synthetic/reference cases. Corrupt and unsupported fixtures must produce the expected typed failures with bounded attempts and complete diagnostics. Unexpected infrastructure/programming exceptions must remain distinguishable from damaged input. Record unsupported classes explicitly.

**Blocking consequence:** Block real-input admission for any unverified syntax and GPU training that depends on it. Repair or reject affected instances under the documented recovery policy; do not claim broad support from one uncompressed sample or turn a package/configuration defect into widespread fallback predictions.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.

## Comments

### 2026-09-11 — Implementation and private integration evidence

Implemented the private real-DICOM integration boundary in
`src/rsna_knee/real_inference.py` and the CLI shim
`src/rsna_knee/real_inference_cli.py`. It builds a checksum-covered manifest
from the ticket-03 inventory and selection, rechecks archive identity, decodes
all admitted entries through the pinned pydicom policy, validates identity and
declared pixel semantics, applies versioned geometry and preprocessing rules,
and sends the resulting studies through the existing hierarchical inference
and submission path. Private manifests and diagnostics retain source mappings,
typed failures, exclusions, warnings, and sampling indices; public evidence is
aggregate-only.

The implementation freezes explicit-VR-little-endian decoder support, MR
single-frame grayscale semantics, geometry tolerances and a three-slice
minimum, stored-padding/modality/polarity/normalization behavior, nearest
aspect-preserving 128 by 128 materialization, and post-materialization eight-
slice/sixteen-series packing. It does not import report or training
orchestration, and its cache configuration excludes sampling and model
architecture choices.

The bounded Slurm adapter is `slurm/run_ticket05_real_inference.sbatch` and
requests one `cluster1-cpu` node, 16 CPUs, 64 GB, six hours, and no exclusive
allocation. Private job 6913 completed in 551.6 seconds over 158 studies, 965
series, and 31,486 decoded slices with no rejected series, decoder failures,
fallback rows, or warnings. The exact output contract produced 158 rows and
13 columns with finite bounded scores. No competition data was added as a Git
fixture.

The repository suite passes 69 tests and mypy is clean. Public-safe details
and the unresolved representative-scope, compressed-syntax, repeatability,
and model-quality limitations are recorded in
`docs/evidence/ticket-05-real-inference.md`. V04–V06 are not marked fully
verified by this implementation note; only the demonstrated selected subset
and synthetic policy cases are reported.
