# 17: Train one fold with admissible weak labels

**What to build:** Complete one training-and-evaluation pass using clean gold-plus-report supervision through the existing image baseline.

**Blocked by:** 15: Evaluate fold-specific deterministic report extraction

**Status:** resolved

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [x] Supervision admission is separate from the imaging Dataset and verifies eligible/excluded fold identities and complete upstream lineage.
- [x] Reference labels override weak disagreements; only confident admissible binary weak labels contribute to masked BCE and unknown/uncertain/failed entries contribute zero.
- [x] Record weak-label version, mixture policy, actual sample/supervised-entry exposures and source disagreements privately.
- [x] Reject missing or contaminated lineage in tests; held-out report-only studies remain excluded and fallback priors remain reference-only.
- [x] Keep architecture, preprocessing and loss unchanged; this integration slice does not itself prove report supervision improves performance.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V15 — Derived-supervision provenance and isolation

- [x] For each extractor/rule/teacher/pseudo-label/student artifact, resolve versioned code/configuration, source and checkpoint hashes, fold/repeat, eligible/excluded study-set identities and all upstream adaptation inputs. Verify zero prohibited held-out input influence for the consuming fold and no unresolved lineage. Enforce reference precedence and zero-mask unknown/uncertain/failed states. Frozen evaluation-only outputs must be distinguishable from admitted training labels.

**Blocking consequence:** Reject artifacts with ambiguous or contaminated lineage from that fold's training and model-selection evidence. Invalidate and rerun materially affected comparisons after repair; an apparent AUC gain cannot override the gate. Clean reference-only work remains possible.

## Completion evidence

- [x] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.

## Comments

### 2026-09-21 — Fold-scoped weak-supervision admission and synthetic pass

Implemented the Ticket 17 supervision seam in `src/rsna_knee/supervision.py`.
`admit_fold_supervision()` accepts only a reproducibility-verified Ticket 15
deterministic extraction run with matching fold, report-set, extractor and
held-out exclusion lineage. The admission result is separate from image
loading and resolves reference/weak target vectors with reference precedence.
Only authoritative confidence-one positive/negative calls are masked into
the existing BCE; all other report states remain zero-masked. Held-out and
missing report-only identities are rejected or excluded explicitly.

`train_one_fold(..., supervision=...)` consumes the resolved artifact without
changing the image model, preprocessing, loss, optimizer or final-checkpoint
evaluation. Private run/checkpoint evidence records weak-label version,
mixture policy, lineage, disagreements and actual sample/supervised-entry
exposures. Astra outputs remain inadmissible because Ticket 16 did not
predeclare a training rule.

Added `weak_supervision_cli.py`, the bounded
`slurm/run_ticket17_train_weak_fold.sbatch` adapter, README documentation and
public evidence in `docs/evidence/ticket-17-weak-supervision.md`. Synthetic
tests cover clean admission, masking, reference precedence, lineage rejection
and one fold training/evaluation pass. No private data was processed and no
Ticket 17 Slurm allocation or report-supervision benefit was claimed.

### 2026-09-21 — One real fold with the frozen Ticket 15 artifact

The pre-existing Ticket 15 frozen deterministic report labels were confirmed
admissible under the committed `report-only-confident-binary-v1` policy. The
Ticket 16 Astra outputs were not used because Ticket 16 had no predeclared
training-admission rule; no Astra-derived policy, threshold, or target choice
was introduced.

Exactly one real fold-0 run was submitted through
`slurm/run_ticket17_train_weak_fold.sbatch` as Slurm job `7056`. It completed
on `cluster1-cpu` in `00:11:30` with one node, 16 CPUs, 64 GB requested
memory, no exclusive allocation, and exit code `0:0`. The full fold contained
2,885 training studies: 38 gold/reference and 2,847 eligible/admitted
report-only studies. The existing representative image integration covered
38 gold and 65 report-only training studies; 2,782 eligible report-only rows
were recorded as image-scope exclusions. The run used 456 gold supervised
entries plus 115 weak supervised entries; 665 weak entries were masked. There
were zero gold-vs-weak disagreements, so no conflict override was needed.

The canonical fold checksum, frozen report semantic digest, extractor
fingerprints, training configuration fingerprint, final checkpoint checksum,
100-step fixed budget, finite-loss result, final/latest checkpoint reload,
held-out separation, and 20-study held-out evaluation artifact were all
independently verified. No performance improvement claim is made. Public
aggregate evidence is in `docs/evidence/ticket-17-weak-supervision.md`; the
detailed private-safe execution report is beside `tsk.txt` as
`/home/ch.jitheswar/tsk-report.md`.

## Answer

Ticket 17 is resolved. Exactly one real fold-0 gold-plus-admissible-report
training pass completed successfully using the frozen Ticket 15 artifact.
Lineage, masking, held-out isolation, checkpoint reload, and execution
evidence were verified. Astra labels were excluded, no report-supervision
benefit claim was made, and Ticket 18 was not started.
