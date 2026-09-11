# 07: Recover damaged inputs without losing study rows

**What to build:** Produce complete study predictions with explicit recovery evidence when slices, series or entire studies are unusable.

**Blocked by:** 04: Freeze canonical study folds; 05: Run a real MRI study through inference

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Use bounded deterministic decoder attempts; continue with surviving valid slices/series and explicit masks, respecting geometry and minimum-slice policy.
- [ ] Compute Laplace-smoothed priors solely from known eligible training reference labels per target; use 0.5 if unavailable and never use weak or held-out labels.
- [ ] Test partial corruption, missing series and all-invalid studies; no required row disappears and global configuration/programming failures fail explicitly rather than producing universal fallback.
- [ ] Log study, series, affected files, decoder/error, recovery, remaining usable-series count and fallback status privately.
- [ ] Report full-study fallback numerator/denominator separately from slice/series loss: zero preferred, below 0.5% requires every case investigated and unavoidable, 0.5–1% normally blocks promotion, above 1% blocks it.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.

## Comments

### 2026-09-11 — Recoverable inference implementation

Implemented the ticket-07 recovery seam across `rsna_knee.recovery` and the
private real-DICOM runner. Real inference now continues with surviving decoded
slices when the frozen minimum geometry policy remains satisfied, keeps
partial series distinct from rejected series, and retains one row for every
manifest study. All-invalid studies use a `ReferenceFallbackPrior` with
per-target Laplace smoothing from explicitly eligible reference labels;
unavailable targets use `0.5`, while held-out and non-eligible rows do not
contribute.

Private evidence records study/series/source-entry failures, decoder and
bounded attempts, typed reasons, recovery actions, remaining usable series,
slice/series loss, and fallback status. Public evidence reports fallback
numerator/denominator/rate separately from slice and series loss. Unexpected
decoder infrastructure or programming failures remain fatal. The new bounded
adapter is `slurm/run_ticket07_recovery.sbatch`.

Synthetic focused tests cover partial corruption, missing series, empty
studies, prior isolation, CLI gold-row filtering, private diagnostics, and
fatal unexpected decoder failures. Public-safe scope and unresolved private
verification gates are recorded in
`docs/evidence/ticket-07-recovery.md`.

No private ticket-07 execution was performed, so full-corpus recovery rate,
hidden-set fallback investigation, V17 promotion, and approval-dependent
delivery remain unresolved.

### 2026-09-11 — Boundary tightening and final local verification

Fallback computation now requires an explicit eligible set for direct/API
inputs. Fold-shaped CLI inputs can derive training-reference rows and exclude
one selected held-out fold; report-only rows never enter the prior. Private
evidence also records affected files for geometry/preprocessing rejections and
marks every nonzero fallback case pending operator investigation rather than
claiming it unavoidable. Model, cache, and unexpected decoder failures remain
fatal.

Final local verification: 89 tests passed, mypy passed, the recovery Slurm
adapter parsed with `bash -n`, and `git diff --check` passed. No private Slurm
execution was performed.
