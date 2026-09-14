# 16: Compare one approved local language-model extractor

**What to build:** Evaluate one specifically approved pretrained report extractor against the deterministic baseline under the same clean protocol.

**Blocked by:** 01: Verify the competition delivery contract; 15: Evaluate fold-specific deterministic report extraction

**Status:** ready-for-human

**Authorization:** Implementation was explicitly requested on 2026-09-14. The owner approval required before model use, private-data processing and Slurm execution remains outstanding.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Before download/use, propose exact model/version/hash, source, license, eligibility, public-availability cutoff, leakage assessment, execution location, data-transfer behavior and compute needs.
- [ ] Approval must name a specific model or tightly defined family/version; materially different weights require a new proposal and any GPU work retains its separate approval/preflight gate.
- [ ] Keep report processing local unless a separately approved compliant transfer exists; record prompts, inference settings and reproducibility limits.
- [ ] Compare identical eligible evaluation cohorts and five-state outcomes, including negation/history/uncertainty, coverage, failures, auditability and latency/cost; freeze prompts before held-out evaluation.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V13 — External data/model eligibility and pretrained exception

- [ ] Before use, preserve the exact source/model/version/hash, license, competition eligibility, public availability relative to the applicable cutoff, intended purpose, leakage assessment, reproduction method, compute and execution/transfer policy. Obtain approval for the specific model or tightly defined family/version, including the deterministic comparator for the language-model exception. Demonstrate local processing or separately approved compliant transfer; changing to a materially different resource requires a new proposal.

**Blocking consequence:** Block downloading/using the proposed learned resource in experiments until its review and approval are complete, and block any unapproved report transmission. Continue with the competition-data-only mechanics or clean deterministic work; no generic ImageNet/LLM/medical-pretraining exception is silently assumed.

### V15 — Derived-supervision provenance and isolation

- [ ] For each extractor/rule/teacher/pseudo-label/student artifact, resolve versioned code/configuration, source and checkpoint hashes, fold/repeat, eligible/excluded study-set identities and all upstream adaptation inputs. Verify zero prohibited held-out input influence for the consuming fold and no unresolved lineage. Enforce reference precedence and zero-mask unknown/uncertain/failed states. Frozen evaluation-only outputs must be distinguishable from admitted training labels.

**Blocking consequence:** Reject artifacts with ambiguous or contaminated lineage from that fold's training and model-selection evidence. Invalidate and rerun materially affected comparisons after repair; an apparent AUC gain cannot override the gate. Clean reference-only work remains possible.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.

## Comments

### 2026-09-14 — approval-gated implementation prepared

Implemented the ticket-16 mechanics without downloading or using a learned model:

- added an exact `google/flan-t5-small` proposal with immutable revision, weight
  SHA-256, source, license, cutoff, eligibility/leakage assessment, purpose,
  local transfer policy and bounded `cluster1-cpu` resources;
- added owner-approval validation, local artifact verification, CPU-only
  offline loading, frozen prompt/settings, five-state semantic JSON parsing
  with six-state failure-safe artifacts and private audit digests;
- added same-manifest/same-report-cohort comparison against ticket 15 with
  coverage, uncertainty/failure states, auditability, latency, token/artifact
  cost and repeatability metrics; and
- added synthetic tests, a local CLI and a bounded Slurm adapter.

Focused ticket-16 tests pass (`11 passed`), the full suite passes (`262 passed`),
and mypy passes. V13 remains pending the owner's specific model eligibility
approval; no private report, checkpoint, learned metric or Slurm execution was
created by this change. V15 is enforced as a fail-closed lineage/fold-boundary
contract but still needs private learned-artifact evidence.

### 2026-09-14 — extraction-state semantics tightened

The frozen model prompt now offers only the five semantic states: positive,
negative, not-mentioned, uncertain and unknown. `extraction-failed` remains in
the six-state artifact contract but is reserved for malformed, invalid or
otherwise unusable pipeline output. A model-selected `extraction-failed` is
converted into a policy-origin `invalid-target-output` failure rather than
being accepted as semantic uncertainty. Focused Ticket 16 tests pass (`11`),
and the complete suite passes (`262`).

The detailed handoff report is outside the repository at
`/home/ch.jitheswar/ticket-16-report.md`.
