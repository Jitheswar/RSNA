# 16: Compare one approved local language-model extractor

**What to build:** Evaluate one specifically approved pretrained report extractor against the deterministic baseline under the same clean protocol.

**Blocked by:** 01: Verify the competition delivery contract; 15: Evaluate fold-specific deterministic report extraction

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

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
