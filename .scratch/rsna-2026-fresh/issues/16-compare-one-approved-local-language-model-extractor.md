# 16: Compare one approved local language-model extractor

**What to build:** Evaluate one specifically approved pretrained report extractor against the deterministic baseline under the same clean protocol.

**Blocked by:** 01: Verify the competition delivery contract; 15: Evaluate fold-specific deterministic report extraction

**Status:** resolved

**Authorization:** The owner-approved Astra Medium identity, manual private-report handoff, frozen prompt, and execution scope were recorded before the completed run. No API, model download, or unapproved transfer was used.

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

- [x] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Final aggregate evidence is linked below; the scientific outcome remains mixed/review required.

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

### 2026-09-14 — Astra Medium Codex handoff supersedes local FLAN path

Ticket 16's intended learned extractor is now the explicitly identified
`Astra Medium` model selected through the Codex subscription, with reasoning
and effort fixed to `medium`. The old FLAN-T5/local-checkpoint implementation
is deprecated compatibility code only; its runtime CLI and Slurm adapter fail
closed and no model loader remains in the intended path.

Added `rsna_knee.astra_handoff` and its CLI. The handoff requires an
owner-approved prompt artifact and records its text SHA-256/fingerprint before
batch preparation. It creates stable private batches of at most 16 unique
reports, binds Ticket 15's canonical fold cohorts and gold-label precedence,
validates an exact twelve-target response schema, rejects context mixing, and
turns malformed/incomplete output into all-masked `extraction-failed` audit
records. Raw outputs and private audit artifacts are immutable; public status
is text-free. Resume skips completed batches. No Astra invocation, private
report interpretation, weak-label training, prompt tuning or Ticket 17 work
was performed.

Readiness evidence is in `docs/evidence/ticket-16-local-language-model.md` and
the detailed owner report is outside the repository at
`/home/ch.jitheswar/ticket-16-astra-medium-readiness.md`.

## Answer

### Closure decision — 2026-09-21

Ticket 16 is resolved/closed with the final outcome:

`TICKET 16 SCIENTIFIC EXPERIMENT COMPLETE — MIXED / REVIEW REQUIRED`

The authoritative corrected scientific report is outside the repository at
`/home/ch.jitheswar/tsk2-report.md`. The corrected aggregation definitions and
regression coverage are implemented in
`src/rsna_knee/astra_aggregation.py` and
`tests/test_astra_aggregation.py`.

Operational completion was verified at 276/276 Astra batches over 4,407
studies, with zero validation failures and zero retries. On the held-out gold
cohort, the common-target macro selective accuracy was Astra 0.8614 versus
Ticket 15 0.8961; micro selective accuracy was Astra 325/370 (87.8%) versus
Ticket 15 68/78 (87.2%). Astra materially increased weak-label coverage for
all 12 targets, but performance was target-dependent. Astra and Ticket 15
agreed on all 75 held-out target calls where both were supervised, with zero
disagreements.

The evidence does not generally approve Astra labels for training: Ticket 16
did not predeclare an admission rule, and no such policy is invented here.
The frozen Astra outputs, prompt, fingerprints, Ticket 15 artifacts, and gold
labels remain unchanged. No new scientific decision, Astra rerun, report
reinterpretation, training run, or Ticket 17 work was performed.
