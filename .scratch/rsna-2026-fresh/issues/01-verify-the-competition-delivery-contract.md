# 01: Verify the competition delivery contract

**What to build:** Establish a dated authoritative delivery contract before any competition delivery relies on it.

**Blocked by:** None (dependency-free; implementation authorization is still required).

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [x] Record the exact twelve targets and order, study mapping, macro ROC-AUC, notebook constraints, deadlines, selection limits and permitted inputs from official sources.
- [ ] Distinguish verified facts from inaccessible or unresolved rules; confirm owner eligibility before delivery without accepting binding terms on their behalf.
- [x] Preserve internal readiness before October 15 and a dynamic bottleneck-driven plan; verification changes require reconciliation before dependent action.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V12 — Current rules, account eligibility and delivery constraints

- [ ] Retain dated official evidence for entry/team and final deadlines, metric/schema, notebook internet/runtime requirements, applicable submission/final-selection limits, allowed training/inference inputs, resource restrictions and publication obligations. Before delivery, confirm the owner's account has accepted the rules and is eligible; agents must not accept binding terms on the owner's behalf without authorization. Record any material rule change and reconcile the spec before dependent action.

**Blocking consequence:** Block competition uploads/submissions or resource/publication actions whose eligibility is unverified or conflicts with the rules. Local gold-only synthetic/mechanics work may continue only within separately authorized scope. Do not infer detailed permissions from a generic overview allowance or an inaccessible rules link.

## Completion evidence

- [x] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.

## Comments

### 2026-09-11 — Implementation evidence

Created [`delivery-contract.md`](../delivery-contract.md) as the dated V12 research record. It includes the official target vector and study mapping, macro ROC-AUC, exact submission schema, notebook limits, UTC deadlines, team/submission/final-selection limits, permitted training and inference inputs, data-security and winner-delivery obligations, source-access limitations, owner-confirmation gates, and a bottleneck-driven reconciliation plan.

The contract distinguishes verified public facts from unresolved account, team, resource-license, hidden-runtime, and Kaggle-execution facts. No rules were accepted, account or team state was changed, data or models were uploaded, or competition submission was made. Owner eligibility and the exact Kaggle delivery bundle remain open V12/V08/V13/V16/V17 gates and are intentionally not marked passed by this artifact.

The first and third acceptance criteria are evidenced by the contract. The owner-eligibility portion of the second criterion remains a required pre-delivery confirmation. The completion evidence is recorded in the artifact without exposing restricted data.
