# 12: Verify the baseline on Kaggle

**What to build:** Establish the first Kaggle-verified baseline using the exact frozen bundle and preserve its reproducible delivery path.

**Blocked by:** 01: Verify the competition delivery contract; 11: Freeze a locally validated offline candidate

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Prepare the specific notebook/checkpoint upload and execution request with verified local evidence; obtain owner approval before any upload or competition submission.
- [ ] Execute only the approved combination, record accepted output, environment, actual workload and whole-notebook runtime, and tie evidence to immutable hashes.
- [ ] Do not equate example-only execution with measured hidden-scale compliance or claim unknown hidden fallback rates are zero.
- [ ] Append promotion evidence without mutating frozen content; preserve the last verified bundle and distinguish verification from owner-selected final status.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V08 — Kaggle environment and exact submission contract

- [ ] Produce a separately pinned offline inference environment that loads the exact frozen core/model/configuration without report dependencies or runtime downloads. Synthetic packaging checks and then an explicitly approved Kaggle execution must generate submission.csv with StudyInstanceUID first, the exact twelve target columns in the agreed order, no extra index column, each expected identifier exactly once, and twelve finite scores in [0,1]. Record the actual workload, environment and accepted output evidence for the exact bundle.

**Blocking consequence:** Block Kaggle-verified status and competition delivery claims until the approved actual execution succeeds; a local pass can establish only local validation. Missing dependencies, malformed output or a different tested bundle require repair and re-verification. Any new upload/submission remains subject to approval; tiny-example success does not pass full-scale runtime verification in V16.

### V12 — Current rules, account eligibility and delivery constraints

- [ ] Retain dated official evidence for entry/team and final deadlines, metric/schema, notebook internet/runtime requirements, applicable submission/final-selection limits, allowed training/inference inputs, resource restrictions and publication obligations. Before delivery, confirm the owner's account has accepted the rules and is eligible; agents must not accept binding terms on the owner's behalf without authorization. Record any material rule change and reconcile the spec before dependent action.

**Blocking consequence:** Block competition uploads/submissions or resource/publication actions whose eligibility is unverified or conflicts with the rules. Local gold-only synthetic/mechanics work may continue only within separately authorized scope. Do not infer detailed permissions from a generic overview allowance or an inaccessible rules link.

### V16 — Whole-notebook runtime and inference cost

- [ ] Benchmark every stage from startup/setup through CSV creation on documented hardware and representative workload, including long-tail studies, model loading and ensembling. Record peak VRAM, package/model size, input counts and observed versus projected times. A locally promoted candidate must project no more than 25,920 seconds at approximately 1,300 studies unless the owner explicitly approves the engineering-margin risk; observed Kaggle evidence must state which workload actually ran and remain within the 32,400-second hard limit. An example-only run is not hidden-scale proof.

**Blocking consequence:** Block normal promotion above the 7h12m engineering target until optimized or the specific margin risk is approved. Execution beyond nine hours is ineligible and cannot be waived by internal approval. Block full-scale compliance claims when evidence is only a tiny example; keep unknown hidden runtime explicit and preserve the previous verified candidate.

### V17 — Candidate completeness, fallback and recovery evidence

- [ ] Verify every expected study has one valid row, the exact contract from V08, immutable bundle checksums and complete scientific/execution provenance. Generate a private per-failure report and aggregate full-study fallback numerator/denominator; inspect every case below 0.5 percent before judging it unavoidable. Zero fallback is the target; 0.5–1 percent normally blocks promotion and above one percent blocks it. Demonstrate save/reload and interruption recovery without loss of the last complete checkpoint, with approximately fifteen-minute rollback where practical and any measured exception explicit. Each promotion level must have evidence for that exact bundle.

**Blocking consequence:** Block serious-candidate promotion for unexplained failures, breached thresholds, missing rows, corrupted bundles, unreconstructable provenance or unproven recovery. Keep diagnostic output distinct from a candidate. Inaccessible hidden failure details remain unknown rather than counted as zero; do not claim their investigation is complete. Preserve the previous known-good immutable candidate, and require separate owner approval for final selection.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.
