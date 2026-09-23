# 24: Select and deliver the final candidate

**What to build:** Deliver the owner's selected final frozen candidate while preserving the previous Kaggle-verified submission path.

**Blocked by:** 12: Verify the baseline on Kaggle; 23: Compare fold ensembles with all-data refitting

**Status:** ready-for-human

**Authorization:** The owner selected Ticket 23's recommendation and later authorized one full hidden-test Kaggle submission while prohibiting model or pipeline changes. The frozen Ticket 12 v13 notebook is bound to a representative workload and cannot run the full hidden test as-is. A new hidden-test request/notebook binding is required and conflicts with the no-pipeline-change constraint, so no submission attempt was made. Human direction is needed before changing that frozen delivery binding.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [x] Compare the available candidate paths and record the owner's selection of `ticket-11-local-candidate-20260911-v1`, manifest SHA-256 `cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc`. The selected candidate, notebook, adapter and runtime are bound in the previously approved Ticket 12 v13 request; no new combination was uploaded.
- [ ] Partial: recheck the current official overview and the exact candidate/request/execution hashes, provenance, representative output, recovery, fallback, runtime and resources. The full rules page did not return readable content, and the available Kaggle execution used a representative 158-study workload rather than the competition's full hidden test.
- [ ] Partial: preserve the prior approved exact-bundle Kaggle execution record. No full hidden-test output or new competition submission was produced; the representative 158-row output is not treated as a complete competition submission. Hidden runtime and fallback behavior remain unknown.
- [x] Keep the frozen bundle unchanged and preserve its earlier approved Kaggle execution and recovery evidence. The owner selection is recorded separately from the immutable candidate manifest.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V08 — Kaggle environment and exact submission contract

- [ ] Partial: the selected bundle's pinned offline environment and exact output contract passed its approved Ticket 12 v13 representative execution (158 rows, 13 columns, exact header/order, finite bounded scores). It has not produced a full hidden-test output, so final competition-output completeness remains unverified.

**Blocking consequence:** Block Kaggle-verified status and competition delivery claims until the approved actual execution succeeds; a local pass can establish only local validation. Missing dependencies, malformed output or a different tested bundle require repair and re-verification. Any new upload/submission remains subject to approval; tiny-example success does not pass full-scale runtime verification in V16.

### V12 — Current rules, account eligibility and delivery constraints

- [ ] Partial: the official overview was rechecked on 2026-09-23 for the macro-AUC objective, 9-hour CPU/GPU notebook limit, internet-disabled requirement, `submission.csv` filename and October deadlines. The rules page returned no readable body; the exact Ticket 12 v13 owner approval records rule acceptance, account eligibility and a rules recheck on 2026-09-12. Full current rules text and current account state are not independently reverified here.

**Blocking consequence:** Block competition uploads/submissions or resource/publication actions whose eligibility is unverified or conflicts with the rules. Local gold-only synthetic/mechanics work may continue only within separately authorized scope. Do not infer detailed permissions from a generic overview allowance or an inaccessible rules link.

### V16 — Whole-notebook runtime and inference cost

- [x] For the selected exact bundle, retain the approved representative Kaggle timing and resource profile: 2,932.069 seconds for 158 studies, 26,716,372,992-byte peak RAM, zero VRAM, and 9,462.947 seconds projected at 1,300 studies. This is within the 25,920-second engineering target and 32,400-second hard limit; hidden-set timing is still unknown.

**Blocking consequence:** Block normal promotion above the 7h12m engineering target until optimized or the specific margin risk is approved. Execution beyond nine hours is ineligible and cannot be waived by internal approval. Block full-scale compliance claims when evidence is only a tiny example; keep unknown hidden runtime explicit and preserve the previous verified candidate.

### V17 — Candidate completeness, fallback and recovery evidence

- [ ] Partial: the exact selected bundle produced 158 representative rows with zero missing/duplicate rows, zero decode failures and zero full-study fallback; save/reload and interruption recovery preserved all rows and byte-identical output. No per-study or aggregate fallback result exists for the full hidden test set.

**Blocking consequence:** Block serious-candidate promotion for unexplained failures, breached thresholds, missing rows, corrupted bundles, unreconstructable provenance or unproven recovery. Keep diagnostic output distinct from a candidate. Inaccessible hidden failure details remain unknown rather than counted as zero; do not claim their investigation is complete. Preserve the previous known-good immutable candidate, and require separate owner approval for final selection.

## Completion evidence

- [ ] Partial: record the owner selection and representative evidence without exposing restricted data. Full hidden-test output and any later delivery execution remain unclaimed.

## Answer

The owner selected the Ticket 23 recommendation: the unchanged three-fold gold-only candidate `ticket-11-local-candidate-20260911-v1` (manifest SHA-256 `cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc`). The exact candidate, notebook, adapter and runtime were previously bound to the Ticket 12 v13 approval and completed Kaggle execution. The local loader revalidated the frozen candidate manifest, and the v13 execution record matches its checksum. The v13 request had `upload.submission_allowed=false`; no leaderboard action was performed.

This selects the delivery candidate; it does not claim final hidden-test output, hidden-set runtime/fallback behavior, or leaderboard submission. The existing v13 output contains 158 representative studies. Current overview requirements and the remaining V12/V08/V17 limits are recorded in [selection evidence](../../../docs/evidence/ticket-24-candidate-selection.md) and the [external report](../../../../ticket-24-report.md). The candidate manifest remains immutable.

## Comments

- 2026-09-23 — Owner instructed: “use best recc dont ask me”. Applied as selection of the exact Ticket 11 three-fold gold-only candidate recommended in Ticket 23. The representative Kaggle execution and its existing approval scope were preserved; no new hidden-test execution or competition submission was performed.
- 2026-09-23 — Owner requested one full hidden-test Kaggle submission and prohibited model or pipeline changes. Read-only preflight confirmed the frozen Ticket 12 v13 notebook is bound to `workload_kind=representative` and mount `rsna-knee-ticket12-representative-v12`, including pinned representative-input hashes; its notebook hard-codes the v13 request identity. Reusing it would select the 158-study representative dataset (or fail its input checks), rather than run the full hidden input. No kernel push or competition submission was made, preserving the requested single attempt. Kaggle documents that code-competition submissions rerun notebooks on hidden data and failed submissions count toward the team's daily limit ([debug guide](https://www.kaggle.com/code-competition-debugging)). A full hidden run under the current constraint needs a new hidden-test request/notebook binding, which changes the frozen delivery pipeline; no such change was made.
