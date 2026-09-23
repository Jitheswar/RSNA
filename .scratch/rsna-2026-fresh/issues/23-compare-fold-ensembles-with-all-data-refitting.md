# 23: Compare fold ensembles with all-data refitting

**What to build:** Produce an evidence-based final delivery strategy using available validated methods without requiring advanced research to succeed.

**Blocked by:** 11: Freeze a locally validated offline candidate

**Status:** resolved

**Resolution:** The no-refit delivery recommendation is settled; direct quality and resource measurements for an all-data refit remain partial and are carried forward as unknowns for Ticket 24.

**Authorization:** Planning-only originally. The owner's direct instruction to implement Ticket 23 on 2026-09-23 authorized this implementation and any bounded CPU work under the existing resource cap. No refit job was submitted because the available evidence did not justify one. GPU training and other separately approval-gated actions remain unauthorized.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Partial: compare the existing three-fold candidate with one explicit all-data refit proposal across OOF evidence, update budget, runtime, memory, checkpoint size and packaging. The refit was not run, so its exact OOF score and resource cost remain unknown; OOF for the deployed fold average is also not directly estimated by the held-out single-fold predictions.
- [x] Define a one-model refit at the established 100-successful-update budget, using only the 58 trusted reference-label studies and the fixed gold-only baseline configuration. No report, pseudo-label, distillation, or external learned supervision is admitted.
- [x] Do not run a refit solely to complete the checklist. Current OOF supplies no performance reason to spend on a refit, and the alternative methods have not earned adoption; the decision and measurement limits are recorded without claiming a refit would score worse.
- [x] Do not assign existing OOF scores to hypothetical refitted weights. Compute and record Laplace-smoothed fallback priors from all trusted reference labels, and leave the frozen candidate unchanged.

## Completion evidence

- [x] Record the delivery recommendation, measured scope, unresolved comparisons and limits without exposing restricted data. No future execution or approval-dependent check is marked passed by this evidence.

## Answer

Keep the existing three-fold gold-only candidate as the delivery path for Ticket 24 to consider because its offline delivery path has measured resource margin. This is not a claim that the candidate is high-quality or outperforms an all-data refit. No all-data refit was run. The gold-only model's fold-specific development OOF pooled macro AUC was 0.464313 (95% study-bootstrap interval 0.404–0.524), below the fold-prevalence-only comparator at 0.474353; the within-fold permutation test gave p=0.918. Ticket 18 report supervision, Ticket 19 the wider encoder, and Ticket 22 distillation were inconclusive, Ticket 20 ran no combined arm, and Ticket 21 admitted no pseudo-labels. This selection evidence provides no validated performance reason to spend on an all-data refit or replace the candidate; it is not a direct quality estimate of the deployed three-model average, and it does not establish that a refit would score worse.

The proposed refit budget is one model, 100 successful optimizer updates, batch size 2, the fixed 8/16/32 gold-only baseline configuration and final-checkpoint policy, with all 58 trusted reference-label studies. This matches the existing per-model update budget, not the three-model ensemble's 300 aggregate updates. No refit wall time, peak memory, inference runtime, VRAM, package size or OOF score exists. The measured candidate costs, comparison limits and all-reference Laplace priors are recorded in [public evidence](../../../docs/evidence/ticket-23-delivery-strategy.md) and the [external report](../../../../ticket-23-report.md). The current immutable candidate and its verified delivery path remain unchanged; final owner selection stays with Ticket 24.

## Comments

- 2026-09-23 — Owner directly instructed implementation of Ticket 23. Review covered the corrected version-2 OOF evidence, the frozen candidate and its approved Kaggle execution, plus Tickets 18–22. No refit or GPU job was submitted. The strategy records the all-reference priors without changing the frozen candidate.
