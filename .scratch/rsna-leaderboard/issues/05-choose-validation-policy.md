# Choose validation and evidence requirements for model promotion

Parent: ../map.md
Type: grilling
Labels: wayfinder:grilling
Mode: HITL
Status: resolved
Assignee: root
Blocked by: 01, 02

## Question

Which validation and promotion policy can support honest decisions with only 58 reference-labelled studies, repeated development, rare positives, possible repeated-patient leakage, and fold-dependent score offsets? Decide the primary and diagnostic comparisons, label/extractor isolation, uncertainty reporting, limits on adaptive comparisons and public submissions, treatment of disagreement between local and public scores, and the evidence needed to say an image model learns useful signal. Do not describe reused folds as an untouched test or require an arbitrary significance threshold that the cohort cannot support.

## Comments

### Policy presented for owner review, 2026-09-25

Two live questions were put to the owner: accept the evidence policy below or require stronger promotion evidence; and investigate local/public disagreement using local evidence as the primary selection basis, or let public score drive selection. The owner subsequently answered both in the visible chat; see the resolution below.

- Retain the corrected version-2 three-fold manifest for comparable development. Reconcile every study and target against the source. If a reliable patient/group identity or duplicate study contradicts isolation, repair the split and rerun affected comparisons before any promotion; retain old evidence as superseded.
- Primary competition-shaped estimate: pooled twelve-target reference OOF macro AUC. Always show per-target class support, fold metrics, pair-weighted within-fold AUC, and paired study-bootstrap uncertainty (2,000 resamples with undefined resamples counted). Resample groups instead of studies when reliable grouping exists. Cross-fold offsets require investigation; do not choose a new metric because it favors a candidate.
- Freeze the hypothesis, code/config, adaptation boundaries, training budget/checkpoint rule, comparator, and promotion criteria before each comparison. All fitted preprocessing, report rules, prompts, thresholds, feature adaptation, and pseudo-label sources must exclude the consuming held-out fold. Repeated outer-fold development is disclosed; no pristine internal-test claim.
- Training-only memorization and nonzero gradient checks establish optimization competence, not generalization. Assess useful image information using reference OOF against constant and fold-prevalence controls, within-fold discrimination, and an image-ablation/shuffle diagnostic with recorded uncertainty. A tiny positive delta or memorization success alone is insufficient.
- Distinguish a provisional experiment candidate from a confirmed scientific improvement and from the final submission choice. A promising candidate can progress despite a wide interval; an inconclusive interval remains explicitly inconclusive. Hard integrity, eligible-input, output-completeness, and runtime requirements cannot be waived for AUC.
- Evaluate one declared modeling change per paired comparison; complete the planned folds before interpreting an improvement. Use the same reference exposure and seed where possible, record changed compute/exposures, and inspect rare-target tradeoffs. Do not search unrestricted per-target ensemble weights on 58 studies.
- Public-score conflict triggers checks of cohort shift, pipeline, rank offsets, fallback behavior, and uncertainty before additional tuning. Submission counts and the final shortlist are set in the delivery ticket within the current official allowance; public score is secondary under the recommended choice.

This policy governs the experiment plan; it does not authorize specific experiment execution.

## Answer

Resolved by live owner exchange on 2026-09-25. The owner answered **“yes to both”** to using the existing three folds, simple baselines, uncertainty and explicit inconclusive-result labels, and to investigating conflicting public scores while prioritizing reproducible local evidence. Adopt the policy recorded above. Promising but inconclusive candidates may progress as experiments without being represented as proven improvements. Preserve fold isolation and disclose repeated development use of the reference cohort. Delivery and GPU execution choices remain in their respective tickets.
