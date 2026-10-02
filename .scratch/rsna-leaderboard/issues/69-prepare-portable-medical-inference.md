# 69: Prepare portable medical inference

**Type:** task
**Status:** claimed
**Assignee:** campaign agent
**Blocked by:** 68

## Scope

Prepare an additive CPU inference adapter while existing extraction and fitting
continue, without modifying their frozen sources. Match the strict pinned
BiomedCLIP vision recipe, anatomical pooling and final32-hidden heads; explicit
single-fold OOF prediction and three-fold probability mean are separate modes.
Encode each valid slice once, preserve masks/descriptors and invalid zeros,
process head rows in the fixed batch-two order, and return explicit fallback
flags with mandatory eligible-reference priors for all-invalid studies.

Hash-gate restricted external/head checkpoint loads, exact architecture/runtime,
finite FP32 CPU outputs and pinned identity. No text construction, download,
new model fitting, scientific normalization or implicit arbitrary fallback.
Verify numerical dispatch and malformed-input/load safeguards using synthetic
fixtures. Original-MRI adapter replay, real package construction, cross-runtime
Kaggle verification and accuracy/anchor admission remain separately gated by
actual complete Ticket68 receipts and new resource decisions. No speculative
candidate upload, submission or promotion.

## Acceptance

- [x] Portable pinned vision/head loading and inference implemented.
- [x] Explicit OOF/ensemble/fallback/input identities covered by meaningful tests.
- [ ] Original-MRI adapter equivalence and full delivery resource admission pass.
- [ ] Exact offline package and competition runtime verified before use.

## Comments

- Additive `biomedclip_inference.py` implemented,26 focused synthetic tests
  and module type check pass. Existing frozen sources remain unchanged.
  Native adapter equivalence, actual fitted-head packaging and Kaggle runtime
  remain pending; no model gain or competition request follows from these tests.
- Exact scored-source audit confirms five embedded cells match saved
  script354125287/v2/score0.936; the23-cell D4 cache has different lineage
  and cannot currently prove same-cohort scored-baseline predictions.
- [Mechanics and remaining gates](../../../docs/evidence/ticket-69-portable-medical-inference.md).
