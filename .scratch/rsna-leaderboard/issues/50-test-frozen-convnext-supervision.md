# 50: Test frozen ConvNeXt supervision

Type: task
Status: resolved
Labels: ready-for-agent
Parent: [RSNA leaderboard campaign](../spec.md)
Blocked by: 49

## Purpose

Train one independent frozen ConvNeXt Tiny representation with the same admitted gold/report supervision, folds, 1,500 successful steps, 32-unit head, anatomical pooling, Adam learning rate, seeds and additive report sampling used in the completed Ticket 44 ResNet comparison. Measure representation quality and diversity without changing the public ensemble recipe or Ticket 39.

## Requirements

- [x] Require complete independently verified 4,228-study feature/pooled manifests before freezing exact training inputs and launching.
- [x] Freeze one CPU plan for six matched fold/arm runs, final-step checkpoint selection, immutable source, strict report admission and exact comparator hashes. No pixel/encoder training, new downloads, GPU use, network, publication or final selection.
- [x] Verify precomputed anatomical pooling reproduces direct per-step head updates, gold exposure matches between arms, report pool coverage is complete, ablation preserves masks/descriptors and exact checkpoint reload reproduces held-out predictions.
- [x] Train six fixed-budget heads in one bounded CPU allocation; retain losses, initialization/exposure hashes, final states, predictions and Slurm terminal evidence.
- [x] Evaluate complete 58-study OOF predictions, paired intervals, per-fold effects, within-fold learning controls and a fixed 50/50 ResNet-report / ConvNeXt-report probability blend. No tuning on reference outcomes.
- [x] Record a development disposition without claiming an untouched test estimate, competition score or final promotion.

## Predeclared scientific choices

Feature extraction is [Ticket 49](49-materialize-frozen-convnext-cohort.md); backbone identity and preprocessing are its immutable inputs. Anatomical pooling yields 3,100 values (four planes, each 768 image features plus six metadata states and one presence flag). New head is Linear(3100,32), ReLU, Linear(32,12), final checkpoint at 1,500 successful optimizer steps. Adam learning rate 0.001, constant schedule, batch size two gold studies plus two admitted report studies in the treatment. Initialization seed 20260911, gold batch seed 20260912, report offset 34034034. Reference-only and report heads receive identical initial state and ordered gold exposures per fold. Report-only holdout exclusion and masks remain the existing Ticket 30/Ticket 36 policy.

Primary comparison: ConvNeXt report head versus the fixed Ticket 44 report-supervised ResNet at 1,500 steps. Secondary comparisons: report versus gold-only ConvNeXt, gold-only ConvNeXt versus corresponding ResNet, fixed 50/50 report-head blend versus ResNet, and feature-zero ablations of both new heads. Paired study bootstrap: 2,000 resamples, seed 20260930; 95% intervals with undefined draws retained. Within-fold permutations: 10,000, same seed. No blend-weight or class-weight search.

A provisional development review requires a positive primary macro delta, lower paired 95% bound above zero, and positive deltas in all three folds. A qualifying blend may warrant a separate diversity experiment, using the same criterion. Otherwise preserve an inconclusive/reject disposition. Every outcome remains development evidence on the adaptively reused 58-study cohort; this ticket never promotes a final competition candidate.

## Implementation boundary

`frozen_head_experiment.py` is a separate explicitly identified experiment module. It reuses the existing pooling, masked BCE and deterministic batch sampler while preserving the existing registered ResNet/DINO training and checkpoint contracts. The standalone head checkpoint identifies its external feature/checkpoint provenance; it is not mislabeled as an existing ResNet checkpoint and cannot establish end-to-end competition delivery by itself.

## Comments

2026-09-30: Five focused tests passed: exact direct/prepooled optimizer-update equivalence, descriptor-preserving ablation, matched pair initialization/gold order/complete weak exposure, final checkpoint reload and wrong-identity rejection, and overlapping training-partition rejection. No real-data training has started. Exact training plan and admission will be frozen after Ticket 49 verification completes.

2026-09-30: Exact v2 training plan SHA-256 `924aa958c9fc0fc50b709a0b5057c1272833d69aaceb0c100b870ee861bb47d9`, source snapshot and six original comparator run hashes are frozen under `artifacts/private/ticket50-convnext-supervision/`. The image-scoped admitted report pools contain **2,731 / 2,846 / 2,749** studies for folds 0/1/2. Job **7938** was submitted once under the owner's existing zero-spend campaign authorization: one CPU node, sixteen CPUs, 64 GiB, two hours maximum, no requeue, zero GPUs and no execution-time network. A read-only systemd controller monitor preserves terminal evidence.

Preparation v1 is preserved and superseded before any execution. V2 fixes the per-fold analysis lookup to the existing comparator schema's `delta.folds`; scientific choices, gate criteria, resource limits and planned output predictions are unchanged. Python compilation, Bash parsing and `git diff --check` passed. The five focused head tests passed in 3.76 seconds; no registered ResNet/DINO training contracts were modified.

## Answer

Job **7938** completed `0:0` after **85 allocated CPU-node seconds**, zero GPUs and zero external spend. All six matched heads completed 1,500 successful optimizer steps. All three admitted report pools were fully exposed. The new gold and report exposure sequence hashes exactly match the corresponding frozen Ticket 44 ResNet runs, and gold/report pairs share an identical new head initialization. All **116 held-out predictions** replay exactly from the six final standalone checkpoints (maximum absolute error zero).

| Predeclared comparison | Baseline AUC | Candidate AUC | Delta | Paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| ConvNeXt report versus ConvNeXt gold | 0.603156 | 0.659947 | +0.056791 | +0.015940 to +0.096772 |
| ConvNeXt report versus fixed ResNet report | 0.660737 | 0.659947 | -0.000791 | -0.041491 to +0.041929 |
| Fixed 50/50 report-head blend versus ResNet report | 0.660737 | 0.665014 | +0.004277 | -0.022133 to +0.032276 |

**Disposition: inconclusive_no_promotion.** The primary representation comparison and the fixed blend both have intervals crossing zero and improve only fold 0. Their remaining two fold effects are negative, failing the predeclared gate. Report supervision improves the new head over gold-only in all folds on this adaptively reused development cohort; that secondary result does not establish improvement over the retained comparator or a competition score.

Both new heads retain measurable image signal under the declared controls. Full-minus-feature-zero deltas are **+0.081862** gold-only (95% interval +0.016933 to +0.146319) and **+0.138387** report-supervised (+0.075207 to +0.202854). Pair-weighted within-fold AUC is **0.614361 / 0.677889**; 10,000 within-fold label-vector permutation upper-tail p-values are **0.00019998 / 0.00009999**. The ablation preserves descriptors and plane presence.

Full analysis, fold/per-target effects, support and undefined resamples are `artifacts/private/ticket50-convnext-supervision/analysis.json`; readable summary is `analysis.md`. Exact plan/source/approval, standalone final head states, exposure counters, independent cross-architecture exposure check, five focused checks and terminal controller evidence are preserved in the same run root. Code elapsed **81.855 seconds**, peak process RSS **570,048,512 bytes**. No registered ResNet/DINO checkpoint contract was changed, no public ensemble recipe was altered, and no final candidate was selected.
