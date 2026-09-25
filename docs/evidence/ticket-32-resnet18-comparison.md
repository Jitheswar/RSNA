# RSNA Ticket 32: ResNet-18 three-fold comparison

Recorded: 2026-09-25. Aggregate evidence only; no study identifiers, report text, source images, or per-study predictions are included.

## Scope and execution

Ticket 32 completed the predeclared corrected three-fold reference OOF comparison against the preserved corrected v2 mechanics baseline. The cohort contained 58 studies (20/19/19 held out by folds) under fold manifest SHA-256 `8460234d0fcbfd2eb9ca3a3e4165929f50a86acac37d2cd03be1164c2f287fb8`.

The frozen configuration used a frozen `torchvision.ResNet18_Weights.IMAGENET1K_V1` encoder (`pytorch/vision@v0.22.1`, SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`), 224×224 whole-field inputs, eight slices and sixteen series per study, plane-stratified series pooling, and an independently trained fold head. The CPU head training used Adam at 0.001, batch size 2, 100 optimizer steps per fold, and seeds 20260911/20260912. The encoder remained frozen; held-out outcomes were used for prediction/evaluation only. Only identity-matched frozen features were reused across folds.

The three folds ran sequentially in Slurm job `7169` on `cluster1-cpu` with 16 CPUs, 64 GiB, and a six-hour wall-time limit. No GPU or external spend was used. All folds completed 100/100 steps with zero skipped batches. Analysis job `7173` replayed the checkpoints, recomputed the OOF comparison and controls, and performed the zero-image-feature ablation. The failed analysis attempts `7170` and `7171` stopped before image loading or metric computation. Checkpoint replay differed from saved held-out outputs by at most `5.96e-8`.

| Fold | Training studies | Held-out studies | Run ID | Final checkpoint SHA-256 |
| ---: | ---: | ---: | --- | --- |
| 0 | 38 | 20 | `5c04da85b882a75c978004d99a2e966b9b8dbe062e90830359a3c66f63242e75` | `c687c925464965f974bca0994e71171fd09a1a93add3793234e66045060ac697` |
| 1 | 39 | 19 | `5a2850a299f276a59a03ae182c13ed2150a54bbe7dfe1978f87aa6409acccde5` | `14ebcac1ce89e1fabec07908768825f0d19019a5779ecc919b6d2281f71c3ba7` |
| 2 | 39 | 19 | `6afcc6544eba98e6bc61b90a52276057e62c6f4c424f6bdd9df5fd4907e223c1` | `3bfcb068e8b48b91a3d63558506dedb6d88aae948f4839d1082e15550509a706` |

The focused controlled comparison-boundary check recorded in the frozen plan was `tests/test_image_model_comparison.py`: 2 passed in 2.43 seconds. This validates comparison-boundary software behavior only; it is not performance evidence.

## Decision

The ResNet-18 result is provisional and inconclusive. Its pooled point estimate is positive, all three fold deltas are positive, and the within-fold permutation diagnostic meets the predeclared signal rule. The paired improvement interval crosses zero, the candidate OOF interval includes 0.5, and the full-versus-zero-image ablation interval crosses zero. Do not promote it as a confirmed improvement or final candidate. No supervision, DINOv2, or fine-tuning branch was started. Final freeze, publication, redistribution, and release remain blocked by the separate rights and winner-release gate.

Reliable patient grouping was unavailable, so uncertainty was resampled by study and repeated-patient leakage cannot be ruled out. This is repeated development cross-validation on the 58-study cohort, not an untouched test or a clinical-generalization estimate.

Private source artifacts, full configuration, exact run records, and detailed analysis are retained under `artifacts/private/ticket32-resnet18/`; the frozen run plan is `artifacts/private/ticket32-resnet18/ticket32-run-plan.json`.

---

## Detailed comparison results

- Corrected reference OOF studies: `58`
- Fold manifest checksum: `8460234d0fcbfd2eb9ca3a3e4165929f50a86acac37d2cd03be1164c2f287fb8`
- Candidate pooled twelve-target macro AUC: `0.518125`
- Corrected mechanics baseline pooled macro AUC: `0.464313`
- Candidate minus baseline pooled macro AUC: `0.053812`
- Paired 95% study-bootstrap interval: `-0.004524` to `0.115394`; `1999` defined and `1` undefined of `2000`
- Candidate pair-weighted within-fold macro AUC: `0.546154`
- Candidate cross-fold macro AUC: `0.504182`
- Candidate training-prevalence-only pooled macro AUC: `0.474353`
- Constant-score macro AUC: `0.500000`
- Candidate minus zero-image-ablation pooled macro AUC: `0.015748`; paired 95% study-bootstrap interval `-0.045430` to `0.077447` (1999 defined, 1 undefined)
- Repeated-patient leakage cannot be ruled out because reliable group identity is unavailable; bootstrap resampling is by study.
- Repeated development CV is not an untouched test or a clinical-generalization estimate.

### Fold results

| Fold | Baseline macro AUC | ResNet-18 macro AUC | Delta | Held-out studies |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.492283 | 0.497681 | 0.005398 | 20 |
| 1 | 0.421119 | 0.601088 | 0.179968 | 19 |
| 2 | 0.488201 | 0.547983 | 0.059782 | 19 |

### Per-target comparison

| Target | Positives | Negatives | Baseline AUC | ResNet-18 AUC | Delta | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 24 | 34 | 0.462010 | 0.500000 | 0.037990 | -0.129032 to 0.202802; undefined 0 |
| MCL | 9 | 49 | 0.351474 | 0.446712 | 0.095238 | -0.217568 to 0.420880; undefined 1 |
| Medial Meniscus | 26 | 32 | 0.443510 | 0.556490 | 0.112981 | -0.153846 to 0.365254; undefined 0 |
| Lateral Meniscus | 23 | 35 | 0.427329 | 0.409938 | -0.017391 | -0.203129 to 0.181838; undefined 0 |
| Medial OA | 15 | 43 | 0.480620 | 0.674419 | 0.193798 | 0.008504 to 0.358244; undefined 0 |
| Lateral OA | 11 | 47 | 0.411992 | 0.495164 | 0.083172 | -0.009310 to 0.161029; undefined 0 |
| PF OA | 21 | 37 | 0.465894 | 0.483912 | 0.018018 | -0.253005 to 0.282680; undefined 0 |
| Effusion | 35 | 23 | 0.577640 | 0.490683 | -0.086957 | -0.320956 to 0.152632; undefined 0 |
| Synovitis | 27 | 31 | 0.445639 | 0.585424 | 0.139785 | -0.076930 to 0.354886; undefined 0 |
| Baker's | 12 | 46 | 0.467391 | 0.568841 | 0.101449 | -0.085558 to 0.277316; undefined 0 |
| Contusion | 19 | 39 | 0.497976 | 0.511471 | 0.013495 | -0.288372 to 0.324251; undefined 0 |
| Fracture | 18 | 40 | 0.540278 | 0.494444 | -0.045833 | -0.244457 to 0.153869; undefined 0 |

### Candidate OOF details

#### Gold-only OOF comparison

- Repeat: `repeat-1`
- Fold manifest checksum: `8460234d0fcbfd2eb9ca3a3e4165929f50a86acac37d2cd03be1164c2f287fb8`
- Fold generation method: `gold:iterative-multilabel-greedy-v2(support-seeded;phase2=marginal-total-imbalance;fold-size-cap);report-only:stable-study-or-reliable-group-identifier-hash-v1;groups:unsplit-v1`
- Fold manifest superseded: `false`
- Held-out predictions: `58`
- Pooled macro ROC-AUC: `0.518125`
- Fold macro ROC-AUCs: `0.497681`, `0.601088`, `0.547983`
- Mean fold macro ROC-AUC: `0.548917`
- Fold macro standard deviation: `0.042221`
- Pair-weighted within-fold macro ROC-AUC: `0.546154`
- Cross-fold macro ROC-AUC: `0.504182`
- Mean within-fold pair share: `0.332361`
- Training-prevalence-only pooled macro ROC-AUC: `0.474353`
- Within-fold permutation test: p_upper `0.039296` over `10000` resamples (null mean `0.500384`, 95th percentile `0.542918`); measurable image signal: `true`
- Study bootstrap pooled macro 95% interval: `0.471360` to `0.560698` (1 undefined of 2000)
- Study bootstrap within-fold macro 95% interval: `0.496224` to `0.597603` (1 undefined of 2000)

| Target | Pooled AUC | Within-fold AUC | Cross-fold AUC | Within share | Prevalence baseline | Positives | Negatives | Fold 0 | Fold 1 | Fold 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 0.500000 | 0.572491 | 0.464351 | 0.329657 | 0.443015 | 24 | 34 | 0.395604 | 0.636364 | 0.688889 |
| MCL | 0.446712 | 0.367347 | 0.486395 | 0.333333 | 0.493197 | 9 | 49 | 0.490196 | 0.395833 | 0.208333 |
| Medial Meniscus | 0.556490 | 0.642599 | 0.513514 | 0.332933 | 0.476562 | 26 | 32 | 0.585859 | 0.670455 | 0.677778 |
| Lateral Meniscus | 0.409938 | 0.426415 | 0.401852 | 0.329193 | 0.437888 | 23 | 35 | 0.414141 | 0.590909 | 0.256410 |
| Medial OA | 0.674419 | 0.739535 | 0.641860 | 0.333333 | 0.492248 | 15 | 43 | 0.693333 | 0.800000 | 0.728571 |
| Lateral OA | 0.495164 | 0.575581 | 0.455072 | 0.332689 | 0.462282 | 11 | 47 | 0.390625 | 0.937500 | 0.483333 |
| PF OA | 0.483912 | 0.521236 | 0.465251 | 0.333333 | 0.490991 | 21 | 37 | 0.538462 | 0.380952 | 0.642857 |
| Effusion | 0.490683 | 0.492537 | 0.489758 | 0.332919 | 0.475776 | 35 | 23 | 0.343750 | 0.797619 | 0.363636 |
| Synovitis | 0.585424 | 0.570397 | 0.592857 | 0.330944 | 0.453405 | 27 | 31 | 0.272727 | 0.744444 | 0.727273 |
| Baker's | 0.568841 | 0.673913 | 0.516304 | 0.333333 | 0.492754 | 12 | 46 | 0.609375 | 0.733333 | 0.683333 |
| Contusion | 0.511471 | 0.538462 | 0.497976 | 0.333333 | 0.482456 | 19 | 39 | 0.714286 | 0.371795 | 0.500000 |
| Fracture | 0.494444 | 0.433333 | 0.525000 | 0.333333 | 0.491667 | 18 | 40 | 0.523810 | 0.153846 | 0.615385 |

### Fold score scale

Correlation of fold mean score with fold training prevalence over fold x target cells: Pearson `0.890529`, Spearman `0.847259`. Fold-standardized pooled macro (diagnostic only): `0.553969`.

| Target | Fold 0 mean / SD | Fold 1 mean / SD | Fold 2 mean / SD | Fold 0 train prev. | Fold 1 train prev. | Fold 2 train prev. | Between/within SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 0.4373 / 0.0105 | 0.4619 / 0.0074 | 0.4042 / 0.0023 | 0.4474 | 0.4103 | 0.3846 | 3.500048 |
| MCL | 0.1669 / 0.0156 | 0.2637 / 0.0161 | 0.2344 / 0.0131 | 0.1579 | 0.1538 | 0.1538 | 2.718151 |
| Medial Meniscus | 0.2937 / 0.0233 | 0.4561 / 0.0182 | 0.4236 / 0.0190 | 0.4474 | 0.4615 | 0.4359 | 3.478802 |
| Lateral Meniscus | 0.4371 / 0.0198 | 0.3961 / 0.0157 | 0.4579 / 0.0170 | 0.3684 | 0.3846 | 0.4359 | 1.463979 |
| Medial OA | 0.2177 / 0.0252 | 0.2441 / 0.0219 | 0.2127 / 0.0122 | 0.2632 | 0.2564 | 0.2564 | 0.695690 |
| Lateral OA | 0.1762 / 0.0181 | 0.2970 / 0.0124 | 0.2588 / 0.0097 | 0.1842 | 0.2051 | 0.1795 | 3.764155 |
| PF OA | 0.4661 / 0.0259 | 0.4010 / 0.0117 | 0.4125 / 0.0045 | 0.3684 | 0.3590 | 0.3590 | 2.025441 |
| Effusion | 0.6655 / 0.0194 | 0.5829 / 0.0118 | 0.5951 / 0.0134 | 0.6053 | 0.5897 | 0.6154 | 2.448721 |
| Synovitis | 0.3270 / 0.0307 | 0.4645 / 0.0224 | 0.4584 / 0.0200 | 0.4737 | 0.4359 | 0.4872 | 2.603184 |
| Baker's | 0.1724 / 0.0228 | 0.2597 / 0.0212 | 0.2234 / 0.0125 | 0.2105 | 0.2051 | 0.2051 | 1.897605 |
| Contusion | 0.4027 / 0.0232 | 0.3531 / 0.0145 | 0.3554 / 0.0199 | 0.3158 | 0.3333 | 0.3333 | 1.188549 |
| Fracture | 0.2724 / 0.0217 | 0.3204 / 0.0145 | 0.2854 / 0.0112 | 0.3158 | 0.3077 | 0.3077 | 1.284290 |

Undefined one-class AUCs remain undefined; no reduced-target macro is used as the primary metric. Within- and cross-fold AUCs partition the pooled positive/negative pairs exactly; cross-fold pairs compare scores from different fold models. The prevalence baseline uses no images.

This is repeated development cross-validation for model selection. It is not an untouched internal test and does not establish clinical generalization.


### Image ablation

The paired bootstrap reports full ResNet-18 minus the zero-image-feature ablation: pooled delta `0.015748`, 95% interval `-0.045430` to `0.077447` (1 undefined of 2,000). The ablated head was unchanged; actual masks and series descriptors were retained. Held-out cache hits were 20, 19 and 19 by fold, with no materialized features. Replay differences from saved held-out outputs were at most `5.96e-8`.

### Decision

The ResNet-18 comparison is **provisional and inconclusive**. The pooled candidate-minus-baseline delta is positive (`+0.053812`), all three fold deltas are positive, and the within-fold label-permutation diagnostic indicates measurable signal (`p=0.0393`). However, the paired 95% interval crosses zero (`-0.004524` to `0.115394`), the candidate OOF interval includes `0.5`, and the full-versus-zero-image-ablation interval crosses zero. Do not promote this model as a confirmed improvement or final candidate. The existing evidence policy was applied without adding a numeric threshold.

Reliable patient grouping is unavailable, so uncertainty is study-resampled and repeated-patient leakage cannot be ruled out. This is repeated development CV, not an untouched test or a clinical-generalization estimate.
