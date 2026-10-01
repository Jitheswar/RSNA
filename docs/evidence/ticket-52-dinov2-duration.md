# Ticket 52: longer frozen DINOv2 head training

The 1,500-step frozen DINOv2 gold-only model does not establish an improvement over the preserved, equally trained ResNet control. Its pooled development macro AUC is **0.609957**, versus **0.628390** for ResNet. The paired difference is **−0.018433**, with 95% study-bootstrap interval **[−0.084528, +0.041067]**. Two of three fold differences are negative. The predeclared advancement gate fails: **inconclusive, no promotion**.

The longer run does improve over its own 100-step result: **+0.067139**, interval **[+0.004611, +0.124742]**, with positive differences in all three folds. This answers the duration question without establishing representation superiority. Full minus zero-image-feature AUC is **+0.090086**, interval **[+0.002956, +0.175483]**. Pair-weighted within-fold macro AUC is **0.623579**, with one-sided whole-label-vector permutation **p = 0.00019998** over 10,000 resamples. Constant scores yield AUC 0.5.

The new experiment reused the exact admitted DINOv2 feature cache and corrected 58-study reference folds. It changed only head duration to 1,500 successful Adam steps, retaining anatomical pooling, the 32-unit head, batch size two, learning rate `0.001`, constant schedule, masked BCE, initialization seed `20260911`, batch-order seed `20260912` and final-step selection. All fitting used training-fold reference studies. No encoder fitting, image encoding, reports, new weights or parameter search occurred.

Each fold first reproduced the preserved 100-step predictions with **maximum absolute error 0** and matched its ordered reference-exposure digest before continuing. All three final checkpoints also reloaded exactly. A separate read-only validator rehashed all **277 frozen files**, checked all fold partitions and 3,000 reference exposures per fold, and reproduced all **58 final held-out predictions with maximum error 0**.

Slurm CPU job **7953** completed `0:0` in **39 allocated wall seconds** on one nonexclusive node with sixteen CPUs, 64 GiB and zero GPUs. Code elapsed was **35.46 seconds**, peak RSS **710,860 KiB**, external spend **$0**. The frozen one-hour maximum was not exhausted. The source and resource plan was registered before launch; an earlier incomplete preparation failed on relative mask paths before any allocation and is retained separately.

These adaptively reused development folds are not an untouched evaluation set, and their AUC is not a Kaggle score. No final selection or redistribution is authorized. Existing DINOv2 release-rights restrictions remain.

Private evidence: `artifacts/private/ticket52-dinov2-long-head/plan.json`, `job-7953/evidence.json`, `independent-validation.json` and `controller-7953/terminal-controller.txt`. Plan SHA-256: `377d95efe7e86f1e342a8a5878ea61fef48da9b85de257c471932df06ddad3f0`. [Issue 52](../../.scratch/rsna-leaderboard/issues/52-test-longer-dinov2-gold-head-training.md) records scope and disposition.
