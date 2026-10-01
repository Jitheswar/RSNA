# 52: Test longer DINOv2 gold-head training

Type: task
Status: resolved
Labels: ready-for-agent
Parent: [RSNA leaderboard campaign](../spec.md)

## Purpose

Ticket 44 established that 1,500 head steps improved the frozen ResNet gold-only development result relative to 100 steps. Ticket 38's frozen DINOv2 alternative used only 100 steps. Test this duration hypothesis once, using the existing DINOv2 feature cache and the same corrected 58-study reference folds.

## Frozen scope

- One separate private CPU experiment: three reference-only DINOv2 heads, 1,500 successful steps each, unchanged 384-dimensional encoder features, anatomical pooling, 32-unit head, Adam `1e-3`, batch size two, constant schedule, masked BCE, seeds `20260911` / `20260912`, final checkpoint. No encoder training, image encoding, new weights, reports or hyperparameter sweep.
- Before continuing beyond step 100, reproduce each corresponding Ticket 38 held-out score within `1e-6` and match the exact ordered gold exposure digest. Fail closed on mismatch. Reload all final head checkpoints and reproduce the 58 predictions within `1e-6`. Keep held-out inputs out of fitting.
- Primary comparison: preserved Ticket 44 ResNet gold-only 1,500-step predictions versus new DINOv2 1,500-step predictions. Secondary duration comparison: preserved Ticket 38 100-step DINOv2 versus new long heads. Image ablation zeros the image feature blocks while retaining metadata and presence. Reuse the existing pooled/within-fold metrics, target support, paired 2,000-study bootstrap and 10,000 within-fold label-vector permutations, seed `20260930`.
- Additional evaluation is warranted only if the primary paired 95% lower bound is positive and all three fold differences are positive. Otherwise record inconclusive, with no candidate promotion. These folds have been adaptively reused; this experiment is development evidence.
- One sequential CPU Slurm job, one nonexclusive `cluster1-cpu` node, sixteen CPUs, 64 GiB, one-hour request, zero GPUs, zero external spend, no network during execution, one launch attempt. Freeze input, runtime and source hashes before fitting.
- Apply the owner's existing broad campaign authorization to a new exact private CPU plan, as in Tickets 44 and 48–50. Do not claim Ticket 52-specific owner words. No GPU cap expansion, Ticket 39/40 change, final selection, publication or DINOv2 redistribution.

## Acceptance criteria

- [x] Freeze the plan, exact resource admission, source snapshot and input hashes before launch.
- [x] Verify source replay, fold isolation, final checkpoint replay and complete OOF coverage.
- [x] Preserve terminal allocation, controls, paired uncertainty, cost and a recorded disposition.

## Comments

2026-09-30: Prepared while Ticket 51 remains pending its concrete GPU-budget decision. The purpose is to isolate head-training duration; existing DINOv2 rights restrictions and all prior candidate decisions remain in force.

2026-09-30: Before launch, all 277 source/runtime/input files match the frozen plan SHA-256 `377d95efe7e86f1e342a8a5878ea61fef48da9b85de257c471932df06ddad3f0`; resource-admission record SHA-256 `94328393f7ad02d2d690bdfbd2424e954343fadc7ef7d506499b2d1be3d423ae`. Python syntax and Slurm launcher syntax pass. An incomplete preparation with unresolved relative mask-file paths was retained separately; no training or allocation occurred during that preparation.

## Answer

CPU job **7953** completed `0:0` in **39 allocated wall seconds**, zero GPUs and zero spend. All 277 frozen files remain unchanged. Each 100-step source replay matched exactly; independent replay of all three final heads reproduced all 58 OOF scores with maximum absolute error **0**. Training exposure and fold-isolation checks pass.

DINOv2 1,500-step macro AUC **0.609957** is below the preserved ResNet 1,500-step **0.628390**: difference **−0.018433**, paired 95% interval **[−0.084528, +0.041067]**, two negative folds. Disposition **inconclusive_no_promotion**. Compared to its own 100-step result, DINOv2 improves **+0.067139**, interval **[+0.004611, +0.124742]**, all positive folds. The image-ablation interval is positive; within-fold permutation p is **0.00019998**. The secondary duration improvement does not override the primary advancement gate.

[Readable evidence](../../../docs/evidence/ticket-52-dinov2-duration.md) and private `artifacts/private/ticket52-dinov2-long-head/` preserve complete controls, support, uncertainty, source hashes and terminal allocation. Reused development CV, no final selection, no Kaggle submission and unresolved release rights remain explicit.
