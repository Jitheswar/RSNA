# 44: Test full weak-cohort exposure with frozen image features

**What to build:** A separate, matched three-fold CPU experiment tests whether the short Ticket 36 training budget prevented most admitted report studies from contributing to the model.

**Blocked by:** 36: [Compare expanded supervision across all three folds](36-compare-expanded-supervision-across-three-folds.md); exact owner approval for this new scientific configuration and CPU allocation.

**Parent:** [RSNA leaderboard campaign with zero external spend](../spec.md)

**Status:** resolved

## Evidence for the question

Ticket 36 used 100 successful optimizer steps per fold. The additive sampler drew two weak studies per step, so its run records show only 200 weak sample exposures per fold from admitted pools of 2,731, 2,846 and 2,749 studies. The cyclic shuffled sampler (`_batch_indices`) visits every pool member once before cycling; these runs therefore exposed at most 7.3%, 7.0% and 7.3% of the eligible weak studies. The three-fold pooled macro AUC moved from 0.518125 to 0.528005, with paired 95% interval `[-0.035188, +0.055273]`. This is an underexposure hypothesis, not proof that longer training will help.

## Proposed frozen experiment

- Reuse the exact Ticket 36 sealed 4,228-study frozen ResNet-18 feature cohort, corrected 58-study gold folds, Ticket 30 admitted Ticket 16 report labels, reference precedence, preprocessing, model, ImageNet-1K V1 checkpoint and additive weak sampler. Read all prior caches and source artifacts without modifying them.
- Run **1,500 successful optimizer steps** from the same admitted initialization in each of six independent runs: gold-only and gold-plus-report for folds 0, 1 and 2. Keep batch size 2, Adam at `1e-3`, constant schedule, masked BCE, seeds `20260911` and `20260912`, and final-step checkpoint selection. The report arm adds two weak samples per gold batch; 3,000 weak draws per fold cover each admitted weak pool at least once. Match the ordered gold draws and initial model state within every pair.
- Keep held-out gold images and labels out of fitting. Use all 58 held-out predictions exactly once per arm; report pooled and within-fold twelve-target AUC, target support, paired 2,000-resample study bootstrap (seed `20260929`), undefined outcomes, exposure coverage, controls and measured CPU/storage cost. Compare the longer pair and the frozen 100-step Ticket 36 results as development evidence, with no claim that repeated folds are an untouched test.
- Use one sequential Slurm job on `cluster1-cpu`, one node, 16 CPUs, 64 GiB RAM, a two-hour wall-time request, no GPU and zero external spend. Preserve the existing 2 TB soft storage limit and all Ticket 36 checkpoints. No new report extraction, weight download, or Kaggle submission is included.
- Require a new exact artifact/resource approval before execution. Do not reinterpret the Ticket 36 approval as permission for this different step budget. No candidate promotion, final freeze, publication or redistribution is authorized by the experiment.

## Acceptance criteria

- [x] Freeze an immutable plan, source/checkpoint/feature/fold/label hashes, output roots and exact resource request; validate the new approval against them before any training.
- [x] Verify each pair's initial model digest and ordered gold-exposure digest agree, while added weak exposure remains separate. Reject partial or mismatched fold sets.
- [x] Check that each report arm covers every admitted weak study at least once, and record effective supervised target entries separately from sample exposures.
- [x] Complete six resumable final checkpoints with strict fold isolation, reload them, and reproduce all 58 held-out scores.
- [x] Apply the predeclared paired analysis and record an explicit promotion/rejection decision in the final-freeze register. A positive point estimate alone is inconclusive and does not authorize final freeze.

## Comments

### 2026-09-29 — proposed from completed run evidence

The authenticated public leaderboard places the current scored Elyra fallback at 0.498, far below the 0.961 public leader. The underexposed weak cohort is one measurable, low-spend way to test whether the prepared image/label investment can improve an independent candidate while Ticket 39 waits at its separate GPU recovery approval gate. This proposal had not run and had no new owner approval when written.

### 2026-09-29 — bounded CPU execution started

The owner subsequently authorized work broadly with “approved for anything everything” and reiterated the first-place objective. The separate Ticket 44 approval records that this authorization was applied to the exact frozen plan; it does not claim that Ticket 44 was named in the owner's words. The new approval covers checkpoint use and the one-job, two-hour CPU allocation only. Plan SHA-256: `adfe087d19a1685556bfb26a52301f9dfc9ac3461f12799847514c7f15bea89f`; approval SHA-256: `90dedd115315c4969700e8b25ca86dcf0eacac7e1ae904cd0467e62cca6aac73`. Slurm job `7829` started on `cluster1-cpu` at 15:04 IST with 16 CPUs, 64 GiB, no GPU and no requeue. Ticket 39 job `7816` remains queued separately. Results and promotion decision are pending the six completed fold runs.

## Answer

### 2026-09-30 — complete comparison recovered from preserved training runs

Job `7830` completed all six 1,500-step training runs, then failed during analysis because the compute node has no Git executable and all code-revision fields were null. The exact source snapshot still matches its frozen manifest. Recovery job `7908` failed before analysis because its launcher omitted `RSNA_PROJECT_ROOT`; job `7910` failed during replay because the recovery script referenced a nonexistent configuration attribute. Both failures, diagnoses, and immutable requests are preserved. Corrected CPU job `7911` completed `0:0` in 40 seconds on one node, 16 CPUs, 64 GiB, zero GPUs, no requeue, and zero external spend. It loaded the frozen training modules and validated the unchanged plan `988448d210192a513b0b07ab1eb630015680d351f02c3bf7adb2326c2962f29b` and approval.

The recovery creates explicitly derived analysis views. Their source identity is the verified snapshot SHA-256 `84fa4c7b7eb1863a4d9b223854e4032e0a2e89c39229b563db17c2b0f0f19a06`, not an invented Git commit. Original environments and artifact hashes are retained in supplements; original training directories are unchanged. Recorded compute measurements are adapted to the comparator's sidecar schema with source hashes. Every final checkpoint reloaded, and all 116 held-out study outputs across both arms reproduced with maximum absolute error **0**. All initialization and ordered gold-exposure checks passed.

Gold-only pooled macro AUC is **0.6283898645**, and report-supervised AUC is **0.6607373788**. The paired difference is **+0.0323475143**, with 95% interval **[-0.0075421162, +0.0717768306]** over 2,000 defined resamples and zero undefined resamples. Fold differences are approximately **-0.0422, +0.0532, +0.0684**. Disposition: **inconclusive_no_promotion**. Longer training produced stronger development results than the old short runs, but this matched comparison does not establish a reliable incremental benefit from reports. These are adaptively reused development folds.

Each report arm exposed all **2,731 / 2,846 / 2,749** admitted weak studies through 3,000 draws per fold. Effective weak supervised entries were **14,820 / 15,502 / 12,773**. Reference exposures remained matched. Measured total head-training wall time was 963.76 seconds, total CPU time 6,612.72 seconds, and peak RSS 3,947,844 KiB. Full evidence: [comparison](../../../artifacts/private/ticket44-full-weak-exposure/analysis-recovery-v1/comparison.json), [readable report](../../../artifacts/private/ticket44-full-weak-exposure/analysis-recovery-v1/comparison.md), and [decision](../../../artifacts/private/ticket44-full-weak-exposure/analysis-recovery-v1/decision.json). The final-freeze register records this terminal disposition. No final candidate promotion or release occurred.

The predeclared comparison to the frozen 100-step runs and image controls also completed in CPU job `7914` (`0:0`, 45 seconds). Gold-only long-minus-short AUC was **+0.110265**, paired interval **[+0.044802, +0.173599]**; report-supervised was **+0.132732**, interval **[+0.072635, +0.193342]**. Full-minus-zero-feature intervals were positive for both arms. Pair-weighted within-fold AUC was **0.650001 / 0.675163**, with **p = 0.00009999** in each arm's 10,000 within-fold whole-label-vector permutations. Constant predictions yielded AUC 0.5. This resolves the inadequate-duration hypothesis more strongly than the weak-supervision hypothesis: longer head training establishes measurable development image signal, while added reports have an inconclusive incremental effect. [Learning report](../../../artifacts/private/ticket44-full-weak-exposure/learning-controls-v1/report.md) and [controls, support and uncertainty](../../../artifacts/private/ticket44-full-weak-exposure/learning-controls-v1/evidence.json) preserve the evidence.
