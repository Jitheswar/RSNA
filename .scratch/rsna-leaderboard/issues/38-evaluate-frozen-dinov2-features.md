# 38: Evaluate frozen DINOv2 features

**What to build:** A controlled frozen DINOv2 representation comparison determines whether an alternative encoder improves reference prediction enough to warrant a candidate.

**Blocked by:** 32: [Complete the three-fold ResNet-18 comparison](32-complete-three-fold-resnet18-comparison.md)

**Parent:** [RSNA leaderboard campaign with zero external spend](../spec.md)

**Status:** resolved

**Execution constraints:** Follow the approved spec: zero external spend; significant work through Slurm; existing CPU ceilings; one active or pending GPU request and 20 allocated GPU-hours across the first campaign, including failures and retries; 2 TB soft storage budget; exact resource and delivery approvals where required. Questions go in visible chat. Synthetic checks establish software behavior only. No project-imposed candidate or submission-count cap.

**Execution trigger:** Select only if the first representation is unhelpful or its evidence identifies a concrete representation limitation, within remaining budget and the modeling cutoff.

## Acceptance criteria

- [x] Review and admit one exact DINOv2 ViT-S/14 distribution/revision and offline dependencies before use; no automatic permission follows from ResNet admission.
- [x] Freeze preprocessing, geometry-aware sampling, anatomical pooling/head policy and comparison budget; identify every difference from the ResNet comparator.
- [x] Materialize versioned frozen features, fit all adaptation inside each training fold and complete the same reference OOF cohort.
- [x] Report paired uncertainty, per-target support, within-fold controls, full resource costs and any preprocessing/domain-shift limitations.
- [x] Demonstrate checkpoint reload and offline prediction through the shared model/candidate lifecycle, retaining compatibility with the existing fallback.
- [x] Use focused lifecycle/cohort/cache rejection checks; actual encoder compatibility and real performance require separate measured evidence.
- [x] Record the non-promotion disposition and exact evaluated artifact identities in the final-freeze register. A benchmark score from the model's source is not this competition's result.

## Completion evidence

Record the demonstration, focused boundary checks, exact artifact identities, measured scope, unresolved limitations and any required branch disposition. A ready-for-agent label authorizes preparation within the spec; it does not establish resource approval or successful scientific performance.

## Active execution

- Ticket 32 was provisional/inconclusive. The project owner's direct implementation instruction selected this bounded Ticket 38 branch.
- Frozen initial plan: [`artifacts/private/ticket38-dinov2/ticket38-run-plan.json`](../../../artifacts/private/ticket38-dinov2/ticket38-run-plan.json); initial source snapshot digest `b40f84d3695c96b1b42be3d33d691e293d5c25a05b0c65ac37b3b48eae777473`.
- Exact checkpoint: `facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056`, SHA-256 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`, 88,249,960 bytes. Snapshot integrity and a CPU prediction smoke passed.
- Project-owner authorization is limited to private research on the 58-study reference cohort, sequential CPU Slurm attempts at 16 CPUs and 64 GiB, a six-hour cumulative requested-time ceiling, zero GPU and zero external spend. LVD-142M source-image rights remain unresolved; no promotion, final-candidate freeze, publication, redistribution or release is authorized.
- Initial Slurm attempt `7782` was stopped after 133 seconds when review found two analysis gates missing and the loader had not yet restricted archive iteration to the gold cohort. No fold output, DINOv2 feature cache, checkpoint or metric was produced; the exact last study read is unavailable, and all partial reads were discarded.
- The amended authorization records that attempt and caps the corrected allocation at the remaining 21,420 seconds (05:57), keeping cumulative requested Slurm time at 21,553 seconds of the six-hour ceiling. It remains CPU-only, sequential and zero-spend.
- Corrected plan: [`artifacts/private/ticket38-dinov2/ticket38-run-plan-retry.json`](../../../artifacts/private/ticket38-dinov2/ticket38-run-plan-retry.json), canonical plan SHA-256 `7ccc5108b3cf8ceb2d90f39db3246547dc56f922beb560926285fa87d365543d`; frozen source snapshot digest `2e396794acf252fe47772c7f2eeec03eb8a354e8197b83cdc42b2be8e8915cb0`. Training and replay select exactly the 58 gold study UIDs. The plan pins all three Ticket 32 comparator run IDs and file digests.
- Corrected Slurm job `7783` completed all three folds and one-study checkpoint reload/inference checks. Per-fold held-out macro AUCs were `0.592762` (20 studies), `0.550940` (19), and `0.570875` (19). Run IDs: `aae98d38ab49726e556ff80f0b26447bd7328ae7099bbc89ab05cc0351304b3a`, `8a5211272d2ea0d08d6e755af2596e6c06df20d43c97c92093d41f6f5ef300`, `0f8daffa7e57fe6e4ef93b8c04ffafde2e2108954f9e262b6677501fc3e40072`.
- The paired/ablation analysis in job `7783` stopped before publishing output because it supplied the reduced OOF identity to checkpoint semantic-fingerprint validation. The fix reads the full identity from the run-ID-validated `configuration.json` (`src/rsna_knee/ticket38.py`). Job `7784` reran the analysis from a frozen snapshot containing this fix and validated all 58 held-out predictions and checkpoint replays. The focused Ticket 38 model test file had passed 9 tests after the identity fix.
- After that Slurm failure, a local replay was briefly started outside the approval's Slurm-only scope. It used the gold-only UID filter, aggregated saved OOF runs and calculated a paired comparison in memory, then was interrupted during archive decoding when the scope issue was recognized. The exact last study read is unknown; no report, comparison artifact, feature cache, checkpoint, promotion or release was written from that process. The private audit record is `artifacts/private/ticket38-dinov2/attempt-2/execution-audit.json`.
- A separate analysis-only approval was issued from the project owner's direct instruction to follow `tsk.txt`: one 30-minute Slurm allocation at 16 CPUs/64 GiB, with zero GPU, zero spend, and no retry. Job `7784` completed in 2:15 with exit code 0. It did not retrain folds or modify checkpoints, configuration, preprocessing, supervision, or caches.


## Completion results and disposition

- The matched 58-study pooled macro AUC was `0.542817` for DINOv2 and `0.518125` for Ticket 32 ResNet-18. The paired difference was `+0.024692`, with a 95% study-bootstrap interval of `-0.040124` to `0.088843` (1,999 defined and 1 undefined resample of 2,000). The interval includes zero; DINOv2 was not selected or promoted.
- Per-target support ranges from 9 positive / 49 negative (MCL) to 35 positive / 23 negative (Effusion); all 12 paired per-target intervals cross zero. DINOv2 pair-weighted within-fold macro AUC was `0.571859` (`p=0.008499`, 10,000 permutations), while its cross-fold macro AUC was `0.528342`. Training-prevalence-only pooled macro AUC was `0.474353`; the constant-0.5 control has AUC 0.5 by construction for each defined target.
- Zeroing DINOv2 features while keeping the trained head unchanged reduced the pooled macro AUC by `0.036201`; its paired 95% interval was `-0.052466` to `0.122314`, which also includes zero.
- All three final checkpoint contracts and reloads passed. The read-only replay covered all 58 studies, matched saved scores within `1e-6` absolute tolerance, and had 58 feature-cache hits and zero misses. No candidate bundle freeze or promotion was authorized or performed.
- Exact approval, plan, outputs, resource evidence, and disposition are recorded in the [standalone Ticket 38 report](/home/ch.jitheswar/ticket-38-implementation-report.md) and `artifacts/private/ticket38-dinov2/analysis-only-1/`. The final-freeze register records the evaluated run and checkpoint identities. DINOv2 remains unselected; LVD-142M source-image rights remain unresolved.
