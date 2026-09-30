# 31: Run one ResNet-18 fold through the complete lifecycle

**What to build:** One admitted reference fold trains a head on frozen ResNet-18 features, predicts held-out studies, reloads its checkpoint and produces an offline candidate output through the shared lifecycle.

**Blocked by:** 28: [Preserve the baseline through shared model construction](28-preserve-baseline-through-shared-model-construction.md); 29: [Diagnose real-image learning within campaign limits](29-diagnose-real-image-learning-within-campaign-limits.md)

**Parent:** [RSNA leaderboard campaign with zero external spend](../spec.md)

**Status:** resolved

**Disposition:** completed — one-fold experiment closed; non-promotion.

**Execution constraints:** Follow the approved spec: zero external spend; significant work through Slurm; existing CPU ceilings; one active or pending GPU request and 20 allocated GPU-hours across the first campaign, including failures and retries; 2 TB soft storage budget; exact resource and delivery approvals where required. Questions go in visible chat. Synthetic checks establish software behavior only. No project-imposed candidate or submission-count cap.

## Acceptance criteria

- [x] Before downloading or using the proposed ResNet-18 ImageNet weights, verify current applicable competition rules, exact source/revision or artifact identity, full digest, rights and the required specific resource approval. Inaccessible eligibility remains an explicit gate. The approval is limited to this experiment and does not resolve release rights.
- [x] Resolve the learning diagnosis before expansion. Freeze the fold, preprocessing, geometry-aware sampling, seed, head training and checkpoint policy before looking at held-out outcomes.
- [x] Preserve the whole knee field with documented resizing/padding and grayscale normalization; the new 224-pixel representation has a distinct identity from the existing 128-pixel cache.
- [x] Extract frozen slice features and preserve anatomical plane/series and missingness information in pooling; retain a simple pooling comparator.
- [x] Fit every learned transform and head only on the training fold. The external frozen encoder may predict held-out features without adaptation.
- [x] Cache features with source, encoder, preprocessing, sampling, representation and dtype identities; stale or changed identities are rejected or rematerialized rather than reused silently. The partition hit counter has a reporting bug documented below; total hits and cache identity are valid.
- [x] Produce one real fold's complete held-out predictions, learning and resource evidence, checkpoint reload and offline study output; explicitly label the result as one fold, not full OOF.
- [x] Prefer the bounded CPU path for the first slice. A GPU alternative requires the existing exact resource/preflight gate and the shared campaign ledger; this ticket does not depend on a later GPU experiment.
- [x] Register this selected experiment and every scientific rerun with the final-freeze register. Record its outcome without asserting a competition gain from lifecycle success.
- [x] Obtain approval for the exact pending resource record before acquiring or using the checkpoint. Any resulting eligibility is limited to the one reference-fold experiment described there; the existing exact-resource criterion and V13 remain fully operative, including independent full-digest verification against the acquired bytes before model loading.
- [x] Keep this artifact out of final candidate freeze, winner publication, redistribution, and release until checkpoint/derivative redistribution rights and the competition winner-release rules are separately resolved and separately approved. Experimental-use approval does not satisfy this gate; release remains blocked.

## Completion evidence

Record the demonstration, focused boundary checks, exact artifact identities, measured scope, unresolved limitations and any required branch disposition. A ready-for-agent label authorizes preparation within the spec; it does not establish resource approval or successful scientific performance.

## Comments

### 2026-09-25 — Synthetic preparation complete; weight rights remain unresolved

Added `tests/test_ticket31_synthetic_lifecycle.py` because the existing suite did not cover the Ticket 31 ResNet-18 lifecycle. It trains fold 0 for one optimizer step using nine generated DICOM studies (six training, three held-out) and the random fixture encoder, reloads the final checkpoint, replays all held-out studies, and verifies the one-study offline CSV/evidence. Reloaded scores matched the training held-out predictions within `1e-6`; the evidence correctly labels the output as one fold, not full OOF.

Focused result: `./.venv/bin/python -m pytest -q tests/test_ticket31_synthetic_lifecycle.py` — **1 passed in 5.46s**. This establishes synthetic software behavior only. It is not real-fold, resource, generalization, or performance evidence. The run used CPU only, no Slurm or GPU, no external spend, no competition data, and no pretrained weights. Pytest retained synthetic artifacts under `/tmp/pytest-of-ch.jitheswar/pytest-567/test_ticket31_synthetic_fold_r0`.

The current Kaggle overview allows freely public pretrained models, but current full rules could not be extracted in this review. The September 11 rule capture records both a winner-license wording conflict and an exception for incompatible pretrained models without resolving whether this exact ImageNet checkpoint qualifies. TorchVision warns that terms can derive from training data; ImageNet's agreement restricts database use to non-commercial research and education but does not address these weights or their publication. The rights remain ambiguous, so no artifact-specific approval record was prepared and no download/use occurred. A written official host/sponsor clarification is required before the real fold.

See [the detailed report](../../../docs/research/RSNA-ticket-31-report.md) and [the requested working report](</home/ch.jitheswar/RSNA-ticket-31-report.md>). Ticket 31 remains open; the real fold, resource record, and final-freeze registration are outstanding.

### 2026-09-25 — Experimental-use and winner-release gates separated; owner decision pending

Prepared the exact pending owner-approval record at [`../resource-approvals/ticket31-resnet18-imagenet1k-v1.json`](../resource-approvals/ticket31-resnet18-imagenet1k-v1.json). It proposes use of `torchvision.ResNet18_Weights.IMAGENET1K_V1` only for one pre-frozen CPU reference fold, at zero external cost, without downloading ImageNet images, without held-out adaptation, and with local/offline processing after checkpoint acquisition. The record includes the exact TorchVision source/version, URL, expected byte count and full SHA-256, provenance and its known license/training-recipe gaps. The expected digest is not an observed digest: after owner approval, acquisition must independently verify and record the complete digest and byte count before the weights are loaded.

The proposal is based on the official competition overview and host interpretation already reviewed in the report. The September 25 comment's request for further host/sponsor clarification before the fold is superseded for this bounded experiment only: the already-documented written host reply #3517696 is the competition-policy interpretation used in this proposal, and no artifact-specific host ruling is claimed. It is a competition/project decision for this experiment only, not a third-party rights grant or legal guarantee. No ImageNet rights question is represented as resolved. Even if approved, these weights remain ineligible for final candidate freeze, winner publication, redistribution, or release until the separate source-rights and winner-release question is resolved and separately approved. The existing exact-resource acceptance criterion and V13 are unchanged; the two new unchecked criteria explicitly retain both gates.

Current disposition: **pending owner approval of the exact JSON record**. No weights or ImageNet images have been downloaded, the approval record is not yet executable, and the real fold has not run. Ticket 31 remains open and `ready-for-agent` does not itself approve this resource.

### 2026-09-25 — Owner approved bounded experiment; fold preregistered

TSK6 explicitly approves the exact resource record for one Ticket 31 reference-fold CPU experiment only. Updated the record with `decision: approved`, `rights_status: eligible` for this scope, the owner's approval identity and timestamp. The checkpoint was acquired from the recorded URL; the acquisition helper matched the expected values, then independent `sha256sum` and `stat` checks verified SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` and 46,830,571 bytes. Those observations were recorded before model loading. No ImageNet image was downloaded.

Before inspecting held-out outcomes, froze fold 0 of the corrected v2 split, the implementation's Ticket 31 defaults (100 Adam steps, learning rate 0.001, batch size 2, seeds 20260911/20260912), 224-pixel whole-field preprocessing, frozen encoder and anatomical pooling. The private plan and Slurm job script are under `artifacts/private/ticket31-resnet18/`; the planned allocation is one nonexclusive `cluster1-cpu` node, 16 CPUs, 64 GB and six hours. A final-freeze register entry records the run as not promoted and preserves the separate release block.

The fold run is now the only pending step. Its completion evidence will be added here and to both reports before any candidate or release action. No final candidate freeze, winner publication, redistribution or release is authorized.

### 2026-09-25 — First staging attempt stopped before head training

Slurm job `7167` ran on `cluster1-cpu` for 9m13s and exited `1:0` with `refusing to overwrite existing experiment directory`. The feature-cache path was nested under the experiment output path, so feature materialization created the directory before the training runner's fresh-output guard. The run materialized 58 identity-scoped frozen-feature entries, then stopped before head optimization; no run metrics or fold checkpoint were produced. The recorded peak RSS was 27,453,616,128 bytes (about 25.6 GiB of 64 GiB); the logged path-layout error caused the stop, not memory exhaustion. The cached features are retained under the same frozen feature identity.

The scientific plan remains unchanged. The corrected invocation will write the experiment into a separate output directory and reuse the verified matching feature cache. It remains the same fold 0, same training config, same one-fold authorization, and same CPU-only scope; this path correction does not add a fold or experiment.

### 2026-09-25 — Corrected fold completed; candidate and release gates remain closed

Slurm job `7168` completed with exit `0:0` after 7m22s on `cluster1-cpu` (`KLEF-C-NODE2`), using 16 CPUs and 64 GiB. Preflight passed. The frozen fold 0 run completed 100/100 Adam steps with no skipped batches, using 38 reference-training studies and 20 held-out studies. No GPU was requested or used; external spend was zero. The encoder remained frozen and held-out data was used only for prediction/evaluation.

The 20 held-out predictions define all 12 target AUCs; macro AUC is `0.4976806547210959`. Training masked loss moved from `0.6966536045074463` to `0.5687010288238525`. This is one fold only and is not evidence of full OOF performance or a competition gain. The final checkpoint is 45,615,081 bytes with SHA-256 `17220c47d15e786e870eb0d46aa088314973b7929094ede697585c148211e872`.

Offline lifecycle evidence ties the checkpoint to a one-row held-out-study CSV (12 finite scores; 5 valid series; 0 rejected; no fallback; no runtime network download). The output is explicitly marked `one-held-out-study-from-one-fold;not-full-oof`. The run, metrics, checkpoint, submission, and evidence are under `artifacts/private/ticket31-resnet18/fold-0-complete/`.

Cache evidence has 58 total hits and 0 materializations for 38 training plus 20 held-out studies. The JSON's `partitions.training.cache_hits` incorrectly says 58 because the code computes that field after adding held-out hits; the correct partition split is 38 training / 20 held-out. This is a reporting-counter bug and does not change cached features, fitting, predictions, or metrics. The final-freeze register records the correction and result. The pretrained checkpoint remains excluded from final freeze, publication, redistribution, and release pending a separate rights/rules decision.


### 2026-09-25T12:32:32Z — TSK7: completed and closed as a one-fold non-promotion

Ticket 31 is complete as the one-fold experiment it specified. Fold 0 produced macro AUC `0.4976806547210959` on 20 held-out studies. The corrected v2 mechanics baseline on the same fold was `0.4922830053252847`, a single-fold delta of `+0.0053976493958112215`. This is not a promotion result. It is also not enough evidence to reject the ResNet-18 route: one fold does not establish the predeclared pooled OOF estimate.

The existing gate in [Ticket 05](05-choose-validation-policy.md) says to complete planned folds before interpreting an improvement; it also says a tiny positive delta alone is insufficient and establishes no universal numeric threshold. [Ticket 06](06-choose-first-learning-campaign.md) explicitly says a single-fold positive is insufficient for promotion and calls for comparable OOF evidence. Ticket 32 is the predeclared corrected three-fold comparison. Therefore TSK7's condition for proceeding is met by the gate itself, not by Ticket 31 execution success. The conditional instruction authorizes a separate, CPU-only Ticket 32 scope, recorded at [`../resource-approvals/ticket32-resnet18-imagenet1k-v1.json`](../resource-approvals/ticket32-resnet18-imagenet1k-v1.json); the Ticket 31 approval remains one-fold-only and unchanged.

Ticket 31's release disposition remains blocked: no final-candidate freeze, publication, redistribution or release is authorized. The triage `Status: ready-for-agent` remains the canonical role label; this separate `Disposition` records that the ticket's experiment is completed.


### 2026-09-25T13:11:19Z — TSK7 follow-through recorded

Ticket 31 remains closed as a one-fold non-promotion. The predeclared gate required full planned-fold evidence before interpreting an improvement, so Ticket 32 ran under its separate three-fold CPU approval. Ticket 32 completed with a provisional/inconclusive result; it does not alter Ticket 31's one-fold scope, non-promotion disposition, or separate release block. See the TSK7 section in the report and the completed [Ticket 32 issue](32-complete-three-fold-resnet18-comparison.md).

### 2026-09-25 — Closed by owner request

The owner requested closure after completion of the one-fold experiment. The result remains a non-promotion; the separate final-candidate, publication, redistribution, and release gate remains blocked.
