# 45: Evaluate an attributed public-ensemble baseline

Type: task
Status: resolved
Labels: ready-for-agent
Parent: [RSNA leaderboard campaign](../spec.md)

## Purpose

Evaluate the already prepared public ensemble as a stronger scored baseline while preserving the existing 0.498 Elyra fallback. This separate baseline uses public pretrained inference assets; it does not establish fold-isolated performance of Ticket 39 or authorize final-candidate selection. The upstream reported public score is an observation about its author's run until this account obtains its own score.

## Requirements

- [x] Retain upstream source credit and Apache-2.0 notices; preserve recorded component provenance and unresolved release limitations.
- [x] Verify all required asset mounts and complete GPU inference on the three visible test studies, with strict schema, finite scores, exact order and no silent fallback.
- [x] Pin local request/source identity, downloaded remote code/metadata and actual canonical notebook handle. Preserve every failed observation and handle reconciliation.
- [x] Save and verify the exact full-workload version, retaining the tested model/inference logic and a bounded hidden-test execution limit.
- [x] Sequentially reserve all GPU work against the 72,000-second shared campaign, one active request at a time; verify free quota and paid scaling disabled. Record measured allocation separately from conservative charges.
- [x] Submit the exact verified notebook version through the code-competition API and record its actual submission reference, terminal status, score and any runtime failure. Do not submit the three-row visible CSV as if it covered the hidden test.
- [x] Preserve the original scored fallback and record this baseline's provisional disposition without claiming first place or selecting a final candidate.

## Execution evidence

2026-09-30: private internet-off GPU preflight version 1 of `jitheswar/rsna-public-ensemble-gpu-preflight-attributed` completed. All model arms ran, with `full_ensemble_complete=true`; code wall time 159.49 seconds on two T4 GPUs. Remote code cells match the frozen local notebook. Its full 3,600 GPU-second reservation is conservatively charged because code timing alone does not prove the VM allocation interval. Raw evidence remains unchanged.

Saved version 2 was accepted and is running under the same canonical handle. Its full-workload request is `artifacts/private/public-notebook-adoption/scored-baseline-v1/request.json`, SHA-256 `fe2a61c5d181eade46e6d959ddf45fcf59856d1d87d04e8416205ed69301fb8c`. The visible save reserves 3,600 GPU seconds; the subsequent scoring stage requires its own 57,600-second reservation after the save terminates and all source/output checks pass. Total worst-case charges fit the existing cap. The version uses a 900-second visible watchdog and a 28,000-second hidden watchdog inside an eight-hour reservation, with strict complete-fusion rejection and no model fitting.

Free quota and paid-scaling state are captured before saving. The typed SDK duration reports 108,000 allowed seconds; its raw JSON serializer incorrectly drops whole days, so resource decisions use normalized `total_seconds()` values. Paid scaling is false. No competition submission has yet been issued at this checkpoint.

Artifacts: `artifacts/private/public-notebook-adoption/gpu-inference-preflight-v1/` and `artifacts/private/public-notebook-adoption/scored-baseline-v1/`. Context: [public ensemble route](../../../docs/research/public-ensemble-route-2026-09-29.md).

### 2026-09-30 16:34 IST — saved version verified and scoring accepted

Version 2 completed. Its remote code exactly matches the local artifact, source metadata confirms current version 2 and private internet-off T4 execution, and its visible `submission.csv` is byte-identical to preflight version 1. Complete-fusion and output checks pass. Normalized account quota increased by 170.211 wall seconds; rounding up and charging both assigned GPUs gives **342 GPU seconds**. Before scoring, the shared campaign held **10,126 charged seconds**, zero reservations, and **61,874 remaining**.

Exactly one code-competition request was accepted for kernel version 2 and output `submission.csv`; it did not upload the local three-row CSV. Submission reference **56705536**, script version ID **354125287**, submitted **2026-09-30T11:03:59.940Z**, description “Attributed public ensemble baseline, exact GPU-validated version 2”. Initial authoritative status is **PENDING**, with no public or private score. The historical scored fallback is still submission **56515942**, public score **0.498**.

Scoring has a separate immutable request SHA-256 `c5e24ed163ab4e674ceee2671b955b7f0dac5cfc2fd7ef4e82d5e9dc04efb3c4` and **57,600 GPU seconds** reserved; worst-case charged plus reserved is **67,726 of 72,000**, leaving **4,274 seconds**. Do not release the reservation from intent or elapsed waiting: reconcile it after the exact scoring handle is terminal with allocation evidence, or retain a conservative charge when complete allocation cannot be established.

The independent user service `rsna-public-ensemble-scoring-56705536.service` polls only this accepted reference each minute, writes `scoring-status.json` and `scoring-observations.jsonl`, and exits on an authoritative terminal status. It cannot launch or retry work. Root continuation must inspect the exact submission and the monitor's live PID/status before treating a wait as verified. No score, rank, final candidate selection, or first-place claim is established while this submission remains pending.

### 2026-09-30 12:09 UTC — terminal-accounting path verified

`scripts/reconcile_public_ensemble_scoring.py` now reads only the exact accepted reference and verifies its source URL/version against the immutable request. If it remains pending, it records the observation and leaves the ledger unchanged. Once the authenticated status is COMPLETE or ERROR, it preserves an immutable normalized API response and invokes `record_kaggle_scoring_terminal_without_allocation` in the repository environment. This path charges the **entire 57,600 GPU-second reservation**, records actual VM allocation timing as **unknown**, and closes the exact one-attempt resource request. It never treats submission waiting time, code timing or free quota as an allocation interval; there is no increase in remaining budget from the conversion. Changed evidence, wrong submission/request identity, nonterminal status and repeat charges are rejected.

The first local helper invocation exposed a missing NumPy dependency in the isolated Kaggle SDK environment before any API call or ledger mutation. The operational repair keeps existing environments separate: SDK Python performs authenticated observation; repository Python applies validated accounting only after terminal evidence. No package installation or scientific configuration change was made. The repaired helper observed **56705536 PENDING** at **12:09:31 UTC** and reported `ledger_changed=false`. The original monitoring service remains live.

**25 focused tests passed** for campaign and terminal-accounting boundaries, including exact identity, both assigned GPUs, full unknown exposure, unchanged headroom, refusal of pending evidence, artifact integrity and no double charge. No scoring attempt was added or retried. A terminal COMPLETE status without a valid score or with an error description must still be treated as a delivery failure in the scientific disposition; accounting termination alone does not establish model success.


## Answer

Authenticated submission **56705536** is **COMPLETE**, public score **0.936**, private score unavailable, empty error description. The exact source URL still points to script version **354125287** of saved notebook version 2. A fresh authenticated observation at **2026-09-30 16:04 UTC** reconfirmed these facts and the already reconciled ledger. No duplicate scoring submission was issued.

Terminal evidence is `artifacts/private/public-notebook-adoption/scored-baseline-v1/terminal-scoring-evidence.json`; reconciliation observations and the unchanged accepted request remain beside it. With hidden VM allocation timing unavailable, the full **57,600 GPU-second** scoring reservation was conservatively charged. Waiting duration and visible code runtime were not substituted for hidden allocation evidence.

**Disposition: scored public baseline, provisional; final selection pending.** The earlier submission **56515942**, public score **0.498**, remains preserved. The new public score is evidence for this account's stronger baseline; it does not prove final private rank, first place, or independent local OOF improvement. Ticket 47's separately bounded visible inference can now proceed after exact resource and source gates.
