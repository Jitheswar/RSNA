# RSNA campaign handoff — October 2, 2026

Objective remains first place on **both public and final private leaderboards**.
Neither is achieved. Last authenticated official snapshot **01:08:08 UTC**:
**Elyra rank 2,005 /4,808, 0.936**; public leader **rhoskeri /0.961**.
Best exact submission **56705536 /script 354125287 /v2**; earlier **0.498**
fallback preserved. Private score/rank and final selections remain unavailable.

## Resume order

1. Read root instructions and [previous handoff](session-handoff-20261001.md)
   for older dispositions. Inspect Git, Slurm and persistent services. Standing
   owner authorization is **“approve anything as long as we win 1st place and
   fits kaggle comp dont ask me”**, zero external spend. Do not ask again.
2. Continue registered [Ticket 64](issues/64-test-long-full-backbone-report-adaptation.md).
   Root `artifacts/private/ticket64-full-backbone-comparison-v1/`, plan SHA-256
   `557230fec672f6db9252b1895a4d174652fe217c0388a8c9598bd749183aa985`.
   First full fold-0/stage-0 job **8032** and matched-control job **8033**
   accepted/running; actual allocations match. Their intents are consumed.
   Inspect exact logs and `controller-JOB/` terminal evidence; never resubmit
   either request because Slurm purges its record or startup is quiet.
3. Persistent **`rsna-ticket64-comparison-coordinator-v1.service`** advances
   only the 24 registered treatment stages. `coordinator-state.json` records
   exact schedule. It waits for successful control, then fills the free slot
   with fold 1; extends completed folds before starting fold 2. Every launch
   uses the frozen executable and original exact terminal/checkpoint/resource
   guards. Single file lock, at most two owner CPU jobs, no retry/requeue,
   no intermediate scoring, no conditional final fit or candidate promotion.
   Any failed/ambiguous stage stops the coordinator. Diagnose before a distinct
   recovery request; do not overwrite frozen fit source or launch intents.
4. Scientific budget stays **7,500 additional updates per arm/fold**, warm
   Ticket 55 report heads, original ImageNet encoder, audited original pools,
   two draws/source, head 1e-4/full encoder 1e-5, FP32/no augmentation/fixed
   normalization. **1,000 updates/stage**, eight stages/fold, last 500; global
   scientific config stays 7,500 through resumes. Full allocations 16 CPUs /
   64 GiB /5.5 h, watchdog 19,740 s; control all three folds in two hours.
   Final checkpoints alone produce held-out predictions. Do not change
   duration/rates/weights based on intermediate losses or runtime.
5. Before disposition, admit and execute the prepared independent original-MRI
   final prediction replay and registered comparison recomputation. The plan
   reserves **5,400 CPU-node seconds**, one 16-CPU/64-GiB bounded job; freeze
   verifier inputs/source/launcher before requesting it. Replay all 58 held-out
   source studies for both arms within **1e-6**, strict reload all six finals,
   reconstruct original targets/source exposures, verify treatment encoder
   updates/control encoder/fixed buffers, and recompute the five registered
   comparisons. The fit script's `analyze SHA` checks terminals/exposures and
   computes comparisons but explicitly does not satisfy original-source replay.
   No final comparison exists yet. Positive primary bootstrap lower bound and
   all positive fold directions plus image controls permit provisional review
   only; separate fixed-anchor comparison/offline delivery gates still apply.
6. Observe the separate TPU environment request **136722939 /v1**,
   `jitheswar/rsna-tpu-environment-admission-v1`, root
   `artifacts/private/ticket60-tpu-environment-v1/`. Last observed **QUEUED**
   at **06:33:21 UTC**; actual hardware remains unverified. One 600-second
   server/540-second code watchdog, private/offline, GPU off, no attachments,
   training or spend. Existing intent is consumed: never push again. Observer
   `rsna-ticket60-tpu-environment-v1-observer.service` retains terminal.
   On terminal use Kaggle SDK Python with `scripts/verify_ticket60_tpu_environment.py`;
   verify exact v1 source, actual TPU XLA/global devices/matmul and quota;
   conservatively charge 600 s if unresolved. Passing enables separate real-MRI
   pilot admission only. No GPU launch while this accelerator request queues.
7. Preserve completed [Ticket 59](issues/59-diagnose-hidden-frontier-execution-failure.md)
   accessible input probes. All original DINO input jobs **7995 /7996** pass
   4,407 studies/819,078 DICOMs with zero decode/fallback events; independent
   integrity **8012** hashes every original payload and archive CRC, exact
   terminal/source passes. Full CoAt **8014** now completes `0:0`, **8,770
   CPU-node seconds**, all 4,407 studies, 52–94 windows, zero preparation
   failures/fallbacks/warnings. Independent summary verifies producer hash
   records, without pixel replay. One consumed attempt each; no relaunch.
   The actual hidden **56739912 /354194017 /v3** failure remains unreproduced,
   no matching traceback/cause/repair, no scoring retry or promotion. Preserve
   its full unknown **57,600 GPU-second** charge.
8. Run relevant checks after new changes, update evidence/tracker/freeze
   register, commit/push private `origin/master`, and verify equal heads.
   Complete the goal only after both actual first-place ranks are proven.

## Completed prerequisites

- Full-backbone CPU pilot **8016 /COMPLETED /0:0**, **127 CPU-node seconds**,
  four training reports, 232 valid/512 padded slices; native initial and
  fresh-checkpoint logits exact. Every encoder stage updates, fixed buffers,
  median **5.955313 s/update**, peak **17.88 GB**. Source/terminal admitted.
  [Evidence](../../docs/evidence/ticket-61-full-backbone-cpu-cost.md).
- Exact pixels **8017 /COMPLETED /0:0**, **2,080 CPU-node seconds**, 4,228
  studies/187,272 valid slices, **37,586,239,488 bytes**. Every pixel/mask/
  descriptor/source mapping replays fresh native packing exactly. Verified
  complete read/reconstruction **189.317 s**, native replay **704.605 s**.
  [Ticket 62](issues/62-materialize-exact-mri-pixel-cohort.md) resolved;
  [evidence](../../docs/evidence/ticket-62-exact-mri-pixel-cohort.md).
- Actual shared training cost **8018 /COMPLETED /0:0**, **2,170 CPU-node
  seconds**, plan `40f5268a019a9d247cb3ced8d0d7a62c419bd7b1e544a1370af7f884d65f7e39`.
  Complete pixel/admitted all-fold sources, fresh-process exact staged versus
  continuous model/Adam/scheduler/progress/sampler replay, changed encoder and
  fixed buffers independently pass at **06:06:37 UTC**. Max startup **372.921
  s**, diagnostic interval **8.745448 s**, checkpoint **135,071,169 bytes**,
  peak **16,871,268,352 bytes**. One closed consumed request, no AUC.
  [Ticket 63](issues/63-measure-shared-full-backbone-training-delivery-cost.md)
  resolved; [evidence](../../docs/evidence/ticket-63-shared-full-backbone-training-cost.md).
- Planned-pause checkpoint regression first reproduced stale step-0 after
  step-2 pause, then passed with exact durable publication and continuous
  replay. **27 relevant lifecycle/training tests**, **five resource-admission
  tests**, **seven coordinator safety tests** pass. Typing **84 source files**.
  Previous complete pixel/model/native adapter suite: **57 passed**.

## Resources and delivery

Fresh frozen admission includes **537,988,466,127 checkpoint bytes**, all six
fits plus conditional final fit/history/temp files, under fresh project/external
model two-terabyte storage accounting. Maximum route CPU-node reserve **653,400
s /181.5 h**; conservative calendar **516,600 s /143.5 h** plus one review day
before October 11. Each job <=16 CPUs/64 GiB/six hours, nonexclusive; at most
**two active/pending owner CPU jobs**. Independent replay **5,400 s** and offline
delivery **7,200 s** reserved. Conditional additional full fit and GPU delivery
need separate exact recipes/decisions; they are not authorized by Ticket 64.

GPU ledger unchanged: **129,637 charged /136,800 cap /zero reserved /7,163
remaining seconds**. Preserve unknown charges. TPU reserves **600 seconds**,
charged zero/hardware unverified. No external spend. Old continuation
coordinator remains stopped; Ticket 64 is the only active fit coordinator.

Science freeze October 11, internal delivery before October 15; recorded
competition final deadline October 22 at 23:59 UTC. Five submissions/day,
two final selections, nine-hour offline notebook bound. Refresh official
rules/state before delivery. GitHub remains private:
**https://github.com/Jitheswar/RSNA**.

Medical encoder [primary-source research](../../docs/research/medical-image-encoder-options-2026-10-02.md)
finds conditional ungated MIT BiomedCLIP 224-pixel vision-only 512-D admission.
MRI evidence concerns article figures, with no knee gain established. No weights
were downloaded. Keep the registered full-backbone comparison; this note alone
does not admit a new encoder run. Other variants have unresolved gates.


Independent verifier code is now `scripts/validate_ticket64_full_backbone_comparison.py`.
Its `prepare` rejects unfinished fits and creates no request; six admission/
prediction-shape checks pass. After all six fits, run the frozen fit `analyze SHA`
under its project/source environment, then verifier `prepare`, inspect frozen
request/launcher, `launch REPLAY_SHA`, and independently reconcile actual replay
terminal before disposition. No verifier resource request exists yet. It uses
the admitted original Ticket 62 archive reader, strict final checkpoints, native
parent/full/control/zero-feature predictions, original source/mask exposures,
exact Adam duration/rates, all encoder scopes and independent recomputation of
all five stored/native comparisons. Real final replay remains pending.

At **06:33 UTC** both jobs have begun successful optimizer updates with finite
logged losses: full fold 0 **29** and control fold 0 **1,425**. These are progress
observations only, without held-out scoring or scientific selection. Runtime
source is frozen; preserve exact remaining stages and watch their terminal.
