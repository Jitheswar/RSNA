# Ticket 56 — CPU report-adaptation timing

The real-MRI pilot passed: caching the unfitted encoder prefix reduced median
update time from **1.315271 s** to **0.508122 s**, approximately **2.59x** faster,
while native and cached execution produced **identical logits and final weights**.
This establishes a feasible route to plan, without establishing a predictive gain.

## Actual bounded execution

Slurm job **7982** completed `0:0` on **KLEF-C-NODE1**, one nonexclusive node,
**16 CPUs / 64 GiB**, no GPU, requeue or restarts. Start **06:39:24 UTC**, end
**06:39:52 UTC**: **28 allocated CPU-node seconds**; code elapsed **24.166 s**.
Peak process RSS **1,861,849,088 bytes**, external spend **$0**. Terminal controller
evidence was saved before purge.

Four fold-0 training reports were selected by valid-slice count, without labels,
at the registered 10th/40th/70th/100th order statistics. Their actual counts are
**32 / 40 / 48 / 112**, **232 slices** total, with **14 admitted supervised entries**
per four-study update. Gold and fold-0 held-out studies are excluded. Source
checksums, read-only series-cache admission, packing masks and anatomical
descriptors agree with the existing 224px whole-field representation.

The existing ImageNet encoder and independently verified long fold-0 report head
initialize both modes identically. Only ResNet-18 layer4 and MLP32 adapt, using
Adam, layer4 learning rate **1e-5**, head rate **1e-4**, and admitted masked report
BCE. Each mode performs **one warmup plus four timed updates**. All gradients and
losses are finite; both parameter groups change. Earlier encoder weights and all
normalization buffers remain unchanged. Native/cached final-state error is **0.0**
under the unchanged **1e-6** gate. Fitted probe weights are discarded; no candidate
checkpoint or AUC was produced.

The deterministic prefix is **(232,256,14,14)** FP32, extracted in **2.784796 s**.
A temporary **46,563,456-byte** activation file writes in **0.079013 s**, reads
in **0.062182 s**, and matches its original array hash after readback. The temporary
file was removed. The whole cohort was not extracted by this pilot.

## Cost scenario and limits

The registered scenario uses **1,500 adaptation updates per fold**, two reference
and two report draws per update, actual fold-specific valid-slice counts and a
**2x allowance**. This is a cost scenario, not an automatically selected training
budget. Projected training per fold is **1,228.987 / 1,177.521 / 1,166.693 s**.
Projected frozen-prefix extraction is **4,495.813 s**, simple sequential cache I/O
**227.948 s**; total **8,296.962 s / 2.305 CPU-node hours**. It passes the prospective
six-hour-per-fold and 24-node-hour cost ceilings.

The **187,272-slice** cohort would require **37,586,239,488 bytes** of FP32 activation
payload. A fresh apparent-size measurement covers **533,058,112,898 bytes** inside
RSNA and **16,397,455,028 bytes** in the admitted external Qwen snapshot. Adding
activation payload yields **587,041,807,414 bytes**, below the **2 TB** soft ceiling;
the filesystem has about **23.5 TB** available. Full preparation must refresh
storage and allow metadata/checkpoints and other tracked campaign artifacts.

These projections come from **four reports and repeated warm-cache updates**.
Linear scaling, full-cohort extraction and random-read throughput remain unmeasured.
Scheduling, final evaluation, repeated random-read I/O and checkpoint/delivery
overhead are omitted. The cost gate permits separate full-route planning; it does
not authorize claiming complete training, a clean generalization gain or first place.
No GPU accounting, competition submission or final selection changed.

## Validation and next action

Exact plan SHA-256:
`b650a094a5539974af86e4a87a29746ac8077616c9687b16ad9d53440728b163`.
Independent preparation verified **95 frozen files**, compiled **81 Python files**,
checked launcher syntax and training/report exclusions. Post-run validation
rehashed those files, inspected the exact successful allocation and independently
recomputed all cost and storage projections. Actual parity is established by the
inspected immutable runner's real-update checks; discarded probe weights were
not separately replayed.

Private root: `artifacts/private/ticket56-cpu-layer4-pilot/`, preserving plan, source
snapshot, accepted launch, controller observations, timings and `validation.json`.
[Runner](../../scripts/ticket56_cpu_layer4_pilot.py),
[standing-authorized concrete resource decision](../../.scratch/rsna-leaderboard/resource-approvals/ticket56-cpu-layer4-pilot-20261001.json),
[resolved issue](../../.scratch/rsna-leaderboard/issues/56-measure-report-supervised-encoder-adaptation-cost.md).

Next, separately freeze full deterministic prefix extraction and verify actual
cohort-wide hashes, masks and throughput before adapting three fold-specific
encoders. A future training choice must be registered before its held-out predictions.
The pending D4 scoring request remains separate; its score and final private ranking
are not established by this CPU result.
