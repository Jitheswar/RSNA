# Ticket 23 delivery strategy evidence

Recorded 2026-09-23. This public-safe record contains aggregate reference
counts only; it contains no study identifiers, source rows, reports, or raw
paths.

## Decision

Keep the existing three-fold gold-only candidate as the delivery path for
Ticket 24 to consider. Do not run an all-data refit for this ticket. The
verified candidate already has a small package and substantial runtime margin,
while current development OOF does not show measurable image signal. No
validated result supports a model replacement. This is not a claim that the
existing model is high-quality or outperforms an unmeasured all-data refit;
owner selection remains separate.

## Comparison

| Dimension | Existing three-fold candidate | Proposed one-model all-data refit |
| --- | --- | --- |
| Selection evidence | Gold-only fold OOF pooled macro AUC 0.464313; study-bootstrap 95% interval 0.404–0.524. The fold-prevalence-only score was 0.474353, and the predeclared within-fold permutation test gave p=0.918. | No refit was trained, so its exact weights have no OOF score. The existing OOF describes model selection and cannot be attached to refitted weights. |
| Training budget | Three fold models, each with 100 successful optimizer updates at batch size 2: 300 updates total. | Proposed same gold-only baseline configuration and final-checkpoint policy, one model, 100 successful updates. This matches the existing per-model budget, not the ensemble's 300 aggregate updates. |
| Admissible supervision | Each OOF fold used trusted reference labels outside that fold. | All 58 trusted reference-label studies. No report-derived labels, pseudo-labels, distillation targets, or external learned resources. |
| Inference and memory | Three-model mean-probability inference. Local 158-study run: 604.504 s, 2,071,003,136-byte peak RSS, with 4,973.767 s projected at 1,300 studies. The approved Kaggle run: 2,932.069 s, 26,716,372,992-byte peak RAM, with 9,462.947 s projected at 1,300 studies. | No same-workload runtime or memory measurement exists. A single forward path may be cheaper, but no speedup or memory reduction is claimed. |
| Checkpoints and package | Three checkpoints total 314,944 bytes. The local model-plus-bundle measurement is 953,991 bytes; the Kaggle execution bundle measurement is 940,350 bytes. | No package was built. One comparable checkpoint would be about 105 KB based on the existing checkpoint files, but exact refit and package sizes remain unmeasured. |
| Packaging | Three fold checkpoint files and probability-mean aggregation are already packaged, tested, and Kaggle-executed. | Would package one all-data checkpoint and use one model. This removes two checkpoint members and the ensemble aggregation, but has no validated deployment artifact. |

The Ticket 10 OOF predictions come from the model that held each study out.
They do not directly evaluate the deployed three-model average, because that
average includes other fold models that saw the study during training. This
limits the quality comparison between the exact deployed ensemble and any
all-data refit.

The runtime records are from different environments and measurement paths;
they are kept separate. The all-data row is a bounded proposal, not measured
cost evidence. No CPU refit or GPU job was submitted.

## Why no refit was run

The development results do not supply a performance reason to refit. Ticket
10's fold-prevalence-only comparator exceeds the fold-specific gold-only OOF
macro, and the within-fold permutation test found no detectable image signal.
These OOF predictions do not directly evaluate the deployed three-model
average, so they are evidence against spending on an unearned model change,
not a quality estimate of the exact deployed ensemble.
The wider encoder, additive report supervision, and distilled student also
produced inconclusive development-CV deltas; none was adopted. Ticket 20 did
not run a combined arm, and Ticket 21 admitted no pseudo-labels. The existing
candidate is already below the runtime and package limits on the measured
representative workload. A refit merely to create a single checkpoint would
not resolve those quality or validation gaps.

If a single-model refit is reconsidered, the bounded proposal is one CPU job
using the fixed 8/16/32 baseline, batch size 2, Adam at learning rate 0.001
with the existing constant schedule, 100 successful optimizer updates, and
the final checkpoint. It uses only the 58 trusted gold-reference studies. The
job must remain on `cluster1-cpu`, one non-exclusive node, no more than 16 CPUs,
64 GB requested memory, six hours, and actual memory bounded. Its exact
training duration, peak memory, checkpoint size, runtime, and OOF score would
be measured only if it is later executed. The current three-fold candidate is
preserved unchanged.

## All-reference fallback priors

These Laplace-smoothed values use every trusted reference label in the
canonical version-2 fold manifest (`8460234d…f287fb8`), alpha=1, and all 58
complete gold studies. They are emergency fallback scores for a future
all-data candidate; they are not model predictions or OOF estimates.

| Target | Positive | Negative | Known | Smoothed prior |
| --- | ---: | ---: | ---: | ---: |
| ACL | 24 | 34 | 58 | 0.416667 |
| MCL | 9 | 49 | 58 | 0.166667 |
| Medial Meniscus | 26 | 32 | 58 | 0.450000 |
| Lateral Meniscus | 23 | 35 | 58 | 0.400000 |
| Medial OA | 15 | 43 | 58 | 0.266667 |
| Lateral OA | 11 | 47 | 58 | 0.200000 |
| PF OA | 21 | 37 | 58 | 0.366667 |
| Effusion | 35 | 23 | 58 | 0.600000 |
| Synovitis | 27 | 31 | 58 | 0.466667 |
| Baker's | 12 | 46 | 58 | 0.216667 |
| Contusion | 19 | 39 | 58 | 0.333333 |
| Fracture | 18 | 40 | 58 | 0.316667 |

These values do not mutate the frozen candidate. Its existing fallback remains
the configured midpoint prior; the representative executions used no
full-study fallback. A future candidate would need to embed and verify this
prior as part of its own immutable configuration.

## Limits and handoff

- The OOF set is development model-selection evidence, not an untouched test;
  patient-level separation is unavailable.
- Exact OOF performance for the deployed all-fold average and all-data refit
  is unknown.
- No refit training time, peak memory, full inference time, VRAM, or package
  size was measured.
- The approved representative Kaggle execution does not prove hidden-set
  runtime or fallback behavior. No leaderboard submission, final model
  selection, or export is claimed.
- Ticket 24 remains the owner-controlled selection and delivery step.
