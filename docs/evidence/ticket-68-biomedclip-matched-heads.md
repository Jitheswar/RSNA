# Ticket 68: matched medical-head workflow

The scientific workflow and waiting controller are frozen. **No medical head
has fitted, no fitting resource decision exists, and no prediction gain is
established.** Actual admission requires Ticket 67's complete independently
verified feature bank and successful terminal. Controller
`rsna-ticket68-biomedclip-heads-v1.service`, PID **324630**, is live in phase
`wait_for_complete_bank`; it currently leaves Ticket 67 scheduling intact.

Private controller root `artifacts/private/ticket68-biomedclip-head-controller-v1/`.
Admission SHA **`ae31e9e0a1cf6e52c75fdd5e79f5e0dd22717c316b93f5d426a61c2bdd51ef67`**.
Preserve the four frozen Ticket 68 repository scripts and controller snapshot.
Future fit execution root `artifacts/private/ticket68-biomedclip-matched-heads-v1/`
will be created only after bank completion and fresh resource admission. Its
actual plan SHA, launchers and resource decision are not available yet.

## Matched scientific contract

Inspection of Ticket 55's actual worker and receipts establishes **7,500 total
updates from random initialization**, with an independently verified original
1,500-step prefix. Its final heads are the matched ResNet comparators; no
learned head is transferred across representations. This differs from Ticket
64's separately registered **7,500 additional updates from those final heads**.

Medical heads reproduce the original model constructor's RNG consumption,
then use its random 2,076→32→12 MLP weights. Actual initial head SHA
`ea67e0aa11250dfccbfd2e593a78c445684bd8970ac7e9e8151bbc62e3c52d52`
matches all six historical ResNet heads. This was verified without fitting.
An actual metadata preflight independently reconstructed all three target/mask
tensors and all six 7,500-step source-exposure totals, matching the original
records with every held-out target masked.

Six medical fits: reference-only and additive report-supervision for each
original corrected fold. Fixed **7,500 steps**, Adam **1e-3**, no schedule or
augmentation, batch two per source, initialization seed20260911, batch seed
20260912, weak offset34034034, original audited admission and ordered pools.
Whole-field224, strict frozen512/L2 medical representation, unfitted anatomical
pooling; no fitted normalization, layer choice, rate or duration sweep. Final
step only, no intermediate AUC or adaptation. Historical ResNet fits are reused.

Primary is medical report minus matched ResNet report; fixed secondaries are
medical reference minus ResNet reference, medical report minus medical reference,
both medical zero-image controls, and one predeclared half medical/ResNet report
blend. Mean12 ROC AUC, **2,000 paired study-bootstrap resamples**, seed20261002,
95% interval and all three fold directions. Positive interval lower bound and
positive fold directions, with report image signal, permit **provisional review
only**. Gold58 is reused adaptive development evidence with no patient groups.
Scored-anchor comparison, offline delivery and real leaderboard ranks remain
separate unverified gates. No automatic promotion or submission.

## Complete admission and automatic scheduling

Proposed fit allocation **1,800 s**, code watchdog **1,740 s**; measured matched
ResNet six-head cost yields conservative **1,207.101 s** forecast. Separate
original-MRI verifier allocation **10,800 s**, watchdog **10,740 s**. Conservative
medical native decoding/inference forecast plus retained existing **5,400 s**
other native-replay allowance totals **9,100.521 s**. Proposed new reserves:
**12,600 CPU-node seconds**, **19,800 calendar seconds**, including two one-hour
queue allowances. These are prospective bounds, not owner numeric approvals
or an actual consumed allocation. Fresh full-route calendar/storage checks
run after the bank finishes, protecting prior fitting/delivery/review reserves,
2 TB storage and the October 11 science freeze. Failed admission does not fit.

After successful bank completion and restored original fit scheduling, the
controller freezes the real completed input bytes and its exact new resource
decision. It temporarily owns combined scheduling, advances unchanged Ticket
64 stages itself, allows one experiment job and at most two pending+active owner
CPU jobs, then restores the original coordinator on completion or exit. Every
job is 16 CPUs/64 GiB, nonexclusive, one attempt/no requeue or retry. Existing
final-verification watcher remains intact. GPU/TPU budgets and zero spend stay
unchanged. Do not duplicate consumed intents or manually restart a fit scheduler
while a combined controller owns scheduling.

## Independent completion gate

Separate process verifies every frozen input/runtime/source and all six final
checkpoints, original targets/masks/initial weights/RNG, ordered source sequences,
7500-step counts, finite Adam states, matched parent weights, stored logits,
ablations and all declared comparisons. It reconstructs **all 58 held-out
studies from original MRI**, checks their exact cached pixels/masks/descriptors/
indices, runs strict official medical vision and original ImageNet ResNet,
and checks medical feature bags and all medical/parent/zero-image predictions
within **1e-6**. Native ranks must reproduce all six fixed comparisons. Actual
successful fit and verifier terminals are mandatory; transient observations
never authorize retry. This proves only the held-out native vision scope,
not a fresh all-4,228-study native vision replay.

**21 focused tests pass**, covering exact historical initialization, malformed
Adam moments/steps/rate, complete-bank gating, slot/order guards, interval/fold
directions, finite prediction membership, exact successful terminal/bounds,
failure restoration and consumed-controller refusal. All four scripts parse;
future generated launchers are hash-checked and syntax-checked during admission.

Submission target remains **the next 24–48 hours**, conditional on measured gain
and offline delivery. No submission is queued. If this route fails the fixed
gates, the longer full-backbone comparison remains the next pending opportunity.
