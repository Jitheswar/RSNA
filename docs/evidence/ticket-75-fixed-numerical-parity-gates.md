# Ticket75 fixed numerical parity gates

Previous goal turn classification: progress. Commita9888e2 implemented strict
model-only CPU delivery preparation and passed203 pilot checks. This turn begins
from the clean worktree and polls actual training jobs8132/8133 and active
T73/T74v3/corrected-reference controllers. All remain live; the real CPU reference
still has no accepted job. No expired observation or missing receipt was treated
as a reason to restart an accepted process.

The new tpu_pilot_parity module implements the prospectively declared numerical
gates without exposing a configurable tolerance. CPU replay requires all eight
prescribed twelve-target FP32 logits and absolute error at most1e-6, with no
relative tolerance. CPU2.8-versus-TPU CPU-converted FP32 tensors use the fixed
reference-relative formula1e-6 +1e-4*abs(CPU_reference). Nonfloating tensors,
configuration/progress primitives, mapping/sequence types, full keys and ordered
model metadata compare exactly. Floating losses are represented as FP32 tensors;
configuration floats do not get a numerical tolerance.

Comparison diagnostics use float64 arithmetic on the original FP32 endpoints
to avoid overflow in finite extreme-value differences. This does not change
model/optimizer computation precision or the fixed threshold formula. Reports
retain per-tensor shape/dtype/element count, finite support, mismatched elements,
maximum absolute error, relative error on nonzero references and zero-reference
support. Nonfinite endpoints fail; their finite support is explicit rather than
silently removed from acceptance. A numerical failure raises PilotParityFailure
with the complete report and exact tolerance fields for the future failure
recorder. It cannot become a passing receipt or permission to widen/retry.

The first-three-update entry point requires both complete step3 states, every
original model/buffer/metadata tensor against the frozen initialization's scope,
all three full ordered parameter-gradient records, all three FP32 global losses,
and all eight prescribed step3 logits. Ordinary Adam requires exact original
parameter groups/order/defaults/rates, complete FP32 moment shapes, nonnegative
second moments and exact scalar step3. Thus mutually omitted gradients/moments
cannot pass merely because both sides have the same incomplete inventory. The
future worker must supply the actual source-bound initial model scope and
actual host/device observations; a synthetic declared scope is not backend or
registered-source proof.

Validation:31 new parity checks plus203 existing pilot checks pass together
(234total). Coverage includes relative/absolute boundaries and near-zero support,
CPU-only absolute replay rejection despite passing cross-backend tolerance,
rank7 target/nonfinite errors, full failure-report support and every tensor's
errors, missing keys/shapes/dtypes/metadata, changed config/counters/sequences,
non-CPU/unsupported values, empty comparisons, float32 extreme-value overflow,
all three gradient/loss records, mutually missing gradients/Adam, changed group
rates and scalar steps, nonnegative moments and original full-backbone tensor
inventory. Tests use synthetic CPU values, actual CPU ordinary Adam in a small
fixture, and actual original full-backbone architecture with synthetic fixed
gradients/logits. They establish no actual CPU2.8, TPU or real-MRI execution.
An initial syntax error in the new first-three wrapper prevented collection;
it was corrected before the final successful targeted and combined runs.
Mypy passes96 source modules, files compile and diff checks pass.

All receipts explicitly leave actual CPU2.8 runtime, actual TPU execution and
original MRI identity unverified. Actual host reference execution, full worker/
rank capture/global writer, fresh PJRT supervisor, complete private bundle,
quota/vacancy admission, TPU execution and independent native/license/runtime
verification remain pending. This module is editable until a separate accepted
request freezes it; no frozen live route or original scientific configuration
was modified. No new allocation, upload or daily scoring submission occurred.
First place remains unproven and the original campaign goal stays active.
