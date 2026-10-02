# Ticket75 collective finite admission before optimizer mutation

Previous goal turn classification: progress. Commit21915ea changed the editable
adapter's immediate fixed telemetry gates and passed129 CPU checks. The live
training handles were re-polled at this turn's start; jobs8132 and8133 remain
RUNNING, and the T73 scheduler, T74v3 waiter and corrected CPU-reference waiter
remain active. No observation timeout, retry or duplicate launch is involved.

Manual comparison with the fixed pilot design found a concrete protocol gap:
the adapter previously checked collective finiteness only after ordinary Adam
and scheduler had already advanced. The design requires a collective finite
guard before any parameter update. The adapter is still editable and is not
frozen into any accepted TPU request. No existing scientific worker, accepted
source snapshot or resource owner needs to change to correct this gap.

The adapter now executes original forward/backward and exactly one mean gradient
reduction, synchronizes, and constructs a device-only finite flag covering loss,
every reduced gradient and model parameter, existing Adam first/second moments,
nonnegative second moments and unchanged encoder normalization. All eight flags
are summed and synchronized; a count other than8 stops the consumed update
before calling optimizer.step or scheduler.step. This also checks reduction
overflow rather than assuming finite local gradients imply finite reduced ones.
The existing post-update collective check remains to detect failures produced by
the update itself. Missing gradients fail rather than being omitted.

Intentional pre-update guard/CPU scalar-readback work has separate raw metric
boundaries and timing. Training compile/execute/fallback deltas combine only
before→after-backward and before-optimizer→after-training intervals. The
intervening guard cannot become an apparent unsupported training scalar fallback,
but its raw metrics and actual elapsed time remain retained. Both pre/post guard
costs stay inside the full synchronized update interval and the unchanged fixed
timing envelope. Missing metric evidence remains unknown; counter reset and
unobserved fallback operations still fail. The pre-update collective rank count
and phase metrics remain available in partial in-memory failure evidence.

Validation:142 pilot checks pass. New tests exercise the actual CPU full-backbone
finite flag with empty/initialized ordinary Adam state, NaN loss/model/moments,
infinite gradients, negative second moments, changed normalization and missing
gradients. They prove that inspecting the flag does not mutate model/Adam state.
Two additional tests run actual synthetic32-pixel CPU forward/backward with only
backend collective/synchronization boundaries mocked: a reported failed other
rank and actual nonfinite gradients both leave every model tensor, Adam state
and scheduler position unchanged, retain failed-rank evidence, and prevent retry.
These are meaningful CPU behavior checks, not actual PJRT synchronization,
eight-rank training or real224-MRI TPU proof. Phase-parser checks distinguish
intentional guard metrics from unsupported training fallback and reject absent
execute evidence or backwards counters. Mypy passes93 source modules; updated
files compile and diff checks pass.

The corrected real-MRI CPU reference remains waiting for an actual free owner
slot and has no accepted job yet. Full worker/supervisor, fresh-process checkpoint
and per-rank RNG recovery, exact CPU2.8 reference, bounded private attachment,
quota/vacancy admission, actual TPU execution and independent delivery remain
pending. This turn corrected a prerequisite; it did not establish an improved
candidate, admit a TPU allocation or consume a daily scoring submission. First
place on either leaderboard remains unproven and the original goal stays active.
