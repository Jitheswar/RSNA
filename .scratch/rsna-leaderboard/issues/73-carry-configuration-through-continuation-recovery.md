# 73: Carry configuration through full-continuation recovery

Type: task
Status: claimed
Labels: ready-for-agent

## Evidence and scope

T72 audit and fixed T71 rejection remain complete. Full continuationv3
c0cafdac0916705cb672be6f2c45e7db75a661841c8efa6706fbc2cf8a5a1eaf launched
8111/8113.8111 failed before updates because the new artifact directory lacked
immutable configuration.json, actual FAILED1:0/349 CPU-node seconds. All seeds
unchanged, no new fold0 training artifacts. Preserve failed intent and terminal.
Its coordinator stopped and cannot be restarted as an unconsumed action.

Atomic exact original fold1 config handoff let existing8113 resume beyond3000
without restart or changed science. Observe its same handle; never assume its
terminal from time or a checkpoint. Generic real-trainer regression reproduces
this missing-config seam and proves exact resumed state/updates after carrying
original bytes. New helper scripts/carry_training_configuration.py, three guards.

Create a separately frozen execution recovery preserving the original7500
additional updates/fold, optimizer/config/RNG/source identity, all controls,
final-only statistics and original native-verification gates. Carry and hash
original config plus each checkpoint as one resumable input. Existing8113 may
only be inherited through actual terminal/state proof, or explicitly admitted
live inheritance that gates its next stage on successful terminal/input closure.
Do not repeat any update already performed, shrink the budget, silently edit
frozen plans or relabel8072/8074 historical terminals. Retain39600 conservative
missing-allocation charge and349 measured new failed allocation. New complete
resource/calendar/storage decision under standing authorization, zero spend.

## Acceptance

- [x] Capture actual failed terminal/trace and no-update boundary.
- [x] Preserve live8113 via exact original configuration handoff.
- [x] Real shared-trainer regression proves missing prerequisite and exact recovery.
- [ ] Freeze complete configuration/checkpoint migration and recovered execution route.
- [ ] Admit/launch only distinct bounded unused execution and restore scheduling.
- [ ] Complete full scientific budgets and independent original-MRI final comparison.
