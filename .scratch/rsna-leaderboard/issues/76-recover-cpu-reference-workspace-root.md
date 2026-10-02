# 76: Recover the actual copied CPU reference workspace root

Type: task
Status: claimed
Labels: ready-for-agent
Parent: [RSNA campaign](../spec.md)

## Actual failure and scope

Original CPU reference8131 genuinely failed1:0 after88 CPU-node seconds before
input export or any optimizer update. Its copied helper inferred the snapshot
as workspace because the launcher omitted RSNA_PROJECT_ROOT. Preserve its
consumed intent, source/plan, raw terminal/observation and stderr. No requeue or
same-plan retry. The original frozen science/reference worker is unchanged.

Use a separately identified1800-second CPU recovery request with the required
workspace-root environment. Retain the whole old joint route, failed charge,
additional7200-second T74 analysis and every final-fit/delivery/review reserve.
Admit an additional100MB source/history allowance. Amend only the unused T74
waiting lifecycle through a separately frozen v3 owner so future admission
includes the new recovery. No TPU admission, scoring or weight change.

## Acceptance

- [x] Reproduce the exact copied-helper failure without MRI loading or fitting.
- [x] Prove the original manifest hash and helper paths pass with only the root environment corrected.
- [x] Preserve8131 and independently check actual failed terminal/zero-update evidence.
- [x] Retain complete joint route and separately admit recovery calendar/storage.
- [x] Start exact separately frozen T74v3 reserve owner and preserve stopped v2 source.
- [x] Freeze corrected CPU v2 request with original worker bytes unchanged.
- [x] Start exact one-attempt recovery controller waiting for an actual free owner slot.
- [ ] Complete the real64-MRI source/reference job and verify its actual successful terminal.
- [ ] Produce restricted-load portable states under the corrected request identity.

## Comments

October3 IST: fast actual import harness fails with the exact old fold-manifest
error when the root variable is omitted or wrong, and passes with the original
manifest SHA6db019d04ca94a5af688f860db2a46becc1b432972fdf925642254e2cb5a6bb9 when
only RSNA_PROJECT_ROOT points to the project. New corrected snapshot also passes.
Recovery plan45f34a3ff1de8570b71bbae25695e3237013834da5c5e369a2a40a0242f8c35d;
decision13435027cfd592f5b4e440546335eae13d081d0995912864a8ccec58703022c5.
Actual T74v3 PID1261544 and recovery controller PID1265703 live, no consumed new
CPU intent/accepted job yet. Jobs8128 fold0/stage6 and8132 fold1/stage6 RUNNING.
17focused tests and mypy93 PASS. Whole conservative finishOct9 11:47UTC and
storage1791028146436B, no completed files deleted or old future reserves reused.
Actual full reference completion remains pending. No daily scored submission.
