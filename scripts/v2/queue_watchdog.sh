#!/usr/bin/env bash
# Queue watchdog (login node, detached): every 10 min
#  1) a task in running/ whose Slurm job is gone (time limit / crash) goes back to pending/ (no lost work items);
#  2) keep TARGET v2-gpu workers (running+pending) while tasks are pending.
cd /home/ch.jitheswar/RSNA
Q=artifacts/private/v2/queue
TARGET=${TARGET:-3}
LOG=artifacts/private/v2/logs/watchdog.log
while [ "$(date +%Y%m%d)" -le 20261023 ]; do
  # one job list per pass; if squeue fails (controller outage) skip the pass -- an empty reply is NOT "job gone"
  if ! jobs=$(squeue -u "$USER" -h -o %i 2>/dev/null); then echo "$(date) squeue failed, skip" >> $LOG; sleep 600; continue; fi
  for f in $(ls -1 $Q/running/ 2>/dev/null | grep '\.task\.'); do
    job=${f##*.task.}; job=${job%%-*}
    if ! grep -qx "$job" <<< "$jobs"; then
      base=${f%%.task.*}.task
      if [ ! -e "$Q/pending/$base" ]; then mv "$Q/running/$f" "$Q/pending/$base" && echo "$(date) requeued $base (job $job gone)" >> $LOG
      else mv "$Q/running/$f" "$Q/failed/$f"; fi
    fi
  done
  n_jobs=$(squeue -u "$USER" -h -n v2-gpu 2>/dev/null | wc -l)
  n_pend=$(ls -1 $Q/pending/ 2>/dev/null | grep -c '\.task$')
  if [ "$n_pend" -gt 0 ] && [ "$n_jobs" -lt "$TARGET" ]; then
    out=$(sbatch slurm/v2_gpu_worker.sbatch 2>&1); echo "$(date) workers $n_jobs < $TARGET with $n_pend pending -> $out" >> $LOG
  fi
  sleep 600
done
