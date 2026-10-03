#!/usr/bin/env bash
# GPU worker: claims *.task scripts from the queue (atomic mv) and runs them; exits after IDLE_MAX s idle.
set -uo pipefail
Q=artifacts/private/v2/queue
IDLE_MAX=${IDLE_MAX:-5400}
ME="${SLURM_JOB_ID:-local}-$(hostname -s)"
idle=0
echo "[worker $ME] start $(date -u) gpus=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true
while true; do
  task=""
  for f in $(ls -1 "$Q/pending/" 2>/dev/null | grep '\.task$' | sort); do
    if mv "$Q/pending/$f" "$Q/running/$f.$ME" 2>/dev/null; then task="$f"; break; fi
  done
  if [ -z "$task" ]; then
    sleep 30; idle=$((idle+30))
    if [ "$idle" -ge "$IDLE_MAX" ]; then echo "[worker $ME] idle exit $(date -u)"; exit 0; fi
    continue
  fi
  idle=0
  echo "[worker $ME] run $task $(date -u)"
  t0=$(date +%s)
  bash "$Q/running/$task.$ME" > "$Q/logs/$task.$ME.log" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then mv "$Q/running/$task.$ME" "$Q/done/$task.$ME"; else mv "$Q/running/$task.$ME" "$Q/failed/$task.$ME"; fi
  echo "[worker $ME] end $task rc=$rc ${dt}s $(date -u)"
done
