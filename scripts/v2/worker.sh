#!/usr/bin/env bash
# GPU worker: claims *.task scripts from the queue (atomic mv) and runs them; exits after IDLE_MAX s idle.
set -uo pipefail
Q=artifacts/private/v2/queue
IDLE_MAX=${IDLE_MAX:-5400}
ME="${SLURM_JOB_ID:-local}-$(hostname -s)"
idle=0
echo "[worker $ME] start $(date -u) gpus=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true
left_s() {  # seconds left in this Slurm job (large if not under Slurm)
  [ -z "${SLURM_JOB_ID:-}" ] && { echo 999999; return; }
  local L; L=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null | tr -d ' ')
  local d=0 h=0 m=0 s=0
  case "$L" in *-*) d=${L%%-*}; L=${L#*-};; esac
  IFS=: read -r a b c <<< "$L"
  if [ -n "$c" ]; then h=$a; m=$b; s=$c; elif [ -n "$b" ]; then m=$a; s=$b; else s=$a; fi
  echo $(( 10#$d*86400 + 10#$h*3600 + 10#$m*60 + 10#$s ))
}
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
MIN_LEFT=${MIN_LEFT:-$([ "${GPU_MEM:-81920}" -lt 40000 ] && echo 36000 || echo 14400)}  # A30: 10 h, A100: 4 h
while true; do
  if [ "$(left_s)" -lt "$MIN_LEFT" ]; then echo "[worker $ME] <$MIN_LEFT s left in job: exit without claiming $(date -u)"; exit 0; fi
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
