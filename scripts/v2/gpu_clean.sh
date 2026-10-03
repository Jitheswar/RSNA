#!/usr/bin/env bash
# Kill stale compute processes on this job's GPU (tasks run sequentially per worker, so any app here is stale).
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  if [ "$pid" != "$$" ] && ps -o user= -p "$pid" 2>/dev/null | grep -q "$USER"; then
    echo "[gpu_clean] killing stale GPU process $pid"; kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null
  fi
done
true
