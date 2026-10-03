#!/usr/bin/env bash
# usage: train_task.sh <run_name> <extra train.py args...>   (copies volumes to node-local disk first)
set -euo pipefail
cd /home/ch.jitheswar/RSNA
bash scripts/v2/gpu_clean.sh
RUN=$1; shift
LOCAL=/tmp/rsna_v2_$USER/vol384
mkdir -p "$(dirname $LOCAL)"
(
  flock 9
  if [ ! -f "$LOCAL/.complete" ]; then
    echo "[copy] $(date -u) syncing volumes to $LOCAL"
    mkdir -p "$LOCAL"
    rsync -a --info=stats1 artifacts/private/v2/vol384/vol/ "$LOCAL/"
    touch "$LOCAL/.complete"
    echo "[copy] $(date -u) done"
  fi
  ~/rsna-gpu2/bin/python scripts/v2/fix_laterality.py --vol-dir "$LOCAL" --flip-csv artifacts/private/v2/lat_flip_v1.csv
) 9>"$(dirname $LOCAL)/.copy.lock"
nvidia-smi --query-gpu=name,memory.used --format=csv
~/rsna-gpu2/bin/python scripts/v2/train.py --vol-dir "$LOCAL" --out artifacts/private/v2/runs/$RUN "$@"
