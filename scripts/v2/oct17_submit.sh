#!/usr/bin/env bash
# Detached Oct 17 morning submissions: 3-arm (R7 + S6fall + all s_mm2 models, 1/3 each) and the no-OAI-lineage hedge
# (S6fall + all models, 0.5/0.5). Waits for the quota reset AND for every pending submission to finish (a pending hidden run
# mounts the latest model-dataset version; never re-release under it).
cd /home/ch.jitheswar/RSNA
V=artifacts/private/v2; L=$V/logs; KG=~/rsna-gpu2/bin/kaggle
until [ "$(date +%Y%m%d%H%M)" -ge 202610170536 ]; do sleep 60; done
until ! $KG competitions submissions rsna-knee-abnormality-detection 2>/dev/null | sed -n 3,12p | grep -q PENDING; do sleep 300; done
CK="$V/runs/s_mm2/fold*/best.pt $V/runs/s_mm2_full_s1/full/last.pt $V/runs/s_mm2_full_s2/full/last.pt $V/runs/s_mm2_full_s3/full/last.pt $V/runs/s_mm2_384_full_s1/full/last.pt $V/runs/s_mm2_384/fold*/best.pt"
for run in s_mm2_s2 s_mm2_s3 s_mm2_384_s2 s_mm2_384_s3; do for f in 0 1 2 3 4; do
  ls $V/queue/done | grep -q "_${run}_f${f}.task" && [ -f $V/runs/$run/fold$f/best.pt ] && CK="$CK $V/runs/$run/fold$f/best.pt"
done; done
N=$(ls $CK | wc -l); echo "$(date) $N models: $CK" >> $L/oct17_submit.log
W_S6=0.5 S6=1 bash scripts/v2/submit_only.sh "$CK" "BLEND 2 arms, NO OAI lineage: nartaa S6fall + v2 $N models, 0.5/0.5" rsna-v2-rehearsal2 1.0 --only > $L/submit42.log 2>&1 &
sleep 120
BASE=r7 S6=1 W_S6=0.3333 bash scripts/v2/submit_only.sh "$CK" "BLEND 3 arms (OAI-free): attributed heliosli R7 + nartaa S6fall + v2 $N models, 1/3 each" rsna-v2-timing 0.5 --blend > $L/submit43.log 2>&1
wait; echo "$(date) done" >> $L/oct17_submit.log
