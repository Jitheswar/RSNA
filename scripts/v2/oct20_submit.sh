#!/usr/bin/env bash
# Oct 20 morning: (1) ours-only s_r6 (R6-distilled labels) 6 models; (2) Raptor + R6flip + ours [s_r6 6 + 24 s_mm2] 1/3.
cd /home/ch.jitheswar/RSNA
V=artifacts/private/v2; L=$V/logs; KG=~/rsna-gpu2/bin/kaggle
until [ $(ls $V/queue/done | grep -c "_s_r6_") -ge 6 ]; do sleep 120; done
until [ "$(date +%Y%m%d%H%M)" -ge 202610200536 ]; do sleep 60; done
until ! $KG competitions submissions rsna-knee-abnormality-detection 2>/dev/null | sed -n 3,12p | grep -q PENDING; do sleep 300; done
NEW="$V/runs/s_r6/fold*/best.pt $V/runs/s_r6_full_s1/full/last.pt"
OLD="$V/runs/s_mm2/fold*/best.pt $V/runs/s_mm2_s2/fold*/best.pt $V/runs/s_mm2_384/fold*/best.pt $V/runs/s_mm2_s3/fold*/best.pt $V/runs/s_mm2_full_s1/full/last.pt $V/runs/s_mm2_full_s2/full/last.pt $V/runs/s_mm2_full_s3/full/last.pt $V/runs/s_mm2_384_full_s1/full/last.pt"
echo "$(date) new $(ls $NEW | wc -l) old $(ls $OLD | wc -l)" >> $L/oct20_submit.log
bash scripts/v2/submit_only.sh "$NEW" "OURS-ONLY s_r6: 6 models on R6-distilled labels (0.6 stud_mm2 + 0.4 rank R6)" rsna-v2-only 1.0 --only > $L/submit52.log 2>&1 &
sleep 120
BASE=raptor-r6 bash scripts/v2/submit_only.sh "$NEW $OLD" "BLEND Raptor + R6flip + ours [6 s_r6 + 24 s_mm2], 1/3" rsna-v2-blend 0.3333 --blend > $L/submit53.log 2>&1
wait; echo "$(date) done" >> $L/oct20_submit.log
