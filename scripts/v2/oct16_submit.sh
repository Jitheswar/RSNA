#!/usr/bin/env bash
# Detached Oct 16 morning submissions (after the 05:30 IST quota reset): ours-only hedge + R7 blend with all finished folds.
cd /home/ch.jitheswar/RSNA
V=artifacts/private/v2; L=$V/logs
until [ "$(date +%Y%m%d%H%M)" -ge 202610160536 ]; do sleep 60; done
ck() {  # finished checkpoints only (task in queue/done)
  local out="$V/runs/s_mm2/fold*/best.pt $V/runs/s_mm2_full_s1/full/last.pt $V/runs/s_mm2_full_s2/full/last.pt $V/runs/s_mm2_384/fold*/best.pt"
  for full in s_mm2_full_s3 s_mm2_384_full_s1; do ls $V/queue/done | grep -q "_${full}_f0.task" && out="$out $V/runs/$full/full/last.pt"; done
  local runs="s_mm2_s2 s_mm2_384_s2 s_mm2_s3"; [ "$(ls $V/queue/done | grep -c _s_mm3_f)" -ge 5 ] && ~/rsna-gpu2/bin/python scripts/v2/gate_family.py s_mm3 >> $L/oct16_submit.log 2>&1 && runs="$runs s_mm3"
  for run in $runs; do for f in 0 1 2 3 4; do
    ls $V/queue/done | grep -q "_${run}_f${f}.task" && [ -f $V/runs/$run/fold$f/best.pt ] && out="$out $V/runs/$run/fold$f/best.pt"
  done; done; echo "$out"; }
CK=$(ck); N=$(ls $CK 2>/dev/null | wc -l)
echo "$(date) $N models: $CK" >> $L/oct16_submit.log
bash scripts/v2/submit_only.sh "$CK" "OURS-ONLY hedge (no OAI lineage): v2 s_mm2(+mm3) families, $N models" rsna-v2-only 1.0 --only > $L/submit39.log 2>&1 &
sleep 120
BASE=r7 bash scripts/v2/submit_only.sh "$CK" "BLEND: attributed heliosli R7 + v2 s_mm2(+mm3) families ($N models), rank weight 0.5" rsna-v2-rehearsal 0.5 --blend > $L/submit40.log 2>&1
wait; echo "$(date) done" >> $L/oct16_submit.log
