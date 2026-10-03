#!/usr/bin/env bash
# Build cal_v1 labels + merged aux grades, then queue teacher runs (T1 convnext_small, T2 effnetv2_s), interleaved.
set -euo pipefail
cd /home/ch.jitheswar/RSNA
PY=~/rsna-gpu2/bin/python
$PY scripts/v2/labels_eval.py --write > /dev/null 2>&1 || true
SRC="sol61:ordinal:2 astra:categorical:1 pubv2:prob:1"
[ -f artifacts/private/v2/labels/qwen38_grades.csv ] && SRC="$SRC qwen38:ordinal:1"
[ -f artifacts/private/v2/labels/gemma4_grades.csv ] && SRC="$SRC gemma4:ordinal:1"
$PY scripts/v2/build_labels.py --sources $SRC --out labels/cal_v1.csv --fold-specific
$PY - <<'PY'
import pandas as pd
V='artifacts/private/v2/labels/'
s=pd.read_csv(V+'sol61_grades.csv').set_index('StudyInstanceUID')
try: q=pd.read_csv(V+'qwen38_grades.csv').set_index('StudyInstanceUID')
except FileNotFoundError: q=s.iloc[:0]
m=pd.concat([s, q[~q.index.isin(s.index)]])
m.reset_index().to_csv(V+'aux_grades_v1.csv', index=False); print('aux grades', len(m))
PY
L=artifacts/private/v2/labels/cal_v1_f{fold}.csv
A=artifacts/private/v2/labels/aux_grades_v1.csv
for f in 0 1 2 3 4; do
  $PY scripts/v2/qtrain.py 01${f}a t1_cnxs320_cal1 $f -- --labels $L --aux-grades $A --backbone convnext_small.fb_in22k_ft_in1k_384 --img 320 --k 12 --k-sag 20 --k-cor 12 --k-ax 10 --n-series 6 --epochs 10 --bs 2 --accum 4 --workers 14 --micro 32
  $PY scripts/v2/qtrain.py 01${f}b t2_effv2s320_cal1 $f -- --labels $L --aux-grades $A --backbone tf_efficientnetv2_s.in21k_ft_in1k --img 320 --k 12 --k-sag 20 --k-cor 12 --k-ax 10 --n-series 6 --epochs 10 --bs 2 --accum 4 --workers 14 --micro 32 --lr-enc 2e-4 --micro 32
done
ls artifacts/private/v2/queue/pending
