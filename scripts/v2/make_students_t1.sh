#!/usr/bin/env bash
# Student round for the T1 family: labels = MIX*cal_v2 + (1-MIX)*mean(teacher OOF); queue 5 folds (T1 recipe).
# usage: make_students_t1.sh <tag> <mix> <prio> <teacher run dirs (relative to v2)...>
set -euo pipefail
cd /home/ch.jitheswar/RSNA
PY=~/rsna-gpu2/bin/python
TAG=$1; MIX=$2; P=$3; shift 3
$PY scripts/v2/fuse_labels.py --report ${REPORT:-labels/cal_v3.csv} --oof "$@" --mix $MIX --out labels/stud_${TAG}.csv
L=artifacts/private/v2/labels/stud_${TAG}.csv
A=artifacts/private/v2/labels/aux_grades_v2.csv
for f in 0 1 2 3 4; do
  $PY scripts/v2/qtrain.py ${P} s_${TAG} $f -- --labels $L --aux-grades $A --backbone convnext_small.fb_in22k_ft_in1k_384 \
    --img 320 --k 12 --k-sag 20 --k-cor 12 --k-ax 10 --n-series 6 --epochs 10 --bs 2 --accum 4 --workers 14 --micro 32
done
