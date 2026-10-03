#!/usr/bin/env bash
# Student round: labels = MIX*report_cal + (1-MIX)*mean(teacher OOF); queue student runs.
# usage: make_students.sh <tag> <mix> <report_labels.csv (relative to v2)> <teacher run dirs...>
set -euo pipefail
cd /home/ch.jitheswar/RSNA
PY=~/rsna-gpu2/bin/python
TAG=$1; MIX=$2; REP=$3; shift 3
$PY scripts/v2/fuse_labels.py --report $REP --oof "$@" --mix $MIX --out labels/stud_${TAG}.csv
A=artifacts/private/v2/labels/aux_grades_v1.csv
L=artifacts/private/v2/labels/stud_${TAG}.csv
for f in 0 1 2 3 4; do
  $PY scripts/v2/qtrain.py 02${f}a s1_cnxs320_${TAG} $f -- --labels $L --aux-grades $A --backbone convnext_small.fb_in22k_ft_in1k_384 --img 320 --k 12 --n-series 6 --epochs 10 --bs 2 --accum 4 --workers 14
  $PY scripts/v2/qtrain.py 02${f}b s2_effv2s320_${TAG} $f -- --labels $L --aux-grades $A --backbone tf_efficientnetv2_s.in21k_ft_in1k --img 320 --k 12 --n-series 6 --epochs 10 --bs 2 --accum 4 --workers 14 --lr-enc 2e-4
done
