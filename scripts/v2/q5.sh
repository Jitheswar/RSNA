#!/usr/bin/env bash
# Queue a 5-fold run: q5.sh <prio(3 chars)> <run_name> <labels (may contain {fold})> <recipe> [extra args...]
# recipes: coat384  = CoAtNet-2 384 bag recipe;  cnxs384 = ConvNeXt-S 384 bag recipe;  effv2m = EffNetV2-M 384 bag recipe
set -euo pipefail
cd /home/ch.jitheswar/RSNA
P=$1; RUN=$2; LAB=$3; REC=$4; shift 4
BAG="--zoom 1.14 --k 12 --k-sag 6 --k-cor 4 --k-ax 4 --ktr-sag 3 --ktr-cor 2 --ktr-ax 2 --random-pos --head mil --pos-weight --aug none --n-series 6 --workers 14"
case $REC in
  coat384) R="--backbone coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --img 384 --lr-enc 3e-5 --lr-head 1e-3 --epochs 16 --bs 4 --accum 2 --micro 28 $BAG";;
  cnxs384) R="--backbone convnext_small.fb_in22k_ft_in1k_384 --img 384 --lr-enc 5e-5 --lr-head 1e-3 --epochs 16 --bs 4 --accum 2 --micro 28 $BAG";;
  effv2m)  R="--backbone tf_efficientnetv2_m.in21k_ft_in1k --img 384 --lr-enc 1e-4 --lr-head 1e-3 --epochs 16 --bs 4 --accum 2 --micro 28 $BAG";;
  *) echo "unknown recipe $REC"; exit 1;;
esac
for f in 0 1 2 3 4; do
  ~/rsna-gpu2/bin/python scripts/v2/qtrain.py $P $RUN $f -- --labels "$LAB" $R "$@"
done
