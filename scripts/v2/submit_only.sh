#!/usr/bin/env bash
# usage: submit_only.sh "<ckpt globs>" "<message>" [slug=rsna-v2-only] [weight=1.0] [--blend]
# Packages checkpoints into jitheswar/rsna-v2-models (new version), (re)builds + pushes the notebook, waits for the
# run to finish, then submits its submission.csv to the competition.
set -euo pipefail
cd /home/ch.jitheswar/RSNA
CK=$1; MSG=$2; SLUG=${3:-rsna-v2-only}; W=${4:-1.0}; MODE=${5:---only}
KG=~/rsna-gpu2/bin/kaggle
bash scripts/v2/release.sh "$CK" "$MSG"
for i in $(seq 1 60); do s=$($KG datasets status jitheswar/rsna-v2-models 2>&1 | tail -1); [ "$s" = "ready" ] && break; sleep 20; done
echo "dataset: $s"
if [ "$MODE" = "--only" ]; then ~/rsna-gpu2/bin/python notebooks/v2/build.py $SLUG $W --only; else ~/rsna-gpu2/bin/python notebooks/v2/build.py $SLUG $W; fi
out=$($KG kernels push -p notebooks/v2/$SLUG 2>&1 | tail -1); echo "$out"
ver=$(echo "$out" | grep -o "version [0-9]*" | grep -o "[0-9]*")
for i in $(seq 1 720); do st=$($KG kernels status jitheswar/$SLUG 2>&1 | tail -1); case "$st" in *COMPLETE*|*ERROR*|*CANCEL*) break;; esac; sleep 30; done
echo "kernel: $st (version $ver)"
case "$st" in *COMPLETE*) $KG competitions submit rsna-knee-abnormality-detection -k jitheswar/$SLUG -v $ver -f submission.csv -m "$MSG" 2>&1 | tail -2;; *) echo "NOT submitted";; esac
