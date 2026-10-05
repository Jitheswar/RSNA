#!/usr/bin/env bash
# usage: submit_nb.sh "<message>" <slug> <weight> [--only|--blend]
# Like submit_only.sh but reuses the current jitheswar/rsna-v2-models version (no re-packaging): (re)builds + pushes
# the notebook, waits for the commit run, then submits its submission.csv.
set -euo pipefail
cd /home/ch.jitheswar/RSNA
MSG=$1; SLUG=$2; W=$3; MODE=${4:---blend}
KG=~/rsna-gpu2/bin/kaggle
# team-shared daily quota: skip (before spending Kaggle GPU on a commit run) if no slot is left today
REM=$($KG competitions submission-limits rsna-knee-abnormality-detection 2>/dev/null | grep -o "Remaining today: [0-9]*" | grep -o "[0-9]*$")
if [ "${REM:-1}" -lt 1 ]; then echo "NOT submitted (no submission slots left today; team quota)"; exit 0; fi
echo "submission slots remaining today: ${REM:-unknown}"
if [ "$MODE" = "--only" ]; then ~/rsna-gpu2/bin/python notebooks/v2/build.py $SLUG $W --only; else ~/rsna-gpu2/bin/python notebooks/v2/build.py $SLUG $W; fi
out=$($KG kernels push -p notebooks/v2/$SLUG 2>&1 | tail -1); echo "$out"
ver=$(echo "$out" | grep -o "version [0-9]*" | grep -o "[0-9]*")
for i in $(seq 1 720); do st=$($KG kernels status jitheswar/$SLUG 2>&1 | tail -1); case "$st" in *COMPLETE*|*ERROR*|*CANCEL*) break;; esac; sleep 30; done
echo "kernel: $st (version $ver)"
case "$st" in
  *COMPLETE*)
    sub=$($KG competitions submit rsna-knee-abnormality-detection -k jitheswar/$SLUG -v $ver -f submission.csv -m "$MSG" 2>&1); echo "$sub" | tail -3
    # the new CLI prints "N submissions remaining today" (no "Successfully submitted") -> explicit marker for automations
    if echo "$sub" | grep -q -i -E "successfully submitted|remaining today"; then echo "Successfully submitted"; else echo "NOT submitted (submit error)"; fi;;
  *) echo "NOT submitted";;
esac
