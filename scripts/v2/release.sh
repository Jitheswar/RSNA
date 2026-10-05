#!/usr/bin/env bash
# usage: release.sh "<ckpt globs>" "<version message>"   -> new version of private dataset jitheswar/rsna-v2-models
# Every release carries a unique release_tag.txt; we wait until Kaggle serves that tag from the latest version
# (`datasets status` can say "ready" for the PREVIOUS version), and build.py bakes the tag into the notebook, which
# refuses to run on a stale mount.
set -euo pipefail
cd /home/ch.jitheswar/RSNA
KG=~/rsna-gpu2/bin/kaggle
D=artifacts/private/v2/kaggle/rsna-v2-models
export RELEASE_TAG="r$(date +%Y%m%d-%H%M%S)-$RANDOM"
rm -rf "$D"; mkdir -p "$D"
~/rsna-gpu2/bin/python scripts/v2/package_models.py --ckpts $1 --out "$D" ${KINFER:+--k-infer $KINFER} ${ENSJSON:+--ens-json $ENSJSON} ${WEIGHTS:+--weight $WEIGHTS} 2>&1 | grep -v -i warn
ls -la "$D" | head -30
(cd "$D" && $KG datasets version -p . -m "$2 [$RELEASE_TAG]" 2>&1 | tail -2)
T=$(mktemp -d)
for i in $(seq 1 90); do
  rm -f "$T/release_tag.txt"
  $KG datasets download jitheswar/rsna-v2-models -f release_tag.txt -p "$T" --force >/dev/null 2>&1 || true
  [ -f "$T/release_tag.txt.zip" ] && (cd "$T" && unzip -o -q release_tag.txt.zip) || true
  served=$(cat "$T/release_tag.txt" 2>/dev/null || echo none)
  st=$($KG datasets status jitheswar/rsna-v2-models 2>&1 | tail -1)
  if [ "$served" = "$RELEASE_TAG" ] && [ "$st" = "ready" ]; then echo "release $RELEASE_TAG live (check $i)"; exit 0; fi
  sleep 20
done
echo "ERROR: release $RELEASE_TAG not served after 30 min (served: $served, status: $st)"; exit 1
