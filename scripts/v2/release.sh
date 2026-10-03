#!/usr/bin/env bash
# usage: release.sh "<ckpt globs>" "<version message>"   -> new version of private dataset jitheswar/rsna-v2-models
set -euo pipefail
cd /home/ch.jitheswar/RSNA
D=artifacts/private/v2/kaggle/rsna-v2-models
rm -rf "$D"; mkdir -p "$D"
~/rsna-gpu2/bin/python scripts/v2/package_models.py --ckpts $1 --out "$D" 2>&1 | grep -v -i warn
ls -la "$D" | head -30
cd "$D" && ~/rsna-gpu2/bin/kaggle datasets version -p . -m "$2" 2>&1 | tail -2
