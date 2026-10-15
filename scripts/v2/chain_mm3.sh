#!/usr/bin/env bash
# Detached: when both Gemma-4 all-studies shards are done -> labels stud_mm3 (avg Qwen+Gemma mm, Opus overrides)
# -> queue s_mm3 5-fold (s_mm2 recipe) at the head of the GPU queue.
cd /home/ch.jitheswar/RSNA
V=artifacts/private/v2; Q=$V/queue; L=$V/logs/chain_mm3.log
until [ "$(ls $Q/done | grep -c '0001_mm_gemma4_all_s[01]')" -ge 2 ]; do sleep 120; done
echo "$(date) gemma shards done" >> $L
~/rsna-gpu2/bin/python - >> $L 2>&1 <<'PY'
import json
from pathlib import Path
V = Path("artifacts/private/v2/llm")
def load(fs):
    d = {}
    for f in fs:
        for l in open(V / f):
            r = json.loads(l)
            if r.get("parsed"): d[r["StudyInstanceUID"]] = r["parsed"]
    return d
q = load(["mm-qwen38-all-s0.jsonl", "mm-qwen38-all-s1.jsonl"]); g = load(["mm-gemma4-all-s0.jsonl", "mm-gemma4-all-s1.jsonl"])
n = 0
with open(V / "mm-qg-all.jsonl", "w") as fh:
    for u, pq in q.items():
        pg = g.get(u)
        p = {t: {"p": (pq[t]["p"] + pg[t]["p"]) / 2 if pg else pq[t]["p"]} for t in pq}
        n += pg is not None
        fh.write(json.dumps({"StudyInstanceUID": u, "parsed": p, "model": "avg(qwen38,gemma4)"}) + "\n")
print(f"mm-qg-all: {len(q)} studies, gemma averaged on {n}")
PY
~/rsna-gpu2/bin/python scripts/v2/build_mm_labels.py --mm $V/llm/mm-qg-all.jsonl $V/llm/mm-opus-gold.jsonl $V/llm/mm-opus-all.jsonl \
  --students runs/s_mix1 runs/s_mix1_384 runs/s_mix1_pw runs/s_mix1_k24 runs/s_dinob_mix1 --fallback labels/stud_mix1e.csv --out labels/stud_mm3.csv >> $L 2>&1 || exit 1
for f in 0 1 2 3 4; do cat > $Q/pending/0002${f}_s_mm3_f${f}.task <<T
#!/usr/bin/env bash
set -euo pipefail
cd /home/ch.jitheswar/RSNA
bash scripts/v2/train_task.sh s_mm3 --folds $f --labels artifacts/private/v2/labels/stud_mm3.csv --aux-grades artifacts/private/v2/labels/aux_grades_v2.csv --backbone convnext_small.fb_in22k_ft_in1k_384 --img 320 --k 12 --k-sag 24 --k-cor 16 --k-ax 14 --pos-weight --n-series 6 --epochs 10 --bs 2 --accum 4 --workers 14 --micro 32 --seed 42
T
chmod +x $Q/pending/0002${f}_s_mm3_f${f}.task; done
echo "$(date) queued s_mm3 5-fold" >> $L
