"""Collect subagent multimodal label outputs into a JSONL ({StudyInstanceUID, parsed: {target: {p}}}, like mm_label).

python scripts/v2/mm_collect.py --tag opus-gold --out artifacts/private/v2/llm/mm-opus-gold.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_label import TARGETS  # noqa: E402

V2 = Path("artifacts/private/v2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = V2 / "mm" / args.tag
    rows, bad, missing = {}, 0, []
    for b in sorted(glob.glob(str(root / "batches/batch_*.json"))):
        want = [it["uid"] for it in json.load(open(b))]
        o = root / "out" / Path(b).name
        if not o.exists():
            missing.append(Path(b).name)
            continue
        try:
            got = json.load(open(o))
        except Exception:
            bad += 1
            continue
        for u in want:
            r = got.get(u)
            if isinstance(r, dict) and all(t in r for t in TARGETS):
                rows[u] = {"StudyInstanceUID": u, "parsed": {t: {"p": int(r[t])} for t in TARGETS}, "model": "claude-opus-5.5",
                           "source": args.tag}
            else:
                bad += 1
    with open(args.out, "w") as fh:
        for r in rows.values():
            fh.write(json.dumps(r) + "\n")
    print(f"collected {len(rows)} studies -> {args.out}; bad/incomplete {bad}; missing batch outputs {len(missing)}")


if __name__ == "__main__":
    main()
