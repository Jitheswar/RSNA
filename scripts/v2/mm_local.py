"""Multimodal (image + report) panel-style labels from a LOCAL open-weight VLM via vLLM (no API, no usage limit).

Same prompt and montages as the Opus / frontier runs (mm_label.build_prompt, artifacts/private/v2/mm/montages/<uid>/
{sagittal,coronal,axial}.png). Output rows match mm-opus-*.jsonl: {StudyInstanceUID, parsed: {target: {p}}, model}.

python scripts/v2/mm_local.py --model google/gemma-4-31B-it --gold --out artifacts/private/v2/llm/mm-gemma4-gold.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_label import TARGETS  # noqa: E402
import re
from mm_label import build_prompt  # noqa: E402

V2 = Path("artifacts/private/v2")


def parse(text: str):
    """Last JSON object in the output that has all 12 targets with a numeric p (thinking blocks stripped)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    for s in reversed([m.start() for m in re.finditer(r"\{", text)]):
        depth = 0
        for i in range(s, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                try:
                    obj = json.loads(text[s:i + 1])
                except json.JSONDecodeError:
                    break
                if isinstance(obj, dict) and all(isinstance(obj.get(t), dict) and isinstance(obj[t].get("p"), (int, float))
                                                 for t in TARGETS):
                    return {t: {"p": float(obj[t]["p"])} for t in TARGETS}
                break
    return None


def image_part(p: Path) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gold", action="store_true", help="only the reference-labelled studies")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--chunk", type=int, default=256, help="studies per llm.chat call (results appended per chunk)")
    args = ap.parse_args()

    import pandas as pd
    from vllm import LLM, SamplingParams

    df = pd.read_csv(V2 / "train.csv")
    if args.gold:
        df = df[df.ACL.notna()]
    else:  # gold first so a partial run is already evaluable
        df = pd.concat([df[df.ACL.notna()], df[df.ACL.isna()]])
    df = df[[(V2 / "mm/montages" / u / "sagittal.png").exists() for u in df.StudyInstanceUID]]
    if args.limit:
        df = df.head(args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["StudyInstanceUID"] for l in out.read_text().splitlines()
                if json.loads(l).get("parsed") is not None}
    todo = df[~df.StudyInstanceUID.isin(done)]
    print(f"{len(done)} done, {len(todo)} to label", flush=True)
    if not len(todo):
        return
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.92, trust_remote_code=True, limit_mm_per_prompt={"image": 3, "video": 0})
    sp = SamplingParams(temperature=0.6 if args.think else 0.0, top_p=0.95,
                        max_tokens=8192 if args.think else 2048, seed=0)
    kw = {"chat_template_kwargs": {"enable_thinking": bool(args.think)}}
    fmt = ('\n\nAnswer ONLY with one JSON object: {"<target>": {"p": <0-100>, "why": "<reason>"}, ...} with exactly '
           "these keys: " + ", ".join(TARGETS) + ".")
    rows = list(todo.itertuples(index=False))
    n_ok = 0
    for c in range(0, len(rows), args.chunk):
        part = rows[c:c + args.chunk]
        msgs = []
        for r in part:
            d = V2 / "mm/montages" / r.StudyInstanceUID
            imgs = [image_part(d / f"{pl}.png") for pl in ("sagittal", "coronal", "axial") if (d / f"{pl}.png").exists()]
            msgs.append([{"role": "user", "content": imgs + [{"type": "text", "text": build_prompt(r.Report) + fmt}]}])
        outs = llm.chat(msgs, sp, use_tqdm=True, **kw)
        with out.open("a") as fh:
            for r, o in zip(part, outs):
                text = o.outputs[0].text
                obj = parse(text)
                n_ok += obj is not None
                fh.write(json.dumps({"StudyInstanceUID": r.StudyInstanceUID, "parsed": obj, "raw": text[-3000:],
                                     "model": args.model}) + "\n")
        print(f"chunk {c // args.chunk}: parsed {n_ok}/{c + len(part)}", flush=True)


if __name__ == "__main__":
    main()
