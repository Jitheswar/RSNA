"""Severity-aware report labelling with a local open-weight LLM (vLLM, offline; no data leaves the cluster).

For each training report the model returns, per target, an English evidence phrase, an ordinal severity
grade on a target-specific scale, and the probability that the expert image panel graded the target
positive under the host's definitions. Output: JSONL with the raw text and the parsed object.

Usage: python llm_label.py --model ~/models/hf/Qwen3.8-27B --out artifacts/private/v2/llm/qwen38.jsonl [--think]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

SYSTEM = """You are an expert musculoskeletal radiologist. You will read one knee MRI report (it may be written in any language) and estimate how an expert reading panel graded the MRI images for 12 targets.

The panel (two subspecialist MSK radiologists with third-reader adjudication) read the IMAGES, not the report, using the strict criteria below. Borderline or equivocal findings were graded NEGATIVE. Reports often describe lower-grade findings that the panel would call negative, and sometimes omit findings the panel would call positive.

Positive definitions:
1. ACL: high-grade partial tear or complete (full-thickness) tear of the anterior cruciate ligament. Sprain, mucoid degeneration, ganglion, low-grade partial tear -> negative.
2. MCL: high-grade acute partial tear or complete tear of the medial collateral ligament. Grade 1 sprain, periligamentous edema only, chronic thickening or old injury -> negative.
3. Medial Meniscus: definite tear of the medial meniscus: signal unequivocally reaching an articular surface, or specific tear morphology (radial, horizontal, vertical/longitudinal, complex, flap, bucket-handle, root tear, displaced fragment, macerated or absent meniscal tissue). Intrasubstance degeneration (grade 1-2 signal) -> negative.
4. Lateral Meniscus: same as 3 for the lateral meniscus.
5. Medial OA: high-grade cartilage loss in the medial femorotibial compartment (deep/>50% thickness loss, grade 3-4, full-thickness defects, bone exposure; usually with osteophytes). Mild or superficial chondromalacia/thinning -> negative.
6. Lateral OA: same as 5 for the lateral femorotibial compartment.
7. PF OA: same as 5 for the patellofemoral compartment (patella and/or trochlea).
8. Effusion: moderate or large joint effusion. None, physiological, trace or small -> negative.
9. Synovitis: synovial inflammation or synovial thickening/proliferation (effusion-synovitis, Hoffa-synovitis, synovial hypertrophy).
10. Baker's: moderate or large popliteal (Baker's) cyst. Small cysts -> negative.
11. Contusion: bone-marrow edema-like (bone bruise/contusion) injury without a discrete fracture line.
12. Fracture: acute fracture with a cortical break or discrete fracture line (including avulsion, impaction, osteochondral, Segond and tibial plateau fractures). Old healed fractures -> negative.

For each target give:
- "ev": the most relevant phrase(s) of the report translated to English (at most 20 words), or "" if the report does not address the target.
- "g": severity grade on the target's scale below, or -1 if the report does not address the target at all.
- "p": integer probability 0-100 that the expert panel graded the target POSITIVE on the images.

Severity scales:
- ACL: 0 normal/intact; 1 degeneration, thickening, ganglion or sprain without tear; 2 low-grade partial tear; 3 partial tear of unspecified extent; 4 high-grade partial tear; 5 complete tear, rupture or non-visualized ligament; 6 reconstructed with intact graft; 7 reconstructed with torn graft.
- MCL: 0 normal; 1 chronic thickening, old injury or periligamentous edema only; 2 grade 1 sprain; 3 grade 2 sprain or partial tear of unspecified extent; 4 high-grade partial tear; 5 complete tear (grade 3).
- Medial Meniscus, Lateral Meniscus: 0 normal; 1 degeneration or intrasubstance signal; 2 possible/probable tear or signal possibly reaching a surface; 3 definite tear; 4 complex, displaced, bucket-handle or root tear, or macerated meniscus; 5 post-meniscectomy without a definite new tear.
- Medial OA, Lateral OA, PF OA: 0 normal cartilage; 1 mild chondropathy (grade 1, signal change or softening); 2 moderate (grade 2, superficial or <50% thickness loss); 3 severe (grade 3, >50% thickness loss); 4 full-thickness loss, bone exposure (grade 4) or advanced osteoarthritis.
- Effusion: 0 none or physiological; 1 trace, minimal or small; 2 present, size unspecified; 3 moderate; 4 large or massive.
- Synovitis: 0 none; 1 mild or possible synovial thickening; 2 synovitis or synovial hypertrophy; 3 marked or extensive synovitis.
- Baker's: 0 none; 1 small; 2 present, size unspecified; 3 moderate; 4 large or ruptured.
- Contusion: 0 none; 1 minimal, subtle or non-traumatic (degenerative/reactive) marrow edema; 2 bone bruise or contusion; 3 extensive or multiple contusions.
- Fracture: 0 none; 1 old/healed fracture or stress reaction; 2 possible or occult fracture; 3 definite acute fracture.

Also give "side": "left", "right" or "unknown" (the knee side stated in the report) and "postop": true or false (prior knee surgery such as ACL reconstruction or meniscectomy).

Answer with only one JSON object of this form (no markdown):
{"side": "...", "postop": false, "ACL": {"ev": "...", "g": 0, "p": 0}, "MCL": {...}, "Medial Meniscus": {...}, "Lateral Meniscus": {...}, "Medial OA": {...}, "Lateral OA": {...}, "PF OA": {...}, "Effusion": {...}, "Synovitis": {...}, "Baker's": {...}, "Contusion": {...}, "Fracture": {...}}"""


def user_msg(report: str) -> str:
    return f"Report:\n<<<\n{report.strip()}\n>>>"


def parse(text: str):
    """Extract the last balanced JSON object from the model output; validate the 12 targets."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for s in starts:
        depth = 0
        for i in range(s, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[s:i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and all(t in obj for t in TARGETS):
                        ok = all(isinstance(obj[t], dict) and "p" in obj[t] and "g" in obj[t] for t in TARGETS)
                        if ok:
                            return obj
                    break
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", default="artifacts/private/v2/train.csv")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gold-first", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    from vllm import LLM, SamplingParams

    df = pd.read_csv(args.train)
    if args.gold_first:
        df = pd.concat([df[df.ACL.notna()], df[df.ACL.isna()]])
    if args.limit:
        df = df.head(args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r.get("parsed") is not None:
                done.add(r["StudyInstanceUID"])
    todo = df[~df.StudyInstanceUID.isin(done)]
    print(f"{len(done)} done, {len(todo)} to label", flush=True)
    if not len(todo):
        return
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.92, enable_prefix_caching=True, trust_remote_code=True,
              limit_mm_per_prompt={"image": 0, "video": 0})
    max_tokens = args.max_tokens or (8192 if args.think else 2048)
    sp = SamplingParams(temperature=0.6 if args.think else 0.0, top_p=0.95, max_tokens=max_tokens, seed=0)
    msgs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(r)}] for r in todo.Report]
    kw = {"chat_template_kwargs": {"enable_thinking": bool(args.think)}}
    outs = llm.chat(msgs, sp, use_tqdm=True, **kw)
    n_ok = 0
    with out.open("a") as fh:
        for uid, o in zip(todo.StudyInstanceUID, outs):
            text = o.outputs[0].text
            obj = parse(text)
            n_ok += obj is not None
            fh.write(json.dumps({"StudyInstanceUID": uid, "parsed": obj, "raw": text[-6000:],
                                 "n_tokens": len(o.outputs[0].token_ids)}) + "\n")
    print(f"parsed {n_ok}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
