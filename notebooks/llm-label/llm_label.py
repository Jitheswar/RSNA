"""Score report findings with a local LLM via yes/no next-token log-probabilities.

For each report the shared prefix (instructions + report) is encoded once; each
of the twelve findings is then scored by P(" Sí") / (P(" Sí") + P(" No")) after
a short question suffix, reusing the prefix KV cache. CPU-friendly; shardable.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
QUESTIONS = {
    "ACL": "¿Hay rotura (completa o parcial) del ligamento cruzado anterior?",
    "MCL": "¿Hay lesión o rotura (incluido esguince de grado I-III) del ligamento colateral medial / interno?",
    "Medial Meniscus": "¿Hay rotura del menisco interno / medial?",
    "Lateral Meniscus": "¿Hay rotura del menisco externo / lateral?",
    "Medial OA": "¿Hay artrosis o condropatía del compartimento femorotibial medial / interno?",
    "Lateral OA": "¿Hay artrosis o condropatía del compartimento femorotibial lateral / externo?",
    "PF OA": "¿Hay artrosis o condropatía femoropatelar / rotuliana?",
    "Effusion": "¿Hay derrame articular?",
    "Synovitis": "¿Hay sinovitis o engrosamiento sinovial?",
    "Baker's": "¿Hay quiste de Baker / poplíteo?",
    "Contusion": "¿Hay contusión ósea o edema óseo?",
    "Fracture": "¿Hay fractura (incluida fractura por insuficiencia o estrés)?",
}
SYSTEM = ("Eres un radiólogo musculoesquelético experto. Lee el informe de resonancia magnética "
          "de rodilla y responde a la pregunta solo con Sí o No.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--gold-only", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--cuda", action="store_true", help="fp16, sharded over visible GPUs")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.cuda:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    dev = model.get_input_embeddings().weight.device
    model.eval()
    yes = tok.encode("Sí", add_special_tokens=False)[0]
    no = tok.encode("No", add_special_tokens=False)[0]

    df = pd.read_csv(args.train_csv)
    if args.gold_only:
        df = df.dropna(subset=["ACL"])
    df = df.iloc[args.shard :: args.nshards]
    done = set()
    if args.out.exists():
        done = set(pd.read_csv(args.out).StudyInstanceUID)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(",".join(["StudyInstanceUID", *[f'"{t}"' for t in TARGETS]]) + "\n")

    for row in df.itertuples():
        if row.StudyInstanceUID in done:
            continue
        t0 = time.time()
        fids = [tok(tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": f"Informe:\n{row.Report}\n\nPregunta: {QUESTIONS[t]}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False), return_tensors="pt").input_ids[0]
            for t in TARGETS]
        n = min(len(f) for f in fids)
        common = 0
        while common < n - 1 and all(f[common] == fids[0][common] for f in fids):
            common += 1
        with torch.no_grad():
            cache = model(fids[0][None, :common].to(dev), use_cache=True).past_key_values
        probs = []
        for fid in fids:
            with torch.no_grad():
                logits = model(fid[None, common:].to(dev), past_key_values=copy.deepcopy(cache), use_cache=True).logits[0, -1]
            lp = torch.log_softmax(logits.float(), -1)
            probs.append(float(torch.sigmoid(lp[yes] - lp[no])))
        with args.out.open("a") as f:
            f.write(",".join([row.StudyInstanceUID, *[f"{p:.6f}" for p in probs]]) + "\n")
        print(f"{row.StudyInstanceUID[-8:]} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
