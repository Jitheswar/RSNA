"""Severity-aware report labels from a frontier LLM via the Codex CLI (rules s.8.2: LLM subscriptions allowed).

Batches reports, enforces a strict JSON schema, runs read-only/ephemeral in an empty working directory so the
model only sees report text. Output JSONL rows: {StudyInstanceUID, parsed, model, effort}.

python scripts/v2/frontier_label.py --model gpt-6.1-sol --effort high --batch 20 --workers 6 \
    --out artifacts/private/v2/llm/sol61-high.jsonl [--gold-first] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_label import SYSTEM, TARGETS  # noqa: E402

CODEX = str(Path.home() / ".local/bin/codex")


def schema():
    fobj = {"type": "object", "additionalProperties": False, "required": ["ev", "g", "p"],
            "properties": {"ev": {"type": "string"}, "g": {"type": "integer"}, "p": {"type": "integer"}}}
    item = {"type": "object", "additionalProperties": False,
            "required": ["id", "side", "postop"] + TARGETS,
            "properties": {"id": {"type": "string"}, "side": {"type": "string", "enum": ["left", "right", "unknown"]},
                           "postop": {"type": "boolean"}, **{t: fobj for t in TARGETS}}}
    return {"type": "object", "additionalProperties": False, "required": ["results"],
            "properties": {"results": {"type": "array", "items": item}}}


def build_prompt(batch):
    head = SYSTEM.split("Answer with only one JSON object")[0]
    parts = [head.strip(), "",
             f"You are given {len(batch)} independent reports, each with an id. Label every report separately "
             "(do not let one report influence another). Do not run any commands or tools; just answer.",
             "Return a JSON object {\"results\": [...]} with exactly one entry per report, in the given order, "
             "each entry holding \"id\", \"side\", \"postop\" and the 12 target objects {\"ev\",\"g\",\"p\"}.", ""]
    for rid, rep in batch:
        parts.append(f"### id={rid}\n<<<\n{rep.strip()}\n>>>\n")
    return "\n".join(parts)


def run_batch(batch_rows, model, effort, schema_path, timeout):
    ids = [f"r{i + 1}" for i in range(len(batch_rows))]
    prompt = build_prompt([(rid, rep) for rid, (_, rep) in zip(ids, batch_rows)])
    with tempfile.TemporaryDirectory(prefix="fl-") as td:
        out = Path(td) / "out.json"
        cmd = [CODEX, "exec", "-m", model, "-c", f'model_reasoning_effort="{effort}"', "-s", "read-only",
               "--ephemeral", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules", "-C", td,
               "--output-schema", schema_path, "-o", str(out), "--color", "never", "-"]
        t0 = time.time()
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
        dt = time.time() - t0
        if not out.exists():
            raise RuntimeError(f"no output (rc={p.returncode}): {p.stderr[-800:]}")
        obj = json.loads(out.read_text())
    res = {r["id"]: r for r in obj.get("results", [])}
    rows = []
    for rid, (uid, _) in zip(ids, batch_rows):
        r = res.get(rid)
        rows.append({"StudyInstanceUID": uid, "parsed": r, "model": model, "effort": effort, "batch_s": round(dt, 1)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-6.1-sol")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", default="artifacts/private/v2/train.csv")
    ap.add_argument("--gold-first", action="store_true")
    ap.add_argument("--gold-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()
    import pandas as pd
    df = pd.read_csv(args.train)
    if args.gold_only:
        df = df[df.ACL.notna()]
    elif args.gold_first:
        df = pd.concat([df[df.ACL.notna()], df[df.ACL.isna()]])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r.get("parsed"):
                done.add(r["StudyInstanceUID"])
    todo = df[~df.StudyInstanceUID.isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    rows = list(zip(todo.StudyInstanceUID, todo.Report))
    batches = [rows[i:i + args.batch] for i in range(0, len(rows), args.batch)]
    print(f"{len(done)} done; {len(rows)} to label in {len(batches)} batches", flush=True)
    sp = out.with_suffix(".schema.json")
    sp.write_text(json.dumps(schema()))
    lock = threading.Lock()
    n_ok = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(run_batch, b, args.model, args.effort, str(sp), args.timeout): i for i, b in enumerate(batches)}
        for k, f in enumerate(as_completed(futs)):
            try:
                res = f.result()
            except Exception as exc:
                print(f"batch {futs[f]} failed: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
                continue
            with lock, out.open("a") as fh:
                for r in res:
                    fh.write(json.dumps(r) + "\n")
                    n_ok += r["parsed"] is not None
            print(f"[{k + 1}/{len(batches)}] ok={n_ok} elapsed={time.time() - t0:.0f}s batch_s={res[0]['batch_s']}", flush=True)
    print(f"finished: {n_ok}/{len(rows)} parsed", flush=True)


if __name__ == "__main__":
    main()
