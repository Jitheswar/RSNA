"""Summarise every v2 run/fold: best epoch, best weak AUC, gold AUC of that fold, epochs done."""
import glob, json, os
rows = []
for lg in sorted(glob.glob("artifacts/private/v2/runs/*/fold*/log.json")):
    run, fold = lg.split("/")[-3], lg.split("/")[-2]
    try:
        L = json.load(open(lg))
    except Exception:
        continue
    L = [r for r in L if "weak_auc" in r]
    if not L:
        continue
    b = max(L, key=lambda r: r["weak_auc"])
    rows.append((run, fold, len(L), b["epoch"], b["weak_auc"], b.get("gold_auc_fold", float("nan"))))
print(f"{'run':28s} {'fold':6s} {'ep':>3s} {'best':>4s} {'weak':>7s} {'gold':>7s}")
for r in rows:
    print(f"{r[0]:28s} {r[1]:6s} {r[2]:3d} {r[3]:4d} {r[4]:7.4f} {r[5]:7.4f}")
