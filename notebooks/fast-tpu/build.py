"""Build a Kaggle TPU v5e-8 runner with train.py and train_tpu.py embedded.

usage: build.py <slug-suffix> <folds csv> [extra train_tpu args...]
"""
import base64, json, sys
from pathlib import Path
here = Path(__file__).parent
suffix, folds, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
files = {n: base64.b64encode(Path(f"scripts/fast/{n}").read_bytes()).decode() for n in ("train.py", "train_tpu.py")}
run = f'''# Kaggle TPU v5e-8 fold runner (streams logs live).
import base64, glob, os, subprocess, sys, time
os.makedirs("/kaggle/working/src", exist_ok=True)
for name, blob in {files!r}.items():
    open(f"/kaggle/working/src/{{name}}", "wb").write(base64.b64decode(blob))
def _find(name, marker):
    for pat in [f"/kaggle/input/{{name}}", f"/kaggle/input/*/{{name}}", f"/kaggle/input/*/*/{{name}}"]:
        for c in sorted(glob.glob(pat)):
            if os.path.exists(os.path.join(c, marker)):
                return c
    raise FileNotFoundError(name)
VOL, GOLD = _find("rsna-fast-vols", "index.csv"), _find("rsna-fast-gold", "index.csv")
COMP = _find("rsna-knee-abnormality-detection", "train.csv")
os.environ["TIMM_WEIGHTS_DIR"] = _find("rsna-timm-weights", "convnext_nano.in12k_ft_in1k.safetensors")
os.environ["HF_HUB_OFFLINE"] = "1"
print("inputs", VOL, GOLD, COMP, flush=True)
import torch, torch_xla, timm
print("torch", torch.__version__, "xla", torch_xla.__version__, "timm", timm.__version__, flush=True)
T_START = time.time()
BUDGET = 8.4 * 3600
def _run(fold, out, extra_args):
    cmd = [sys.executable, "-u", "/kaggle/working/src/train_tpu.py", "--fold", str(fold),
           "--vol-root", f"{{VOL}},{{GOLD}}", "--labels", f"{{VOL}}/llm_labels_v2.csv",
           "--train-csv", f"{{COMP}}/train.csv", "--series-csv", f"{{COMP}}/train_series.csv",
           "--out", out] + extra_args
    t0 = time.time()
    r = subprocess.run(cmd, cwd="/kaggle/working/src")
    print(f"fold {{fold}} -> {{out}} exit {{r.returncode}} in {{time.time() - t0:.0f}}s", flush=True)
    return r.returncode, time.time() - t0
smoke_args = {extra!r}[:]
for flag, val in (("--epochs", "1"), ("--limit", "64")):
    if flag in smoke_args:
        smoke_args[smoke_args.index(flag) + 1] = val
    else:
        smoke_args += [flag, val]
rc, _ = _run(0, "/kaggle/working/smoke", smoke_args)
if rc != 0:
    raise SystemExit("in-session smoke failed; not starting full folds")
longest = 0.0
for fold in "{folds}".split(","):
    left = BUDGET - (time.time() - T_START)
    if longest and left < 1.1 * longest:
        print(f"skip fold {{fold}}: {{left:.0f}}s left < 1.1 x {{longest:.0f}}s", flush=True)
        continue
    rc, took = _run(fold, "/kaggle/working/out", {extra!r})
    longest = max(longest, took)
'''
out = here / suffix
out.mkdir(exist_ok=True)
(out / "run.py").write_text(run)
meta = {"id": f"jitheswar/rsna-fast-tpu-{suffix}", "title": f"rsna-fast-tpu-{suffix}", "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
        "enable_tpu": True, "enable_internet": False, "machine_shape": "TpuV5E8",
        "dataset_sources": ["jitheswar/rsna-fast-vols", "jitheswar/rsna-fast-gold", "jitheswar/rsna-timm-weights"],
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
print(out)
