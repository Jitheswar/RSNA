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
print("inputs", VOL, GOLD, COMP, flush=True)
import torch, torch_xla, timm
print("torch", torch.__version__, "xla", torch_xla.__version__, "timm", timm.__version__, flush=True)
for fold in "{folds}".split(","):
    t0 = time.time()
    cmd = [sys.executable, "-u", "/kaggle/working/src/train_tpu.py", "--fold", fold,
           "--vol-root", f"{{VOL}},{{GOLD}}", "--labels", f"{{VOL}}/llm_labels_v2.csv",
           "--train-csv", f"{{COMP}}/train.csv", "--series-csv", f"{{COMP}}/train_series.csv",
           "--out", "/kaggle/working/out"] + {extra!r}
    r = subprocess.run(cmd, cwd="/kaggle/working/src")
    print(f"fold {{fold}} exit {{r.returncode}} in {{time.time() - t0:.0f}}s", flush=True)
'''
out = here / suffix
out.mkdir(exist_ok=True)
(out / "run.py").write_text(run)
meta = {"id": f"jitheswar/rsna-fast-tpu-{suffix}", "title": f"rsna-fast-tpu-{suffix}", "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
        "enable_tpu": True, "enable_internet": True, "machine_shape": "TpuV5E8",
        "dataset_sources": ["jitheswar/rsna-fast-vols", "jitheswar/rsna-fast-gold"],
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
print(out)
