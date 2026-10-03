"""Build per-fold-pair Kaggle runner scripts with train.py embedded."""
import base64, json, sys
from pathlib import Path
here = Path(__file__).parent
src = base64.b64encode(Path("scripts/fast/train.py").read_bytes()).decode()
run = (here / "run.py").read_text()
for folds in sys.argv[1:]:
    tag = folds.replace(",", "")
    out = here / f"k{tag}"
    out.mkdir(exist_ok=True)
    body = run.replace('shutil.copy(f"{VOL}/train.py", "/kaggle/working/train.py")',
                       f'import base64\nopen("/kaggle/working/train.py","wb").write(base64.b64decode("{src}"))')
    body = body.replace('os.environ.get("FOLDS", "0,1")', f'os.environ.get("FOLDS", "{folds}")')
    (out / "run.py").write_text(body)
    meta = json.loads((here / "kernel-metadata.json").read_text())
    meta["id"] = f"jitheswar/rsna-fast-train-f{tag}"
    meta["title"] = f"rsna-fast-train-f{tag}"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(out)
