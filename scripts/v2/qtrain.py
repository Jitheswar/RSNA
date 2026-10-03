"""Queue one GPU task per fold for a training config.

python scripts/v2/qtrain.py <prio> <run_name> <folds e.g. 0,1,2,3,4> -- <train.py args...>
Task files: artifacts/private/v2/queue/pending/<prio><fold>_<run>_f<fold>.task (lower prio string runs first).
"""
import shlex, sys
from pathlib import Path
i = sys.argv.index("--")
prio, run, folds = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
extra = " ".join(shlex.quote(a) for a in sys.argv[i + 1:])
q = Path("artifacts/private/v2/queue/pending")
for f in folds:
    p = q / f"{prio}{f}_{run}_f{f}.task"
    p.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd /home/ch.jitheswar/RSNA\n"
                 f"bash scripts/v2/train_task.sh {run} --folds {f} {extra}\n")
    print("queued", p.name)
