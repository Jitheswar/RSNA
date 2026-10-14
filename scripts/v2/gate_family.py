"""Exit 0 if adding family RUN (its finished folds) to s_mm2 + s_mm2_384 does not lower gold58 within-fold AUC."""
import sys
sys.path.insert(0, "scripts/v2")
import glob
import pandas as pd
import mm_eval as m

run = sys.argv[1]
tr = pd.read_csv(m.V2 / "train.csv").set_index("StudyInstanceUID")
g0 = tr[tr.ACL.notna()][m.T].astype(int)
fs = sorted(glob.glob(str(m.V2 / "runs" / run / "fold*/oof.csv")))
new = pd.concat([pd.read_csv(f) for f in fs]).set_index("StudyInstanceUID")[m.T]
idx = g0.index.intersection(new.index)
g = g0.loc[idx]
fo = pd.read_csv(m.V2 / "folds.csv").set_index("StudyInstanceUID")["fold"].loc[idx].values
base = [m.oof(r).loc[idx] for r in ("s_mm2", "s_mm2_384")]
a = m.wf_auc(sum(base) / 2, g, fo)
b = m.wf_auc((sum(base) + new.loc[idx]) / 3, g, fo)
print(f"{run}: n={len(idx)} base {a:.4f} +family {b:.4f}")
sys.exit(0 if b >= a else 1)
