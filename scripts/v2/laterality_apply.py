"""Apply the laterality checker to every training study; compare with report-stated side where available."""
import glob, json, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "scripts/v2")
from laterality_clf import net, slices_for
V2 = "artifacts/private/v2"
torch.set_num_threads(32)
fr = []
for m in sorted(glob.glob(f"{V2}/vol384/meta-train-*.csv")):
    try: fr.append(pd.read_csv(m))
    except Exception: pass
meta = pd.concat(fr).drop_duplicates("SeriesInstanceUID", keep="last")
model = net(); model.load_state_dict(torch.load(f"{V2}/lat_clf.pt")); model.eval()
rows = []
for st, g in meta[meta.plane == "Coronal"].groupby("StudyInstanceUID"):
    se = g.sort_values("depth", ascending=False).iloc[0]
    vol = np.load(f"{V2}/vol384/vol/{se.SeriesInstanceUID}.npy", mmap_mode="r")
    with torch.no_grad():
        p = torch.sigmoid(model(slices_for(vol)[None])[0, 0]).item()
    rows.append((st, se.side, se.side_src, p))
d = pd.DataFrame(rows, columns=["StudyInstanceUID", "side", "side_src", "p_flipped"]).set_index("StudyInstanceUID")
d.to_csv(f"{V2}/lat_all.csv")
print(d.groupby("side_src").p_flipped.describe(percentiles=[.5, .9, .99]).round(3))
for thr in [0.5, 0.9, 0.99]:
    print(thr, d[d.p_flipped > thr].side_src.value_counts().to_dict())
