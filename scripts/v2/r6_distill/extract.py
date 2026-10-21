import zipfile, sys, os
from concurrent.futures import ProcessPoolExecutor
ZIP="/home/ch.jitheswar/RSNA/rsna-knee-abnormality-detection.zip"; OUT="/home/ch.jitheswar/RSNA/scratch_r6/dcm"
def work(k, n):
    z=zipfile.ZipFile(ZIP); c=0
    for i,info in enumerate(z.infolist()):
        nm=info.filename
        if not nm.startswith("train_series/") or nm.endswith("/"): continue
        if hash(nm.split("/")[1]) % n != k: continue
        p=os.path.join(OUT,nm)
        if os.path.exists(p) and os.path.getsize(p)==info.file_size: continue
        os.makedirs(os.path.dirname(p),exist_ok=True)
        with z.open(info) as s, open(p,"wb") as d: d.write(s.read())
        c+=1
    return c
if __name__=="__main__":
    os.environ["PYTHONHASHSEED"]="0"
    n=int(sys.argv[1])
    with ProcessPoolExecutor(n) as ex: print(sum(ex.map(work, range(n), [n]*n)))
