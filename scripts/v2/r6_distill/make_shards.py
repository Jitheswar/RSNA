import os, sys, pandas as pd
R="/home/ch.jitheswar/RSNA/scratch_r6"; V2="/home/ch.jitheswar/RSNA/artifacts/private/v2"
n=int(sys.argv[1])
tr=pd.read_csv(f"{V2}/train.csv",dtype=str); ser=pd.read_csv(f"{V2}/train_series.csv",dtype=str)
ss=pd.read_csv(f"{V2}/sample_submission.csv",nrows=1)
ids=tr.StudyInstanceUID.tolist()
for k in range(n):
    inp=f"{R}/in{k}"; root=f"{inp}/rsna-knee-abnormality-detection"; os.makedirs(root+"/test_series",exist_ok=True)
    for d in ["rsna-orthofoundation-fold0-epoch7-20261004","rsna-knee-r6-frozen-fe055406-w1"]:
        if not os.path.exists(f"{inp}/{d}"): os.symlink(f"{R}/input/{d}",f"{inp}/{d}")
    my=ids[k::n]
    pd.DataFrame({"StudyInstanceUID":my}).to_csv(root+"/test.csv",index=False)
    ser[ser.StudyInstanceUID.isin(set(my))].to_csv(root+"/test_series.csv",index=False)
    sub=pd.DataFrame({"StudyInstanceUID":my})
    for c in ss.columns[1:]: sub[c]=0.5
    sub.to_csv(root+"/sample_submission.csv",index=False)
    for st in my:
        p=f"{root}/test_series/{st}"
        if not os.path.exists(p): os.symlink(f"{R}/dcm/train_series/{st}",p)
    for nm in ["train.csv","train_series.csv"]:
        if not os.path.exists(f"{root}/{nm}"): os.symlink(f"{V2}/{nm}",f"{root}/{nm}")
    print(k,len(my))
