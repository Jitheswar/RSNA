# Kaggle 2xT4 trainer: runs two folds concurrently, one per GPU.
import os, subprocess, sys, shutil
FOLDS = os.environ.get("FOLDS", "0,1").split(",")
VOL = "/kaggle/input/rsna-fast-vols"
COMP = "/kaggle/input/rsna-knee-abnormality-detection"
shutil.copy(f"{VOL}/train.py", "/kaggle/working/train.py")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm==1.0.19"], check=False)
procs = []
for gpu, fold in enumerate(FOLDS):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [sys.executable, "/kaggle/working/train.py", "--fold", fold, "--fp16", "--workers", "2",
           "--vol-root", VOL, "--labels", f"{VOL}/llm_labels_v2.csv",
           "--train-csv", f"{COMP}/train.csv", "--series-csv", f"{COMP}/train_series.csv",
           "--out", "/kaggle/working/out"] + sys.argv[1:]
    procs.append(subprocess.Popen(cmd, env=env, stdout=open(f"/kaggle/working/log{fold}.txt", "w"),
                                  stderr=subprocess.STDOUT))
for p in procs:
    p.wait()
for fold in FOLDS:
    print(open(f"/kaggle/working/log{fold}.txt").read()[-3000:])
