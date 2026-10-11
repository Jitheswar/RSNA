# RSNA v2: fine-tune the full-data ConvNeXt-S student on multimodal-mixed labels, preparing studies from raw DICOMs
# on the fly (identical to the submission's preparation path). Output: /kaggle/working/run/full/last.pt
import glob, os, shutil, subprocess, sys, zipfile
src = os.path.dirname([p for p in glob.glob('/kaggle/input/**/rawprep.py', recursive=True)][0])
comp = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/train_series.csv', recursive=True)
        if os.path.isdir(os.path.join(os.path.dirname(p), 'train_series'))][0]
code = '/kaggle/working/code'; os.makedirs(code, exist_ok=True)
for f in glob.glob(src + '/*'):
    if os.path.isfile(f): shutil.copy(f, code)
meta_dir = '/kaggle/working/meta'
if os.path.isdir(src + '/meta'): shutil.copytree(src + '/meta', meta_dir, dirs_exist_ok=True)
elif os.path.exists(src + '/meta.zip'): zipfile.ZipFile(src + '/meta.zip').extractall(meta_dir)
print('src', src, 'comp', comp, 'meta files', len(glob.glob(meta_dir + '/**/*.csv', recursive=True)), flush=True)
meta_glob = sorted(glob.glob(meta_dir + '/**/meta-train-*.csv', recursive=True))
cmd = [sys.executable, code + '/train.py', '--comp', comp, '--vol-dir', '/nonexistent',
       '--meta', os.path.dirname(meta_glob[0]) + '/meta-train-*.csv', '--series-csv', code + '/train_series.csv',
       '--labels', code + '/stud_mm1.csv', '--folds-csv', code + '/folds.csv', '--train-csv', code + '/train.csv',
       '--aux-grades', code + '/aux_grades_v2.csv', '--init-ckpt', code + '/init_s_mix1e_full_s1.pt',
       '--out', '/kaggle/working/run', '--full', '--seed', '45', '--amp', 'fp16',
       '--backbone', 'convnext_small.fb_in22k_ft_in1k_384', '--img', '320', '--k', '12', '--k-sag', '24', '--k-cor', '16',
       '--k-ax', '14', '--pos-weight', '--n-series', '6', '--epochs', '3', '--lr-enc', '3e-5', '--lr-head', '1.5e-4',
       '--bs', '2', '--accum', '4', '--workers', '4', '--micro', '16']
print(' '.join(cmd), flush=True)
p = subprocess.run(cmd, cwd=code)
print('exit', p.returncode, flush=True)
for f in glob.glob(code + '/*.pt'): os.remove(f)  # keep the output small
