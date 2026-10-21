# Local (cluster) run of heliosli's public R6 stage on a shard of TRAIN studies posing as "test".
# cell3.py / cell4.py: stages 3-4 of heliosli's public R6 notebook (notebooks/v2/rsna-r6-only), unchanged.
import sys, os, time, builtins
k=sys.argv[1]; R="/home/ch.jitheswar/RSNA/scratch_r6"
NB_T0=time.time(); builtins.display=print
src=open(f"{R}/cell3.py").read()+"\n"+open(f"{R}/cell4.py").read()
src=src.replace("INPUT = Path('/kaggle/input')",f"INPUT = Path('{R}/in{k}')")
src=src.replace("WORK = Path('/kaggle/working')",f"WORK = Path('{R}/work{k}'); WORK.mkdir(exist_ok=True)")
src=src.replace("TMP = Path('/kaggle/temp') if Path('/kaggle/temp').is_dir() else Path('/tmp')",f"TMP = Path(os.environ.get('R6_TMP','{R}/tmp{k}'))")
src=src.replace("DEADLINE_HOURS = 8.3","DEADLINE_HOURS = 30")
if os.environ.get("SKIP_WHEELS"):
    src=src.replace("subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *wheels], check=True)","print('skip wheels', len(wheels))")
src=src.replace("'--loader-workers', '2'","'--loader-workers', '6'")
exec(compile(src,"r6stage","exec"))
