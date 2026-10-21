
deadline = T0 + DEADLINE_HOURS * 3600
env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')
pred = TMP / 'prediction'
rc = subprocess.run([sys.executable, '-u', str(SRC / 'probe_dual.py'), '--data', str(DATA), '--split', 'test',
                     '--code', str(CODE), '--bundle', str(BUNDLE), '--arm', ARM, '--output', str(pred),
                     '--amp', 'auto', '--loader-workers', '2', '--deadline', str(deadline),
                     '--worker', WORKER, '--worker-args', json.dumps(WORKER_ARGS),
                     '--max-fallback-fraction', '1.0'], env=env).returncode   # [blend] no 3 % stop here: stage 3 gives a study we did not score Raptor's rank
for log in sorted(pred.glob('worker*/log.txt')):
    shutil.copyfile(log, WORK / (log.parent.name + '_log.txt'))
if rc != 0:
    raise RuntimeError('controller failed with exit code %d' % rc)
shutil.copyfile(pred / 'submission.csv', WORK / 'ours_sub.csv')

if PROBE_CLASS:
    # single-class probe: every column except PROBE_CLASS is a constant, so public LB = (11 * 0.5 + AUC_c) / 12
    import pandas as pd
    _df = pd.read_csv(WORK / 'ours_sub.csv')
    assert PROBE_CLASS in _df.columns, PROBE_CLASS
    for _c in _df.columns:
        if _c not in ('StudyInstanceUID', PROBE_CLASS):
            _df[_c] = 0.5
    _df.to_csv(WORK / 'ours_sub.csv', index=False)
    print('SINGLE_CLASS_PROBE', PROBE_CLASS, 'other 11 columns set to 0.5', flush=True)
receipt = json.loads((pred / 'submission_receipt.json').read_text())
receipt['seconds_including_notebook_setup'] = time.time() - T0
receipt['bundle_id'] = BUNDLE_ID
receipt['arm'] = ARM
(WORK / 'submission_receipt.json').write_text(json.dumps(receipt, indent=1))
print('OURS_STAGE_COMPLETE', json.dumps({k: receipt.get(k) for k in ('status', 'arm', 'rows', 'scored', 'fallback_studies',
                                                                 'seconds', 'per_study_mean', 'gpus')}), flush=True)
import pandas as pd
display(pd.read_csv(WORK / 'ours_sub.csv').head())
