"""Prepare a verified sidecar without changing/copying original CPU MRI arrays."""
from datetime import datetime,timezone
from pathlib import Path
import sys

PROJECT = next(p for p in Path(__file__).resolve().parents if (p/'.git').exists())
sys.path.insert(0,str(PROJECT/'src'))
from prepare_ticket77_portable_reference import CPU,CPU_SHA,verified_sources
from ticket75_cpu_reference import digest,read,require,write
from rsna_knee.tpu_pilot_bank import FIELDS,SCHEMA,FrozenPilotInputBank

ROOT = PROJECT/'artifacts/private/ticket75-input-bank-manifest-v2'


def prepare():
    require(not ROOT.exists(), 'Unused immutable input-bank preparation required; never retry partial preparation')
    # This checks actual successful CPU terminal, exact accepted request/job,
    # source/result/checkpoint/MRI hashes and all original preparation bindings.
    # It must pass before any preparation directory or receipt is created.
    _,result = verified_sources()
    bank_root = CPU/'attachment'
    selection_path = bank_root/'selection.json'
    require(digest(selection_path) == digest(PROJECT/'artifacts/private/ticket75-real-mri-selection-v1/selection.json'),
        'Exact original selection bytes required; never reconstruct/resample the subset')
    bound = {str(Path(row['path']).resolve()):row['sha256'] for row in result['attachment_files']}
    rows = []
    for index in range(64):
        for field in FIELDS:
            relative = f'studies/{index:02d}/{field}.npy'
            path = bank_root/relative
            require(path.resolve().is_relative_to(bank_root.resolve()) and str(path.resolve()) in bound
                and digest(path) == bound[str(path.resolve())], 'Exact successful native CPU array bytes required')
            rows.append(dict(study_index=index,field=field,path=relative,sha256=bound[str(path.resolve())],bytes=path.stat().st_size))
    require(digest(selection_path) == bound[str(selection_path.resolve())], 'Original selected-byte inventory changed')
    sources = [Path(__file__),PROJECT/'src/rsna_knee/tpu_pilot_bank.py',PROJECT/'src/rsna_knee/tpu_pilot_inputs.py',
        CPU/'plan.json',CPU/'reference-result.json',CPU/'terminal-verification.json',CPU/'accepted-launch.json',
        CPU/'source-replay.json',selection_path,PROJECT/'scripts/prepare_ticket77_portable_reference.py']
    bindings = [dict(path=str(path),sha256=digest(path)) for path in sources]
    ROOT.mkdir(mode=0o700)
    path = ROOT/'input-bank-manifest.json'
    write(path,dict(schema_version=SCHEMA,selection=dict(path='selection.json',sha256=digest(selection_path)),arrays=rows))
    checksum = digest(path)
    bank = FrozenPilotInputBank(bank_root,manifest_path=path,expected_manifest_sha256=checksum)
    final = bank.verify_final_bytes()
    require(all(digest(row['path']) == row['sha256'] for row in bindings), 'Source/terminal bindings changed during sidecar verification')
    write(ROOT/'prepared-validation.json',dict(status='COMPLETE',created_at_utc=datetime.now(timezone.utc).isoformat(),
        original_CPU_plan_sha256=CPU_SHA,original_CPU_job_id=read(CPU/'accepted-launch.json')['job_id'],
        manifest_path=str(path),manifest_sha256=checksum,bank_root=str(bank_root),source_bindings=bindings,
        all384_lossless_fields_verified=True,actual_bank_array_bytes=bank.array_bytes,final_byte_validation=final,
        original_MRI_replayed_by_verified_CPU_job=64,original_arrays_copied=False,original_array_bytes_modified=False,
        final_attachment_prepared=False,uploaded=False,accelerator_launch_admitted=False,
        actual_CPU28_execution=False,actual_TPU_execution=False,competition_submission=False,candidate_promotion=False))
    print(dict(status='INPUT_BANK_SIDECAR_ONLY',manifest_sha256=checksum,actual_array_bytes=bank.array_bytes))


if __name__ == '__main__':
    prepare()
