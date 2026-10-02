"""Freeze additive runtime, original-MRI byte and full-member proof for Ticket70."""

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
PARENT = PROJECT / "artifacts/private/ticket70-scored-anchor-gold58-v2"
DEST = PROJECT / "artifacts/private/ticket70-scored-anchor-gold58-v3"
INPUT = PROJECT / "artifacts/private/public-frontier-adoption/full-cpu-input-probe-v2"
PARENT_SHA = "6f40239b3e5e57b7534bdbc97be7b84e92efaa2226d5330e9bbf0d67b690a886"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def input_aggregate(rows):
    """Bind sorted relative names, lengths and original extraction content hashes."""
    names = [row[0] for row in rows]
    if not rows or len(set(names)) != len(names):
        raise ValueError("Original DICOM membership must be nonempty and unique")
    for name, size, checksum in rows:
        parts = PurePosixPath(name).parts
        if (len(parts) != 2 or name != "/".join(parts) or any(p in (".", "..", "") for p in parts)
                or "\\" in name or "\0" in name or not name.endswith(".dcm")
                or type(size) is not int or size <= 0 or not re.fullmatch("[0-9a-f]{64}", checksum)):
            raise ValueError("Unsafe or malformed original DICOM identity")
    payload = "".join(canonical(row) + "\n" for row in sorted(rows))
    return {"dicom_files": len(rows), "dicom_bytes": sum(row[1] for row in rows),
            "aggregate_sha256": hashlib.sha256(payload.encode()).hexdigest()}


def recipe_constants(code):
    result = {}
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in ("ORIGINAL_CELLS", "SOURCE_CELLS"):
                if key in result:
                    raise ValueError("Ambiguous frozen recipe constant")
                result[key] = ast.literal_eval(node.value)
    if set(result) != {"ORIGINAL_CELLS", "SOURCE_CELLS"} or any(len(v) != 5 for v in result.values()):
        raise ValueError("Exactly five original and relocated cells required")
    return result


def replace_once(code, old, new):
    if code.count(old) != 1:
        raise ValueError("Runtime instrumentation insertion is missing or ambiguous")
    return code.replace(old, new)


def instrument(code, inputs):
    original = recipe_constants(code)
    code = replace_once(code, "    view = Path('/tmp/rsna_exact_scored_gold58')\n",
        "    _rsna70_runtime = _rsna70_runtime_receipt(torch)\n"
        "    _rsna70_inputs = _rsna70_verify_inputs(comp, ids, sha)\n"
        "    view = Path('/tmp/rsna_exact_scored_gold58')\n")
    code = replace_once(code, "    def profile(frame,event,arg):\n",
        "    def profile(frame,event,arg):\n        _rsna70_capture_dino(frame,event,arg)\n")
    start = code.index("    class Monitor:\n")
    end = code.index("    sys.setprofile(profile)\n", start)
    code = code[:start] + "    stdout, stderr = sys.stdout, sys.stderr\n    monitor = _Rsna70Monitor(stdout, 'stdout')\n    error_monitor = _Rsna70Monitor(stderr, 'stderr')\n    sys.stdout, sys.stderr = monitor, error_monitor\n" + code[end:]
    code = replace_once(code, "        sys.stdout = stdout\n        torch.load = original_load\n",
        "        monitor.finish()\n        error_monitor.finish()\n"
        "        sys.stdout, sys.stderr = stdout, stderr\n        torch.load = original_load\n")
    code = replace_once(code, "    if not globals().get('_RSNA_FULL_ENSEMBLE_COMPLETE')",
        "    _rsna70_execution = _rsna70_verify_execution()\n"
        "    if not globals().get('_RSNA_FULL_ENSEMBLE_COMPLETE')")
    code = replace_once(code, "        'loaded_weights':loaded,'coat_receipt':coat_receipt,'fallback_events':monitor.fallbacks,\n",
        "        'loaded_weights':loaded,'coat_receipt':coat_receipt,'fallback_events':monitor.fallbacks,\n"
        "        'runtime_receipt':_rsna70_runtime,'input_receipt':_rsna70_inputs,\n"
        "        'execution_receipt':_rsna70_execution,\n")
    result = "_RSNA70_EXPECTED_INPUTS = " + repr(inputs) + "\n" + RUNTIME_PROOF + "\n" + code
    if recipe_constants(result) != original:
        raise ValueError("Runtime instrumentation changed frozen scientific cells")
    compile(result, "ticket70-verified-runtime-v3", "exec")
    return result


RUNTIME_PROOF = r'''
import hashlib as _rsna70_hashlib
import importlib.metadata as _rsna70_metadata
import json as _rsna70_json
import os as _rsna70_os
import platform as _rsna70_platform
import re as _rsna70_re
import threading as _rsna70_threading
import time as _rsna70_time
from pathlib import Path as _Rsna70Path
_rsna70_work = _Rsna70Path('/kaggle/working')
_rsna70_log = _rsna70_work / 'development-execution.jsonl'
_rsna70_dino = []
_RSNA70_REJECT_PATTERNS = (
    r'MEMBER .* failed', r'dropped -- costs one vote',
    r'degenerate predictions; not banked', r'public-frontier vote omitted',
    r'surrendering .* member', r'public-frontier fallback not emitted',
    r'study failed:', r'CoAtNet branch failed', r'legacy checkpoint disabled',
    r'legacy complement unavailable', r'\[DINOsaur V4\.2\].*FALLBACK',
    r'legacy study .* fallback:', r'calibration skipped', r'final fusion failed',
    r'CoAtNet output unavailable')
def _rsna70_canonical(value):
    return _rsna70_json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def _rsna70_rejection(line):
    return next((pattern for pattern in _RSNA70_REJECT_PATTERNS if _rsna70_re.search(pattern,line)),None)
class _Rsna70Monitor:
    def __init__(self,stream,name):
        self.stream=stream; self.name=name; self.tail=''; self.fallbacks=0
        self.lock=_rsna70_threading.RLock()
    def __getattr__(self,name): return getattr(self.stream,name)
    def _record(self,line):
        reason=_rsna70_rejection(line)
        if reason: self.fallbacks+=1
        row={'stream':self.name,'line':line,'rejection':reason}
        data=(_rsna70_canonical(row)+'\n').encode()
        descriptor=_rsna70_os.open(str(_rsna70_log),_rsna70_os.O_WRONLY|_rsna70_os.O_CREAT|_rsna70_os.O_APPEND,0o600)
        try:
            if _rsna70_os.write(descriptor,data)!=len(data):
                raise RuntimeError('Execution proof log write was incomplete')
        finally: _rsna70_os.close(descriptor)
    def write(self,text):
        with self.lock:
            self.tail+=text
            while '\n' in self.tail:
                line,self.tail=self.tail.split('\n',1)
                self._record(line)
            return self.stream.write(text)
    def flush(self): return self.stream.flush()
    def finish(self):
        with self.lock:
            if self.tail: self._record(self.tail); self.tail=''
            self.stream.flush()
def _rsna70_runtime_receipt(torch):
    packages={name:_rsna70_metadata.version(name) for name in
        ('numpy','pandas','pydicom','torch','torchvision','timm','scipy','Pillow')}
    for name in ('opencv-python','opencv-python-headless'):
        try: packages[name]=_rsna70_metadata.version(name)
        except _rsna70_metadata.PackageNotFoundError: packages[name]=None
    record={'schema_version':'rsna-ticket70-actual-runtime-v3','python':_rsna70_platform.python_version(),
        'packages':packages,'torch_cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'gpu_count':torch.cuda.device_count(),'gpus':[{'index':i,'name':torch.cuda.get_device_name(i),
        'capability':list(torch.cuda.get_device_capability(i)),
        'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]}
    path=_rsna70_work/'development-runtime.json'
    path.write_text(_rsna70_json.dumps(record,indent=2,sort_keys=True)+'\n')
    return {'file':path.name,'sha256':_rsna70_hashlib.sha256(path.read_bytes()).hexdigest()}
def _rsna70_verify_inputs(comp,ids,sha):
    started=_rsna70_time.monotonic()
    expected=_RSNA70_EXPECTED_INPUTS
    if ids!=[row['study_uid'] for row in expected['studies']]:
        raise RuntimeError('Frozen original-MRI study order changed')
    actual=[]
    base=comp/'train_series'
    for entry in expected['studies']:
        root=base/entry['study_uid']; rows=[]
        for path in sorted(root.rglob('*.dcm')):
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise RuntimeError('Unsafe original MRI source')
            relative=path.relative_to(root).as_posix()
            if len(path.relative_to(root).parts)!=2:
                raise RuntimeError('Original MRI directory layout changed')
            rows.append([relative,path.stat().st_size,sha(path)])
        payload=''.join(_rsna70_canonical(row)+'\n' for row in sorted(rows))
        observed={'study_uid':entry['study_uid'],'dicom_files':len(rows),'dicom_bytes':sum(row[1] for row in rows),
            'aggregate_sha256':_rsna70_hashlib.sha256(payload.encode()).hexdigest()}
        if observed!=entry: raise RuntimeError('Original MRI DICOM bytes/membership changed')
        actual.append(observed)
    identity={'schema_version':expected['schema_version'],'studies':actual}
    identity_sha=_rsna70_hashlib.sha256(_rsna70_canonical(identity).encode()).hexdigest()
    if identity_sha!=expected['identity_sha256']: raise RuntimeError('Original MRI aggregate identity changed')
    record={'schema_version':'rsna-ticket70-original-mri-byte-proof-v3','status':'COMPLETE',
        'study_count':len(actual),'dicom_files':sum(row['dicom_files'] for row in actual),
        'dicom_bytes':sum(row['dicom_bytes'] for row in actual),'identity_sha256':identity_sha,
        'studies':actual,'hash_elapsed_seconds':_rsna70_time.monotonic()-started,
        'source_basis':'T59 original ZIP extraction SHA256 and CRC-checked reads; complete mounted bytes freshly hashed before inference',
        'target_labels_uploaded':False}
    path=_rsna70_work/'development-original-mri-proof.json'
    path.write_text(_rsna70_json.dumps(record,indent=2,sort_keys=True)+'\n')
    return {'file':path.name,'sha256':sha(path),'identity_sha256':identity_sha,
        'dicom_files':record['dicom_files'],'dicom_bytes':record['dicom_bytes']}
def _rsna70_capture_dino(frame,event,arg):
    if event=='return' and frame.f_code.co_name=='infer_from_package' and frame.f_code.co_filename=='<ticket70-source-0>':
        local=frame.f_locals
        planned=[member['id'] for members in local.get('groups',{}).values() for member in members]
        completed=[member['id'] for member in local.get('per_member',[])]
        expected_public=[member['id'] for member in local.get('members',[])]
        completed_public=[member['id'] for member in local.get('public_frontier_members',[])]
        _rsna70_dino.append({'planned':sorted(planned),'completed':sorted(completed),
            'expected_public':sorted(expected_public),'completed_public':sorted(completed_public)})
def _rsna70_verify_execution():
    events=[]; lines=0
    with _rsna70_log.open() as handle:
        for text in handle:
            row=_rsna70_json.loads(text); reason=_rsna70_rejection(row['line']); lines+=1
            if reason!=row['rejection']: raise RuntimeError('Execution guard log differs')
            if reason: events.append({'stream':row['stream'],'reason':reason,
                'line_sha256':_rsna70_hashlib.sha256(row['line'].encode()).hexdigest()})
    if events: raise RuntimeError('Exact original ensemble used a failed/partial member, study or calibration')
    if len(_rsna70_dino)!=1: raise RuntimeError('Unique actual complete DINO return proof required')
    dino=_rsna70_dino[0]
    if (not dino['planned'] or len(set(dino['planned']))!=len(dino['planned'])
        or dino['planned']!=dino['completed'] or not dino['expected_public']
        or len(set(dino['expected_public']))!=len(dino['expected_public'])
        or dino['expected_public']!=dino['completed_public']):
        raise RuntimeError('Actual DINO member completion differs from the original plan')
    return {'schema_version':'rsna-ticket70-complete-member-proof-v3','rejection_events':events,
        'execution_log_file':_rsna70_log.name,'execution_log_sha256':_rsna70_hashlib.sha256(_rsna70_log.read_bytes()).hexdigest(),
        'logged_lines':lines,'stdout_and_stderr':True,'unterminated_tails_checked':True,'dino':dino}
'''


def original_inputs(parent):
    import csv
    import io
    evidence = {}
    validation = json.loads((INPUT / "independent-summary-validation.json").read_text())
    if validation["status"] != "COMPLETE" or validation["study_count"] != 4407 or not validation["disjoint_complete_membership"]:
        raise ValueError("Complete original extraction cohort verification required")
    if digest(INPUT / "plan.json") != validation["campaign_plan_sha256"]:
        raise ValueError("Original extraction campaign plan differs")
    evidence[str(INPUT / "independent-summary-validation.json")] = digest(INPUT / "independent-summary-validation.json")
    evidence[str(INPUT / "plan.json")] = digest(INPUT / "plan.json")
    with zipfile.ZipFile(PROJECT / "rsna-knee-abnormality-detection.zip") as archive:
        train = archive.read("train.csv")
        if hashlib.sha256(train).hexdigest() != parent["train_csv_sha256"]:
            raise ValueError("Original reference metadata differs")
        rows = list(csv.DictReader(io.StringIO(train.decode())))
        targets = recipe_targets(PARENT / "development.ipynb")
        ids = [row["StudyInstanceUID"] for row in rows if all(row[t] in ("0", "1") for t in targets)]
        if len(ids) != 58 or len(set(ids)) != 58 or hashlib.sha256("".join(uid + "\n" for uid in ids).encode()).hexdigest() != parent["development_uid_sha256"]:
            raise ValueError("Frozen reference membership differs")
        expected = {}
        for info in archive.infolist():
            parts = PurePosixPath(info.filename).parts
            if len(parts) == 4 and parts[0] == "train_series" and parts[1] in ids and parts[3].endswith(".dcm"):
                if info.filename in expected:
                    raise ValueError("Duplicate original DICOM ZIP entry")
                expected[info.filename] = (info.file_size, info.CRC)
    by_uid = {uid: [] for uid in ids}
    found = set()
    for index, observed in enumerate(validation["partitions"]):
        root = INPUT / f"partition-{index}"
        result_path = root / "probe-result.json"
        result = json.loads(result_path.read_text())
        terminal = root / f"controller-{result['job_id']}" / "terminal-controller.txt"
        fields = dict(token.split("=", 1) for token in terminal.read_text().split() if "=" in token)
        if (digest(result_path) != observed["result_sha256"] or result["status"] != "COMPLETE"
                or result["plan_sha256"] != digest(root / "plan.json")
                or result["job_id"] != observed["job_id"] or not result["cpu_input_guard_passed"]
                or digest(terminal) != observed["terminal_sha256"] or fields.get("JobState") != "COMPLETED"
                or fields.get("ExitCode") != "0:0" or fields.get("JobId") != result["job_id"]):
            raise ValueError("Original extraction terminal/result differs")
        source = root / "source-mri-entries.json"
        if digest(source) != result["source_entries_sha256"]:
            raise ValueError("Original DICOM content manifest changed")
        for path in (result_path, root / "plan.json", terminal, source):
            evidence[str(path)] = digest(path)
        for row in json.loads(source.read_text()):
            name = row["entry"]
            if name not in expected:
                continue
            if name in found or (row["bytes"], row["archive_crc32"]) != expected[name]:
                raise ValueError("Original DICOM membership/CRC/size differs")
            found.add(name)
            parts = PurePosixPath(name).parts
            by_uid[parts[1]].append(["/".join(parts[2:]), row["bytes"], row["actual_sha256"]])
    if found != set(expected):
        raise ValueError("Complete reference DICOM manifest coverage required")
    studies = [{"study_uid": uid, **input_aggregate(by_uid[uid])} for uid in ids]
    identity = {"schema_version": "rsna-ticket70-frozen-original-mri-identity-v3", "studies": studies}
    return {**identity, "identity_sha256": hashlib.sha256(canonical(identity).encode()).hexdigest()}, evidence


def recipe_targets(path):
    code = notebook_code(json.loads(Path(path).read_text()))
    nodes = [n for n in ast.parse(code).body if isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "TARGETS"]
    if len(nodes) != 1:
        raise ValueError("Unique exact target schema required")
    return ast.literal_eval(nodes[0].value)


def notebook_code(notebook):
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def parent_inputs():
    if digest(PARENT / "request.json") != PARENT_SHA:
        raise ValueError("Immutable pinned parent request changed")
    parent = json.loads((PARENT / "request.json").read_text())
    validation = json.loads((PARENT / "preparation-validation.json").read_text())
    if validation["status"] != "PINNED_ENVELOPE_PREPARATION_VALIDATED" or validation["request_sha256"] != PARENT_SHA:
        raise ValueError("Validated immutable parent required")
    for field, name in (("notebook_sha256", "development.ipynb"), ("metadata_sha256", "kernel-metadata.json")):
        if digest(PARENT / name) != parent[field]:
            raise ValueError("Frozen parent source/settings changed")
    for name, checksum in parent["frozen_files"].items():
        if digest(PARENT / name) != checksum:
            raise ValueError("Frozen parent source changed")
    return parent


def prepare():
    if DEST.exists():
        raise FileExistsError("Immutable verified-runtime preparation already exists")
    parent = parent_inputs()
    inputs, evidence = original_inputs(parent)
    notebook = json.loads((PARENT / "development.ipynb").read_text())
    if len(notebook["cells"]) != 1 or notebook["cells"][0]["cell_type"] != "code":
        raise ValueError("Unique original wrapper cell required")
    code = instrument(notebook_code(notebook), inputs)
    notebook["cells"][0]["source"] = code.splitlines(keepends=True)
    metadata = json.loads((PARENT / "kernel-metadata.json").read_text())
    metadata.update(id="jitheswar/rsna-exact-scored-anchor-gold58-v3")
    DEST.mkdir(mode=0o700)
    shutil.copytree(PARENT / "source-snapshot", DEST / "source-snapshot")
    shutil.copyfile(Path(__file__), DEST / "source-snapshot" / Path(__file__).name)
    for name in ("NOTICE.md", "LICENSE-APACHE-2.0.txt"):
        shutil.copyfile(PARENT / name, DEST / name)
    write(DEST / "development.ipynb", notebook)
    write(DEST / "kernel-metadata.json", metadata)
    write(DEST / "original-mri-contract.json", inputs)
    request = {**parent, "schema_version": "rsna-ticket70-exact-scored-anchor-verified-runtime-v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "kernel_id": metadata["id"],
        "parent_request_sha256": PARENT_SHA, "supersedes": str(PARENT),
        "metadata_sha256": digest(DEST / "kernel-metadata.json"),
        "notebook_sha256": digest(DEST / "development.ipynb"),
        "original_mri_contract_sha256": digest(DEST / "original-mri-contract.json"),
        "original_mri_identity_sha256": inputs["identity_sha256"],
        "original_mri_dicom_files": sum(row["dicom_files"] for row in inputs["studies"]),
        "original_mri_dicom_bytes": sum(row["dicom_bytes"] for row in inputs["studies"]),
        "original_mri_evidence": evidence,
        "input_hash_wall_seconds_allowance": 180,
        "forecast_with_input_hash_allowance_seconds": parent["forecast_wall_seconds"] + 180,
        "only_wrapper_changes": "Record installed runtime, hash all original Gold58 DICOM bytes before inference, check stdout/stderr including tails and unique complete original DINO members; five scientific cells unchanged",
        "historical_weight_byte_parity_proven": False,
        "resource_decision_status": "Prepared verified-runtime envelope only; no admission/reservation/launch",
        "frozen_files": {str(p.relative_to(DEST)): digest(p) for p in (DEST / "source-snapshot").iterdir()}}
    if request["forecast_with_input_hash_allowance_seconds"] >= request["execution_watchdog_seconds"]:
        raise ValueError("Additional proof forecast exceeds original execution watchdog")
    write(DEST / "request.json", request)
    (DEST / "request-checksum.txt").write_text(digest(DEST / "request.json") + "\n")
    verify()


def verify():
    parent = parent_inputs()
    request = json.loads((DEST / "request.json").read_text())
    if digest(DEST / "request.json") != (DEST / "request-checksum.txt").read_text().strip():
        raise ValueError("Immutable verified-runtime request changed")
    inputs, evidence = original_inputs(parent)
    if (json.loads((DEST / "original-mri-contract.json").read_text()) != inputs
            or digest(DEST / "original-mri-contract.json") != request["original_mri_contract_sha256"]
            or request["original_mri_evidence"] != evidence):
        raise ValueError("Frozen original MRI byte contract differs")
    code = notebook_code(json.loads((DEST / "development.ipynb").read_text()))
    if code != instrument(notebook_code(json.loads((PARENT / "development.ipynb").read_text())), inputs):
        raise ValueError("Additive execution instrumentation differs")
    metadata = json.loads((PARENT / "kernel-metadata.json").read_text())
    metadata.update(id="jitheswar/rsna-exact-scored-anchor-gold58-v3")
    if json.loads((DEST / "kernel-metadata.json").read_text()) != metadata:
        raise ValueError("Pinned container/attachment envelope changed")
    if (request["parent_request_sha256"] != PARENT_SHA or request["kernel_id"] != metadata["id"]
            or digest(DEST / "development.ipynb") != request["notebook_sha256"]
            or digest(DEST / "kernel-metadata.json") != request["metadata_sha256"]):
        raise ValueError("Verified-runtime request identity differs")
    for key in ("assigned_gpu_count", "session_timeout_seconds", "execution_watchdog_seconds", "proposed_reserved_gpu_seconds",
                "maximum_launch_attempts", "model_fitting", "competition_submission", "final_selection", "external_spend_usd", "source_cells"):
        if request[key] != parent[key]:
            raise ValueError("Verified-runtime envelope changes scientific/resource scope")
    for name, checksum in request["frozen_files"].items():
        if digest(DEST / name) != checksum:
            raise ValueError("Verified-runtime source snapshot changed")
    record = {"status": "VERIFIED_RUNTIME_ENVELOPE_PREPARATION_VALIDATED", "request_sha256": digest(DEST / "request.json"),
        "parent_request_sha256": PARENT_SHA, "scientific_cells_unchanged": 5, "original_mri_contract_studies": len(inputs["studies"]),
        "original_mri_dicom_files": sum(row["dicom_files"] for row in inputs["studies"]),
        "original_mri_dicom_bytes": sum(row["dicom_bytes"] for row in inputs["studies"]),
        "source_payloads_freshly_rehashed_during_preparation": False, "mounted_mri_bytes_verified": False,
        "historical_weight_byte_parity_proven": False, "runtime_execution_verified": False, "launched": False,
        "resource_decision_admitted": False}
    path = DEST / "preparation-validation.json"
    serialized = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != serialized:
        raise ValueError("Immutable preparation validation changed")
    if not path.exists(): path.write_text(serialized)
    print(json.dumps(record))


if __name__ == "__main__":
    if not sys.argv[1:]: prepare()
    elif sys.argv[1:] == ["--verify"]: verify()
    else: raise SystemExit("Use no arguments to prepare, or --verify")
