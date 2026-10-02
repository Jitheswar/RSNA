"""Independently verify one terminal exact-anchor development run; never launch it."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from statistics import NormalDist

import numpy as np
import pandas as pd


PROJECT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
ROOT = PROJECT / "artifacts/private/ticket70-scored-anchor-gold58-v3"
REQUEST_SHA = "255eff3d3ea66deb926a9a4507391322bd19bd66b8a9edd061a470a518cd2924"
ASSETS = PROJECT / "artifacts/private/ticket70-asset-lineage-audit-v1"
MODEL_AUDIT = PROJECT / "artifacts/private/ticket70-model-asset-audit-v1/pinned-files.json"
MODEL_AUDIT_SHA = "05c8ff8072533b26001d129ee61e50eeb38172a65df5e6b342ee588641e7cc6b"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
REJECT_PATTERNS = (
    r"MEMBER .* failed", r"dropped -- costs one vote", r"degenerate predictions; not banked",
    r"public-frontier vote omitted", r"surrendering .* member", r"public-frontier fallback not emitted",
    r"study failed:", r"CoAtNet branch failed", r"legacy checkpoint disabled", r"legacy complement unavailable",
    r"\[DINOsaur V4\.2\].*FALLBACK", r"legacy study .* fallback:", r"calibration skipped",
    r"final fusion failed", r"CoAtNet output unavailable",
    r"[1-9]\d* series had a slice that would not decode", r"time budget reached during decode")
RAD_HASHES = {
    "ResNet50.pt": "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734",
    "v52_radimagenet_heads.pt": "0f465649799ecfbccaac1767844639e7ced44e1bc9babde6e4bac7c5d9b89eaa",
    "v52_e11_heads.pt": "ad9f19af73bfdf4e49263c0e45060dc3cb239e1195039b26dc8c0a3a6bcd1a8a"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def checked_file(root, name):
    path = root / name
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
            "Exact retained regular file required")
    return path


def notebook_cells(notebook):
    return [(cell["cell_type"], "".join(cell["source"])) for cell in notebook["cells"]]


def constants(code):
    found = {}
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"ORIGINAL_CELLS", "SOURCE_CELLS", "_RSNA70_EXPECTED_INPUTS"}:
                require(name not in found, "Ambiguous embedded frozen recipe/input constant")
                found[name] = ast.literal_eval(node.value)
    require(set(found) == {"ORIGINAL_CELLS", "SOURCE_CELLS", "_RSNA70_EXPECTED_INPUTS"},
            "Embedded original/relocated recipe and MRI contract required")
    require(len(found["ORIGINAL_CELLS"]) == len(found["SOURCE_CELLS"]) == 5, "Exactly five frozen cells required")
    return found


def verify_source(root, request, accepted, terminal, remote, source_validation):
    require(accepted["request_sha256"] == digest(root / "request.json") == terminal["request_sha256"],
            "Accepted and terminal request identity differs")
    require(accepted["response_sha256"] == digest(root / "push-response.json"), "Accepted push bytes differ")
    push = read(root / "push-response.json")
    handle = push["ref"].removeprefix("/code/")
    require(not push.get("error") and push["kernelId"] == accepted["kernel_numeric_id"]
            and push["versionNumber"] == accepted["kernel_version"] == 1
            and accepted["launch_attempt"] == terminal["launch_attempt"] == 1
            and handle == request["kernel_id"] == accepted["kernel_id"] == terminal["kernel_id"]
            and terminal["kernel_numeric_id"] == accepted["kernel_numeric_id"]
            and terminal["kernel_version"] == 1, "Exact one-attempt remote kernel identity differs")
    require(terminal["status"] == terminal["response"]["status"] == "COMPLETE", "Actual matching COMPLETE kernel terminal required")
    require(source_validation["status"] == "PASS" and all(source_validation.get(key) is True for key in
            ("exact_code_cells", "pinned_docker", "private_internet_off", "attachments_canonical_refs_verified")),
            "Lifecycle exact remote source/settings proof required")
    metadata = remote["metadata"]
    local = read(root / "kernel-metadata.json")
    require(metadata["id"] == accepted["kernel_numeric_id"] and metadata["ref"] == accepted["kernel_id"]
            and metadata["currentVersionNumber"] == 1, "Independent remote numeric identity/version differs")
    require(metadata["isPrivate"] is True and metadata["enableGpu"] is True
            and metadata["enableTpu"] is False and metadata["enableInternet"] is False
            and metadata["machineShape"] == "NvidiaTeslaT4"
            and metadata["dockerImage"] == local["docker_image"] == request["docker_image"],
            "Independent pinned offline two-T4 runtime settings differ")
    for remote_key, local_key in (("datasetDataSources", "dataset_sources"), ("modelDataSources", "model_sources"),
                                  ("competitionDataSources", "competition_sources"), ("kernelDataSources", "kernel_sources")):
        expected = (["/".join(ref.split("/")[:2]) for ref in local[local_key]]
                    if local_key == "dataset_sources" else local[local_key])
        require(metadata[remote_key] == expected, "Independent exact canonical attachments differ")
    remote_source = remote["blob"]["source"]
    require(hashlib.sha256(remote_source.encode()).hexdigest() == source_validation["remote_source_sha256"],
            "Lifecycle retained remote source hash differs")
    local_notebook = read(root / "development.ipynb")
    require(notebook_cells(local_notebook) == notebook_cells(json.loads(remote_source)), "Independent actual remote code cells differ")
    code = "\n".join(source for cell_type, source in notebook_cells(local_notebook) if cell_type == "code")
    recipe = constants(code)
    for index, (original, relocated) in enumerate(zip(recipe["ORIGINAL_CELLS"], recipe["SOURCE_CELLS"])):
        row = request["source_cells"][index]
        require(row == {"index": index, "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                        "relocated_sha256": hashlib.sha256(relocated.encode()).hexdigest()},
                "Frozen embedded scientific recipe differs")
        require((root / f"source-snapshot/original-cell-{index}.py").read_text() == original
                and (root / f"source-snapshot/relocated-cell-{index}.py").read_text() == relocated,
                "Retained original/relocated cell bytes differ")
    require(recipe["_RSNA70_EXPECTED_INPUTS"] == read(root / "original-mri-contract.json"),
            "Embedded native MRI byte contract differs")
    return recipe["SOURCE_CELLS"]


def verify_original_mri(proof, contract, request):
    require(proof["schema_version"] == "rsna-ticket70-original-mri-byte-proof-v3" and proof["status"] == "COMPLETE"
            and proof["study_count"] == 58 and proof["target_labels_uploaded"] is False,
            "Actual complete original MRI byte proof required")
    require(proof["studies"] == contract["studies"] and len(proof["studies"]) == 58,
            "Actual original MRI byte membership/order differs")
    ids = [row["study_uid"] for row in proof["studies"]]
    require(len(set(ids)) == 58 and hashlib.sha256("".join(uid + "\n" for uid in ids).encode()).hexdigest()
            == request["development_uid_sha256"], "Actual native MRI cohort identity differs")
    for row in proof["studies"]:
        require(type(row["dicom_files"]) is int and row["dicom_files"] > 0
                and type(row["dicom_bytes"]) is int and row["dicom_bytes"] > 0
                and re.fullmatch("[0-9a-f]{64}", row["aggregate_sha256"]), "Malformed native DICOM byte identity")
    identity = {"schema_version": "rsna-ticket70-frozen-original-mri-identity-v3", "studies": proof["studies"]}
    require(hashlib.sha256(canonical(identity).encode()).hexdigest() == contract["identity_sha256"]
            == proof["identity_sha256"] == request["original_mri_identity_sha256"], "Native MRI aggregate hash differs")
    require(proof["dicom_files"] == sum(row["dicom_files"] for row in proof["studies"]) == request["original_mri_dicom_files"]
            and proof["dicom_bytes"] == sum(row["dicom_bytes"] for row in proof["studies"]) == request["original_mri_dicom_bytes"],
            "Native MRI whole-cohort file/byte counts differ")
    require(isinstance(proof["hash_elapsed_seconds"], (int, float)) and not isinstance(proof["hash_elapsed_seconds"], bool)
            and math.isfinite(proof["hash_elapsed_seconds"]) and proof["hash_elapsed_seconds"] >= 0,
            "Actual MRI byte hashing elapsed time is invalid")
    return ids


def verify_runtime(runtime):
    require(runtime["schema_version"] == "rsna-ticket70-actual-runtime-v3"
            and re.fullmatch(r"\d+\.\d+\.\d+", runtime["python"]), "Actual installed runtime proof required")
    for name in ("numpy", "pandas", "pydicom", "torch", "torchvision", "timm", "scipy", "Pillow"):
        require(isinstance(runtime["packages"].get(name), str) and runtime["packages"][name], "Missing actual package version")
    require(any(runtime["packages"].get(name) for name in ("opencv-python", "opencv-python-headless")),
            "Actual OpenCV package version required")
    require(isinstance(runtime["torch_cuda"], str) and runtime["torch_cuda"]
            and type(runtime["cudnn"]) is int and runtime["cudnn"] > 0, "Actual CUDA/cuDNN runtime required")
    require(runtime["gpu_count"] == 2 and len(runtime["gpus"]) == 2
            and [row["index"] for row in runtime["gpus"]] == [0, 1], "Exactly two actual CUDA GPUs required")
    require(all("T4" in row["name"] and row["capability"] == [7, 5]
                and type(row["total_memory_bytes"]) is int and row["total_memory_bytes"] >= 15_000_000_000
                for row in runtime["gpus"]), "Actual assigned Tesla T4 hardware differs")


def verify_execution(output, execution):
    require(execution["schema_version"] == "rsna-ticket70-complete-member-proof-v3"
            and execution["rejection_events"] == [] and execution["stdout_and_stderr"] is True
            and execution["unterminated_tails_checked"] is True, "Actual complete-member and fallback proof required")
    require(execution["execution_log_file"] == "development-execution.jsonl", "Unexpected actual execution proof path")
    log = checked_file(output, execution["execution_log_file"])
    require(digest(log) == execution["execution_log_sha256"], "Actual execution log bytes changed")
    lines = 0
    with log.open() as handle:
        for text in handle:
            row = json.loads(text)
            require(row["stream"] in {"stdout", "stderr"} and isinstance(row["line"], str)
                    and row["rejection"] is None, "Execution proof log is malformed or rejected")
            require(not any(re.search(pattern, row["line"]) for pattern in REJECT_PATTERNS),
                    "Independent execution log contains member/study/calibration fallback")
            lines += 1
    require(lines == execution["logged_lines"] and lines > 0, "Complete execution log line count differs")
    proof = execution["dino"]
    for planned, completed in (("planned", "completed"), ("expected_public", "completed_public")):
        require(isinstance(proof[planned], list) and proof[planned] and proof[planned] == sorted(proof[planned])
                and len(set(proof[planned])) == len(proof[planned]) and proof[planned] == proof[completed],
                "Actual original DINO member completion differs")
    require(set(proof["expected_public"]).issubset(proof["planned"]), "Public DINO plan is outside the actual ensemble")
    return proof


def load_inventories(request):
    inventories = {}
    for name, expected_sha in request["asset_lineage_files"].items():
        path = checked_file(ASSETS, name)
        require(digest(path) == expected_sha, "Pinned authenticated asset-lineage evidence changed")
        if not name.endswith("--pinned-files.json"):
            continue
        inventory = read(path)
        pin = inventory["ref"] + "/" + str(inventory["version"])
        require(pin in request["dataset_version_pins"], "Pinned inventory version differs")
        files = {}
        for page in inventory["pages"]:
            for row in page["datasetFiles"]:
                require(row["name"] not in files and type(row["totalBytes"]) is int and row["totalBytes"] > 0,
                        "Pinned complete asset file inventory is ambiguous")
                files[row["name"]] = row["totalBytes"]
        inventories[inventory["ref"]] = files
    require(set(inventories) == {pin.rsplit("/", 1)[0] for pin in request["dataset_version_pins"]},
            "Complete pinned attachment inventory required")
    require(digest(MODEL_AUDIT) == MODEL_AUDIT_SHA, "Pinned model materialization inventory changed")
    model = read(MODEL_AUDIT)
    require(model["model_source"] == "metaresearch/dinov2/PyTorch/small/1", "Exact pinned DINO model required")
    model_files = {}
    for page in model["pages"]:
        for row in page["files"]:
            require(row["name"] not in model_files and type(row["size"]) is int and row["size"] > 0,
                    "Complete pinned model inventory is ambiguous")
            model_files[row["name"]] = row["size"]
    inventories[model["model_source"]] = model_files
    return inventories


def mounted_asset(path, inventories):
    pure = PurePosixPath(path)
    require(pure.is_absolute() and ".." not in pure.parts and "\\" not in path and "\0" not in path,
            "Unsafe loaded checkpoint path")
    for ref, files in inventories.items():
        parts = ref.split("/")
        owner, slug = parts[:2]
        mounts = (("/kaggle/input/datasets/" + ref + "/", "/kaggle/input/" + slug + "/") if len(parts) == 2
                  else ("/kaggle/input/models/" + ref.lower() + "/", "/kaggle/input/" + "/".join(parts[1:]).lower() + "/",
                        "/kaggle/input/dinov2-small/pytorch/small/1/", "/kaggle/input/dinov2-small/", "/kaggle/input/dinov2/"))
        for mount in mounts:
            if path.startswith(mount):
                relative = path[len(mount):]
                require(relative in files, "Loaded checkpoint is absent from the pinned version inventory")
                return ref, relative, files[relative]
    raise ValueError("Loaded checkpoint is outside every exact pinned dataset mount")


def verify_weights(loaded, inventories, dino):
    require(isinstance(loaded, dict) and loaded, "Actual original loaded checkpoint hashes required")
    matched = []
    for path, row in loaded.items():
        ref, relative, size = mounted_asset(path, inventories)
        require(type(row["bytes"]) is int and row["bytes"] == size
                and re.fullmatch("[0-9a-f]{64}", row["sha256"]), "Actual checkpoint hash/size differs from pinned inventory")
        name = PurePosixPath(relative).name
        if name in RAD_HASHES:
            require(row["sha256"] == RAD_HASHES[name], "Explicit original RadImageNet checkpoint SHA differs")
        matched.append({"dataset_ref": ref, "relative_path": relative, **row})
    names = [PurePosixPath(row["relative_path"]).name for row in matched]
    required = {"raptor_ft_coatnet_v5_full_swa.pt", "raptor_ft_coatnet_v4_full.pt", *RAD_HASHES,
                *("m_f" + str(fold) + ".pt" for fold in range(5))}
    require(required.issubset(names), "A required original Raptor/A5/RadImageNet checkpoint was not actually loaded")
    expected_dino = {name for name in inventories["pilkwang/rsna-knee-weights"] if name.endswith(".pt")}
    actual_dino = {name for name in names if re.fullmatch("m_[0-9a-f]{10}\\.pt", name)}
    require(actual_dino == expected_dino and len(actual_dino) == len(dino["expected_public"]),
            "Complete original DINO checkpoint inventory was not actually loaded")
    return matched


def read_predictions(path, ids):
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str}, float_precision="round_trip")
    values = frame[TARGETS].to_numpy(np.float64) if set(TARGETS).issubset(frame.columns) else None
    require(frame.columns.tolist() == ["StudyInstanceUID"] + TARGETS and frame.StudyInstanceUID.tolist() == ids
            and values.shape == (58, 12) and np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all(),
            "Actual exact-anchor stage prediction identity/schema/values differ")
    return frame


def probit(values):
    clipped = np.clip(np.asarray(values, np.float64), 1e-4, 1 - 1e-4)
    return np.asarray([NormalDist().inv_cdf(float(v)) for v in clipped.flat]).reshape(clipped.shape)


def original_raptor_rank(primary, legacy):
    rank = lambda a: a.argsort(0).argsort(0).astype(np.float64) / 57
    p = rank(np.clip(primary, 0, 1)); l = rank(np.clip(legacy, 0, 1)); mixed = p.copy()
    weights = {"MCL": .16, "Medial OA": .10, "PF OA": .12, "Effusion": .10, "Synovitis": .16,
               "Baker's": .12, "Contusion": .12, "ACL": .10, "Lateral Meniscus": .10,
               "Medial Meniscus": .10, "Lateral OA": .10, "Fracture": .10}
    for column, name in enumerate(TARGETS):
        weight = weights[name]; corr = float(np.corrcoef(p[:, column], l[:, column])[0, 1])
        if not np.isfinite(corr): continue
        if corr > .992: weight *= .5
        elif corr < .65: weight *= .4
        mixed[:, column] = (1 - weight) * probit(p[:, column]) + weight * probit(l[:, column])
    return rank(mixed).astype(np.float32)


def exact_outer_fusion(transformer, primary, legacy):
    coat = pd.DataFrame(original_raptor_rank(primary, legacy), columns=TARGETS)
    transformer_rank = transformer[TARGETS].rank(method="average", pct=True)
    coat_rank = coat.rank(method="average", pct=True)
    weights = {"Lateral Meniscus": .65, "Fracture": .65, "Medial Meniscus": .65, "Lateral OA": .60,
               "PF OA": .60, "ACL": .60, "Medial OA": .60, "Effusion": .60, "MCL": .58,
               "Contusion": .55, "Synovitis": .55, "Baker's": .55}
    fused = transformer_rank.copy()
    for name in TARGETS:
        weight = weights[name]
        fused[name] = (1 - weight) * probit(transformer_rank[name].to_numpy()) + weight * probit(coat_rank[name].to_numpy())
    return fused.rank(method="average", pct=True).to_numpy(np.float64)


def verify_outputs(output, request, contract, source_cells, inventories):
    require(output.is_dir() and not output.is_symlink(), "Retained terminal output directory required")
    require(not list(output.rglob("submission.csv")), "Development-only execution must not leave a submission file")
    receipt_path = checked_file(output, "exact-scored-anchor-receipt.json")
    receipt = read(receipt_path)
    require(receipt["schema_version"] == "rsna-ticket70-exact-scored-anchor-output-v1" and receipt["status"] == "COMPLETE"
            and receipt["study_count"] == 58 and receipt["development_uid_sha256"] == request["development_uid_sha256"]
            and receipt["assigned_gpu_count"] == 2 and receipt["model_fitting"] is False
            and receipt["competition_submission"] is False and receipt["fallback_events"] == 0,
            "Actual complete fixed-scope development receipt required")
    require(receipt["coat_receipt"] == {"completed": True, "legacy_success_count": 58},
            "Actual full original Raptor primary/complement execution required")
    require(receipt["original_cells_sha256"] == [row["original_sha256"] for row in request["source_cells"]],
            "Actual execution recipe identity differs")
    require(type(receipt["elapsed_seconds"]) in (float, int) and math.isfinite(receipt["elapsed_seconds"])
            and 0 < receipt["elapsed_seconds"] < request["execution_watchdog_seconds"], "Actual execution exceeded the fixed watchdog")
    files = {}
    for key, name in (("runtime_receipt", "development-runtime.json"), ("input_receipt", "development-original-mri-proof.json")):
        require(receipt[key]["file"] == name, "Actual evidence receipt path differs")
        path = checked_file(output, name)
        require(digest(path) == receipt[key]["sha256"], "Actual runtime/native MRI receipt bytes differ")
        files[name] = digest(path)
    runtime = read(output / "development-runtime.json")
    verify_runtime(runtime)
    mri = read(output / "development-original-mri-proof.json")
    ids = verify_original_mri(mri, contract, request)
    require(all(receipt["input_receipt"][key] == mri[key] for key in ("identity_sha256", "dicom_files", "dicom_bytes")),
            "Actual MRI receipt summary differs")
    dino = verify_execution(output, receipt["execution_receipt"])
    loaded = verify_weights(receipt["loaded_weights"], inventories, dino)
    files["development-execution.jsonl"] = digest(output / "development-execution.jsonl")
    require(len(receipt["stages"]) == 5, "Every original inference stage receipt is required")
    frames = []
    for index, stage in enumerate(receipt["stages"]):
        name = f"development-stage-{index}.csv"; path = checked_file(output, name)
        require(stage["index"] == index and stage["recipe_sha256"] == hashlib.sha256(source_cells[index].encode()).hexdigest()
                and stage["sha256"] == digest(path) and isinstance(stage["cache_config"], dict),
                "Actual original stage source or output bytes differ")
        frames.append(read_predictions(path, ids)); files[name] = digest(path)
    final_name = "development-exact-scored-anchor.csv"
    final = checked_file(output, final_name); final_frame = read_predictions(final, ids)
    require(digest(final) == receipt["final_sha256"] == files["development-stage-4.csv"] == files["development-stage-3.csv"],
            "Actual final/audit stage bytes differ")
    raw = checked_file(output, "development-raptor-raw.npz")
    with np.load(raw, allow_pickle=False) as saved:
        require(set(saved.files) == {"primary", "legacy", "study_uids"}
                and saved["study_uids"].astype(str).tolist() == ids, "Actual Raptor raw cohort mapping differs")
        primary, legacy = saved["primary"], saved["legacy"]
        for values in (primary, legacy):
            require(values.shape == (58, 12) and values.dtype.kind == "f" and np.isfinite(values).all()
                    and (values >= 0).all() and (values <= 1).all(), "Actual raw Raptor probabilities differ")
        recomputed = exact_outer_fusion(frames[2], primary, legacy)
    actual = final_frame[TARGETS].to_numpy(np.float64)
    require(np.array_equal(actual, recomputed), "Independent original Raptor/probit final-rank fusion differs")
    files[final_name] = digest(final); files["development-raptor-raw.npz"] = digest(raw)
    audit = read(checked_file(output, "submission_audit.json"))
    require(audit["rows"] == 58 and audit["full_ensemble_complete"] is True,
            "Original final source audit did not complete the full ensemble")
    files["submission_audit.json"] = digest(output / "submission_audit.json")
    return {"output_files_sha256": files, "output_sha256": digest(final),
            "output_receipt_sha256": digest(receipt_path), "original_mri_studies": 58,
            "original_mri_dicom_files": mri["dicom_files"], "original_mri_dicom_bytes": mri["dicom_bytes"],
            "original_mri_evidence": {"path": str(output / "development-original-mri-proof.json"),
                                      "sha256": files["development-original-mri-proof.json"]},
            "runtime_evidence": {"path": str(output / "development-runtime.json"), "sha256": files["development-runtime.json"]},
            "loaded_checkpoint_inventory": loaded, "all_original_dino_members": len(dino["planned"]),
            "raptor_raw_and_final_fusion_recomputed": True, "historical_weight_byte_parity_proven": False,
            "base_model_files_individually_hashed": False,
            "strict_full_dino_state_load_verified_by_exact_source": True,
            "asset_envelope_verification_scope": "Exact pinned model/dataset attachments and saved container; actual inference checkpoint hashes and pinned inventory sizes. DINO initialization/config files are version-pinned without individual hashes; strict full Model state loading replaces backbone parameters and persistent buffers before inference.",
            "runtime": runtime, "visible_elapsed_seconds": receipt["elapsed_seconds"]}


def verify(root=ROOT):
    root = Path(root).resolve()
    require(digest(checked_file(root, "request.json")) == REQUEST_SHA
            == (root / "request-checksum.txt").read_text().strip(), "Exact immutable v3 request required")
    request = read(root / "request.json")
    for name, key in (("development.ipynb", "notebook_sha256"), ("kernel-metadata.json", "metadata_sha256"),
                      ("original-mri-contract.json", "original_mri_contract_sha256")):
        require(digest(checked_file(root, name)) == request[key], "Frozen exact source/settings/MRI contract changed")
    for name, expected in request["frozen_files"].items():
        require(digest(checked_file(root, name)) == expected, "Frozen original scientific source changed")
    for name, expected in request["original_mri_evidence"].items():
        require(digest(Path(name)) == expected, "Original source extraction provenance changed")
    accepted = read(checked_file(root, "accepted-launch.json")); terminal = read(checked_file(root, "kernel-terminal.json"))
    cells = verify_source(root, request, accepted, terminal, read(checked_file(root, "remote-terminal.json")),
                          read(checked_file(root, "remote-source-validation.json")))
    output = root / "terminal-output"
    require(output.resolve().is_relative_to(root), "Output evidence escapes the exact request directory")
    result = verify_outputs(output, request, read(root / "original-mri-contract.json"), cells, load_inventories(request))
    result.update(schema_version="rsna-ticket70-independent-exact-anchor-verification-v3", status="COMPLETE", passed=True,
        observed_at_utc=datetime.now(timezone.utc).isoformat(), request_sha256=REQUEST_SHA,
        kernel_id=accepted["kernel_id"], kernel_numeric_id=accepted["kernel_numeric_id"], kernel_version=1,
        accepted_launch_sha256=digest(root / "accepted-launch.json"), terminal_sha256=digest(root / "kernel-terminal.json"),
        terminal_state="COMPLETE", remote_source_verified=True, remote_settings_verified=True,
        asset_runtime_envelope_verified=True, exact_recipe_verified=True, no_fallback_verified=True,
        remote_source_validation_sha256=digest(root / "remote-source-validation.json"),
        remote_terminal_sha256=digest(root / "remote-terminal.json"), output_directory=str(output),
        clean_oof=False, candidate_promotion=False, competition_submission=False,
        limitations="Gold58 and public parents were adaptively reused; historical latest-attachment lineage is inferred. No historical SHA equality claimed for assets without original source hashes; notebook elapsed is not VM allocation time.")
    return result


def download(root=ROOT):
    """Consume one download intent only after a fresh exact matching terminal check."""
    root = Path(root).resolve()
    require(digest(root / "request.json") == REQUEST_SHA, "Exact immutable v3 request required before download")
    accepted, terminal = read(root / "accepted-launch.json"), read(root / "kernel-terminal.json")
    require(accepted["request_sha256"] == REQUEST_SHA == terminal["request_sha256"]
            and terminal["status"] == terminal["response"]["status"] == "COMPLETE", "Exact matching terminal required before retrieval")
    output = root / "terminal-output"
    require(not output.exists() and not (root / "output-download-intent.json").exists(),
            "Consumed download/output: inspect retained evidence before another retrieval")
    import sys
    sys.path.insert(0, "/home/ch.jitheswar/.venvs/kaggle-cli-2.2.4/lib/python3.11/site-packages")
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    api = KaggleApi(); api.authenticate()
    status = json.loads(api.kernels_status(accepted["kernel_id"]).to_json(ignore_defaults=False))
    query = ApiGetKernelRequest(); query.user_name, query.kernel_slug = accepted["kernel_id"].split("/")
    with api.build_kaggle_client() as client:
        remote = json.loads(client.kernels.kernels_api_client.get_kernel(query).to_json(ignore_defaults=False))
    metadata = remote["metadata"]
    require(status["status"] == "COMPLETE" and metadata["ref"] == accepted["kernel_id"]
            and metadata["id"] == accepted["kernel_numeric_id"] and metadata["currentVersionNumber"] == 1,
            "Fresh exact terminal kernel/version required before retrieval")
    require(not (root / "independent-verification.json").exists(), "Independent output proof already retained")
    with (root / "output-download-intent.json").open("x") as handle:
        json.dump({"request_sha256": REQUEST_SHA, "kernel_id": accepted["kernel_id"], "kernel_numeric_id": accepted["kernel_numeric_id"],
                   "kernel_version": 1, "accepted_launch_sha256": digest(root / "accepted-launch.json"),
                   "terminal_sha256": digest(root / "kernel-terminal.json"), "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                   "download_attempt": 1, "retry_allowed": False}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    api.kernels_output(accepted["kernel_id"], str(output), quiet=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Retrieve terminal output once before verification; no launch")
    args = parser.parse_args()
    if args.download: download()
    result = verify()
    with (ROOT / "independent-verification.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")
    print(json.dumps({"status": result["status"], "passed": result["passed"], "studies": result["original_mri_studies"],
                      "request_sha256": REQUEST_SHA, "candidate_promotion": False, "competition_submission": False}))


if __name__ == "__main__":
    main()
