"""Synthetic complete-run proof must reject byte, source and native-execution tampering."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest


spec = importlib.util.spec_from_file_location("ticket70_output_proof", Path(__file__).resolve().parents[1] / "scripts/verify_ticket70_exact_anchor_outputs.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def synthetic_run(tmp_path):
    ids = [f"synthetic-study-{index:02}" for index in range(58)]
    sources = tuple(f"scientific_cell_{index} = True\n" for index in range(5))
    studies = [dict(study_uid=uid, dicom_files=1, dicom_bytes=10, aggregate_sha256="a" * 64) for uid in ids]
    identity = dict(schema_version="rsna-ticket70-frozen-original-mri-identity-v3", studies=studies)
    contract = {**identity, "identity_sha256": hashlib.sha256(module.canonical(identity).encode()).hexdigest()}
    request = dict(development_uid_sha256=hashlib.sha256("".join(uid + "\n" for uid in ids).encode()).hexdigest(),
        original_mri_identity_sha256=contract["identity_sha256"], original_mri_dicom_files=58,
        original_mri_dicom_bytes=580, execution_watchdog_seconds=7140,
        source_cells=[dict(index=i, original_sha256=hashlib.sha256(source.encode()).hexdigest(),
                           relocated_sha256=hashlib.sha256(source.encode()).hexdigest()) for i, source in enumerate(sources)])
    runtime = dict(schema_version="rsna-ticket70-actual-runtime-v3", python="3.12.13",
        packages={name:"1.0.0" for name in ("numpy", "pandas", "pydicom", "torch", "torchvision", "timm", "scipy", "Pillow", "opencv-python-headless")},
        torch_cuda="12.8", cudnn=9000, gpu_count=2,
        gpus=[dict(index=i, name="Tesla T4", capability=[7, 5], total_memory_bytes=16_000_000_000) for i in range(2)])
    mri = dict(schema_version="rsna-ticket70-original-mri-byte-proof-v3", status="COMPLETE", study_count=58,
        dicom_files=58, dicom_bytes=580, identity_sha256=contract["identity_sha256"], studies=studies,
        hash_elapsed_seconds=1., target_labels_uploaded=False)
    for name, value in (("development-runtime.json", runtime), ("development-original-mri-proof.json", mri)):
        write(tmp_path / name, value)
    log = tmp_path / "development-execution.jsonl"
    log.write_text(module.canonical(dict(stream="stdout", line="Every original member completed", rejection=None)) + "\n")
    dino = dict(planned=["dino-a", "dino-b"], completed=["dino-a", "dino-b"],
                expected_public=["dino-a", "dino-b"], completed_public=["dino-a", "dino-b"])
    execution = dict(schema_version="rsna-ticket70-complete-member-proof-v3", rejection_events=[],
        execution_log_file=log.name, execution_log_sha256=module.digest(log), logged_lines=1,
        stdout_and_stderr=True, unterminated_tails_checked=True, dino=dino)
    inventories = {
        "owner/raptor-primary":{"raptor_ft_coatnet_v5_full_swa.pt":10},
        "owner/raptor-legacy":{"raptor_ft_coatnet_v4_full.pt":11},
        "owner/a5":{f"m_f{i}.pt":12 for i in range(5)},
        "owner/rad":{name:13 for name in module.RAD_HASHES},
        "pilkwang/rsna-knee-weights":{"m_0000000000.pt":14, "m_1111111111.pt":14},
    }
    loaded = {}
    for ref, files in inventories.items():
        for name, size in files.items():
            loaded[f"/kaggle/input/datasets/{ref}/{name}"] = dict(bytes=size, sha256=module.RAD_HASHES.get(name, "b" * 64))
    values = np.repeat(np.linspace(.1, .9, 58)[:, None], 12, axis=1)
    expected = np.repeat((np.arange(1, 59, dtype=float) / 58)[:, None], 12, axis=1)
    stages = []
    for index in range(5):
        frame = pd.DataFrame(values if index < 3 else expected, columns=module.TARGETS)
        frame.insert(0, "StudyInstanceUID", ids)
        path = tmp_path / f"development-stage-{index}.csv"
        frame.to_csv(path, index=False)
        stages.append(dict(index=index, sha256=module.digest(path),
            recipe_sha256=hashlib.sha256(sources[index].encode()).hexdigest(), cache_config={"CACHE_SLICES":8}))
    final = tmp_path / "development-exact-scored-anchor.csv"
    shutil.copyfile(tmp_path / "development-stage-4.csv", final)
    np.savez_compressed(tmp_path / "development-raptor-raw.npz", primary=values.astype(np.float32),
                        legacy=values.astype(np.float32), study_uids=np.array(ids))
    write(tmp_path / "submission_audit.json", dict(rows=58, full_ensemble_complete=True))
    receipt = dict(schema_version="rsna-ticket70-exact-scored-anchor-output-v1", status="COMPLETE", study_count=58,
        development_uid_sha256=request["development_uid_sha256"], assigned_gpu_count=2, model_fitting=False,
        competition_submission=False, fallback_events=0, coat_receipt=dict(completed=True, legacy_success_count=58),
        original_cells_sha256=[row["original_sha256"] for row in request["source_cells"]], elapsed_seconds=100.,
        runtime_receipt=dict(file="development-runtime.json", sha256=module.digest(tmp_path / "development-runtime.json")),
        input_receipt=dict(file="development-original-mri-proof.json", sha256=module.digest(tmp_path / "development-original-mri-proof.json"),
                           identity_sha256=mri["identity_sha256"], dicom_files=58, dicom_bytes=580),
        execution_receipt=execution, loaded_weights=loaded, stages=stages, final_sha256=module.digest(final))
    write(tmp_path / "exact-scored-anchor-receipt.json", receipt)
    return request, contract, sources, inventories, receipt


def test_complete_synthetic_native_run_binds_original_mri_members_assets_and_recomputes_fusion(tmp_path):
    request, contract, sources, inventories, _ = synthetic_run(tmp_path)
    result = module.verify_outputs(tmp_path, request, contract, sources, inventories)
    assert result["original_mri_studies"] == 58
    assert result["all_original_dino_members"] == 2
    assert result["raptor_raw_and_final_fusion_recomputed"] is True
    assert result["historical_weight_byte_parity_proven"] is False
    assert result["base_model_files_individually_hashed"] is False
    assert len(result["output_files_sha256"]) == 11
    assert result["output_sha256"] == module.digest(tmp_path / "development-exact-scored-anchor.csv")


@pytest.mark.parametrize("change", ["wrong_uid", "wrong_count", "wrong_input_hash", "same_length_native_hash",
    "runtime_gpu_count", "runtime_cpu", "missing_version", "legacy_not_complete", "missing_dino", "wrong_checkpoint_size",
    "unknown_checkpoint", "missing_a5", "wrong_rad_sha", "raw_nan", "raw_wrong_uid", "raw_missing_arm",
    "stage_nan", "stage_schema", "stage_tamper", "final_fusion_tamper", "final_byte_tamper", "submission_file",
    "bad_log_hash", "fitting", "source_cell_sha", "native_mri_receipt_path"])
def test_inconsistent_complete_claims_cannot_pass_independent_proof(tmp_path, change):
    request, contract, sources, inventories, receipt = synthetic_run(tmp_path)
    if change.startswith("wrong_") or change == "same_length_native_hash":
        if change == "wrong_uid": contract["studies"][0]["study_uid"] = "incorrect"
        if change == "wrong_count": receipt["input_receipt"]["dicom_files"] += 1
        if change == "wrong_input_hash": receipt["input_receipt"]["identity_sha256"] = "a" * 64
        if change == "same_length_native_hash": contract["studies"][0]["aggregate_sha256"] = "b" * 64
    if change in {"runtime_gpu_count", "runtime_cpu", "missing_version"}:
        path = tmp_path / "development-runtime.json"; runtime = module.read(path)
        if change == "runtime_gpu_count": runtime["gpu_count"] = 1
        if change == "runtime_cpu": runtime["torch_cuda"] = None
        if change == "missing_version": runtime["packages"].pop("torch")
        write(path, runtime); receipt["runtime_receipt"]["sha256"] = module.digest(path)
    if change == "legacy_not_complete": receipt["coat_receipt"]["legacy_success_count"] = 57
    if change == "missing_dino": receipt["execution_receipt"]["dino"]["completed"].pop()
    if change == "wrong_checkpoint_size": next(iter(receipt["loaded_weights"].values()))["bytes"] += 1
    if change == "unknown_checkpoint": receipt["loaded_weights"]["/tmp/unknown.pt"] = dict(bytes=10, sha256="a" * 64)
    if change == "missing_a5":
        receipt["loaded_weights"].pop(next(path for path in receipt["loaded_weights"] if path.endswith("m_f2.pt")))
    if change == "wrong_rad_sha":
        receipt["loaded_weights"][next(path for path in receipt["loaded_weights"] if path.endswith("ResNet50.pt"))]["sha256"] = "c" * 64
    if change.startswith("raw_"):
        path = tmp_path / "development-raptor-raw.npz"
        with np.load(path, allow_pickle=False) as saved: arrays = {key:saved[key].copy() for key in saved.files}
        if change == "raw_nan": arrays["primary"][0, 0] = np.nan
        if change == "raw_wrong_uid": arrays["study_uids"][0] = "wrong"
        if change == "raw_missing_arm": arrays.pop("legacy")
        np.savez_compressed(path, **arrays)
    if change in {"stage_nan", "stage_schema", "stage_tamper"}:
        path = tmp_path / "development-stage-1.csv"
        if change == "stage_tamper": path.write_text(path.read_text() + "\n")
        else:
            frame = pd.read_csv(path)
            if change == "stage_nan": frame.loc[0, "ACL"] = np.nan
            if change == "stage_schema": frame = frame.drop(columns="ACL")
            frame.to_csv(path, index=False); receipt["stages"][1]["sha256"] = module.digest(path)
    if change == "final_fusion_tamper":
        for name in ("development-stage-3.csv", "development-stage-4.csv", "development-exact-scored-anchor.csv"):
            path = tmp_path / name; frame = pd.read_csv(path); frame.loc[[0, 1], "ACL"] = frame.loc[[1, 0], "ACL"].to_numpy()
            frame.to_csv(path, index=False)
        receipt["stages"][3]["sha256"] = module.digest(tmp_path / "development-stage-3.csv")
        receipt["stages"][4]["sha256"] = module.digest(tmp_path / "development-stage-4.csv")
        receipt["final_sha256"] = module.digest(tmp_path / "development-exact-scored-anchor.csv")
    if change == "final_byte_tamper":
        path = tmp_path / "development-exact-scored-anchor.csv"; path.write_text(path.read_text() + "\n")
    if change == "submission_file": (tmp_path / "submission.csv").write_text("forbidden development artifact")
    if change == "bad_log_hash": receipt["execution_receipt"]["execution_log_sha256"] = "a" * 64
    if change == "fitting": receipt["model_fitting"] = True
    if change == "source_cell_sha": receipt["stages"][2]["recipe_sha256"] = "a" * 64
    if change == "native_mri_receipt_path": receipt["input_receipt"]["file"] = "../development-original-mri-proof.json"
    write(tmp_path / "exact-scored-anchor-receipt.json", receipt)
    with pytest.raises((ValueError, KeyError, AttributeError)):
        module.verify_outputs(tmp_path, request, contract, sources, inventories)


@pytest.mark.parametrize("line", ["MEMBER m_a failed on cuda:0", "1 series had a slice that would not decode",
    "12 series had a slice that would not decode", "time budget reached during decode", "[V18] calibration skipped safely",
    "[DINOsaur V4.2] a study FALLBACK", "public-frontier vote omitted"])
def test_independent_guard_rejects_failure_lines_even_when_runtime_labels_the_line_successful(tmp_path, line):
    request, contract, sources, inventories, receipt = synthetic_run(tmp_path)
    path = tmp_path / "development-execution.jsonl"
    path.write_text(module.canonical(dict(stream="stderr", line=line, rejection=None)) + "\n")
    receipt["execution_receipt"]["execution_log_sha256"] = module.digest(path)
    write(tmp_path / "exact-scored-anchor-receipt.json", receipt)
    with pytest.raises(ValueError, match="fallback"):
        module.verify_outputs(tmp_path, request, contract, sources, inventories)


def source_fixture(tmp_path):
    request, contract, sources, _, _ = synthetic_run(tmp_path)
    code = f"ORIGINAL_CELLS = {sources!r}\nSOURCE_CELLS = {sources!r}\n_RSNA70_EXPECTED_INPUTS = {contract!r}\n"
    notebook = dict(cells=[dict(cell_type="code", source=code.splitlines(keepends=True))])
    source = json.dumps(notebook)
    write(tmp_path / "development.ipynb", notebook)
    request.update(kernel_id="jitheswar/synthetic-kernel", docker_image="saved-container-digest")
    local = dict(dataset_sources=["owner/name/1"], model_sources=["metaresearch/dinov2/PyTorch/small/1"],
                 competition_sources=["rsna-knee-abnormality-detection"], kernel_sources=[], docker_image=request["docker_image"])
    write(tmp_path / "kernel-metadata.json", local); write(tmp_path / "original-mri-contract.json", contract)
    write(tmp_path / "request.json", request)
    write(tmp_path / "push-response.json", dict(ref="/code/" + request["kernel_id"], kernelId=123, versionNumber=1))
    accepted = dict(request_sha256=module.digest(tmp_path / "request.json"), response_sha256=module.digest(tmp_path / "push-response.json"),
                    kernel_id=request["kernel_id"], kernel_numeric_id=123, kernel_version=1, launch_attempt=1)
    terminal = dict(**{key:accepted[key] for key in ("request_sha256", "kernel_id", "kernel_numeric_id", "kernel_version", "launch_attempt")},
                    status="COMPLETE", response=dict(status="COMPLETE"))
    remote = dict(metadata=dict(id=123, ref=request["kernel_id"], currentVersionNumber=1, isPrivate=True, enableGpu=True,
        enableTpu=False, enableInternet=False, machineShape="NvidiaTeslaT4", dockerImage=request["docker_image"],
        datasetDataSources=["owner/name"], modelDataSources=local["model_sources"],
        competitionDataSources=local["competition_sources"], kernelDataSources=[]), blob=dict(source=source))
    validation = dict(status="PASS", exact_code_cells=True, pinned_docker=True, private_internet_off=True,
                      attachments_canonical_refs_verified=True, remote_source_sha256=hashlib.sha256(source.encode()).hexdigest())
    snapshot = tmp_path / "source-snapshot"; snapshot.mkdir()
    for index, text in enumerate(sources):
        (snapshot / f"original-cell-{index}.py").write_text(text)
        (snapshot / f"relocated-cell-{index}.py").write_text(text)
    return request, accepted, terminal, remote, validation


def test_exact_authenticated_source_bridge_accepts_only_matching_numeric_version_and_cells(tmp_path):
    values = source_fixture(tmp_path)
    assert len(module.verify_source(tmp_path, *values)) == 5


@pytest.mark.parametrize("change", ["terminal_error", "terminal_numeric", "wrong_version", "internet_on", "container_changed",
    "dataset_version_changed", "source_changed", "push_changed", "source_validation_false", "original_source_changed"])
def test_remote_or_terminal_identity_and_settings_tamper_is_rejected(tmp_path, change):
    request, accepted, terminal, remote, validation = source_fixture(tmp_path)
    if change == "terminal_error": terminal["response"]["status"] = "ERROR"
    if change == "terminal_numeric": terminal["kernel_numeric_id"] = 124
    if change == "wrong_version": remote["metadata"]["currentVersionNumber"] = 2
    if change == "internet_on": remote["metadata"]["enableInternet"] = True
    if change == "container_changed": remote["metadata"]["dockerImage"] = "latest"
    if change == "dataset_version_changed": remote["metadata"]["datasetDataSources"] = ["owner/name/2"]
    if change == "source_changed": remote["blob"]["source"] = remote["blob"]["source"].replace("scientific_cell_0", "different_cell_0")
    if change == "push_changed": write(tmp_path / "push-response.json", dict(ref="/code/different/kernel", kernelId=123, versionNumber=1))
    if change == "source_validation_false": validation["pinned_docker"] = False
    if change == "original_source_changed": (tmp_path / "source-snapshot/original-cell-0.py").write_text("changed")
    with pytest.raises(ValueError): module.verify_source(tmp_path, request, accepted, terminal, remote, validation)
