"""Exercise portable inference mechanics without external weights or real MRI."""

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from rsna_knee.biomedclip_inference import (
    BiomedClipVision, CLIP_MEAN, CLIP_STD, FEATURE_IDENTITY_SHA256, HEAD_ID,
    MedicalFoldHead, MedicalInferenceError, MedicalStudyPredictor, load_medical_head, reference_fallback,
)


class FixtureVision(nn.Module):
    def __init__(self, *, zero=False):
        super().__init__();self.calls=[];self.zero=zero

    def forward(self, images):
        self.calls.append(images.clone())
        result=torch.zeros((len(images),512),dtype=torch.float32)
        if not self.zero:result[:,0]=2;result[:,1]=1
        return result


class RecordingHead(MedicalFoldHead):
    def __init__(self, bias,fold=0,arm="report-supervision"):
        super().__init__(fold,arm);self.batch_sizes=[]
        for value in self.parameters():value.data.zero_()
        self[2].bias.data.fill_(bias)

    def forward(self, values):
        self.batch_sizes.append(len(values));return super().forward(values)


def feature_inputs(batch):
    values=torch.zeros(batch,16,8,512);values[:,0,0,0]=1
    slices=torch.zeros(batch,16,8,dtype=torch.bool);slices[:,0,0]=True
    series=torch.zeros(batch,16,dtype=torch.bool);series[:,0]=True
    descriptors=torch.zeros(batch,16,10);descriptors[:,0,0]=1
    return values,slices,series,descriptors


def predictor():
    heads={i:RecordingHead(bias,i) for i,bias in enumerate((-3.,-1.,1.))}
    priors=torch.tensor([[.1]*12,[.2]*12,[.3]*12])
    return MedicalStudyPredictor(BiomedClipVision(FixtureVision()),heads,priors),heads,priors


def test_encoder_visits_only_valid_indices_with_exact_normalization_and_padding():
    visual=FixtureVision();encoder=BiomedClipVision(visual)
    images=torch.full((2,16,8,1,224,224),.25)
    slices=torch.zeros(2,16,8,dtype=torch.bool);series=torch.zeros(2,16,dtype=torch.bool)
    slices[0,0,0]=slices[0,0,7]=slices[1,15,7]=True;series[0,0]=series[1,15]=True
    slices[0,1,0]=True  # A false series mask makes this slot invalid.
    features=encoder.encode_packed(images,slices,series);valid=slices&series[:,:,None]
    assert len(visual.calls)==3 and all(value.shape==(1,3,224,224) for value in visual.calls)
    expected=(torch.full((1,3,224,224),.25)-torch.tensor(CLIP_MEAN).reshape(1,3,1,1))/torch.tensor(CLIP_STD).reshape(1,3,1,1)
    assert torch.equal(visual.calls[0],expected)
    assert not features[~valid].any() and torch.allclose(torch.linalg.vector_norm(features[valid],dim=-1),torch.ones(3))


def test_zero_or_nonfinite_valid_vision_output_is_rejected():
    images=torch.zeros(1,16,8,1,224,224);_,slices,series,_=feature_inputs(1)
    with pytest.raises(MedicalInferenceError,match="Unit"):
        BiomedClipVision(FixtureVision(zero=True)).encode_packed(images,slices,series)
    images[0,0,0,0,0,0]=float('nan')
    with pytest.raises(MedicalInferenceError,match="Finite"):
        BiomedClipVision(FixtureVision()).encode_packed(images,slices,series)


@pytest.mark.parametrize("bad_output",["nonfinite","wrong_width"])
def test_invalid_projected_outputs_are_rejected(bad_output):
    class BrokenVision(nn.Module):
        def forward(self, images):
            return torch.full((1,511 if bad_output=="wrong_width" else 512),float('nan') if bad_output=="nonfinite" else 1.)
    images=torch.zeros(1,16,8,1,224,224);_,slices,series,_=feature_inputs(1)
    with pytest.raises(MedicalInferenceError,match="projected"):
        BiomedClipVision(BrokenVision()).encode_packed(images,slices,series)


def test_fold_and_probability_mean_are_distinct_and_heads_use_audited_row_groups():
    model,heads,_=predictor();inputs=feature_inputs(5)
    selected=model.predict_features(*inputs,fold=1)
    assert torch.equal(selected.probabilities,torch.full((5,12),float(torch.sigmoid(torch.tensor(-1.)))))
    for head in heads.values():head.batch_sizes.clear()
    ensemble=model.predict_features(*inputs)
    expected=torch.sigmoid(torch.tensor([-3.,-1.,1.])).mean()
    assert torch.equal(ensemble.probabilities,torch.full((5,12),float(expected)))
    assert all(head.batch_sizes==[2,2,1] for head in heads.values()) and not ensemble.fallback_used.any()


def test_all_invalid_rows_keep_their_place_and_receive_explicit_fold_or_mean_priors():
    model,_,priors=predictor();features,slices,series,descriptors=feature_inputs(3)
    features[1]=0;slices[1]=False;series[1]=False
    selected=model.predict_features(features,slices,series,descriptors,fold=2)
    ensemble=model.predict_features(features,slices,series,descriptors)
    assert selected.probabilities.shape==(3,12) and selected.fallback_used.tolist()==[False,True,False]
    assert torch.equal(selected.probabilities[1],priors[2]) and torch.equal(ensemble.probabilities[1],priors.mean(dim=0))


def test_empty_image_studies_never_run_the_encoder_and_still_get_bounded_predictions():
    model,_,priors=predictor();images=torch.zeros(2,16,8,1,224,224)
    result=model.predict_packed(images,torch.zeros(2,16,8,dtype=torch.bool),torch.zeros(2,16,dtype=torch.bool),torch.zeros(2,16,10))
    assert len(model.vision.visual.calls)==0 and result.fallback_used.tolist()==[True,True]
    assert torch.equal(result.probabilities,priors.mean(dim=0).repeat(2,1))


@pytest.mark.parametrize("change",["bad_fold","false_fold","nonunit","invalid_nonzero","wrong_descriptor_width","float_mask"])
def test_inference_rejects_changes_to_the_admitted_contract(change):
    model,_,_=predictor();features,slices,series,descriptors=feature_inputs(1);fold=None
    if change=="bad_fold":fold=3
    if change=="false_fold":fold=False
    if change=="nonunit":features*=2
    if change=="invalid_nonzero":features[0,15,7,0]=1
    if change=="wrong_descriptor_width":descriptors=torch.zeros(1,16,9)
    if change=="float_mask":slices=slices.float()
    with pytest.raises(MedicalInferenceError):model.predict_features(features,slices,series,descriptors,fold=fold)


def test_reference_fallback_uses_only_supplied_binary_reference_rows():
    labels=[[1,0,None]+[0]*9,[0,0,None]+[1]*9]
    priors=reference_fallback(labels)
    assert priors.dtype==torch.float32 and priors.shape==(12,)
    assert priors[:3].tolist()==[.5,.25,.5]
    with pytest.raises(MedicalInferenceError):reference_fallback([])
    with pytest.raises(MedicalInferenceError):reference_fallback([[2]*12])


@pytest.mark.parametrize("change",["missing_fold","bad_prior","wrong_head_width","swapped_folds","mixed_arms"])
def test_predictor_constructor_requires_complete_heads_priors_and_architecture(change):
    _,heads,priors=predictor()
    if change=="missing_fold":heads.pop(2)
    if change=="bad_prior":priors[0,0]=float('nan')
    if change=="wrong_head_width":heads[0][0]=nn.Linear(2075,32)
    if change=="swapped_folds":heads[0],heads[1]=heads[1],heads[0]
    if change=="mixed_arms":heads[1]=RecordingHead(-1.,1,arm="reference-only")
    with pytest.raises(MedicalInferenceError):MedicalStudyPredictor(BiomedClipVision(FixtureVision()),heads,priors)


def checkpoint(tmp_path,change=None):
    identity=dict(head_id=HEAD_ID,feature_identity_sha256=FEATURE_IDENTITY_SHA256,fold=0,arm="report-supervision",plan_sha256="synthetic")
    head=RecordingHead(-1)
    value=dict(schema_version="rsna-frozen-head-checkpoint-v1",head_id=HEAD_ID,identity=identity,checkpoint_kind="final_selected",
        counts=dict(successful_optimizer_steps=7499 if change=="wrong_budget" else 7500),head_state_dict=head.state_dict())
    if change=="wrong_identity":value["identity"]={**identity,"feature_identity_sha256":"different"}
    if change=="bad_weight":value["head_state_dict"]["0.weight"][0,0]=float('nan')
    if change=="invalid_counts":value["counts"]="malformed"
    path=tmp_path/"synthetic.pt";torch.save(value,path)
    return path,hashlib.sha256(path.read_bytes()).hexdigest(),identity


def test_head_loading_is_restricted_hash_gated_and_preserves_torch_rng(tmp_path,monkeypatch):
    path,checksum,identity=checkpoint(tmp_path);before=torch.get_rng_state().clone();original=torch.load;calls=[]
    def load(*args,**kwargs):
        calls.append(kwargs);return original(*args,**kwargs)
    monkeypatch.setattr(torch,"load",load)
    head=load_medical_head(path,expected_sha256=checksum,expected_identity=identity)
    assert torch.equal(before,torch.get_rng_state()) and calls==[dict(map_location="cpu",weights_only=True)]
    assert not head.training and all(not p.requires_grad for p in head.parameters())
    calls.clear()
    with pytest.raises(MedicalInferenceError,match="bytes"):
        load_medical_head(path,expected_sha256="wrong",expected_identity=identity)
    assert calls==[]


def test_head_loading_retains_fp32_under_changed_global_default_dtype(tmp_path):
    path,checksum,identity=checkpoint(tmp_path);original_dtype=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        head=load_medical_head(path,expected_sha256=checksum,expected_identity=identity)
    finally:torch.set_default_dtype(original_dtype)
    assert all(p.dtype==torch.float32 and p.device.type=="cpu" for p in head.parameters())


@pytest.mark.parametrize("change",["wrong_budget","wrong_identity","bad_weight","invalid_counts"])
def test_invalid_final_head_artifacts_are_rejected(tmp_path,change):
    path,checksum,identity=checkpoint(tmp_path,change)
    with pytest.raises(MedicalInferenceError):load_medical_head(path,expected_sha256=checksum,expected_identity=identity)


def test_external_checkpoint_hash_mismatch_is_rejected_before_any_deserialization(tmp_path,monkeypatch):
    path=tmp_path/"untrusted.pt";path.write_bytes(b"not a checkpoint")
    def forbidden(*args,**kwargs):raise AssertionError("Deserialization must not happen")
    monkeypatch.setattr(torch,"load",forbidden)
    with pytest.raises(MedicalInferenceError,match="Pinned"):
        BiomedClipVision.from_checkpoint(path,Path("unused-config"))
