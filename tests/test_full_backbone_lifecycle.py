"""Shared trainer resume, pixel admission and native prediction boundaries."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from rsna_knee.evaluation import predict_training_examples
from rsna_knee.model import build_study_model
from rsna_knee.resnet18 import (
    ANATOMICAL_POOLING_ID, RESNET18_FIXTURE_ENCODER_ID, RESNET18_FULL_FINETUNE_MODEL_ID,
)
from rsna_knee.training import (
    SupervisionProvenance, TrainingConfig, TrainingConfigurationError,
    TrainingDataError, TrainingExample, TrainingInterrupted, load_training_checkpoint,
    train_fixed_budget,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def full_config():
    return TrainingConfig(image_size=(224,224), max_series=1, max_slices=1,
        model_id=RESNET18_FULL_FINETUNE_MODEL_ID, pooling_id=ANATOMICAL_POOLING_ID,
        encoder_id=RESNET18_FIXTURE_ENCODER_ID, successful_optimizer_steps=3,
        encoder_learning_rate=1e-5, learning_rate=1e-4, checkpoint_interval_steps=1,
        head_hidden_size=8)


def full_examples(prefix, weak=False):
    rng = np.random.default_rng(63)
    result = []
    for index in range(2):
        descriptors = np.zeros((1,10),dtype=np.float32);descriptors[0,0]=1
        provenance = (SupervisionProvenance(source_kind='weak',source_id='fixture-report',
            weak_label_version='fixture-v1',source_report_ref='fixture-report',
            source_report_sha256='1'*64) if weak else SupervisionProvenance())
        result.append(TrainingExample(f'{prefix}-{index}',rng.random((1,1,1,224,224),dtype=np.float32),
            np.ones((1,1),dtype=np.bool_),np.ones(1,dtype=np.bool_),
            np.full(12,index,dtype=np.float32),np.ones(12,dtype=np.bool_),
            provenance=provenance,series_descriptors=descriptors))
    return tuple(result)


def test_shared_full_backbone_resume_matches_uninterrupted_mixed_source_training(tmp_path):
    config=full_config();gold=full_examples('gold');weak=full_examples('weak',weak=True)
    torch.manual_seed(64)
    model=build_study_model(config.model_config())
    initial={k:v.clone() for k,v in model.state_dict().items()}
    root=tmp_path/'staged'
    args=dict(additional_examples=weak,additional_batch_order_seed=71,fold=0,
              identity={'pixel_cohort_sha256':'fixture','scientific_plan':'same-plan'})
    with pytest.raises(TrainingInterrupted):
        train_fixed_budget(gold,config,model=model,artifact_dir=root,interrupt_after_steps=1,**args)
    checkpoint=load_training_checkpoint(root/'checkpoints/latest.pt',expected_config=config)
    assert [g['lr'] for g in checkpoint['optimizer_state_dict']['param_groups']]==[1e-5,1e-4]
    assert checkpoint['progress']['successful_optimizer_steps']==1
    resumed=train_fixed_budget(gold,config,artifact_dir=root,resume_from=root/'checkpoints/latest.pt',**args)
    baseline=build_study_model(config.model_config());baseline.load_state_dict(initial,strict=True)
    uninterrupted=train_fixed_budget(gold,config,model=baseline,artifact_dir=tmp_path/'continuous',**args)
    assert resumed.successful_optimizer_steps==uninterrupted.successful_optimizer_steps==3
    assert resumed.loss_history==uninterrupted.loss_history
    assert resumed.exposures_by_source==uninterrupted.exposures_by_source
    for key,value in resumed.model.state_dict().items():
        assert torch.equal(value,uninterrupted.model.state_dict()[key])
    assert not torch.equal(resumed.model.encoder.conv1.weight,initial['encoder.conv1.weight'])
    predictions=predict_training_examples(resumed.model,gold)
    assert set(predictions)=={'gold-0','gold-1'}
    images=torch.from_numpy(np.stack([e.images for e in gold]));slices=torch.from_numpy(np.stack([e.slice_mask for e in gold]))
    series=torch.from_numpy(np.stack([e.series_mask for e in gold]));descriptors=torch.from_numpy(np.stack([e.series_descriptors for e in gold]))
    with torch.no_grad():expected=resumed.model.predict_proba(images,slices,series,descriptors).numpy()
    assert np.array_equal(np.stack([predictions[e.study_instance_uid] for e in gold]),expected)


def test_full_backbone_training_rejects_accelerator_without_its_own_admission():
    with pytest.raises(TrainingConfigurationError,match='scoped to CPU'):
        replace(full_config(),device='cuda')


def test_native_inference_and_candidate_ensemble_keep_anatomical_descriptors():
    from rsna_knee.candidate import _CandidateEnsemble
    from rsna_knee.inference import InferenceEngine
    from rsna_knee.preprocessing import PackedStudy
    config=full_config();model=build_study_model(config.model_config()).eval()
    example=full_examples('native')[0]
    packed=PackedStudy(example.study_instance_uid,example.images,example.slice_mask,
                       example.series_mask,((0,),),example.series_descriptors)
    engine=InferenceEngine(model,config.preprocess_config())
    images=torch.from_numpy(example.images).unsqueeze(0)
    slices=torch.from_numpy(example.slice_mask).unsqueeze(0);series=torch.from_numpy(example.series_mask).unsqueeze(0)
    descriptors=torch.from_numpy(example.series_descriptors).unsqueeze(0)
    with torch.no_grad():
        expected=model.predict_proba(images,slices,series,descriptors).squeeze(0).numpy()
        ensemble=_CandidateEnsemble((model,model))
        assert torch.equal(ensemble.predict_proba(images,slices,series,descriptors).squeeze(0),torch.from_numpy(expected))
    assert np.array_equal(engine._predict_packed(packed),expected)


def test_file_backed_pixels_stack_fingerprint_and_shared_training_without_retained_bags(tmp_path):
    from rsna_knee.pixel_cache import PackedPixelArray,write_pixel_entry,PixelCacheError
    from rsna_knee.training import fingerprint_training_examples
    slices=np.zeros((16,8),dtype=np.bool_);slices[0,0]=True
    series=np.zeros(16,dtype=np.bool_);series[0]=True
    descriptors=np.zeros((16,10),dtype=np.float32);descriptors[0,0]=1
    examples=[]
    for index in range(2):
        uid=f'pixel-{index}';path=tmp_path/f'{uid}.npz'
        metadata={'study_instance_uid':uid,'pixel_identity_sha256':'pixels','source_identity_sha256':'source',
                  'materialization_plan_sha256':'plan','job_id':'123','source_kind':'ticket35-prepared-series-cache-read-only'}
        checksum=write_pixel_entry(path,pixels=np.full((1,1,224,224),0.2+index*0.3,dtype=np.float32),
            slice_mask=slices,series_mask=series,series_descriptors=descriptors,metadata=metadata)
        pixels=PackedPixelArray(path,checksum,uid,'pixels','source','plan')
        examples.append(TrainingExample(uid,pixels,slices,series,np.full(12,index,dtype=np.float32),
            np.ones(12,dtype=np.bool_),series_descriptors=descriptors))
    assert all(isinstance(e.images,PackedPixelArray) for e in examples)
    eager=[replace(e,images=np.asarray(e.images)) for e in examples]
    assert fingerprint_training_examples(examples)==fingerprint_training_examples(eager)
    config=TrainingConfig(image_size=(224,224),channels=(1,1,1),successful_optimizer_steps=1)
    result=train_fixed_budget(examples,config,artifact_dir=tmp_path/'lazy-fit')
    assert result.successful_optimizer_steps==1
    assert all(isinstance(e.images,PackedPixelArray) for e in examples)
    with pytest.raises(TrainingDataError,match='different training study'):
        replace(examples[0],study_instance_uid='wrong-study')
    # Every later read still detects changed bytes; admission is not a permanent cache bypass.
    path=examples[0].images.path
    with path.open('ab') as handle:handle.write(b'corruption')
    with pytest.raises(PixelCacheError):
        np.asarray(examples[0].images)
