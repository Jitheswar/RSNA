"""Named image-model configurations for controlled architecture comparisons."""

from __future__ import annotations

from dataclasses import replace

from .preprocessing import PackingConfig, PreprocessConfig
from .model import ANATOMICAL_POOLING_ID, DINOV2_MODEL_ID, SIMPLE_POOLING_ID
from .dinov2 import (
    DINOV2_ENCODER_ID,
    DINOV2_IMAGE_SIZE,
    DINOV2_POOLING_ID,
    DINOV2_PREPROCESSING_ID,
    DINOV2_SOURCE_REVISION,
    DINOV2_WEIGHTS_SHA256,
)
from .training import TrainingConfig


MECHANICS_BASELINE_VARIANT = "mechanics-baseline"
TICKET19_VARIANT = "ticket19-wider-slice-encoder-v1"
TICKET19_ENCODER_CHANNELS = (16, 32, 64)
TICKET38_DINOV2_VARIANT = "ticket38-frozen-dinov2-small-vits14"
IMAGE_MODEL_VARIANTS = (
    MECHANICS_BASELINE_VARIANT,
    TICKET19_VARIANT,
    TICKET38_DINOV2_VARIANT,
)


def image_model_training_config(
    variant: str,
    *,
    preprocessing_config: PreprocessConfig | None = None,
    packing_config: PackingConfig | None = None,
    pooling_id: str = ANATOMICAL_POOLING_ID,
) -> TrainingConfig:
    """Build the frozen Ticket 19 setting or the established mechanics baseline."""

    packing = packing_config or (
        PackingConfig(
            max_slices=preprocessing_config.max_slices,
            max_series=preprocessing_config.max_series,
        )
        if preprocessing_config is not None
        else PackingConfig()
    )
    effective_preprocessing = (
        replace(
            preprocessing_config,
            max_slices=packing.max_slices,
            max_series=packing.max_series,
        )
        if preprocessing_config is not None
        else None
    )
    baseline = TrainingConfig(
        image_size=(
            effective_preprocessing.image_size
            if effective_preprocessing is not None
            else (128, 128)
        ),
        max_slices=packing.max_slices,
        max_series=packing.max_series,
        preprocessing_config=effective_preprocessing,
    )
    if variant == MECHANICS_BASELINE_VARIANT:
        return baseline
    if variant == TICKET19_VARIANT:
        return replace(baseline, channels=TICKET19_ENCODER_CHANNELS)
    if variant == TICKET38_DINOV2_VARIANT:
        if pooling_id != DINOV2_POOLING_ID:
            raise ValueError("Ticket 38 freezes plane-stratified series pooling")
        source_preprocessing = effective_preprocessing or baseline.preprocess_config()
        dinov2_preprocessing = replace(
            source_preprocessing,
            image_size=DINOV2_IMAGE_SIZE,
            representation_id=DINOV2_PREPROCESSING_ID,
        )
        return replace(
            baseline,
            image_size=DINOV2_IMAGE_SIZE,
            max_slices=source_preprocessing.max_slices,
            max_series=source_preprocessing.max_series,
            channels=(64, 128, 256),
            model_id=DINOV2_MODEL_ID,
            pooling_id=DINOV2_POOLING_ID,
            encoder_id=DINOV2_ENCODER_ID,
            encoder_weights_sha256=DINOV2_WEIGHTS_SHA256,
            encoder_source_revision=DINOV2_SOURCE_REVISION,
            head_hidden_size=32,
            preprocessing_config=dinov2_preprocessing,
        )
    raise ValueError(f"unsupported image-model variant {variant!r}")


__all__ = [
    "IMAGE_MODEL_VARIANTS",
    "MECHANICS_BASELINE_VARIANT",
    "TICKET19_ENCODER_CHANNELS",
    "TICKET19_VARIANT",
    "TICKET38_DINOV2_VARIANT",
    "image_model_training_config",
]
