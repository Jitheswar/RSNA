"""Named image-model configurations for controlled architecture comparisons."""

from __future__ import annotations

from dataclasses import replace

from .preprocessing import PreprocessConfig
from .training import TrainingConfig


MECHANICS_BASELINE_VARIANT = "mechanics-baseline"
TICKET19_VARIANT = "ticket19-wider-slice-encoder-v1"
TICKET19_ENCODER_CHANNELS = (16, 32, 64)
IMAGE_MODEL_VARIANTS = (MECHANICS_BASELINE_VARIANT, TICKET19_VARIANT)


def image_model_training_config(
    variant: str,
    *,
    preprocessing_config: PreprocessConfig | None = None,
) -> TrainingConfig:
    """Build the frozen Ticket 19 setting or the established mechanics baseline."""

    baseline = TrainingConfig(preprocessing_config=preprocessing_config)
    if variant == MECHANICS_BASELINE_VARIANT:
        return baseline
    if variant == TICKET19_VARIANT:
        return replace(baseline, channels=TICKET19_ENCODER_CHANNELS)
    raise ValueError(f"unsupported image-model variant {variant!r}")


__all__ = [
    "IMAGE_MODEL_VARIANTS",
    "MECHANICS_BASELINE_VARIANT",
    "TICKET19_ENCODER_CHANNELS",
    "TICKET19_VARIANT",
    "image_model_training_config",
]
