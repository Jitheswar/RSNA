"""Shared MRI-only mechanics core for the RSNA knee project."""

from .domain import DicomSlice, SeriesRecord, StudyRecord
from .dicom import DicomDecodeError, decode_dataset, read_dicom_slice
from .inference import InferenceBatch, InferenceEngine, StudyPrediction, infer_studies
from .model import HierarchicalStudyModel, MechanicsModel, ModelConfig
from .preprocessing import (
    PackedStudy,
    PreparedSeries,
    PreparedStudy,
    PreprocessConfig,
    PreprocessingError,
    SeriesFailure,
    StudyPackingError,
    pack_study,
    prepare_series,
    prepare_study,
)
from .submission import SubmissionError, validate_predictions, write_submission
from .targets import NUM_TARGETS, SUBMISSION_COLUMNS, TARGET_COLUMNS

__all__ = [
    "DicomSlice",
    "DicomDecodeError",
    "decode_dataset",
    "read_dicom_slice",
    "SeriesRecord",
    "StudyRecord",
    "InferenceBatch",
    "InferenceEngine",
    "StudyPrediction",
    "infer_studies",
    "HierarchicalStudyModel",
    "MechanicsModel",
    "ModelConfig",
    "PackedStudy",
    "PreparedSeries",
    "PreparedStudy",
    "PreprocessConfig",
    "PreprocessingError",
    "SeriesFailure",
    "StudyPackingError",
    "pack_study",
    "prepare_series",
    "prepare_study",
    "SubmissionError",
    "validate_predictions",
    "write_submission",
    "NUM_TARGETS",
    "SUBMISSION_COLUMNS",
    "TARGET_COLUMNS",
]
