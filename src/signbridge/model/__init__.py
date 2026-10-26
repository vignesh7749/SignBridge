"""Module 7: Sign Recognition Model Package (Stage 6).

Provides production static sign classification, model persistence,
dataset splitting with data leakage prevention, and inference interfaces.
"""

from signbridge.model.base import ModelMetadata, PredictionResult
from signbridge.model.classifier import SignClassifier
from signbridge.model.dataset_loader import DatasetSplits, load_and_validate_dataset
from signbridge.model.persistence import load_sign_classifier, save_sign_classifier
from signbridge.model.predictor import SignPredictor, get_default_predictor

__all__ = [
    "ModelMetadata",
    "PredictionResult",
    "SignClassifier",
    "SignPredictor",
    "DatasetSplits",
    "get_default_predictor",
    "load_and_validate_dataset",
    "load_sign_classifier",
    "save_sign_classifier",
]
