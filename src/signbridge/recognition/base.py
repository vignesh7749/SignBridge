"""Module 7: Sign Recognizer Strategy Interface.

Abstract base class allowing interchangeable recognition backends
(Heuristic, KNN, Random Forest, PyTorch MLP, LSTM, Transformer)
without modifying camera, feature, or UI subsystems.
"""

from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np

from signbridge.recognition.confidence import ScoredPrediction


class ISignRecognizer(ABC):
    """Abstract Strategy interface for sign language recognition models."""

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Loads model weights, hyperparameters, and class labels from disk."""
        pass

    @abstractmethod
    def predict(self, feature_vector: np.ndarray) -> ScoredPrediction:
        """Infers sign category and confidence metrics from a normalized feature vector.

        Args:
            feature_vector: 1D array of normalized features (length 63 for 21 3D points).

        Returns:
            ScoredPrediction containing top label, confidence score, and candidate distribution.
        """
        pass

    @property
    @abstractmethod
    def vocabulary(self) -> Sequence[str]:
        """Returns the list of all supported sign tokens in the vocabulary."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Returns True if model weights are successfully loaded into memory."""
        pass
