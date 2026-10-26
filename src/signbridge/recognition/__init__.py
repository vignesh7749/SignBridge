"""Sign recognition interfaces and prediction data structures."""

from signbridge.recognition.base import ISignRecognizer
from signbridge.recognition.confidence import ScoredPrediction

__all__ = ["ISignRecognizer", "ScoredPrediction"]
