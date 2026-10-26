"""Production static sign predictor interface."""

from pathlib import Path
from typing import Sequence, Union
import numpy as np

from signbridge.model.base import ModelMetadata, PredictionResult
from signbridge.model.classifier import SignClassifier
from signbridge.model.persistence import load_sign_classifier

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "sign_classifier"


class SignPredictor:
    """Production static sign predictor.

    Provides a clean, validated prediction interface consuming 94-dimensional
    feature vectors and returning structured classification results.
    """

    def __init__(self, model_dir: Union[Path, str, None] = None) -> None:
        target_dir = Path(model_dir or DEFAULT_MODEL_DIR).resolve()
        if not target_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found at: {target_dir}. "
                "Ensure the model has been trained before initializing SignPredictor."
            )
        self._classifier: SignClassifier = load_sign_classifier(target_dir)

    @property
    def classifier(self) -> SignClassifier:
        return self._classifier

    @property
    def metadata(self) -> ModelMetadata:
        return self._classifier.metadata

    @property
    def vocabulary(self) -> Sequence[str]:
        return self._classifier.vocabulary

    @property
    def is_ready(self) -> bool:
        return self._classifier.is_loaded

    def predict(self, features: np.ndarray) -> PredictionResult:
        """Predicts the static sign for a single 94-dimensional feature vector.

        Args:
            features: 1D (94,) or 2D (1, 94) array of floats.

        Returns:
            PredictionResult containing class_id, class_name, confidence,
            top_margin, top_candidates, and inference_latency_ms.

        Raises:
            ValueError: If features dimension != 94 or features contains NaN/Inf.
            TypeError: If features is not convertible to float32 numpy array.
        """
        return self._classifier.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Computes calibrated probability distributions across all 26 sign classes.

        Args:
            features: 1D (94,) or 2D (N, 94) array of floats.

        Returns:
            2D numpy array of shape (N, 26) with calibrated class probabilities.
        """
        return self._classifier.predict_proba(features)


_cached_predictor: SignPredictor | None = None


def get_default_predictor() -> SignPredictor:
    """Returns a cached singleton SignPredictor instance using the default model path."""
    global _cached_predictor
    if _cached_predictor is None:
        _cached_predictor = SignPredictor()
    return _cached_predictor
