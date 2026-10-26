"""Production SignClassifier inference wrapper."""

import time
from typing import Any, Sequence
import numpy as np

from signbridge.features.schema import FeatureSchema
from signbridge.model.base import ModelMetadata, PredictionResult
from signbridge.recognition.base import ISignRecognizer
from signbridge.recognition.confidence import ScoredPrediction


class SignClassifier(ISignRecognizer):
    """Production static sign classifier.

    Wraps a trained scikit-learn estimator or pipeline, validating 94D input
    integrity, computing class probabilities, and measuring inference latency.
    """

    def __init__(self, model: Any, metadata: ModelMetadata) -> None:
        self._model = model
        self._metadata = metadata
        self._classes = sorted(metadata.class_to_id.keys(), key=lambda k: metadata.class_to_id[k])
        self._num_classes = len(self._classes)

    @property
    def model(self) -> Any:
        return self._model

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def vocabulary(self) -> Sequence[str]:
        """Returns ordered list of recognized sign class labels."""
        return self._classes

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def validate_features(self, features: np.ndarray) -> np.ndarray:
        """Validates and formats the feature vector for inference.

        Args:
            features: 1D (94,) or 2D (N, 94) array of floats.

        Returns:
            Contiguous 2D float32 numpy array with shape (N, 94).

        Raises:
            TypeError: If input is not a numpy array or convertible list.
            ValueError: If dimensions differ from 94 or non-finite values are present.
        """
        if not isinstance(features, np.ndarray):
            try:
                features = np.asarray(features, dtype=np.float32)
            except Exception as e:
                raise TypeError(f"Features must be a numpy ndarray, got {type(features)}") from e

        if features.ndim == 1:
            if features.shape[0] != FeatureSchema.TOTAL_DIMENSIONS:
                raise ValueError(
                    f"Feature vector dimension mismatch: expected {FeatureSchema.TOTAL_DIMENSIONS}, "
                    f"got {features.shape[0]}"
                )
            features = features.reshape(1, -1)
        elif features.ndim == 2:
            if features.shape[1] != FeatureSchema.TOTAL_DIMENSIONS:
                raise ValueError(
                    f"Feature matrix second dimension mismatch: expected {FeatureSchema.TOTAL_DIMENSIONS}, "
                    f"got {features.shape[1]}"
                )
        else:
            raise ValueError(f"Features must be 1D or 2D array, got {features.ndim}D shape {features.shape}")

        if not np.all(np.isfinite(features)):
            raise ValueError("Feature vector contains non-finite values (NaN or Inf)")

        return np.ascontiguousarray(features, dtype=np.float32)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Returns probability distributions over classes for the given feature inputs."""
        X = self.validate_features(features)
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        elif hasattr(self._model, "decision_function"):
            decision = self._model.decision_function(X)
            if decision.ndim == 1:
                # Binary decision
                decision = np.vstack([-decision, decision]).T
            # Softmax
            exp_d = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_d / np.sum(exp_d, axis=1, keepdims=True)
        else:
            # Fallback one-hot based on predict
            preds = self._model.predict(X)
            probs = np.zeros((len(X), self._num_classes), dtype=np.float32)
            for i, p in enumerate(preds):
                probs[i, int(p)] = 1.0
            return probs

    def predict(self, features: np.ndarray) -> PredictionResult:
        """Performs static sign classification on a single 94D feature vector.

        Args:
            features: 1D (94,) or 2D (1, 94) feature vector.

        Returns:
            PredictionResult containing predicted class_id, class_name, confidence,
            ranked top candidates, and measured inference latency in ms.
        """
        X = self.validate_features(features)
        if X.shape[0] != 1:
            raise ValueError(f"predict() expects a single sample, got batch size {X.shape[0]}. Use predict_batch().")

        t0 = time.perf_counter()
        probs_matrix = self.predict_proba(X)
        pred_class_id = int(np.argmax(probs_matrix[0]))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        confidence = float(probs_matrix[0, pred_class_id])
        class_name = self._metadata.id_to_class.get(pred_class_id, str(pred_class_id))

        # Build probabilities dictionary and top candidates
        prob_dict = {
            self._metadata.id_to_class.get(i, str(i)): float(probs_matrix[0, i])
            for i in range(self._num_classes)
        }
        sorted_candidates = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
        top_candidates = sorted_candidates[:3]

        return PredictionResult(
            class_id=pred_class_id,
            class_name=class_name,
            confidence=confidence,
            probabilities=prob_dict,
            top_candidates=top_candidates,
            inference_latency_ms=elapsed_ms,
        )

    def predict_scored(self, features: np.ndarray) -> ScoredPrediction:
        """Adapter satisfying the ISignRecognizer interface."""
        res = self.predict(features)
        probs_arr = np.array([res.probabilities.get(c, 0.0) for c in self._classes], dtype=np.float32)
        return ScoredPrediction(
            label=res.class_name,
            confidence=res.confidence,
            is_valid=True,
            probabilities=probs_arr,
            top_candidates=res.top_candidates,
        )

    def predict_batch(self, features: np.ndarray) -> list[PredictionResult]:
        """Performs batch sign classification across multiple 94D feature vectors."""
        X = self.validate_features(features)
        t0 = time.perf_counter()
        probs_matrix = self.predict_proba(X)
        pred_class_ids = np.argmax(probs_matrix, axis=1)
        total_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_sample_ms = total_elapsed_ms / max(1, len(X))

        results: list[PredictionResult] = []
        for i in range(len(X)):
            cid = int(pred_class_ids[i])
            cname = self._metadata.id_to_class.get(cid, str(cid))
            conf = float(probs_matrix[i, cid])
            p_dict = {
                self._metadata.id_to_class.get(j, str(j)): float(probs_matrix[i, j])
                for j in range(self._num_classes)
            }
            top_cand = sorted(p_dict.items(), key=lambda it: it[1], reverse=True)[:3]
            results.append(
                PredictionResult(
                    class_id=cid,
                    class_name=cname,
                    confidence=conf,
                    probabilities=p_dict,
                    top_candidates=top_cand,
                    inference_latency_ms=per_sample_ms,
                )
            )

        return results

    def load(self, model_path: str) -> None:
        """Replaces the active model with weights from disk."""
        from signbridge.model.persistence import load_sign_classifier
        loaded = load_sign_classifier(model_path)
        self._model = loaded.model
        self._metadata = loaded.metadata
        self._classes = sorted(self._metadata.class_to_id.keys(), key=lambda k: self._metadata.class_to_id[k])
        self._num_classes = len(self._classes)
