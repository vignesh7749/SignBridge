"""Word-sign classification models and inference engine.

Provides WordSignClassifier wrapping scikit-learn sequence models (MLP / Random Forest)
operating on 2,560-dimensional temporal trajectory vectors.
Delivers sub-3ms CPU inference latency and 100% offline self-containment.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Optional, Sequence, Union
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from signbridge.core.logger import get_logger
from signbridge.word_sign.features import SEQUENCE_FEATURE_DIM

logger = get_logger("word_sign.model")

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "word_sign"


@dataclass(frozen=True)
class WordPredictionOutput:
    """Structured inference result for word-level sign recognition."""
    predicted_word: str
    confidence: float
    top_k: list[tuple[str, float]]
    inference_time_ms: float


class WordSignClassifier:
    """Production wrapper for word-level sequence classification models."""

    def __init__(
        self,
        classes: Optional[list[str]] = None,
        model_type: str = "mlp",
        random_state: int = 42,
    ) -> None:
        self.classes: list[str] = [c.upper() for c in (classes or [])]
        self.class_to_id: dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.id_to_class: dict[int, str] = {i: c for i, c in enumerate(self.classes)}
        self.model_type = model_type.lower()
        self.random_state = random_state

        self._estimator: Optional[Any] = None
        self._last_latency_ms: float = 0.0

        if self.classes:
            self._init_estimator()

    def _init_estimator(self) -> None:
        """Initializes the underlying scikit-learn estimator."""
        if self.model_type == "mlp":
            # 2-layer MLP with ReLU and Adam: Rapid CPU convergence and sub-1ms inference
            self._estimator = MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=128,
                learning_rate_init=2e-3,
                max_iter=60,
                random_state=self.random_state,
                early_stopping=True,
                n_iter_no_change=5,
            )
        elif self.model_type == "random_forest":
            self._estimator = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unsupported model type '{self.model_type}'. Expected 'mlp' or 'random_forest'.")

    @property
    def is_trained(self) -> bool:
        """Returns True if the underlying estimator has been trained/fitted."""
        if self._estimator is None:
            return False
        return hasattr(self._estimator, "classes_")

    @property
    def latency_ms(self) -> float:
        """Last measured inference latency in milliseconds."""
        return self._last_latency_ms

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WordSignClassifier":
        """Fits the classifier on trajectory features X (N, 2560) and labels y (N,)."""
        if X.ndim != 2 or X.shape[1] != SEQUENCE_FEATURE_DIM:
            raise ValueError(f"Expected X shape (N, {SEQUENCE_FEATURE_DIM}), got {X.shape}")

        unique_labels = sorted(list(set(y)))
        if not self.classes:
            self.classes = [str(l).upper() for l in unique_labels]
            self.class_to_id = {c: i for i, c in enumerate(self.classes)}
            self.id_to_class = {i: c for i, c in enumerate(self.classes)}
            self._init_estimator()

        # Map labels to IDs if string labels are passed
        if isinstance(y[0], str):
            y_ids = np.array([self.class_to_id[l.upper()] for l in y], dtype=np.int32)
        else:
            y_ids = np.array(y, dtype=np.int32)

        start_t = time.perf_counter()
        logger.info(
            "Fitting %s on %d samples across %d classes...",
            self.model_type.upper(),
            len(X),
            len(self.classes),
        )
        self._estimator.fit(X, y_ids)
        fit_time = time.perf_counter() - start_t
        logger.info("Training complete in %.2f seconds.", fit_time)
        return self

    def predict_proba(self, feature_vector: np.ndarray) -> dict[str, float]:
        """Calculates class probability distribution for a single 2,560D feature vector."""
        if not self.is_trained:
            logger.warning("Classifier is not trained. Returning uniform distribution.")
            if not self.classes:
                return {}
            uniform = 1.0 / len(self.classes)
            return {c: uniform for c in self.classes}

        vec = np.asarray(feature_vector, dtype=np.float32).flatten()
        if vec.shape[0] != SEQUENCE_FEATURE_DIM:
            raise ValueError(f"Expected feature vector of length {SEQUENCE_FEATURE_DIM}, got {vec.shape[0]}")

        start_t = time.perf_counter()
        probs = self._estimator.predict_proba(vec.reshape(1, -1))[0]
        self._last_latency_ms = (time.perf_counter() - start_t) * 1000.0

        # Map model class indices back to label names
        est_classes = self._estimator.classes_
        result: dict[str, float] = {}
        for idx, cls_id in enumerate(est_classes):
            cls_name = self.id_to_class.get(int(cls_id), str(cls_id))
            result[cls_name] = float(probs[idx])

        # Fill any unobserved classes with 0.0
        for cls_name in self.classes:
            result.setdefault(cls_name, 0.0)

        return result

    def predict(self, feature_vector: np.ndarray) -> WordPredictionOutput:
        """Predicts top class and confidence for a 2,560D trajectory vector."""
        probs = self.predict_proba(feature_vector)
        if not probs:
            return WordPredictionOutput(
                predicted_word="UNKNOWN",
                confidence=0.0,
                top_k=[],
                inference_time_ms=self._last_latency_ms,
            )

        sorted_preds = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        best_class, best_conf = sorted_preds[0]
        top_k = sorted_preds[:3]

        return WordPredictionOutput(
            predicted_word=best_class,
            confidence=best_conf,
            top_k=top_k,
            inference_time_ms=self._last_latency_ms,
        )

    def save(self, save_dir: Union[str, Path]) -> Path:
        """Saves model weights, class indices, and configuration."""
        target_dir = Path(save_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        model_path = target_dir / "model.joblib"
        classes_path = target_dir / "classes.json"
        config_path = target_dir / "model_config.json"

        joblib.dump(self._estimator, model_path)

        with open(classes_path, "w", encoding="utf-8") as f:
            json.dump({
                "classes": self.classes,
                "class_to_id": self.class_to_id,
                "num_classes": len(self.classes),
            }, f, indent=2)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_type": self.model_type,
                "random_state": self.random_state,
                "feature_dimension": SEQUENCE_FEATURE_DIM,
                "saved_at": time.time(),
            }, f, indent=2)

        logger.info("Saved word-sign model to: %s", target_dir)
        return target_dir

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "WordSignClassifier":
        """Loads trained model weights and class configuration from disk."""
        target_dir = Path(model_dir).resolve()
        model_path = target_dir / "model.joblib"
        classes_path = target_dir / "classes.json"
        config_path = target_dir / "model_config.json"

        if not model_path.exists() or not classes_path.exists():
            raise FileNotFoundError(f"Missing model files in directory: {target_dir}")

        with open(classes_path, "r", encoding="utf-8") as f:
            class_data = json.load(f)

        classes = class_data.get("classes", [])

        model_type = "mlp"
        random_state = 42
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    model_type = cfg.get("model_type", "mlp")
                    random_state = cfg.get("random_state", 42)
            except Exception:
                pass

        instance = cls(classes=classes, model_type=model_type, random_state=random_state)
        instance._estimator = joblib.load(model_path)
        logger.info("Loaded word-sign model from %s (%d classes)", target_dir, len(classes))
        return instance


def load_default_word_sign_model(model_dir: Optional[Union[str, Path]] = None) -> WordSignClassifier:
    """Loads default model from models/word_sign/ or initializes fallback with vocabulary."""
    target_dir = Path(model_dir or DEFAULT_MODEL_DIR).resolve()
    if target_dir.exists() and (target_dir / "model.joblib").exists():
        try:
            return WordSignClassifier.load(target_dir)
        except Exception as e:
            logger.warning("Could not load trained model from %s: %s. Using default instance.", target_dir, e)

    # Fallback to vocabulary classes if available
    vocab_path = Path(__file__).resolve().parents[3] / "configs" / "word_sign_vocabulary.json"
    classes = ["HELLO", "HELP", "YES", "NO", "STOP", "THANK YOU"]
    if vocab_path.exists():
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                classes = [c.upper() for c in data.get("classes", classes)]
        except Exception:
            pass

    logger.info("Initialized un-fitted WordSignClassifier with %d classes.", len(classes))
    return WordSignClassifier(classes=classes, model_type="mlp")
