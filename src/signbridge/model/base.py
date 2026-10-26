"""Base data contracts and metadata structures for Stage 6 Sign Recognition."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PredictionResult:
    """Immutable result of a single sign classification inference."""

    class_id: int
    class_name: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    top_candidates: list[tuple[str, float]] = field(default_factory=list)
    inference_latency_ms: float = 0.0

    @property
    def top_margin(self) -> float:
        """Difference between the top-1 and top-2 candidate confidence."""
        if len(self.top_candidates) >= 2:
            return self.top_candidates[0][1] - self.top_candidates[1][1]
        return self.confidence

    def to_dict(self) -> dict[str, Any]:
        """Serializes result to a standard dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "top_margin": round(float(self.top_margin), 4),
            "top_candidates": [
                [name, round(float(score), 4)] for name, score in self.top_candidates
            ],
            "inference_latency_ms": round(float(self.inference_latency_ms), 3),
        }


@dataclass
class ModelMetadata:
    """Complete, self-contained provenance and reproducibility metadata for a trained model."""

    model_name: str
    model_version: str
    algorithm: str
    feature_schema_version: str
    feature_dimension: int
    dataset_identity: str
    dataset_manifest_hash: str
    class_to_id: dict[str, int]
    id_to_class: dict[int, str]
    num_classes: int
    train_samples: int
    val_samples: int
    test_samples: int
    random_seed: int
    hyperparameters: dict[str, Any]
    training_timestamp: str
    python_version: str
    numpy_version: str
    scikit_learn_version: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes metadata to JSON-compatible dictionary."""
        d = asdict(self)
        # Ensure int keys in id_to_class are stringified for standard JSON compliance
        d["id_to_class"] = {str(k): v for k, v in self.id_to_class.items()}
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Deserializes metadata from dictionary."""
        id_to_class_raw = data.get("id_to_class", {})
        id_to_class = {int(k): v for k, v in id_to_class_raw.items()}
        return cls(
            model_name=data["model_name"],
            model_version=data["model_version"],
            algorithm=data["algorithm"],
            feature_schema_version=data["feature_schema_version"],
            feature_dimension=int(data["feature_dimension"]),
            dataset_identity=data["dataset_identity"],
            dataset_manifest_hash=data["dataset_manifest_hash"],
            class_to_id=data["class_to_id"],
            id_to_class=id_to_class,
            num_classes=int(data["num_classes"]),
            train_samples=int(data["train_samples"]),
            val_samples=int(data["val_samples"]),
            test_samples=int(data["test_samples"]),
            random_seed=int(data["random_seed"]),
            hyperparameters=data.get("hyperparameters", {}),
            training_timestamp=data["training_timestamp"],
            python_version=data["python_version"],
            numpy_version=data["numpy_version"],
            scikit_learn_version=data["scikit_learn_version"],
            metrics=data.get("metrics", {}),
        )
