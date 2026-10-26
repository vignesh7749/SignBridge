"""Model persistence utilities for SignBridge models."""

import json
from pathlib import Path
from typing import Union
import joblib

from signbridge.features.schema import FeatureSchema
from signbridge.model.base import ModelMetadata
from signbridge.model.classifier import SignClassifier


def save_sign_classifier(classifier: SignClassifier, output_dir: Union[Path, str]) -> Path:
    """Saves model weights, metadata, class mappings, and metrics to disk.

    Output files:
      - model.joblib
      - metadata.json
      - classes.json
      - metrics.json
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Serialize estimator
    model_file = out_path / "model.joblib"
    joblib.dump(classifier.model, str(model_file), compress=3)

    # 2. Serialize metadata
    meta_file = out_path / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(classifier.metadata.to_dict(), f, indent=2)

    # 3. Serialize classes mapping
    classes_file = out_path / "classes.json"
    classes_payload = {
        "class_to_id": classifier.metadata.class_to_id,
        "id_to_class": {str(k): v for k, v in classifier.metadata.id_to_class.items()},
    }
    with open(classes_file, "w", encoding="utf-8") as f:
        json.dump(classes_payload, f, indent=2)

    # 4. Serialize metrics
    metrics_file = out_path / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(classifier.metadata.metrics, f, indent=2)

    return out_path


def load_sign_classifier(model_dir: Union[Path, str]) -> SignClassifier:
    """Loads a saved SignClassifier and its metadata bundle from disk.

    Args:
        model_dir: Path to directory containing model.joblib and metadata.json.

    Returns:
        Instantiated and validated SignClassifier ready for inference.
    """
    target_dir = Path(model_dir).resolve()
    model_file = target_dir / "model.joblib"
    meta_file = target_dir / "metadata.json"

    if not model_file.exists():
        raise FileNotFoundError(f"Model weights file not found: {model_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Model metadata file not found: {meta_file}")

    with open(meta_file, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)

    metadata = ModelMetadata.from_dict(meta_dict)

    if metadata.feature_dimension != FeatureSchema.TOTAL_DIMENSIONS:
        raise ValueError(
            f"Loaded model feature dimension ({metadata.feature_dimension}) does not match "
            f"current FeatureSchema ({FeatureSchema.TOTAL_DIMENSIONS})"
        )

    model = joblib.load(str(model_file))
    return SignClassifier(model=model, metadata=metadata)
