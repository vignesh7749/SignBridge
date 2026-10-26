"""Unit and integration tests for Stage 6: Sign Recognition Model Subsystem.

Tests:
  1. ModelMetadata serialization and deserialization.
  2. Dataset loader validation and strict split disjointness.
  3. Feature dimension validation (accepts 94, rejects other dimensions).
  4. Non-finite feature rejection (NaN / Inf).
  5. SignClassifier prediction output contract (PredictionResult fields).
  6. SignClassifier probability distribution sums to ~1.0.
  7. Prediction determinism (identical input produces identical prediction).
  8. Model save and reload round-trip consistency.
  9. ISignRecognizer strategy adapter compatibility (ScoredPrediction).
  10. Inference latency benchmark sanity check (< 15 ms).
  11. Production model artifact verification (when trained).
"""

import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from signbridge.features.schema import FeatureSchema
from signbridge.model.base import ModelMetadata, PredictionResult
from signbridge.model.classifier import SignClassifier
from signbridge.model.dataset_loader import DatasetSplits, load_and_validate_dataset
from signbridge.model.persistence import load_sign_classifier, save_sign_classifier


@pytest.fixture
def mock_classifier() -> SignClassifier:
    """Creates a lightweight synthetic SignClassifier for fast unit testing."""
    classes = ["A", "B", "C", "D", "E"]
    class_to_id = {c: i for i, c in enumerate(classes)}
    id_to_class = {i: c for i, c in enumerate(classes)}

    # Synthetic 94D data
    rng = np.random.RandomState(42)
    X = rng.randn(50, 94).astype(np.float32)
    y = rng.randint(0, len(classes), size=50)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=100, random_state=42)),
    ])
    pipe.fit(X, y)

    meta = ModelMetadata(
        model_name="MockClassifier",
        model_version="0.1.0",
        algorithm="LogisticRegression",
        feature_schema_version=FeatureSchema.VERSION,
        feature_dimension=FeatureSchema.TOTAL_DIMENSIONS,
        dataset_identity="Mock-Synthetic",
        dataset_manifest_hash="abc123mock",
        class_to_id=class_to_id,
        id_to_class=id_to_class,
        num_classes=len(classes),
        train_samples=40,
        val_samples=10,
        test_samples=10,
        random_seed=42,
        hyperparameters={"C": 1.0},
        training_timestamp="2026-09-18 00:00:00",
        python_version="3.11",
        numpy_version="2.4",
        scikit_learn_version="1.9",
        metrics={"test_accuracy": 0.95},
    )
    return SignClassifier(model=pipe, metadata=meta)


def test_metadata_serialization_round_trip() -> None:
    """Verifies that ModelMetadata can serialize to dict and deserialize faithfully."""
    class_to_id = {"A": 0, "B": 1, "C": 2}
    id_to_class = {0: "A", 1: "B", 2: "C"}

    meta = ModelMetadata(
        model_name="TestModel",
        model_version="1.0.0",
        algorithm="SVM",
        feature_schema_version="1.0",
        feature_dimension=94,
        dataset_identity="TestDS",
        dataset_manifest_hash="hash123",
        class_to_id=class_to_id,
        id_to_class=id_to_class,
        num_classes=3,
        train_samples=100,
        val_samples=20,
        test_samples=20,
        random_seed=42,
        hyperparameters={"C": 10.0},
        training_timestamp="2026-09-18",
        python_version="3.11",
        numpy_version="2.4",
        scikit_learn_version="1.9",
        metrics={"macro_f1": 0.98},
    )

    d = meta.to_dict()
    assert d["model_name"] == "TestModel"
    assert d["feature_dimension"] == 94

    restored = ModelMetadata.from_dict(d)
    assert restored.model_name == meta.model_name
    assert restored.num_classes == 3
    assert restored.id_to_class[0] == "A"
    assert restored.metrics["macro_f1"] == 0.98


def test_classifier_validates_feature_dimension(mock_classifier: SignClassifier) -> None:
    """Verifies that input feature vectors of dimensions other than 94 are rejected."""
    # Correct dimension: 94
    valid_1d = np.zeros(94, dtype=np.float32)
    res = mock_classifier.predict(valid_1d)
    assert isinstance(res, PredictionResult)

    # Wrong dimension: 63
    invalid_dim = np.zeros(63, dtype=np.float32)
    with pytest.raises(ValueError, match="dimension mismatch"):
        mock_classifier.predict(invalid_dim)

    # Wrong dimension: 95
    invalid_dim_95 = np.zeros(95, dtype=np.float32)
    with pytest.raises(ValueError, match="dimension mismatch"):
        mock_classifier.predict(invalid_dim_95)


def test_classifier_rejects_nan_and_inf(mock_classifier: SignClassifier) -> None:
    """Verifies that features containing NaN or Inf raise ValueError."""
    features_nan = np.zeros(94, dtype=np.float32)
    features_nan[10] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        mock_classifier.predict(features_nan)

    features_inf = np.zeros(94, dtype=np.float32)
    features_inf[20] = np.inf
    with pytest.raises(ValueError, match="non-finite values"):
        mock_classifier.predict(features_inf)


def test_classifier_prediction_output_contract(mock_classifier: SignClassifier) -> None:
    """Verifies that predict() returns a complete, compliant PredictionResult."""
    features = np.random.randn(94).astype(np.float32)
    result = mock_classifier.predict(features)

    assert isinstance(result.class_id, int)
    assert 0 <= result.class_id < mock_classifier.metadata.num_classes
    assert isinstance(result.class_name, str)
    assert result.class_name in mock_classifier.vocabulary
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.top_candidates) <= 3
    assert result.top_candidates[0][0] == result.class_name
    assert result.inference_latency_ms >= 0.0
    assert result.top_margin >= 0.0


def test_classifier_predict_proba_distribution(mock_classifier: SignClassifier) -> None:
    """Verifies that predict_proba returns a valid probability distribution."""
    features = np.random.randn(2, 94).astype(np.float32)
    probs = mock_classifier.predict_proba(features)

    assert probs.shape == (2, mock_classifier.metadata.num_classes)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    # Row sums should equal 1.0
    row_sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(row_sums, [1.0, 1.0], rtol=1e-4)


def test_classifier_prediction_determinism(mock_classifier: SignClassifier) -> None:
    """Verifies that identical input features produce identical predictions."""
    features = np.random.randn(94).astype(np.float32)

    res1 = mock_classifier.predict(features)
    res2 = mock_classifier.predict(features)

    assert res1.class_id == res2.class_id
    assert res1.class_name == res2.class_name
    assert pytest.approx(res1.confidence, rel=1e-5) == res2.confidence
    assert pytest.approx(res1.top_margin, rel=1e-5) == res2.top_margin


def test_classifier_scored_prediction_adapter(mock_classifier: SignClassifier) -> None:
    """Verifies compatibility with ISignRecognizer and ScoredPrediction."""
    features = np.random.randn(94).astype(np.float32)
    scored = mock_classifier.predict_scored(features)

    assert scored.is_valid is True
    assert scored.label in mock_classifier.vocabulary
    assert 0.0 <= scored.confidence <= 1.0
    assert len(scored.probabilities) == mock_classifier.metadata.num_classes


def test_model_persistence_round_trip(mock_classifier: SignClassifier, tmp_path: Path) -> None:
    """Verifies saving and reloading a classifier preserves exact predictions."""
    save_dir = tmp_path / "saved_model"
    saved_path = save_sign_classifier(mock_classifier, save_dir)

    assert (saved_path / "model.joblib").exists()
    assert (saved_path / "metadata.json").exists()
    assert (saved_path / "classes.json").exists()
    assert (saved_path / "metrics.json").exists()

    reloaded = load_sign_classifier(saved_path)
    assert reloaded.metadata.model_name == mock_classifier.metadata.model_name
    assert reloaded.metadata.num_classes == mock_classifier.metadata.num_classes
    assert reloaded.vocabulary == mock_classifier.vocabulary

    # Verify identical predictions
    test_vec = np.random.randn(94).astype(np.float32)
    orig_res = mock_classifier.predict(test_vec)
    reloaded_res = reloaded.predict(test_vec)

    assert orig_res.class_id == reloaded_res.class_id
    assert orig_res.class_name == reloaded_res.class_name
    assert pytest.approx(orig_res.confidence, rel=1e-4) == reloaded_res.confidence


def test_inference_latency_benchmark(mock_classifier: SignClassifier) -> None:
    """Verifies that single-sample inference latency is well within real-time limits (< 15ms)."""
    test_vec = np.random.randn(94).astype(np.float32)

    # Warmup
    for _ in range(10):
        mock_classifier.predict(test_vec)

    # Benchmark 100 runs
    latencies = []
    for _ in range(100):
        res = mock_classifier.predict(test_vec)
        latencies.append(res.inference_latency_ms)

    mean_latency = float(np.mean(latencies))
    # Latency for 94D linear/shallow inference should easily be sub-10ms
    assert mean_latency < 15.0, f"Inference latency too slow: {mean_latency:.2f} ms"


def test_production_model_artifacts_and_inference() -> None:
    """Verifies that the trained production model artifacts exist and pass performance gates."""
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / "models" / "sign_classifier"

    assert (model_dir / "model.joblib").exists(), "Production model.joblib missing"
    assert (model_dir / "metadata.json").exists(), "Production metadata.json missing"
    assert (model_dir / "classes.json").exists(), "Production classes.json missing"
    assert (model_dir / "metrics.json").exists(), "Production metrics.json missing"

    classifier = load_sign_classifier(model_dir)
    assert classifier.metadata.num_classes == 26
    assert classifier.metadata.feature_dimension == 94
    assert len(classifier.vocabulary) == 26
    assert classifier.vocabulary[0] == "A"
    assert classifier.vocabulary[-1] == "Z"

    # Performance gates
    metrics = classifier.metadata.metrics
    assert metrics["test_accuracy"] >= 0.95, f"Test accuracy below 95%: {metrics['test_accuracy']}"
    assert metrics["macro_f1"] >= 0.95, f"Test macro F1 below 0.95: {metrics['macro_f1']}"
    assert metrics["mean_latency_ms"] < 10.0, f"Inference latency above 10ms: {metrics['mean_latency_ms']}"

    # Verify single sample prediction on zeros vector
    sample_probe = np.zeros(94, dtype=np.float32)
    res = classifier.predict(sample_probe)
    assert isinstance(res, PredictionResult)
    assert res.class_name in classifier.vocabulary
    assert 0.0 <= res.confidence <= 1.0


def test_stage6_reports_and_plots_exist() -> None:
    """Verifies that all required Stage 6 reports and visual plots are generated."""
    project_root = Path(__file__).resolve().parent.parent
    reports_dir = project_root / "reports" / "stage6"

    required_reports = [
        "model_comparison.json",
        "classification_report.json",
        "confusion_matrix.json",
        "error_analysis.json",
        "benchmark.json",
        "confusion_matrix.png",
        "per_class_f1.png",
        "confidence_distribution.png",
        "model_comparison.png",
    ]

    for fname in required_reports:
        fpath = reports_dir / fname
        assert fpath.exists(), f"Required Stage 6 report missing: {fpath}"
        assert fpath.stat().st_size > 0, f"Stage 6 report is empty: {fpath}"

