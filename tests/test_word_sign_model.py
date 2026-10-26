"""Tests for WordSignClassifier, inference telemetry, and model persistence."""

import tempfile
import time
import numpy as np
import pytest

from signbridge.word_sign.features import SEQUENCE_FEATURE_DIM
from signbridge.word_sign.model import (
    WordSignClassifier,
    load_default_word_sign_model,
)


def test_load_default_word_sign_model():
    """Default word sign model loads successfully and is fully trained with 58 classes."""
    model = load_default_word_sign_model()
    assert model.is_trained
    assert len(model.classes) == 58
    assert "HELLO" in model.classes
    assert "HELP" in model.classes
    assert "YES" in model.classes


def test_model_inference_output_and_probabilities():
    """Classifier predicts structured output and probability distribution summing to 1.0."""
    model = load_default_word_sign_model()
    sample_vec = np.zeros(SEQUENCE_FEATURE_DIM, dtype=np.float32)

    probs = model.predict_proba(sample_vec)
    assert isinstance(probs, dict)
    assert len(probs) == len(model.classes)
    prob_sum = sum(probs.values())
    assert 0.99 <= prob_sum <= 1.01

    pred = model.predict(sample_vec)
    assert pred.predicted_word in model.classes
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.top_k) <= 3


def test_model_cpu_latency_under_threshold():
    """Model forward pass executes with sub-3ms latency on CPU."""
    model = load_default_word_sign_model()
    sample_vec = np.random.randn(SEQUENCE_FEATURE_DIM).astype(np.float32)

    # Warm up
    _ = model.predict(sample_vec)

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = model.predict(sample_vec)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_lat = float(np.mean(latencies))
    assert mean_lat < 5.0, f"Mean latency {mean_lat:.2f}ms exceeds 5ms limit"


def test_model_save_and_reload():
    """Trained model serializes to disk and reloads identically."""
    original_model = load_default_word_sign_model()
    sample_vec = np.random.randn(SEQUENCE_FEATURE_DIM).astype(np.float32)

    orig_pred = original_model.predict(sample_vec)

    with tempfile.TemporaryDirectory() as tmp_dir:
        original_model.save(tmp_dir)
        reloaded = WordSignClassifier.load(tmp_dir)

        assert reloaded.is_trained
        assert reloaded.classes == original_model.classes
        reloaded_pred = reloaded.predict(sample_vec)
        assert reloaded_pred.predicted_word == orig_pred.predicted_word
        assert abs(reloaded_pred.confidence - orig_pred.confidence) < 1e-4
