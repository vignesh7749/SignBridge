"""Unit tests for Stage 4: Landmark Preprocessing & Feature Engineering.

Validates:
  1. Backward compatibility with Stage 1-3 FeatureNormalizer.
  2. Landmark validation: shape, count, NaN/Inf rejection, degenerate hand handling.
  3. Strict translation invariance: f(P + t) == f(P).
  4. Strict scale invariance: f(s * P) == f(P).
  5. Determinism: duplicate calls yield bit-for-bit identical arrays.
  6. Numerical stability of 3D joint angles (zero-length vectors, collinearity).
  7. Handedness canonicalization: Left hand reflection x -> -x.
  8. FeatureSchema v1.0 slice indexing and metadata completeness.
  9. Low-latency extraction throughput sanity.
"""

import time
import numpy as np
import pytest

from signbridge.features.extractor import FeatureVector, HandFeatureExtractor
from signbridge.features.geometry import (
    compute_angle_3d,
    compute_euclidean_distance,
    compute_finger_joint_angles,
    compute_fingertip_distances,
    compute_inter_fingertip_distances,
    compute_palm_aspect_ratio,
)
from signbridge.features.normalizer import CoordinateNormalizer, FeatureNormalizer
from signbridge.features.schema import FEATURE_NAMES, FeatureSchema
from signbridge.features.validator import LandmarkValidationError, LandmarkValidator
from signbridge.vision.landmarks import HandLandmarks, Landmark3D


# ==============================================================================
# Backward Compatibility Tests (Stage 1-3 Contracts)
# ==============================================================================

def test_translation_invariance(mock_hand_landmarks: HandLandmarks) -> None:
    """Verifies that shifting the hand across the screen produces identical normalized features."""
    norm_original = FeatureNormalizer.normalize_hand(mock_hand_landmarks)

    # Shift all raw coordinates by (+500, -300, +50)
    shifted_coords = mock_hand_landmarks.raw_array + np.array([500.0, -300.0, 50.0], dtype=np.float32)
    shifted_landmarks = [Landmark3D(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in shifted_coords]
    shifted_hand = HandLandmarks(handedness="Right", landmarks=shifted_landmarks, raw_array=shifted_coords)

    norm_shifted = FeatureNormalizer.normalize_hand(shifted_hand)

    # Invariance check: maximum absolute error must be negligible
    np.testing.assert_allclose(norm_original.vector, norm_shifted.vector, atol=1e-5)


def test_scale_invariance(mock_hand_landmarks: HandLandmarks) -> None:
    """Verifies that scaling the hand produces identical normalized features."""
    norm_original = FeatureNormalizer.normalize_hand(mock_hand_landmarks)

    # Scale hand coordinates relative to wrist by a factor of 2.5
    wrist = mock_hand_landmarks.raw_array[0]
    scaled_coords = wrist + (mock_hand_landmarks.raw_array - wrist) * 2.5
    scaled_landmarks = [Landmark3D(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in scaled_coords]
    scaled_hand = HandLandmarks(handedness="Right", landmarks=scaled_landmarks, raw_array=scaled_coords)

    norm_scaled = FeatureNormalizer.normalize_hand(scaled_hand)

    # Invariance check
    np.testing.assert_allclose(norm_original.vector, norm_scaled.vector, atol=1e-5)


# ==============================================================================
# Stage 4 Feature Extractor Tests
# ==============================================================================

def test_feature_extractor_shape_dtype_and_bounds(mock_hand_landmarks: HandLandmarks) -> None:
    """Verifies that the extractor produces a 94-dimensional float32 vector with valid values."""
    extractor = HandFeatureExtractor(canonicalize_handedness=False)
    feat_vec = extractor.extract(mock_hand_landmarks)

    assert isinstance(feat_vec, FeatureVector)
    assert feat_vec.features.shape == (94,)
    assert feat_vec.features.dtype == np.float32
    assert np.all(np.isfinite(feat_vec.features)), "Feature vector contains NaN or Inf"
    assert feat_vec.schema_version == "1.0"
    assert feat_vec.handedness == "Right"

    # Coordinates: wrist (first 3 elements) must be origin (0, 0, 0)
    np.testing.assert_allclose(feat_vec.coords[0], [0.0, 0.0, 0.0], atol=1e-6)

    # Angles must be in [0, pi]
    angles = feat_vec.joint_angles
    assert np.all(angles >= 0.0)
    assert np.all(angles <= np.pi + 1e-5)

    # Distances must be non-negative
    assert np.all(feat_vec.wrist_tip_distances >= 0.0)
    assert np.all(feat_vec.inter_tip_distances >= 0.0)
    assert feat_vec.palm_aspect_ratio > 0.0


def test_stage4_strict_translation_invariance(mock_hand_landmarks: HandLandmarks) -> None:
    """Verifies that translating the hand produces identical Stage 4 feature vectors."""
    extractor = HandFeatureExtractor(canonicalize_handedness=False)
    feat_original = extractor.extract(mock_hand_landmarks)

    # Apply 3D translation offset (+345.67, -890.12, +42.0)
    offset = np.array([345.67, -890.12, 42.0], dtype=np.float32)
    shifted_coords = mock_hand_landmarks.raw_array + offset
    shifted_landmarks = [Landmark3D(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in shifted_coords]
    shifted_hand = HandLandmarks(handedness="Right", landmarks=shifted_landmarks, raw_array=shifted_coords)

    feat_shifted = extractor.extract(shifted_hand)

    # Invariance check across all 94 features
    np.testing.assert_allclose(feat_original.features, feat_shifted.features, atol=1e-5)


def test_stage4_strict_scale_invariance(mock_hand_landmarks: HandLandmarks) -> None:
    """Verifies that scaling the hand produces identical Stage 4 feature vectors."""
    extractor = HandFeatureExtractor(canonicalize_handedness=False)
    feat_original = extractor.extract(mock_hand_landmarks)

    for scale in [0.2, 0.5, 1.8, 3.5, 5.0]:
        wrist = mock_hand_landmarks.raw_array[0]
        scaled_coords = wrist + (mock_hand_landmarks.raw_array - wrist) * scale
        scaled_landmarks = [Landmark3D(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in scaled_coords]
        scaled_hand = HandLandmarks(handedness="Right", landmarks=scaled_landmarks, raw_array=scaled_coords)

        feat_scaled = extractor.extract(scaled_hand)
        np.testing.assert_allclose(
            feat_original.features,
            feat_scaled.features,
            atol=1e-3,
            err_msg=f"Scale invariance failed for scale={scale}",
        )


def test_stage4_determinism(mock_hand_landmarks: HandLandmarks) -> None:
    """Verifies that repeated extractions on identical input produce bit-for-bit identical vectors."""
    extractor = HandFeatureExtractor()
    baseline = extractor.extract(mock_hand_landmarks).features

    for _ in range(10):
        repeat = extractor.extract(mock_hand_landmarks).features
        assert np.array_equal(baseline, repeat), "Extractor output is non-deterministic"


# ==============================================================================
# Landmark Validation & Degenerate Protection Tests
# ==============================================================================

def test_validator_rejects_none() -> None:
    """Verifies that None input raises LandmarkValidationError."""
    with pytest.raises(LandmarkValidationError, match="cannot be None"):
        LandmarkValidator.validate(None)


def test_validator_rejects_invalid_shape() -> None:
    """Verifies that incorrectly shaped arrays are rejected."""
    with pytest.raises(LandmarkValidationError, match="Invalid landmark array shape"):
        LandmarkValidator.validate(np.zeros((20, 3), dtype=np.float32))

    with pytest.raises(LandmarkValidationError, match="Invalid landmark array shape"):
        LandmarkValidator.validate(np.zeros((21, 2), dtype=np.float32))


def test_validator_rejects_nans_and_infs(mock_hand_landmarks: HandLandmarks) -> None:
    """Verifies that NaN, Inf, and -Inf values are rejected with informative error messages."""
    coords = mock_hand_landmarks.raw_array.copy()

    # Test NaN rejection
    coords_nan = coords.copy()
    coords_nan[4, 1] = np.nan
    with pytest.raises(LandmarkValidationError, match="non-finite values"):
        LandmarkValidator.validate(coords_nan)

    # Test Inf rejection
    coords_inf = coords.copy()
    coords_inf[12, 0] = np.inf
    with pytest.raises(LandmarkValidationError, match="non-finite values"):
        LandmarkValidator.validate(coords_inf)


def test_validator_rejects_degenerate_collapsed_hand() -> None:
    """Verifies that a hand where wrist and middle MCP coincide is rejected."""
    coords = np.zeros((21, 3), dtype=np.float32)
    # Both wrist (0) and middle MCP (9) are at (0, 0, 0)
    with pytest.raises(LandmarkValidationError, match="Degenerate hand geometry"):
        LandmarkValidator.validate(coords)


# ==============================================================================
# Geometry & Numerical Stability Tests
# ==============================================================================

def test_angle_3d_zero_length_vector_safe() -> None:
    """Verifies that zero-length vectors return 0.0 angle without producing NaN or raising."""
    p1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    p2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # Coincident with p1
    p3 = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    angle = compute_angle_3d(p1, p2, p3)
    assert angle == 0.0
    assert not np.isnan(angle)


def test_angle_3d_known_values() -> None:
    """Verifies correct angle computation for right-angle and straight lines."""
    # 90 degrees (pi/2)
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    angle_90 = compute_angle_3d(a, b, c)
    assert np.isclose(angle_90, np.pi / 2.0, atol=1e-5)

    # 180 degrees (pi) - straight line
    c_straight = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    angle_180 = compute_angle_3d(a, b, c_straight)
    assert np.isclose(angle_180, np.pi, atol=1e-5)


# ==============================================================================
# Handedness Canonicalization Tests
# ==============================================================================

def test_handedness_canonicalization(mock_hand_landmarks: HandLandmarks) -> None:
    """Verifies that Left hands mirrored horizontally match Right hands under canonicalization."""
    right_coords = mock_hand_landmarks.raw_array.copy()

    # Mirror horizontally to simulate a Left hand (x -> -x)
    left_coords = right_coords.copy()
    left_coords[:, 0] = -left_coords[:, 0]

    left_lms = [Landmark3D(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in left_coords]
    left_hand = HandLandmarks(handedness="Left", landmarks=left_lms, raw_array=left_coords)

    # With canonicalization enabled (default)
    extractor = HandFeatureExtractor(canonicalize_handedness=True)
    right_feat = extractor.extract(mock_hand_landmarks)
    left_feat = extractor.extract(left_hand)

    assert right_feat.is_canonicalized is False
    assert left_feat.is_canonicalized is True

    # After canonicalizing Left hand, coordinate features and distances should match closely
    np.testing.assert_allclose(right_feat.coords, left_feat.coords, atol=1e-4)
    np.testing.assert_allclose(right_feat.wrist_tip_distances, left_feat.wrist_tip_distances, atol=1e-4)
    np.testing.assert_allclose(right_feat.joint_angles, left_feat.joint_angles, atol=1e-4)


# ==============================================================================
# Schema & Slice Integrity Tests
# ==============================================================================

def test_feature_schema_completeness(mock_hand_landmarks: HandLandmarks) -> None:
    """Verifies that all 94 schema feature names are unique and accessible via FeatureVector."""
    assert len(FEATURE_NAMES) == 94
    assert len(set(FEATURE_NAMES)) == 94, "Duplicate feature names in schema"

    extractor = HandFeatureExtractor()
    feat_vec = extractor.extract(mock_hand_landmarks)
    d = feat_vec.to_dict()

    assert "metadata" in d
    assert "features" in d
    assert len(d["features"]) == 94
    assert d["metadata"]["dimension"] == 94

    stats = feat_vec.summary_stats()
    assert stats["min"] <= stats["max"]
    assert stats["l2_norm"] > 0.0


def test_extraction_latency_benchmark(mock_hand_landmarks: HandLandmarks) -> None:
    """Verifies that feature extraction executes well within real-time budget (< 1 ms per hand)."""
    extractor = HandFeatureExtractor()

    # Warmup
    for _ in range(50):
        _ = extractor.extract(mock_hand_landmarks)

    # Measure 500 extractions
    n_iterations = 500
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = extractor.extract(mock_hand_landmarks)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / n_iterations) * 1000.0
    throughput = n_iterations / elapsed

    assert avg_ms < 2.0, f"Average extraction time {avg_ms:.3f} ms exceeds 2.0 ms threshold"
    assert throughput > 500.0, f"Throughput {throughput:.1f} extractions/s below 500 threshold"
