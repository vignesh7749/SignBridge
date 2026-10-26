"""Tests for word-sign multi-hand temporal feature extraction and resampling."""

import numpy as np
import pytest

from signbridge.vision.detector import HandDetectionResult
from signbridge.vision.landmarks import BoundingBox, DetectedHand, LandmarkPoint
from signbridge.word_sign.features import (
    FRAME_FEATURE_DIM,
    HAND_FEATURE_DIM,
    SEQUENCE_FEATURE_DIM,
    TARGET_SEQUENCE_LENGTH,
    WordSignFeatureExtractor,
    extract_frame_features,
    extract_hand_features_from_raw,
    extract_single_hand_features,
    resample_sequence_features,
)


def _make_dummy_detected_hand(wrist_offset=(0.5, 0.5, 0.0), scale=0.1) -> DetectedHand:
    """Helper creating a valid DetectedHand fixture."""
    landmarks = []
    raw = np.zeros((21, 3), dtype=np.float32)
    for i in range(21):
        raw[i, 0] = wrist_offset[0] + (i % 5) * 0.02 * scale
        raw[i, 1] = wrist_offset[1] + (i // 5) * 0.03 * scale
        raw[i, 2] = wrist_offset[2]
        landmarks.append(
            LandmarkPoint(
                index=i,
                name=f"pt_{i}",
                x_norm=float(raw[i, 0]),
                y_norm=float(raw[i, 1]),
                z_norm=float(raw[i, 2]),
                x_px=int(raw[i, 0] * 640),
                y_px=int(raw[i, 1] * 480),
                z_depth=0.0,
            )
        )
    return DetectedHand(
        handedness="Right",
        landmarks=landmarks,
        raw_array=raw,
        handedness_confidence=0.98,
        bbox=BoundingBox(0, 0, 100, 100),
    )


def test_extract_single_hand_features_none():
    """Extracting from None hand produces 64-dimensional zeros."""
    feat = extract_single_hand_features(None)
    assert isinstance(feat, np.ndarray)
    assert feat.shape == (HAND_FEATURE_DIM,)
    assert np.all(feat == 0.0)


def test_extract_single_hand_features_valid():
    """Extracting from valid hand sets presence flag and normalized coordinates."""
    hand = _make_dummy_detected_hand()
    feat = extract_single_hand_features(hand)

    assert feat.shape == (HAND_FEATURE_DIM,)
    assert feat[0] == 1.0  # presence indicator
    # Relative coordinates to wrist: landmark 0 relative is (0, 0, 0)
    assert np.allclose(feat[1:4], [0.0, 0.0, 0.0])
    # Distal landmarks are non-zero
    assert np.any(feat[4:] != 0.0)


def test_extract_frame_features_single_and_two_hands():
    """Extracts 128D feature correctly distinguishing single vs two-hand detections."""
    hand1 = _make_dummy_detected_hand((0.3, 0.5, 0.0))
    hand2 = _make_dummy_detected_hand((0.7, 0.5, 0.0))

    # 1. Zero hands
    empty_det = HandDetectionResult(frame_id=0, timestamp=0.0, hands=[])
    feat0 = extract_frame_features(empty_det)
    assert feat0.shape == (FRAME_FEATURE_DIM,)
    assert np.all(feat0 == 0.0)

    # 2. Single hand
    single_det = HandDetectionResult(frame_id=1, timestamp=0.033, hands=[hand1])
    feat1 = extract_frame_features(single_det)
    assert feat1[0] == 1.0  # primary hand present
    assert feat1[64] == 0.0  # secondary hand absent

    # 3. Two hands
    dual_det = HandDetectionResult(frame_id=2, timestamp=0.066, hands=[hand1, hand2])
    feat2 = extract_frame_features(dual_det)
    assert feat2[0] == 1.0  # primary hand present
    assert feat2[64] == 1.0  # secondary hand present


def test_resample_sequence_features_dimensions_and_interpolation():
    """Verifies temporal resampling to exact T=20 time-steps across variable durations."""
    # Test shorter sequence (10 frames)
    short_seq = np.random.randn(10, FRAME_FEATURE_DIM).astype(np.float32)
    res_short = resample_sequence_features(short_seq, target_length=TARGET_SEQUENCE_LENGTH)
    assert res_short.shape == (TARGET_SEQUENCE_LENGTH, FRAME_FEATURE_DIM)

    # Test longer sequence (45 frames)
    long_seq = np.random.randn(45, FRAME_FEATURE_DIM).astype(np.float32)
    res_long = resample_sequence_features(long_seq, target_length=TARGET_SEQUENCE_LENGTH)
    assert res_long.shape == (TARGET_SEQUENCE_LENGTH, FRAME_FEATURE_DIM)

    # Test single frame sequence
    single_seq = np.ones((1, FRAME_FEATURE_DIM), dtype=np.float32) * 5.0
    res_single = resample_sequence_features(single_seq, target_length=TARGET_SEQUENCE_LENGTH)
    assert res_single.shape == (TARGET_SEQUENCE_LENGTH, FRAME_FEATURE_DIM)
    assert np.all(res_single == 5.0)


def test_word_sign_feature_extractor_from_detections():
    """WordSignFeatureExtractor produces exact 2,560D trajectory vector."""
    extractor = WordSignFeatureExtractor()
    hand = _make_dummy_detected_hand()
    detections = [
        HandDetectionResult(frame_id=i, timestamp=i * 0.033, hands=[hand])
        for i in range(15)
    ]

    trajectory = extractor.extract_from_detections(detections)
    assert isinstance(trajectory, np.ndarray)
    assert trajectory.shape == (SEQUENCE_FEATURE_DIM,)
    assert trajectory.dtype == np.float32
