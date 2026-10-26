"""Tests for WordSignTemporalSmoother rolling buffer and duplicate suppression."""

from unittest.mock import MagicMock
import numpy as np
import pytest

from signbridge.vision.detector import HandDetectionResult
from signbridge.vision.landmarks import BoundingBox, DetectedHand, LandmarkPoint
from signbridge.word_sign.features import TARGET_SEQUENCE_LENGTH
from signbridge.word_sign.model import WordPredictionOutput, WordSignClassifier
from signbridge.word_sign.temporal import WordSignTemporalSmoother


def _make_dummy_detection(frame_id: int, timestamp: float) -> HandDetectionResult:
    raw = np.zeros((21, 3), dtype=np.float32)
    raw[0] = [0.5, 0.5, 0.0]
    raw[9] = [0.5, 0.6, 0.0]  # scale
    landmarks = [
        LandmarkPoint(
            index=i, name=f"p{i}",
            x_norm=float(raw[i, 0]), y_norm=float(raw[i, 1]), z_norm=0.0,
            x_px=320, y_px=240, z_depth=0.0
        )
        for i in range(21)
    ]
    hand = DetectedHand(
        handedness="Right",
        landmarks=landmarks,
        raw_array=raw,
        handedness_confidence=0.99,
        bbox=BoundingBox(0, 0, 100, 100),
    )
    return HandDetectionResult(frame_id=frame_id, timestamp=timestamp, hands=[hand])


def test_temporal_smoother_warmup_requires_target_length():
    """Smoother returns empty un-stabilized prediction during initial buffer filling."""
    mock_classifier = MagicMock(spec=WordSignClassifier)
    smoother = WordSignTemporalSmoother(classifier=mock_classifier, min_confidence=0.75, stable_count=3)

    # Feed 10 frames (< 20 required)
    for i in range(10):
        det = _make_dummy_detection(i, i * 0.033)
        pred = smoother.process_frame(det, current_time=i * 0.033)
        assert pred.word == ""
        assert not pred.is_stable
        assert not pred.is_newly_emitted

    mock_classifier.predict.assert_not_called()


def test_temporal_smoother_confidence_filter_rejects_low_confidence():
    """Low-confidence predictions below threshold are not emitted as stable words."""
    mock_classifier = MagicMock(spec=WordSignClassifier)
    mock_classifier.predict.return_value = WordPredictionOutput(
        predicted_word="HELLO",
        confidence=0.60,  # Below 0.75 threshold
        top_k=[("HELLO", 0.60)],
        inference_time_ms=0.5,
    )
    smoother = WordSignTemporalSmoother(classifier=mock_classifier, min_confidence=0.75, stable_count=3)

    for i in range(25):
        det = _make_dummy_detection(i, i * 0.033)
        pred = smoother.process_frame(det, current_time=i * 0.033)
        assert not pred.is_stable
        assert not pred.is_newly_emitted


def test_temporal_smoother_consensus_and_duplicate_suppression():
    """Consensus triggers stable emission, while duplicate suppression prevents spamming."""
    mock_classifier = MagicMock(spec=WordSignClassifier)
    mock_classifier.predict.return_value = WordPredictionOutput(
        predicted_word="HELP",
        confidence=0.92,
        top_k=[("HELP", 0.92)],
        inference_time_ms=0.5,
    )
    smoother = WordSignTemporalSmoother(
        classifier=mock_classifier,
        min_confidence=0.75,
        stable_count=3,
        cooldown_seconds=2.0,
    )

    # Feed 19 frames: buffer filling
    for i in range(19):
        det = _make_dummy_detection(i, i * 0.033)
        pred = smoother.process_frame(det, current_time=i * 0.033)
        assert not pred.is_newly_emitted

    # Frame 20, 21: reaching consensus (count=1, count=2)
    p20 = smoother.process_frame(_make_dummy_detection(20, 20 * 0.033), current_time=20 * 0.033)
    p21 = smoother.process_frame(_make_dummy_detection(21, 21 * 0.033), current_time=21 * 0.033)
    assert not p20.is_newly_emitted
    assert not p21.is_newly_emitted

    # Frame 22: consensus count reaches 3 -> Emits!
    p22 = smoother.process_frame(_make_dummy_detection(22, 22 * 0.033), current_time=22 * 0.033)
    assert p22.is_stable
    assert p22.is_newly_emitted
    assert p22.word == "HELP"

    # Frame 23: within 2.0s cooldown -> suppressed!
    p23 = smoother.process_frame(_make_dummy_detection(23, 23 * 0.033), current_time=23 * 0.033)
    assert p23.is_stable
    assert not p23.is_newly_emitted  # Suppressed by cooldown

    # After cooldown expires (> 2.0 seconds later)
    later_time = 22 * 0.033 + 2.5
    p_later = smoother.process_frame(_make_dummy_detection(50, later_time), current_time=later_time)
    assert p_later.is_newly_emitted


def test_temporal_smoother_reset():
    """Calling reset() clears rolling buffer and emission trackers."""
    mock_classifier = MagicMock(spec=WordSignClassifier)
    mock_classifier.predict.return_value = WordPredictionOutput(
        predicted_word="HELLO", confidence=0.8, top_k=[("HELLO", 0.8)], inference_time_ms=0.2
    )
    smoother = WordSignTemporalSmoother(classifier=mock_classifier)
    for i in range(25):
        smoother.process_frame(_make_dummy_detection(i, i * 0.033), current_time=i * 0.033)

    smoother.reset()
    assert len(smoother._frame_buffer) == 0
    assert smoother._consecutive_count == 0
    assert smoother._last_emitted_word is None
