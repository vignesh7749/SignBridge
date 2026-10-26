"""Unit tests for Stage 3: MediaPipe Hand Detection & Landmark Extraction."""

import numpy as np
import pytest

from signbridge.core.config import VisionConfig
from signbridge.vision.detector import HandDetectionResult, MediaPipeHandDetector
from signbridge.vision.landmarks import (
    FINGER_CHAINS,
    HAND_CONNECTIONS,
    BoundingBox,
    DetectedHand,
    HandLandmarkIndex,
    LandmarkPoint,
)


@pytest.fixture
def mock_detected_hand() -> DetectedHand:
    """Creates a deterministic synthetic DetectedHand object for hardware-independent testing."""
    w, h = 640, 480
    landmarks: list[LandmarkPoint] = []
    raw_coords = np.zeros((21, 3), dtype=np.float32)
    pixel_coords = np.zeros((21, 2), dtype=np.int32)

    for i in range(21):
        x_norm = 0.3 + (i * 0.02)
        y_norm = 0.4 + (i * 0.01)
        z_norm = -0.05 + (i * 0.005)

        x_px = int(round(x_norm * w))
        y_px = int(round(y_norm * h))
        z_depth = z_norm * float(w)

        raw_coords[i] = [x_norm, y_norm, z_norm]
        pixel_coords[i] = [x_px, y_px]

        landmarks.append(
            LandmarkPoint(
                index=i,
                name=HandLandmarkIndex(i).name,
                x_norm=x_norm,
                y_norm=y_norm,
                z_norm=z_norm,
                x_px=x_px,
                y_px=y_px,
                z_depth=z_depth,
            )
        )

    bbox = BoundingBox(
        x_min=int(min(pixel_coords[:, 0]) - 10),
        y_min=int(min(pixel_coords[:, 1]) - 10),
        x_max=int(max(pixel_coords[:, 0]) + 10),
        y_max=int(max(pixel_coords[:, 1]) + 10),
    )

    return DetectedHand(
        handedness="Right",
        handedness_confidence=0.98,
        landmarks=landmarks,
        bbox=bbox,
        raw_array=raw_coords,
        pixel_array=pixel_coords,
        score=0.98,
    )


def test_detected_hand_structure(mock_detected_hand: DetectedHand) -> None:
    """Verifies DetectedHand attributes, landmark count, and shapes."""
    assert mock_detected_hand.handedness == "Right"
    assert mock_detected_hand.handedness_confidence == 0.98
    assert len(mock_detected_hand.landmarks) == 21
    assert mock_detected_hand.raw_array.shape == (21, 3)
    assert mock_detected_hand.pixel_array.shape == (21, 2)
    assert mock_detected_hand.bbox.width > 0
    assert mock_detected_hand.bbox.height > 0


def test_landmark_count_validation() -> None:
    """Verifies that DetectedHand enforces exactly 21 landmarks."""
    short_landmarks = [
        LandmarkPoint(0, "WRIST", 0.5, 0.5, 0.0, 320, 240, 0.0)
    ] * 20  # 20 points instead of 21

    with pytest.raises(ValueError, match="must contain exactly 21 landmarks"):
        DetectedHand(
            handedness="Left",
            handedness_confidence=0.9,
            landmarks=short_landmarks,
            bbox=BoundingBox(0, 0, 10, 10),
            raw_array=np.zeros((20, 3), dtype=np.float32),
            pixel_array=np.zeros((20, 2), dtype=np.int32),
        )


def test_landmark_ordering_and_topology() -> None:
    """Verifies official MediaPipe landmark ordering and finger chains."""
    assert HandLandmarkIndex.WRIST == 0
    assert HandLandmarkIndex.THUMB_CMC == 1
    assert HandLandmarkIndex.THUMB_TIP == 4
    assert HandLandmarkIndex.INDEX_FINGER_MCP == 5
    assert HandLandmarkIndex.INDEX_FINGER_TIP == 8
    assert HandLandmarkIndex.MIDDLE_FINGER_MCP == 9
    assert HandLandmarkIndex.MIDDLE_FINGER_TIP == 12
    assert HandLandmarkIndex.RING_FINGER_MCP == 13
    assert HandLandmarkIndex.RING_FINGER_TIP == 16
    assert HandLandmarkIndex.PINKY_MCP == 17
    assert HandLandmarkIndex.PINKY_TIP == 20

    # Verify finger chain groupings
    assert FINGER_CHAINS["thumb"] == (1, 2, 3, 4)
    assert FINGER_CHAINS["index"] == (5, 6, 7, 8)
    assert FINGER_CHAINS["middle"] == (9, 10, 11, 12)
    assert FINGER_CHAINS["ring"] == (13, 14, 15, 16)
    assert FINGER_CHAINS["pinky"] == (17, 18, 19, 20)

    # Verify connections count (21 standard skeletal bones)
    assert len(HAND_CONNECTIONS) == 21


def test_coordinate_system_projection(mock_detected_hand: DetectedHand) -> None:
    """Verifies that pixel coordinates properly project from normalized coordinates."""
    w, h = 640, 480
    for lm in mock_detected_hand.landmarks:
        # Check normalized ranges
        assert 0.0 <= lm.x_norm <= 1.0
        assert 0.0 <= lm.y_norm <= 1.0

        # Check pixel coordinates
        expected_x_px = int(round(lm.x_norm * w))
        expected_y_px = int(round(lm.y_norm * h))
        assert lm.x_px == expected_x_px
        assert lm.y_px == expected_y_px


def test_multiple_hand_result_container(mock_detected_hand: DetectedHand) -> None:
    """Verifies 0-hand, 1-hand, and 2-hand HandDetectionResult states."""
    # 0 hands
    res_zero = HandDetectionResult(frame_id=1, timestamp=100.0, hands=[])
    assert res_zero.has_hands is False
    assert res_zero.hand_count == 0
    assert res_zero.primary_hand is None

    # 1 hand
    res_one = HandDetectionResult(frame_id=2, timestamp=100.03, hands=[mock_detected_hand])
    assert res_one.has_hands is True
    assert res_one.hand_count == 1
    assert res_one.primary_hand is not None
    assert res_one.primary_hand.handedness == "Right"

    # 2 hands
    left_hand = DetectedHand(
        handedness="Left",
        handedness_confidence=0.95,
        landmarks=mock_detected_hand.landmarks,
        bbox=mock_detected_hand.bbox,
        raw_array=mock_detected_hand.raw_array,
        pixel_array=mock_detected_hand.pixel_array,
        score=0.95,
    )
    res_two = HandDetectionResult(frame_id=3, timestamp=100.06, hands=[mock_detected_hand, left_hand])
    assert res_two.hand_count == 2
    assert res_two.hands[0].handedness == "Right"
    assert res_two.hands[1].handedness == "Left"


def test_detector_invalid_inputs() -> None:
    """Verifies that MediaPipeHandDetector handles None and corrupted frames without crashing."""
    detector = MediaPipeHandDetector(VisionConfig(min_detection_confidence=0.5))

    # None frame
    res1 = detector.detect(None)  # type: ignore[arg-type]
    assert res1.has_hands is False
    assert res1.hands == []

    # Empty frame
    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    res2 = detector.detect(empty_frame)
    assert res2.has_hands is False

    # 1-channel grayscale frame instead of 3-channel BGR
    gray_frame = np.zeros((480, 640), dtype=np.uint8)
    res3 = detector.detect(gray_frame)
    assert res3.has_hands is False

    detector.close()


def test_detector_blank_frame_no_hands() -> None:
    """Verifies that running detection on a blank black frame returns zero hands cleanly."""
    detector = MediaPipeHandDetector(VisionConfig(min_detection_confidence=0.7))
    blank_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

    result = detector.detect(blank_bgr, frame_id=1, timestamp=1.0)
    assert result.has_hands is False
    assert result.hand_count == 0
    assert result.processing_time_ms >= 0.0

    detector.close()


def test_detector_context_manager() -> None:
    """Verifies that MediaPipeHandDetector functions correctly within a context manager."""
    with MediaPipeHandDetector() as detector:
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        res = detector.detect(blank)
        assert res.has_hands is False

    assert detector._is_closed is True
