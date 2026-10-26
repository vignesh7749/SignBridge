"""Multi-hand temporal sequence feature extraction for word-level sign recognition.

Extracts normalized 128-dimensional multi-hand representations per frame:
- Primary hand (64D): 1 presence flag + 63 relative normalized coordinates (21 keypoints x 3)
- Secondary hand (64D): 1 presence flag + 63 relative normalized coordinates (21 keypoints x 3)

Resamples variable frame sequences to a fixed temporal resolution T=20 time-steps,
producing a 2,560-dimensional flattened trajectory vector.

Strict Architectural Boundary:
Preserves and leaves untouched the Stage 4 94D FeatureSchema v1.0 used by the alphabet classifier.
"""

from pathlib import Path
from typing import Optional, Sequence, Union
import cv2
import numpy as np

from signbridge.core.logger import get_logger
from signbridge.vision.detector import HandDetectionResult, IHandDetector, MediaPipeHandDetector
from signbridge.vision.landmarks import DetectedHand

logger = get_logger("word_sign.features")

TARGET_SEQUENCE_LENGTH: int = 20
HAND_FEATURE_DIM: int = 64
FRAME_FEATURE_DIM: int = 128
SEQUENCE_FEATURE_DIM: int = TARGET_SEQUENCE_LENGTH * FRAME_FEATURE_DIM  # 2,560


def extract_single_hand_features(hand: Optional[DetectedHand]) -> np.ndarray:
    """Extracts a 64-dimensional feature vector for a single detected hand.

    Layout:
    - Index 0: Presence indicator (1.0 if detected, 0.0 otherwise)
    - Indices 1..63: 21 relative normalized coordinates (x, y, z) scaled by palm size.
    """
    feat = np.zeros(HAND_FEATURE_DIM, dtype=np.float32)
    if hand is None or hand.raw_array is None or hand.raw_array.shape != (21, 3):
        return feat

    feat[0] = 1.0  # Present

    # Wrist coordinate is landmark 0
    wrist = hand.raw_array[0]
    middle_mcp = hand.raw_array[9]

    # Calculate scale factor as palm span (distance from wrist to middle MCP)
    palm_scale = float(np.linalg.norm(middle_mcp - wrist))
    scale = palm_scale if palm_scale > 1e-4 else 1.0

    # Relative coordinates normalized by palm scale
    rel_coords = (hand.raw_array - wrist) / scale
    feat[1:64] = rel_coords.flatten()

    return feat


def extract_hand_features_from_raw(raw_landmarks: Optional[np.ndarray]) -> np.ndarray:
    """Extracts 64D features directly from a (21, 3) numpy landmark array."""
    feat = np.zeros(HAND_FEATURE_DIM, dtype=np.float32)
    if raw_landmarks is None or raw_landmarks.shape != (21, 3):
        return feat

    feat[0] = 1.0
    wrist = raw_landmarks[0]
    middle_mcp = raw_landmarks[9]
    palm_scale = float(np.linalg.norm(middle_mcp - wrist))
    scale = palm_scale if palm_scale > 1e-4 else 1.0

    rel_coords = (raw_landmarks - wrist) / scale
    feat[1:64] = rel_coords.flatten()
    return feat


def extract_frame_features(detection: Optional[HandDetectionResult]) -> np.ndarray:
    """Extracts 128-dimensional multi-hand features from a single frame detection result."""
    feat = np.zeros(FRAME_FEATURE_DIM, dtype=np.float32)
    if detection is None or not detection.has_hands:
        return feat

    # Primary hand (hands[0])
    primary_hand = detection.hands[0] if len(detection.hands) > 0 else None
    feat[0:64] = extract_single_hand_features(primary_hand)

    # Secondary hand (hands[1])
    secondary_hand = detection.hands[1] if len(detection.hands) > 1 else None
    feat[64:128] = extract_single_hand_features(secondary_hand)

    return feat


def resample_sequence_features(
    sequence: np.ndarray,
    target_length: int = TARGET_SEQUENCE_LENGTH,
) -> np.ndarray:
    """Linearly resamples a variable-length feature sequence (N, D) to (target_length, D).

    Args:
        sequence: Array of shape (N, D).
        target_length: Desired number of temporal timesteps (default 20).

    Returns:
        Array of shape (target_length, D) with float32 precision.
    """
    if sequence is None or len(sequence) == 0:
        return np.zeros((target_length, FRAME_FEATURE_DIM), dtype=np.float32)

    num_frames, feat_dim = sequence.shape
    if num_frames == target_length:
        return sequence.astype(np.float32, copy=True)

    if num_frames == 1:
        # Replicate single frame across all timesteps
        return np.tile(sequence[0], (target_length, 1)).astype(np.float32)

    # Linear temporal interpolation
    target_times = np.linspace(0, num_frames - 1, target_length)
    resampled = np.empty((target_length, feat_dim), dtype=np.float32)

    for step_idx, t in enumerate(target_times):
        low_idx = int(np.floor(t))
        high_idx = min(low_idx + 1, num_frames - 1)
        alpha = t - low_idx

        if low_idx == high_idx:
            resampled[step_idx] = sequence[low_idx]
        else:
            resampled[step_idx] = (1.0 - alpha) * sequence[low_idx] + alpha * sequence[high_idx]

    return resampled


class WordSignFeatureExtractor:
    """Extractor for converting video feeds and landmark streams into 2,560D trajectory vectors."""

    def __init__(self, detector: Optional[IHandDetector] = None) -> None:
        self._detector = detector
        self._owns_detector = False

    def _get_detector(self) -> IHandDetector:
        if self._detector is None:
            self._detector = MediaPipeHandDetector()
            self._owns_detector = True
        return self._detector

    def extract_from_detections(
        self,
        detections: Sequence[Optional[HandDetectionResult]],
        target_length: int = TARGET_SEQUENCE_LENGTH,
    ) -> np.ndarray:
        """Extracts fixed-length trajectory feature vector from a sequence of frame detections.

        Returns:
            Flattened 1D numpy array of length target_length * 128 (default 2,560).
        """
        if not detections:
            return np.zeros(target_length * FRAME_FEATURE_DIM, dtype=np.float32)

        frame_features = [extract_frame_features(det) for det in detections]
        seq_array = np.vstack(frame_features)  # (N, 128)
        resampled = resample_sequence_features(seq_array, target_length=target_length)  # (20, 128)
        return resampled.flatten()

    def extract_from_frames(
        self,
        frames: Sequence[np.ndarray],
        target_length: int = TARGET_SEQUENCE_LENGTH,
    ) -> np.ndarray:
        """Runs detection on a sequence of BGR image frames and returns 2,560D vector."""
        if not frames:
            return np.zeros(target_length * FRAME_FEATURE_DIM, dtype=np.float32)

        detector = self._get_detector()
        detections = []
        for idx, frame in enumerate(frames):
            det = detector.detect(frame, frame_id=idx)
            detections.append(det)

        return self.extract_from_detections(detections, target_length=target_length)

    def extract_from_video_file(
        self,
        video_path: Union[str, Path],
        target_length: int = TARGET_SEQUENCE_LENGTH,
        max_frames: int = 150,
    ) -> np.ndarray:
        """Decodes video file via OpenCV, processes frames, and returns 2,560D vector."""
        path = Path(video_path).resolve()
        if not path.exists():
            logger.warning("Video file does not exist: %s", path)
            return np.zeros(target_length * FRAME_FEATURE_DIM, dtype=np.float32)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.warning("Could not open video file: %s", path)
            return np.zeros(target_length * FRAME_FEATURE_DIM, dtype=np.float32)

        frames = []
        try:
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frames.append(frame)
        finally:
            cap.release()

        return self.extract_from_frames(frames, target_length=target_length)

    def close(self) -> None:
        if self._owns_detector and self._detector is not None:
            self._detector.close()
            self._detector = None
