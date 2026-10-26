"""Module 4: Hand Detection Abstraction & MediaPipe Implementation.

Provides clean separation between MediaPipe's internal graph representations
and SignBridge's domain contracts (DetectedHand, HandDetectionResult).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Optional, Sequence
import cv2
import numpy as np

from signbridge.core.config import VisionConfig
from signbridge.core.exceptions import VisionPipelineError
from signbridge.core.logger import get_logger
from signbridge.vision.landmarks import (
    BoundingBox,
    DetectedHand,
    HandLandmarkIndex,
    LandmarkPoint,
)

logger = get_logger("vision.detector")


@dataclass(frozen=True)
class HandDetectionResult:
    """Contains all hands detected in a single frame, with timing telemetry."""
    frame_id: int
    timestamp: float
    hands: Sequence[DetectedHand] = field(default_factory=list)
    processing_time_ms: float = 0.0
    annotated_frame: Optional[np.ndarray] = None

    @property
    def has_hands(self) -> bool:
        """Returns True if one or more hands are detected."""
        return len(self.hands) > 0

    @property
    def hand_count(self) -> int:
        """Returns the number of detected hands."""
        return len(self.hands)

    @property
    def primary_hand(self) -> Optional[DetectedHand]:
        """Returns the first detected hand, or None if no hands detected."""
        return self.hands[0] if self.hands else None


class IHandDetector(ABC):
    """Abstract interface for hand detection and tracking backends."""

    @abstractmethod
    def detect(
        self,
        bgr_frame: np.ndarray,
        frame_id: int = 0,
        timestamp: Optional[float] = None,
    ) -> HandDetectionResult:
        """Process BGR frame and extract structured hand landmarks."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release backend detector resources."""
        pass

    def __enter__(self) -> "IHandDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class MediaPipeHandDetector(IHandDetector):
    """Production MediaPipe Hands detector implementation.

    Decouples MediaPipe's legacy/tasks solution from the rest of SignBridge.
    Receives raw BGR frames, performs internal color conversion to RGB,
    runs landmark inference, and converts outputs into typed DetectedHand objects.
    """

    def __init__(self, config: Optional[VisionConfig] = None) -> None:
        self.config = config or VisionConfig()
        self._mp_hands = None
        self._is_closed = False
        self._init_detector()

    def _init_detector(self) -> None:
        """Initializes the underlying MediaPipe Hands graph."""
        try:
            import mediapipe as mp

            logger.info(
                "Initializing MediaPipe Hands (max_hands=%d, min_detect=%.2f, min_track=%.2f, complexity=%d)...",
                self.config.max_num_hands,
                self.config.min_detection_confidence,
                self.config.min_tracking_confidence,
                self.config.model_complexity,
            )

            self._mp_hands = mp.solutions.hands.Hands(
                static_image_mode=self.config.static_image_mode,
                max_num_hands=self.config.max_num_hands,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                model_complexity=self.config.model_complexity,
            )
            logger.info("MediaPipe Hands initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize MediaPipe Hands: %s", e)
            raise VisionPipelineError(f"MediaPipe Hands initialization failed: {e}") from e

    def detect(
        self,
        bgr_frame: np.ndarray,
        frame_id: int = 0,
        timestamp: Optional[float] = None,
    ) -> HandDetectionResult:
        """Runs hand detection on an OpenCV BGR frame.

        Args:
            bgr_frame: Input image array in BGR format (H, W, 3).
            frame_id: Monotonic sequence ID of the frame.
            timestamp: Capture timestamp (defaults to time.time()).

        Returns:
            HandDetectionResult containing zero or more DetectedHand instances.
        """
        now = timestamp if timestamp is not None else time.time()
        start_t = time.perf_counter()

        # Input validation
        if bgr_frame is None or not isinstance(bgr_frame, np.ndarray):
            logger.warning("Invalid frame provided to detect(): frame is None or not a numpy array.")
            return HandDetectionResult(frame_id=frame_id, timestamp=now, hands=[], processing_time_ms=0.0)

        if bgr_frame.size == 0 or len(bgr_frame.shape) != 3 or bgr_frame.shape[2] != 3:
            logger.warning("Invalid frame dimensions provided to detect(): shape=%s", bgr_frame.shape if hasattr(bgr_frame, 'shape') else None)
            return HandDetectionResult(frame_id=frame_id, timestamp=now, hands=[], processing_time_ms=0.0)

        if self._is_closed or self._mp_hands is None:
            logger.warning("Attempted to detect with closed MediaPipeHandDetector.")
            return HandDetectionResult(frame_id=frame_id, timestamp=now, hands=[], processing_time_ms=0.0)

        h, w = bgr_frame.shape[:2]

        try:
            # 1. Color-space conversion: BGR -> RGB (MediaPipe requires RGB)
            # Writeable flag set to False improves performance by avoiding memory copy
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            # 2. Run inference
            results = self._mp_hands.process(rgb_frame)

            detected_hands: list[DetectedHand] = []

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness_info in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    classification = handedness_info.classification[0]
                    hand_label = classification.label  # 'Left' or 'Right'
                    hand_conf = float(classification.score)

                    points: list[LandmarkPoint] = []
                    raw_coords = np.zeros((21, 3), dtype=np.float32)
                    pixel_coords = np.zeros((21, 2), dtype=np.int32)

                    xs_px: list[int] = []
                    ys_px: list[int] = []

                    for idx, lm in enumerate(hand_landmarks.landmark):
                        x_norm = float(lm.x)
                        y_norm = float(lm.y)
                        z_norm = float(lm.z)

                        # Calculate pixel coordinates
                        x_px = int(np.clip(round(x_norm * w), 0, w - 1))
                        y_px = int(np.clip(round(y_norm * h), 0, h - 1))
                        z_depth = z_norm * float(w)

                        xs_px.append(x_px)
                        ys_px.append(y_px)

                        raw_coords[idx] = [x_norm, y_norm, z_norm]
                        pixel_coords[idx] = [x_px, y_px]

                        lm_name = HandLandmarkIndex(idx).name
                        points.append(
                            LandmarkPoint(
                                index=idx,
                                name=lm_name,
                                x_norm=x_norm,
                                y_norm=y_norm,
                                z_norm=z_norm,
                                x_px=x_px,
                                y_px=y_px,
                                z_depth=z_depth,
                            )
                        )

                    # Compute bounding box with padding
                    pad_x = int(0.04 * w)
                    pad_y = int(0.04 * h)
                    bbox = BoundingBox(
                        x_min=max(0, min(xs_px) - pad_x),
                        y_min=max(0, min(ys_px) - pad_y),
                        x_max=min(w - 1, max(xs_px) + pad_x),
                        y_max=min(h - 1, max(ys_px) + pad_y),
                    )

                    hand_obj = DetectedHand(
                        handedness=hand_label,
                        handedness_confidence=hand_conf,
                        landmarks=points,
                        bbox=bbox,
                        raw_array=raw_coords,
                        pixel_array=pixel_coords,
                        score=hand_conf,
                    )
                    detected_hands.append(hand_obj)

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            return HandDetectionResult(
                frame_id=frame_id,
                timestamp=now,
                hands=detected_hands,
                processing_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error("Error during MediaPipe hand detection: %s", e)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return HandDetectionResult(
                frame_id=frame_id,
                timestamp=now,
                hands=[],
                processing_time_ms=elapsed_ms,
            )

    def close(self) -> None:
        """Releases underlying MediaPipe graph resources."""
        if not self._is_closed:
            if self._mp_hands is not None:
                try:
                    self._mp_hands.close()
                except Exception as e:
                    logger.warning("Exception closing MediaPipe graph: %s", e)
                self._mp_hands = None
            self._is_closed = True
            logger.info("MediaPipeHandDetector closed and resources freed.")
