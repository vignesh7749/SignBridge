"""Module 5: Hand Landmark Data Structures, Contracts, and Topology.

Defines the official 21 MediaPipe hand landmarks, topological skeletal
connections, coordinate structures (both normalized and pixel space),
and the DetectedHand data contract.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Sequence
import numpy as np


class HandLandmarkIndex(IntEnum):
    """Official MediaPipe 21 Hand Landmark Indices.

    Topology:
        0: Wrist
        1-4: Thumb (CMC -> MCP -> IP -> TIP)
        5-8: Index finger (MCP -> PIP -> DIP -> TIP)
        9-12: Middle finger (MCP -> PIP -> DIP -> TIP)
        13-16: Ring finger (MCP -> PIP -> DIP -> TIP)
        17-20: Pinky finger (MCP -> PIP -> DIP -> TIP)
    """
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# Official MediaPipe Hand Connections (Pairs of landmark indices representing hand bones)
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # Palm / Wrist base
    (0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index finger
    (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
)

FINGER_CHAINS: dict[str, tuple[int, ...]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


@dataclass(frozen=True)
class LandmarkPoint:
    """Represents a single hand keypoint in both normalized and image pixel coordinate spaces."""
    index: int
    name: str

    # Normalized coordinates relative to input frame dimensions [0.0, 1.0]
    x_norm: float
    y_norm: float
    z_norm: float  # Depth relative to wrist landmark (similar scale to x_norm)

    # Pixel space coordinates
    x_px: int
    y_px: int
    z_depth: float  # Metric depth estimate in pixel scale


# Backwards compatibility representation for simple 3D landmark points
@dataclass(frozen=True)
class Landmark3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box surrounding detected hand in image pixel coordinates."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        return max(0, self.y_max - self.y_min)

    @property
    def center(self) -> tuple[int, int]:
        return (self.x_min + self.width // 2, self.y_min + self.height // 2)

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)


@dataclass(frozen=True)
class HandLandmarks:
    """Represents the 21 keypoints of a single hand (base contract)."""
    handedness: str  # 'Left' or 'Right'
    landmarks: Sequence[Any]  # Sequence of 21 landmark items (LandmarkPoint or Landmark3D)
    raw_array: np.ndarray  # Shape (21, 3) normalized float array for feature normalizer

    def __post_init__(self) -> None:
        if len(self.landmarks) != 21:
            raise ValueError(f"Hand must contain exactly 21 landmarks, got {len(self.landmarks)}")
        if self.raw_array.shape != (21, 3):
            raise ValueError(f"raw_array must have shape (21, 3), got {self.raw_array.shape}")


@dataclass(frozen=True)
class DetectedHand(HandLandmarks):
    """Complete data contract for a single detected hand with vision metadata.

    Decouples MediaPipe internal structures from SignBridge downstream modules.
    """
    handedness_confidence: float = 1.0  # Model certainty in handedness [0.0, 1.0]
    bbox: BoundingBox = field(default_factory=lambda: BoundingBox(0, 0, 0, 0))
    pixel_array: np.ndarray = field(default_factory=lambda: np.zeros((21, 2), dtype=np.int32))
    score: float = 1.0  # Overall detection confidence

    def get_landmark(self, index: int | HandLandmarkIndex) -> LandmarkPoint:
        """Retrieves landmark by index (0-20)."""
        idx = int(index)
        if 0 <= idx < 21:
            return self.landmarks[idx]
        raise IndexError(f"Landmark index must be between 0 and 20, got {idx}")
