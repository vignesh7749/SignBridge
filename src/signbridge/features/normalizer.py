"""Module 6: Feature Normalization.

Transforms raw 3D hand landmarks into translation- and scale-invariant numerical vectors.
Guarantees backward compatibility with Stage 1-3 contracts while adding robust
coordinate normalization and handedness canonicalization for Stage 4.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np

from signbridge.vision.landmarks import HandLandmarks


@dataclass(frozen=True)
class NormalizedFeatures:
    """Legacy feature container preserved for Stage 1-3 backward compatibility."""
    vector: np.ndarray  # 1D flattened array of normalized features (length: 63 for 21x3)
    handedness: str
    scale_factor: float


class FeatureNormalizer:
    """Computes scale-invariant, wrist-centered coordinate representations (Legacy)."""

    @staticmethod
    def normalize_hand(hand: HandLandmarks) -> NormalizedFeatures:
        """Normalizes 21 3D points:

        1. Wrist translation subtraction: All points are offset relative to landmark 0 (wrist).
        2. Palm scaling: Scaled by Euclidean distance between wrist (0) and middle MCP (9).
        3. Flattening to 1D vector.
        """
        raw_coords = hand.raw_array.copy()  # Shape: (21, 3)

        # 1. Translation Invariance: center at wrist (landmark 0)
        wrist = raw_coords[0]
        centered = raw_coords - wrist

        # 2. Scale Invariance: normalize by distance between wrist (0) and middle MCP (9)
        palm_dist = np.linalg.norm(centered[9])
        scale_factor = float(palm_dist) if palm_dist > 1e-6 else 1.0
        normalized = centered / scale_factor

        # 3. Flatten to 1D vector (length 63)
        flat_vector = normalized.flatten().astype(np.float32)

        return NormalizedFeatures(
            vector=flat_vector,
            handedness=hand.handedness,
            scale_factor=scale_factor,
        )


class CoordinateNormalizer:
    """Production coordinate normalization for Stage 4 feature extraction.

    Applies:
      1. Translation invariance: Shifts coordinates so wrist (landmark 0) is at origin (0, 0, 0).
      2. Scale invariance: Normalizes coordinate dimensions by the palm anchor distance ||P9 - P0||.
      3. Handedness canonicalization (optional): Reflects Left hand along X-axis (x -> -x)
         so hand signs can be recognized invariant of whether signed with left or right hand.
    """

    EPSILON: float = 1e-6

    @classmethod
    def normalize(
        cls,
        coords: np.ndarray,
        handedness: str = "Right",
        canonicalize_handedness: bool = True,
    ) -> tuple[np.ndarray, float, tuple[float, float, float]]:
        """Transforms (21, 3) landmark coordinates into normalized space.

        Args:
            coords: Sanitized (21, 3) float32 NumPy array.
            handedness: 'Left' or 'Right'.
            canonicalize_handedness: If True, horizontally reflects Left hands (x -> -x)
                                    to right-hand canonical space.

        Returns:
            Tuple of:
              - normalized_coords: (21, 3) float32 array where P0 is (0, 0, 0)
              - scale_factor: Palm distance used for scaling (float)
              - wrist_origin: Original (x, y, z) coordinates of wrist (tuple of 3 floats)
        """
        # 1. Translation Invariance: shift wrist (index 0) to origin
        wrist_origin = (float(coords[0, 0]), float(coords[0, 1]), float(coords[0, 2]))
        centered = coords - coords[0]

        # 2. Scale Invariance: scale by Euclidean distance to middle finger MCP (index 9)
        palm_anchor = centered[9]
        palm_distance = float(np.sqrt(np.dot(palm_anchor, palm_anchor)))
        scale_factor = palm_distance if palm_distance > cls.EPSILON else 1.0

        normalized = (centered / scale_factor).astype(np.float32)

        # 3. Handedness Canonicalization (optional)
        if canonicalize_handedness and handedness.strip().lower() == "left":
            normalized[:, 0] = -normalized[:, 0]

        return normalized, scale_factor, wrist_origin
