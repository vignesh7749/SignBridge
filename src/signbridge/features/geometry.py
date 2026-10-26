"""Geometric Computation Module.

Provides numerically stable 3D geometric operations on hand landmarks:
  - Euclidean distances (fingertips to wrist, inter-fingertip pairs)
  - 3D joint flexion/extension angles with clipping and zero-vector safety
  - Palm aspect ratio and spatial metrics
"""

import numpy as np
from signbridge.features.schema import FeatureSchema


def compute_euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Computes the Euclidean distance between two 3D points.

    Args:
        p1: 1D NumPy array [x, y, z]
        p2: 1D NumPy array [x, y, z]

    Returns:
        float >= 0.0
    """
    diff = p1 - p2
    return float(np.sqrt(np.dot(diff, diff)))


def compute_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-7) -> float:
    """Computes the interior angle at vertex `b` formed by vectors (a - b) and (c - b).

    Numerically safe:
      - If either vector has length < eps, returns 0.0 (avoids division by zero / NaN).
      - Dot product cosine is strictly clamped to [-1.0, 1.0] before calling arccos.
      - Returns angle in radians within [0.0, pi].

    Args:
        a: Point A [x, y, z]
        b: Central vertex B [x, y, z]
        c: Point C [x, y, z]
        eps: Small threshold to detect degenerate zero-length vectors.

    Returns:
        Interior angle in radians in range [0.0, pi].
    """
    u = (a - b).astype(np.float64)
    v = (c - b).astype(np.float64)

    u_norm = float(np.linalg.norm(u))
    v_norm = float(np.linalg.norm(v))

    if u_norm < eps or v_norm < eps:
        return 0.0

    cos_angle = float(np.dot(u, v) / (u_norm * v_norm))
    # Strictly clip to avoid float rounding edge cases like 1.00000000002 -> NaN
    clipped_cos = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.arccos(clipped_cos))


def compute_finger_joint_angles(coords: np.ndarray) -> np.ndarray:
    """Computes the 15 joint flexion/extension angles for all five fingers.

    Triplets are defined in FeatureSchema.JOINT_TRIPLETS:
      - Thumb:  CMC (0-1-2), MCP (1-2-3), IP (2-3-4)
      - Index:  MCP (0-5-6), PIP (5-6-7), DIP (6-7-8)
      - Middle: MCP (0-9-10), PIP (9-10-11), DIP (10-11-12)
      - Ring:   MCP (0-13-14), PIP (13-14-15), DIP (14-15-16)
      - Pinky:  MCP (0-17-18), PIP (17-18-19), DIP (18-19-20)

    Args:
        coords: (21, 3) landmark coordinates.

    Returns:
        1D float32 array of length 15 containing angles in radians [0.0, pi].
    """
    angles = np.zeros(len(FeatureSchema.JOINT_TRIPLETS), dtype=np.float32)
    for idx, (_, a_idx, b_idx, c_idx) in enumerate(FeatureSchema.JOINT_TRIPLETS):
        angles[idx] = compute_angle_3d(coords[a_idx], coords[b_idx], coords[c_idx])
    return angles


def compute_fingertip_distances(coords: np.ndarray) -> np.ndarray:
    """Computes Euclidean distances from the wrist (landmark 0) to each fingertip.

    Fingertips: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20).

    Args:
        coords: (21, 3) landmark coordinates.

    Returns:
        1D float32 array of length 5.
    """
    wrist = coords[0]
    distances = np.zeros(len(FeatureSchema.FINGERTIP_INDICES), dtype=np.float32)
    for i, tip_idx in enumerate(FeatureSchema.FINGERTIP_INDICES):
        distances[i] = compute_euclidean_distance(wrist, coords[tip_idx])
    return distances


def compute_inter_fingertip_distances(coords: np.ndarray) -> np.ndarray:
    """Computes pairwise Euclidean distances between all 5 fingertips.

    Pairs defined in FeatureSchema.INTER_TIP_PAIRS (10 combinations).

    Args:
        coords: (21, 3) landmark coordinates.

    Returns:
        1D float32 array of length 10.
    """
    distances = np.zeros(len(FeatureSchema.INTER_TIP_PAIRS), dtype=np.float32)
    for i, (idx1, idx2) in enumerate(FeatureSchema.INTER_TIP_PAIRS):
        distances[i] = compute_euclidean_distance(coords[idx1], coords[idx2])
    return distances


def compute_palm_aspect_ratio(coords: np.ndarray, eps: float = 1e-7) -> float:
    """Computes the ratio of palm width (Pinky MCP to Index MCP) to palm length (Wrist to Middle MCP).

    Indices:
      - Pinky MCP: 17
      - Index MCP: 5
      - Wrist: 0
      - Middle MCP: 9

    Args:
        coords: (21, 3) landmark coordinates.
        eps: Epsilon threshold for division stability.

    Returns:
        float >= 0.0 representing palm width / palm length.
    """
    palm_width = compute_euclidean_distance(coords[17], coords[5])
    palm_length = compute_euclidean_distance(coords[9], coords[0])
    return float(palm_width / max(palm_length, eps))
