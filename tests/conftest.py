"""Pytest configuration, fixtures, and synthetic landmark generators."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from signbridge.vision.landmarks import HandLandmarks, Landmark3D


@pytest.fixture
def mock_hand_landmarks() -> HandLandmarks:
    """Generates synthetic 21-point hand landmark data."""
    # Synthetic coordinates: 21 points
    coords = np.zeros((21, 3), dtype=np.float32)
    # Wrist at (100, 200, 0)
    coords[0] = [100.0, 200.0, 0.0]
    # Middle MCP (landmark 9) at (100, 250, 0) -> distance = 50.0
    coords[9] = [100.0, 250.0, 0.0]

    for i in range(1, 21):
        if i != 9:
            coords[i] = [100.0 + i * 2.0, 200.0 + i * 3.0, 0.1 * i]

    landmarks_list = [Landmark3D(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in coords]
    return HandLandmarks(handedness="Right", landmarks=landmarks_list, raw_array=coords)
