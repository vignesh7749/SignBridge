"""Landmark Validation Module.

Validates raw landmark inputs for structural conformity, numerical integrity,
finite bounds (zero NaNs/Infs), and degenerate geometric collapses before
downstream feature extraction.
"""

from typing import Any
import numpy as np


class LandmarkValidationError(ValueError):
    """Raised when landmark input fails structural or numerical validation."""
    pass


class LandmarkValidator:
    """Validates raw 21-point hand landmarks for ML feature engineering."""

    EXPECTED_SHAPE: tuple[int, int] = (21, 3)
    MIN_PALM_SPAN: float = 1e-6  # Minimum acceptable distance between wrist and middle MCP

    @classmethod
    def validate(cls, raw_input: Any) -> np.ndarray:
        """Validates landmark input and returns a sanitized (21, 3) float32 NumPy array.

        Accepts:
            - np.ndarray of shape (21, 3)
            - DetectedHand or HandLandmarks instance with .raw_array property
            - Sequence of 21 landmark items with (x, y, z) or (x_norm, y_norm, z_norm)

        Raises:
            LandmarkValidationError: If input is null, improperly shaped, non-numeric,
                                     contains NaN/Inf, or represents a collapsed geometry.
        """
        if raw_input is None:
            raise LandmarkValidationError("Landmark input cannot be None.")

        # Extract underlying array
        if hasattr(raw_input, "raw_array"):
            arr = raw_input.raw_array
        elif isinstance(raw_input, np.ndarray):
            arr = raw_input
        elif hasattr(raw_input, "landmarks"):
            # Handle sequence of landmark objects
            lms = raw_input.landmarks
            if len(lms) != 21:
                raise LandmarkValidationError(f"Expected 21 landmarks, got {len(lms)}.")
            coords = []
            for lm in lms:
                if hasattr(lm, "x_norm") and hasattr(lm, "y_norm") and hasattr(lm, "z_norm"):
                    coords.append([lm.x_norm, lm.y_norm, lm.z_norm])
                elif hasattr(lm, "x") and hasattr(lm, "y") and hasattr(lm, "z"):
                    coords.append([lm.x, lm.y, lm.z])
                else:
                    raise LandmarkValidationError(f"Unrecognized landmark element format: {type(lm)}")
            arr = np.array(coords, dtype=np.float32)
        else:
            try:
                arr = np.asarray(raw_input, dtype=np.float32)
            except Exception as e:
                raise LandmarkValidationError(f"Failed to convert input to NumPy array: {e}") from e

        # Check dimension and shape
        if not isinstance(arr, np.ndarray):
            raise LandmarkValidationError(f"Expected numpy.ndarray, got {type(arr).__name__}.")

        if arr.shape != cls.EXPECTED_SHAPE:
            raise LandmarkValidationError(
                f"Invalid landmark array shape {arr.shape}. Expected {cls.EXPECTED_SHAPE}."
            )

        # Check numeric type
        if not np.issubdtype(arr.dtype, np.number):
            raise LandmarkValidationError(f"Landmarks must be numeric, got dtype {arr.dtype}.")

        # Check for NaN, Inf, -Inf
        if not np.all(np.isfinite(arr)):
            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
            raise LandmarkValidationError(
                f"Landmark coordinates contain non-finite values (NaN: {nan_count}, Inf: {inf_count})."
            )

        # Ensure float32 copy
        sanitized = arr.astype(np.float32, copy=True)

        # Check for degenerate hand collapse (wrist and middle finger MCP coincidental)
        palm_vec = sanitized[9] - sanitized[0]
        palm_dist = float(np.linalg.norm(palm_vec))
        if palm_dist < cls.MIN_PALM_SPAN:
            raise LandmarkValidationError(
                f"Degenerate hand geometry: palm anchor distance {palm_dist:.2e} is below threshold {cls.MIN_PALM_SPAN}."
            )

        return sanitized
