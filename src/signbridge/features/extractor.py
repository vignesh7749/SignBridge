"""Hand Feature Extraction Pipeline.

Transforms raw hand landmarks (DetectedHand / HandLandmarks) into a structured,
deterministic, ML-ready feature vector conforming to FeatureSchema v1.0.
"""

from dataclasses import dataclass, field
import time
from typing import Any
import numpy as np

from signbridge.features.geometry import (
    compute_finger_joint_angles,
    compute_fingertip_distances,
    compute_inter_fingertip_distances,
    compute_palm_aspect_ratio,
)
from signbridge.features.normalizer import CoordinateNormalizer
from signbridge.features.schema import FeatureSchema
from signbridge.features.validator import LandmarkValidator, LandmarkValidationError
from signbridge.vision.landmarks import HandLandmarks


@dataclass(frozen=True)
class FeatureVector:
    """Immutable, typed container for a single extracted hand feature vector."""
    features: np.ndarray  # 1D array of shape (94,), dtype float32
    schema_version: str = FeatureSchema.VERSION
    handedness: str = "Right"
    is_canonicalized: bool = True
    scale_factor: float = 1.0
    wrist_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    timestamp_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.features.ndim != 1 or self.features.shape[0] != FeatureSchema.TOTAL_DIMENSIONS:
            raise ValueError(
                f"Feature vector must have shape ({FeatureSchema.TOTAL_DIMENSIONS},), "
                f"got {self.features.shape}"
            )
        if self.features.dtype != np.float32:
            object.__setattr__(self, "features", self.features.astype(np.float32))

    @property
    def coords(self) -> np.ndarray:
        """Normalized 3D coordinates reshaped to (21, 3)."""
        sl = FeatureSchema.get_slice("coordinates")
        return self.features[sl].reshape(21, 3)

    @property
    def wrist_tip_distances(self) -> np.ndarray:
        """Normalized Euclidean distances from wrist to 5 fingertips (shape: (5,))."""
        sl = FeatureSchema.get_slice("wrist_tip_distances")
        return self.features[sl]

    @property
    def inter_tip_distances(self) -> np.ndarray:
        """Normalized pairwise Euclidean distances between fingertips (shape: (10,))."""
        sl = FeatureSchema.get_slice("inter_tip_distances")
        return self.features[sl]

    @property
    def joint_angles(self) -> np.ndarray:
        """Joint flexion/extension angles in radians [0, pi] (shape: (15,))."""
        sl = FeatureSchema.get_slice("joint_angles")
        return self.features[sl]

    @property
    def palm_aspect_ratio(self) -> float:
        """Ratio of palm width to palm length."""
        sl = FeatureSchema.get_slice("palm_aspect_ratio")
        return float(self.features[sl][0])

    @property
    def l2_norm(self) -> float:
        """Euclidean (L2) norm of the entire feature vector."""
        return float(np.linalg.norm(self.features))

    def summary_stats(self) -> dict[str, float]:
        """Returns statistical summary of the feature vector."""
        return {
            "min": float(np.min(self.features)),
            "max": float(np.max(self.features)),
            "mean": float(np.mean(self.features)),
            "std": float(np.std(self.features)),
            "l2_norm": self.l2_norm,
        }

    def to_dict(self) -> dict[str, Any]:
        """Converts feature vector and metadata to a structured dictionary."""
        feature_names = FeatureSchema.get_feature_names()
        return {
            "metadata": {
                "schema_version": self.schema_version,
                "handedness": self.handedness,
                "is_canonicalized": self.is_canonicalized,
                "scale_factor": self.scale_factor,
                "wrist_center": self.wrist_center,
                "timestamp_ms": self.timestamp_ms,
                "dimension": len(self.features),
            },
            "features": {name: float(val) for name, val in zip(feature_names, self.features)},
        }


class HandFeatureExtractor:
    """Orchestrates landmark validation, coordinate normalization, and geometric feature extraction."""

    def __init__(self, canonicalize_handedness: bool = True) -> None:
        """Initializes the feature extractor.

        Args:
            canonicalize_handedness: If True, horizontally reflects Left hands (x -> -x)
                                    so features reside in canonical right-hand space.
        """
        self.canonicalize_handedness = canonicalize_handedness

    def extract(
        self,
        hand: Any,
        handedness: str | None = None,
        timestamp_ms: float | None = None,
    ) -> FeatureVector:
        """Transforms a detected hand or landmark array into a FeatureVector.

        Pipeline:
          1. Extract handedness metadata (from input object or parameter).
          2. Validate raw landmarks via LandmarkValidator (shape, count, non-finite checks).
          3. Normalize coordinates via CoordinateNormalizer (translation & scale invariance).
          4. Compute geometric features (distances, angles, palm ratio).
          5. Assemble and verify final float32 1D vector of shape (94,).

        Args:
            hand: DetectedHand, HandLandmarks, or (21, 3) NumPy array.
            handedness: Explicit handedness override ('Left' or 'Right'). If None,
                        inferred from `hand.handedness` if available, defaulting to 'Right'.
            timestamp_ms: Optional timestamp in milliseconds. Defaults to current monotonic time.

        Returns:
            FeatureVector with 94 float32 features and extraction metadata.

        Raises:
            LandmarkValidationError: If input landmarks fail structural or numerical validation.
        """
        # Determine handedness
        if handedness is not None:
            effective_handedness = handedness
        elif hasattr(hand, "handedness"):
            effective_handedness = str(hand.handedness)
        else:
            effective_handedness = "Right"

        if timestamp_ms is None:
            timestamp_ms = time.monotonic() * 1000.0

        # Step 1: Validate landmarks
        clean_coords = LandmarkValidator.validate(hand)

        # Step 2: Normalize coordinates
        norm_coords, scale_factor, wrist_origin = CoordinateNormalizer.normalize(
            clean_coords,
            handedness=effective_handedness,
            canonicalize_handedness=self.canonicalize_handedness,
        )

        # Step 3: Compute geometric features from normalized coordinates
        flat_coords = norm_coords.flatten()
        wrist_tip_dists = compute_fingertip_distances(norm_coords)
        inter_tip_dists = compute_inter_fingertip_distances(norm_coords)
        joint_angles = compute_finger_joint_angles(norm_coords)
        palm_ratio = np.array([compute_palm_aspect_ratio(norm_coords)], dtype=np.float32)

        # Step 4: Assemble contiguous 1D array
        feature_array = np.concatenate(
            [flat_coords, wrist_tip_dists, inter_tip_dists, joint_angles, palm_ratio]
        ).astype(np.float32)

        # Step 5: Post-assembly verification
        if feature_array.shape[0] != FeatureSchema.TOTAL_DIMENSIONS:
            raise RuntimeError(
                f"Feature extraction dimension mismatch: expected {FeatureSchema.TOTAL_DIMENSIONS}, "
                f"got {feature_array.shape[0]}"
            )

        if not np.all(np.isfinite(feature_array)):
            raise RuntimeError("Assembled feature vector contains NaN or Inf values.")

        return FeatureVector(
            features=feature_array,
            schema_version=FeatureSchema.VERSION,
            handedness=effective_handedness,
            is_canonicalized=self.canonicalize_handedness and effective_handedness.lower() == "left",
            scale_factor=scale_factor,
            wrist_center=wrist_origin,
            timestamp_ms=timestamp_ms,
        )
