"""Feature extraction, normalization, and geometric representations for SignBridge."""

from signbridge.features.extractor import FeatureVector, HandFeatureExtractor
from signbridge.features.geometry import (
    compute_angle_3d,
    compute_euclidean_distance,
    compute_finger_joint_angles,
    compute_fingertip_distances,
    compute_inter_fingertip_distances,
    compute_palm_aspect_ratio,
)
from signbridge.features.normalizer import (
    CoordinateNormalizer,
    FeatureNormalizer,
    NormalizedFeatures,
)
from signbridge.features.schema import FEATURE_NAMES, FeatureSchema, FeatureSlice
from signbridge.features.validator import LandmarkValidationError, LandmarkValidator

__all__ = [
    # Schema
    "FeatureSchema",
    "FEATURE_NAMES",
    "FeatureSlice",
    # Validation
    "LandmarkValidator",
    "LandmarkValidationError",
    # Normalization
    "FeatureNormalizer",
    "NormalizedFeatures",
    "CoordinateNormalizer",
    # Extraction & Container
    "HandFeatureExtractor",
    "FeatureVector",
    # Geometry operations
    "compute_euclidean_distance",
    "compute_angle_3d",
    "compute_finger_joint_angles",
    "compute_fingertip_distances",
    "compute_inter_fingertip_distances",
    "compute_palm_aspect_ratio",
]
