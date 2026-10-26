"""Feature Schema Definition (Version 1.0).

Defines the exact structure, named ordering, slice indices, and metadata
for the 94-dimensional ML-ready hand feature vector.
"""

from dataclasses import dataclass
from typing import Final, Sequence


@dataclass(frozen=True)
class FeatureSlice:
    """Represents a named contiguous slice within the feature vector."""
    name: str
    start: int
    end: int
    dimension: int

    @property
    def slice(self) -> slice:
        return slice(self.start, self.end)


class FeatureSchema:
    """Schema specification for hand feature vector version 1.0.

    Total Dimensions: 94
    Breakdown:
      - 63: Normalized 3D Coordinates (21 landmarks x [x, y, z])
      -  5: Normalized Fingertip-to-Wrist Euclidean Distances (thumb, index, middle, ring, pinky)
      - 10: Normalized Pairwise Inter-Fingertip Euclidean Distances (C(5, 2) = 10 pairs)
      - 15: Finger Joint Flexion/Extension Angles (3 joint angles per finger in radians [0, pi])
      -  1: Palm Aspect Ratio (palm width / palm length)
    """

    VERSION: Final[str] = "1.0"
    TOTAL_DIMENSIONS: Final[int] = 94

    # Landmark names in official MediaPipe index order (0 to 20)
    LANDMARK_NAMES: Final[tuple[str, ...]] = (
        "wrist",
        "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
    )

    # Fingertip landmark indices: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
    FINGERTIP_INDICES: Final[tuple[int, ...]] = (4, 8, 12, 16, 20)
    FINGER_NAMES: Final[tuple[str, ...]] = ("thumb", "index", "middle", "ring", "pinky")

    # Pairwise fingertip pairs (10 pairs)
    INTER_TIP_PAIRS: Final[tuple[tuple[int, int], ...]] = (
        (4, 8),    # thumb - index
        (4, 12),   # thumb - middle
        (4, 16),   # thumb - ring
        (4, 20),   # thumb - pinky
        (8, 12),   # index - middle
        (8, 16),   # index - ring
        (8, 20),   # index - pinky
        (12, 16),  # middle - ring
        (12, 20),  # middle - pinky
        (16, 20),  # ring - pinky
    )

    # Joint triplets: (vertex_a, vertex_b_center, vertex_c)
    # Angle computed at vertex_b between vector (a - b) and vector (c - b)
    JOINT_TRIPLETS: Final[tuple[tuple[str, int, int, int], ...]] = (
        # Thumb: CMC (0-1-2), MCP (1-2-3), IP (2-3-4)
        ("thumb_cmc_angle", 0, 1, 2),
        ("thumb_mcp_angle", 1, 2, 3),
        ("thumb_ip_angle", 2, 3, 4),
        # Index: MCP (0-5-6), PIP (5-6-7), DIP (6-7-8)
        ("index_mcp_angle", 0, 5, 6),
        ("index_pip_angle", 5, 6, 7),
        ("index_dip_angle", 6, 7, 8),
        # Middle: MCP (0-9-10), PIP (9-10-11), DIP (10-11-12)
        ("middle_mcp_angle", 0, 9, 10),
        ("middle_pip_angle", 9, 10, 11),
        ("middle_dip_angle", 10, 11, 12),
        # Ring: MCP (0-13-14), PIP (13-14-15), DIP (14-15-16)
        ("ring_mcp_angle", 0, 13, 14),
        ("ring_pip_angle", 13, 14, 15),
        ("ring_dip_angle", 14, 15, 16),
        # Pinky: MCP (0-17-18), PIP (17-18-19), DIP (18-19-20)
        ("pinky_mcp_angle", 0, 17, 18),
        ("pinky_pip_angle", 17, 18, 19),
        ("pinky_dip_angle", 18, 19, 20),
    )

    # Slice definitions
    SLICES: Final[dict[str, FeatureSlice]] = {
        "coordinates": FeatureSlice("coordinates", 0, 63, 63),
        "wrist_tip_distances": FeatureSlice("wrist_tip_distances", 63, 68, 5),
        "inter_tip_distances": FeatureSlice("inter_tip_distances", 68, 78, 10),
        "joint_angles": FeatureSlice("joint_angles", 78, 93, 15),
        "palm_aspect_ratio": FeatureSlice("palm_aspect_ratio", 93, 94, 1),
    }

    # Generate full ordered list of 94 feature names
    @classmethod
    def get_feature_names(cls) -> tuple[str, ...]:
        names: list[str] = []

        # 1. Coordinates (63)
        for lm_name in cls.LANDMARK_NAMES:
            for axis in ("x", "y", "z"):
                names.append(f"{lm_name}_{axis}")

        # 2. Wrist to tip distances (5)
        for finger in cls.FINGER_NAMES:
            names.append(f"dist_wrist_{finger}_tip")

        # 3. Inter-fingertip distances (10)
        for idx1, idx2 in cls.INTER_TIP_PAIRS:
            finger1 = cls.LANDMARK_NAMES[idx1].replace("_tip", "")
            finger2 = cls.LANDMARK_NAMES[idx2].replace("_tip", "")
            names.append(f"dist_{finger1}_{finger2}_tip")

        # 4. Joint angles (15)
        for angle_name, _, _, _ in cls.JOINT_TRIPLETS:
            names.append(angle_name)

        # 5. Palm aspect ratio (1)
        names.append("palm_aspect_ratio")

        return tuple(names)

    @classmethod
    def get_slice(cls, name: str) -> slice:
        """Returns the slice object for a named feature block."""
        if name not in cls.SLICES:
            raise KeyError(f"Unknown feature block '{name}'. Available: {list(cls.SLICES.keys())}")
        return cls.SLICES[name].slice


FEATURE_NAMES: Final[tuple[str, ...]] = FeatureSchema.get_feature_names()
assert len(FEATURE_NAMES) == FeatureSchema.TOTAL_DIMENSIONS, (
    f"Expected {FeatureSchema.TOTAL_DIMENSIONS} feature names, got {len(FEATURE_NAMES)}"
)
