"""Module 5: Authoritative Dataset Configuration & Identity Validation.

Centralized type-safe settings and integrity validation for SignBridge datasets.
Defines the single source of truth for:
  - Production dataset root directory
  - Expected dataset identity (name, source, classes, labels, license, sample bounds)
  - Strict failure on missing dataset or identity mismatch (zero silent fallbacks)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


class DatasetConfigError(Exception):
    """Base exception for dataset configuration errors."""
    pass


class DatasetNotFoundError(DatasetConfigError, FileNotFoundError):
    """Raised when the configured dataset directory does not exist."""
    pass


class DatasetIdentityError(DatasetConfigError, ValueError):
    """Raised when dataset content fails identity validation against schema."""
    pass


@dataclass(frozen=True)
class DatasetConfig:
    """Authoritative specification and directory configuration for SignBridge datasets."""

    # Project root
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
    )

    # Dataset Identity
    dataset_name: str = "ASL-Alphabet-Natural-RGB"
    source_repo: str = "Marxulia/asl_sign_languages_alphabets_v03"
    source_url: str = "https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03"
    license: str = "Unspecified / Community Open Access (Requires Verification)"
    dataset_version: str = "1.0.0"
    modality: str = "Natural RGB Photographs"

    # Vocabulary: 26 Classes (A through Z)
    num_classes: int = 26
    expected_classes: tuple[str, ...] = (
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z",
    )

    # Expected sample boundaries
    expected_min_samples: int = 10000
    expected_max_samples: int = 12000

    # Paths (relative to project_root by default)
    raw_dir_override: Path | None = None
    processed_dir_override: Path | None = None
    metadata_dir_override: Path | None = None
    reports_dir_override: Path | None = None

    @property
    def raw_dir(self) -> Path:
        """Absolute path to the raw dataset root directory."""
        if self.raw_dir_override is not None:
            return Path(self.raw_dir_override).resolve()
        return (self.project_root / "datasets" / "external" / "asl_alphabet" / "raw").resolve()

    @property
    def processed_dir(self) -> Path:
        """Absolute path to the processed features directory."""
        if self.processed_dir_override is not None:
            return Path(self.processed_dir_override).resolve()
        return (self.project_root / "datasets" / "processed").resolve()

    @property
    def metadata_dir(self) -> Path:
        """Absolute path to the dataset metadata directory."""
        if self.metadata_dir_override is not None:
            return Path(self.metadata_dir_override).resolve()
        return (self.project_root / "datasets" / "metadata").resolve()

    @property
    def reports_dir(self) -> Path:
        """Absolute path to the dataset reports directory."""
        if self.reports_dir_override is not None:
            return Path(self.reports_dir_override).resolve()
        return (self.project_root / "datasets" / "reports").resolve()

    def validate_identity(self, target_raw_dir: Path | None = None) -> list[str]:
        """Validates that the target directory exists and matches expected dataset identity.

        STRICT NO-FALLBACK GUARANTEE:
        - If the directory does not exist, raises DatasetNotFoundError.
        - If the discovered class labels do not match expected classes, raises DatasetIdentityError.

        Returns:
            sorted list of discovered class names if validation succeeds.
        """
        raw_path = target_raw_dir or self.raw_dir
        if not raw_path.exists():
            raise DatasetNotFoundError(
                f"Authoritative dataset directory not found at: {raw_path}. "
                "Silent fallback is disabled. Ensure the dataset is present."
            )

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        discovered_classes_set = set()
        image_count = 0

        for f in raw_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in image_extensions:
                image_count += 1
                discovered_classes_set.add(f.parent.name)

        discovered_classes = sorted(discovered_classes_set)

        if len(discovered_classes) != self.num_classes:
            raise DatasetIdentityError(
                f"Dataset identity mismatch in {raw_path}: expected {self.num_classes} classes, "
                f"found {len(discovered_classes)}: {discovered_classes}"
            )

        expected_set = set(self.expected_classes)
        if set(discovered_classes) != expected_set:
            missing = expected_set - set(discovered_classes)
            unexpected = set(discovered_classes) - expected_set
            raise DatasetIdentityError(
                f"Dataset label mismatch in {raw_path}: missing classes={sorted(missing)}, "
                f"unexpected classes={sorted(unexpected)}"
            )

        return discovered_classes


# Default global instance
default_dataset_config = DatasetConfig()
