"""Unit and integration tests for Stage 5: Dataset Pipeline.

Tests:
  1. Class mapping consistency and bidirectional lookup.
  2. Raw image discovery and train/test partition tracking.
  3. Preprocessing pipeline on mock dataset (MediaPipe + Stage 4 -> 94D).
  4. Non-finite feature detection and quarantine logging.
  5. Corrupt file and unreadable image handling.
  6. Standalone validator assertions (shape, dtype, classes, NaN/Inf).
  7. Dataset manifest generation and SHA256 integrity.
  8. Dataset statistical analysis computation.
  9. Data leakage protection: train and test splits strictly disjoint.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from scripts.dataset.dataset_stats import compute_dataset_statistics
from scripts.dataset.prepare_dataset import discover_dataset_files, process_dataset
from scripts.dataset.validate_dataset import calculate_sha256, validate_processed_dataset
from signbridge.dataset import (
    DatasetConfig,
    DatasetIdentityError,
    DatasetNotFoundError,
    default_dataset_config,
)
from signbridge.features.schema import FeatureSchema


def test_dataset_config_defaults() -> None:
    """Verifies that default_dataset_config points to authoritative natural RGB dataset."""
    cfg = default_dataset_config
    assert cfg.dataset_name == "ASL-Alphabet-Natural-RGB"
    assert cfg.num_classes == 26
    assert len(cfg.expected_classes) == 26
    assert cfg.expected_classes[0] == "A"
    assert cfg.expected_classes[-1] == "Z"
    assert "J" in cfg.expected_classes
    assert "Z" in cfg.expected_classes
    assert cfg.raw_dir.name == "raw"
    assert "asl_alphabet" in str(cfg.raw_dir)


def test_validate_identity_on_real_dataset() -> None:
    """Verifies that default_dataset_config succeeds when validating authoritative raw_dir."""
    cfg = default_dataset_config
    if cfg.raw_dir.exists():
        classes = cfg.validate_identity()
        assert len(classes) == 26
        assert classes == list(cfg.expected_classes)


def test_validate_identity_not_found(tmp_path: Path) -> None:
    """Verifies that validate_identity raises DatasetNotFoundError on missing directory."""
    cfg = default_dataset_config
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(DatasetNotFoundError, match="Authoritative dataset directory not found"):
        cfg.validate_identity(non_existent)


def test_validate_identity_mismatch(mock_raw_dataset: Path) -> None:
    """Verifies that validate_identity raises DatasetIdentityError on class mismatch."""
    cfg = default_dataset_config
    with pytest.raises(DatasetIdentityError, match="expected 26 classes, found 3"):
        cfg.validate_identity(mock_raw_dataset)


def test_validate_identity_rejects_asl_semcom() -> None:
    """Verifies that authoritative config rejects ASL-SemCom (24 classes A-Y excluding J, Z)."""
    cfg = default_dataset_config
    semcom_raw = cfg.project_root / "datasets" / "external" / "asl_semcom" / "raw"
    if semcom_raw.exists():
        with pytest.raises(DatasetIdentityError, match="expected 26 classes"):
            cfg.validate_identity(semcom_raw)


@pytest.fixture
def mock_raw_dataset(tmp_path: Path) -> Path:
    """Creates a synthetic miniature raw dataset with train/test splits for testing."""
    raw_dir = tmp_path / "raw"
    classes = ["A", "B", "C"]

    for split in ["train", "test"]:
        for cls in classes:
            folder = raw_dir / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            # Create 3 small synthetic RGB images per class
            for i in range(3):
                # Create a simple 200x200 image with a white hand-like rectangle
                img = np.zeros((200, 200, 3), dtype=np.uint8)
                cv2.rectangle(img, (50, 50), (150, 150), (180, 150, 120), -1)
                cv2.imwrite(str(folder / f"sample_{i}.jpg"), img)

    # Add one corrupt file to test quarantine
    corrupt_folder = raw_dir / "train" / "A"
    with open(corrupt_folder / "corrupt.jpg", "w") as f:
        f.write("corrupted image content not a real jpg")

    return raw_dir


def test_discover_dataset_files(mock_raw_dataset: Path) -> None:
    """Verifies that discover_dataset_files correctly groups by class and preserves splits."""
    class_samples, sorted_classes = discover_dataset_files(mock_raw_dataset)

    assert sorted_classes == ["A", "B", "C"]
    assert len(class_samples) == 3

    for cls in sorted_classes:
        samples = class_samples[cls]
        splits = [s for _, s in samples]
        assert "train" in splits
        assert "test" in splits


def test_data_leakage_protection_splits(mock_raw_dataset: Path) -> None:
    """Verifies that train and test sample paths are mutually exclusive."""
    class_samples, _ = discover_dataset_files(mock_raw_dataset)

    for cls, samples in class_samples.items():
        train_paths = {p for p, split in samples if split == "train"}
        test_paths = {p for p, split in samples if split == "test"}
        # Intersection must be completely empty (zero data leakage)
        assert len(train_paths.intersection(test_paths)) == 0


def test_prepare_dataset_synthetic_pipeline(mock_raw_dataset: Path, tmp_path: Path) -> None:
    """Tests the prepare_dataset script end-to-end on synthetic data with custom config."""
    output_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"

    mock_cfg = DatasetConfig(
        dataset_name="Mock-Synthetic-3Class",
        num_classes=3,
        expected_classes=("A", "B", "C"),
        expected_min_samples=1,
        expected_max_samples=100,
        raw_dir_override=mock_raw_dataset,
        processed_dir_override=output_dir,
        reports_dir_override=reports_dir,
    )

    summary = process_dataset(
        raw_dir=mock_raw_dataset,
        output_dir=output_dir,
        reports_dir=reports_dir,
        subset_per_class=None,
        config=mock_cfg,
    )

    # Verify classes.json was created
    classes_file = output_dir / "classes.json"
    assert classes_file.exists()
    with open(classes_file) as f:
        meta = json.load(f)
    assert meta["class_to_id"] == {"A": 0, "B": 1, "C": 2}

    # Verify quarantine log was created and captured the corrupt file
    quarantine_file = reports_dir / "quarantine_log.json"
    assert quarantine_file.exists()
    with open(quarantine_file) as f:
        quarantine = json.load(f)

    # Corrupt file must be recorded in quarantine
    corrupt_entries = [q for q in quarantine if "corrupt.jpg" in q["file_path"]]
    assert len(corrupt_entries) > 0
    assert corrupt_entries[0]["detection_status"] == "rejected"


def test_prepare_dataset_fails_on_identity_mismatch(mock_raw_dataset: Path, tmp_path: Path) -> None:
    """Verifies that prepare_dataset fails immediately when run with default 26-class config on invalid dataset."""
    output_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"

    # Default config expects 26 classes, mock has 3
    with pytest.raises(DatasetIdentityError, match="expected 26 classes, found 3"):
        process_dataset(
            raw_dir=mock_raw_dataset,
            output_dir=output_dir,
            reports_dir=reports_dir,
        )


def test_validator_and_manifest(tmp_path: Path) -> None:
    """Verifies that validate_dataset passes for valid mock processed data."""
    processed_dir = tmp_path / "processed"
    metadata_dir = tmp_path / "metadata"
    processed_dir.mkdir(parents=True, exist_ok=True)

    n_samples = 30
    dim = 94
    features = np.random.randn(n_samples, dim).astype(np.float32)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int64)

    np.save(str(processed_dir / "features.npy"), features)
    np.save(str(processed_dir / "labels.npy"), labels)

    classes_data = {
        "class_to_id": {"A": 0, "B": 1, "C": 2},
        "id_to_class": {"0": "A", "1": "B", "2": "C"},
    }
    with open(processed_dir / "classes.json", "w") as f:
        json.dump(classes_data, f)

    sample_index = [
        {"sample_id": i, "class_name": "A" if l == 0 else ("B" if l == 1 else "C"), "class_id": int(l), "split": "train"}
        for i, l in enumerate(labels)
    ]
    with open(processed_dir / "sample_index.json", "w") as f:
        json.dump(sample_index, f)

    mock_cfg = DatasetConfig(
        dataset_name="Mock-3Class",
        num_classes=3,
        expected_classes=("A", "B", "C"),
        processed_dir_override=processed_dir,
        metadata_dir_override=metadata_dir,
    )

    manifest = validate_processed_dataset(processed_dir, metadata_dir, expected_dim=dim, config=mock_cfg)

    assert manifest["validation"]["status"] == "PASSED"
    assert manifest["feature_spec"]["dimension"] == 94
    assert manifest["dataset_metrics"]["total_processed_samples"] == n_samples
    assert "features.npy" in manifest["checksums"]


def test_validator_fails_on_nan(tmp_path: Path) -> None:
    """Verifies that validate_dataset rejects features with NaN."""
    processed_dir = tmp_path / "processed"
    metadata_dir = tmp_path / "metadata"
    processed_dir.mkdir(parents=True, exist_ok=True)

    features = np.ones((10, 94), dtype=np.float32)
    features[2, 5] = np.nan  # Inject NaN
    labels = np.zeros((10,), dtype=np.int64)

    np.save(str(processed_dir / "features.npy"), features)
    np.save(str(processed_dir / "labels.npy"), labels)

    with open(processed_dir / "classes.json", "w") as f:
        json.dump({"class_to_id": {"A": 0}, "id_to_class": {"0": "A"}}, f)

    with open(processed_dir / "sample_index.json", "w") as f:
        json.dump([{"sample_id": i, "class_name": "A", "class_id": 0, "split": "train"} for i in range(10)], f)

    with pytest.raises(ValueError, match="numerical_finite"):
        validate_processed_dataset(processed_dir, metadata_dir)


def test_validator_fails_on_dimension_mismatch(tmp_path: Path) -> None:
    """Verifies that validate_dataset rejects features with unexpected dimension."""
    processed_dir = tmp_path / "processed"
    metadata_dir = tmp_path / "metadata"
    processed_dir.mkdir(parents=True, exist_ok=True)

    features = np.ones((10, 63), dtype=np.float32)  # 63 instead of 94
    labels = np.zeros((10,), dtype=np.int64)

    np.save(str(processed_dir / "features.npy"), features)
    np.save(str(processed_dir / "labels.npy"), labels)

    with open(processed_dir / "classes.json", "w") as f:
        json.dump({"class_to_id": {"A": 0}, "id_to_class": {"0": "A"}}, f)

    with open(processed_dir / "sample_index.json", "w") as f:
        json.dump([{"sample_id": i, "class_name": "A", "class_id": 0, "split": "train"} for i in range(10)], f)

    with pytest.raises(ValueError, match="features_dimension"):
        validate_processed_dataset(processed_dir, metadata_dir, expected_dim=94)


def test_dataset_statistics_computation(tmp_path: Path) -> None:
    """Verifies that dataset_stats computes valid metrics."""
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)

    features = np.ones((15, 94), dtype=np.float32)
    labels = np.array([0] * 5 + [1] * 5 + [2] * 5, dtype=np.int64)

    np.save(str(processed_dir / "features.npy"), features)
    np.save(str(processed_dir / "labels.npy"), labels)

    with open(processed_dir / "classes.json", "w") as f:
        json.dump({
            "class_to_id": {"A": 0, "B": 1, "C": 2},
            "id_to_class": {"0": "A", "1": "B", "2": "C"},
        }, f)

    with open(processed_dir / "sample_index.json", "w") as f:
        json.dump([{"sample_id": i, "class_name": "A", "class_id": 0, "split": "train"} for i in range(15)], f)

    mock_cfg = DatasetConfig(
        dataset_name="Mock-Stats",
        num_classes=3,
        expected_classes=("A", "B", "C"),
        processed_dir_override=processed_dir,
        reports_dir_override=reports_dir,
    )

    report = compute_dataset_statistics(processed_dir, reports_dir, config=mock_cfg)

    assert report["sample_count"] == 15
    assert report["class_count"] == 3
    assert report["class_balance"]["imbalance_ratio"] == 1.0
    assert (reports_dir / "dataset_stats.json").exists()
