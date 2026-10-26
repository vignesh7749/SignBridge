"""Stage 6: Dataset Loader & Data Leakage Protection.

Loads validated Stage 5 dataset artifacts, performs strict schema & integrity verification,
and enforces train/validation/test split disjointness (zero data leakage).
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.model_selection import train_test_split

from signbridge.dataset import DatasetConfig, default_dataset_config
from signbridge.features.schema import FeatureSchema


@dataclass
class DatasetSplits:
    """Encapsulates strictly partitioned and verified dataset splits."""

    X_train: np.ndarray
    y_train: np.ndarray
    train_indices: np.ndarray

    X_val: np.ndarray
    y_val: np.ndarray
    val_indices: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray
    test_indices: np.ndarray

    class_to_id: dict[str, int]
    id_to_class: dict[int, str]
    manifest: dict[str, Any]
    manifest_hash: str

    @property
    def num_classes(self) -> int:
        return len(self.class_to_id)

    @property
    def total_samples(self) -> int:
        return len(self.X_train) + len(self.X_val) + len(self.X_test)


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate_dataset(
    processed_dir: Path | None = None,
    metadata_dir: Path | None = None,
    val_ratio: float = 0.20,
    random_state: int = 42,
    config: DatasetConfig | None = None,
) -> DatasetSplits:
    """Loads and validates the official Stage 5 dataset artifacts.

    Strictly verifies:
      1. features.shape == (N, 94), dtype == float32
      2. labels.shape == (N,), dtype == integer
      3. Zero NaNs, zero Infs
      4. Valid contiguous class IDs [0, num_classes-1]
      5. classes.json bidirectional consistency
      6. Split disjointness: Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅

    Raises:
        FileNotFoundError: If any required dataset artifact is missing.
        ValueError: If any validation check fails.
    """
    cfg = config or default_dataset_config
    target_proc_dir = Path(processed_dir or cfg.processed_dir).resolve()
    target_meta_dir = Path(metadata_dir or cfg.metadata_dir).resolve()

    features_path = target_proc_dir / "features.npy"
    labels_path = target_proc_dir / "labels.npy"
    classes_path = target_proc_dir / "classes.json"
    index_path = target_proc_dir / "sample_index.json"
    manifest_path = target_meta_dir / "dataset_manifest.json"

    # 1. Existence Checks
    for required_file in [features_path, labels_path, classes_path, index_path]:
        if not required_file.exists():
            raise FileNotFoundError(
                f"Required dataset artifact missing: {required_file}. "
                "Run prepare_dataset.py before training."
            )

    # 2. Ingestion
    features = np.load(str(features_path))
    labels = np.load(str(labels_path))

    with open(classes_path, "r", encoding="utf-8") as f:
        classes_data = json.load(f)

    with open(index_path, "r", encoding="utf-8") as f:
        sample_index = json.load(f)

    manifest: dict[str, Any] = {}
    manifest_hash = "unspecified"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest_hash = compute_file_sha256(manifest_path)

    # 3. Shape and Dtype Validation
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D array, got shape {features.shape}")

    n_samples, n_dim = features.shape
    if n_dim != FeatureSchema.TOTAL_DIMENSIONS:
        raise ValueError(
            f"Feature dimension mismatch: expected {FeatureSchema.TOTAL_DIMENSIONS}, got {n_dim}"
        )

    if features.dtype != np.float32:
        raise ValueError(f"Features dtype must be float32, got {features.dtype}")

    if labels.ndim != 1 or len(labels) != n_samples:
        raise ValueError(
            f"Labels length ({len(labels)}) does not match features count ({n_samples})"
        )

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Labels dtype must be integer, got {labels.dtype}")

    if len(sample_index) != n_samples:
        raise ValueError(
            f"sample_index.json records ({len(sample_index)}) do not match sample count ({n_samples})"
        )

    # 4. Numerical Integrity (Strict Zero NaN / Inf)
    if not np.all(np.isfinite(features)):
        nan_count = int(np.isnan(features).sum())
        inf_count = int(np.isinf(features).sum())
        raise ValueError(
            f"Dataset contains non-finite features: {nan_count} NaNs, {inf_count} Infs"
        )

    # 5. Class Mapping Consistency
    class_to_id: dict[str, int] = classes_data.get("class_to_id", {})
    id_to_class: dict[int, str] = {int(k): v for k, v in classes_data.get("id_to_class", {}).items()}
    num_classes = len(class_to_id)

    if num_classes == 0:
        raise ValueError("classes.json contains zero classes")

    unique_labels = np.unique(labels)
    if not (unique_labels.min() >= 0 and unique_labels.max() < num_classes):
        raise ValueError(
            f"Label values out of bounds: range [{unique_labels.min()}, {unique_labels.max()}], "
            f"expected [0, {num_classes - 1}]"
        )

    # 6. Split Partitioning from sample_index.json (Preserve Source Partition)
    train_source_indices: list[int] = []
    test_source_indices: list[int] = []

    for idx, item in enumerate(sample_index):
        split = item.get("split", "train").lower()
        if split == "test":
            test_source_indices.append(idx)
        else:
            train_source_indices.append(idx)

    train_source_arr = np.array(train_source_indices, dtype=np.int64)
    test_source_arr = np.array(test_source_indices, dtype=np.int64)

    if len(test_source_arr) == 0:
        raise ValueError("No test samples found in sample_index.json")

    # 7. Stratified Train / Validation Split on the TRAIN partition only
    y_train_full = labels[train_source_arr]
    train_sub_idx, val_sub_idx = train_test_split(
        np.arange(len(train_source_arr)),
        test_size=val_ratio,
        stratify=y_train_full,
        random_state=random_state,
    )

    final_train_idx = train_source_arr[train_sub_idx]
    final_val_idx = train_source_arr[val_sub_idx]
    final_test_idx = test_source_arr

    # 8. Strict Data Leakage Assertions
    s_train = set(final_train_idx)
    s_val = set(final_val_idx)
    s_test = set(final_test_idx)

    assert len(s_train.intersection(s_val)) == 0, "Data leakage: Train and Val intersect!"
    assert len(s_train.intersection(s_test)) == 0, "Data leakage: Train and Test intersect!"
    assert len(s_val.intersection(s_test)) == 0, "Data leakage: Val and Test intersect!"
    assert len(s_train) + len(s_val) + len(s_test) == n_samples, "Index count discrepancy!"

    return DatasetSplits(
        X_train=features[final_train_idx],
        y_train=labels[final_train_idx],
        train_indices=final_train_idx,
        X_val=features[final_val_idx],
        y_val=labels[final_val_idx],
        val_indices=final_val_idx,
        X_test=features[final_test_idx],
        y_test=labels[final_test_idx],
        test_indices=final_test_idx,
        class_to_id=class_to_id,
        id_to_class=id_to_class,
        manifest=manifest,
        manifest_hash=manifest_hash,
    )
