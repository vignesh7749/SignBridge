"""Stage 5: Dataset Validator & Manifest Generator.

Performs rigorous standalone validation of the processed SignBridge dataset:
  1. File existence and non-emptiness (features.npy, labels.npy, classes.json, sample_index.json)
  2. Array shapes: features is (N, 94), labels is (N,)
  3. Data types: features is float32, labels is int64
  4. Numerical integrity: 0 NaNs, 0 Infs, 0 -Infs
  5. Label sanity: all labels in [0, C-1], no missing classes
  6. Class mapping alignment: matches classes.json
  7. Predefined group / split preservation (train/test)
  8. SHA256 integrity checksums and manifest creation:
     datasets/metadata/dataset_manifest.json
"""

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.dataset import DatasetConfig, default_dataset_config
from signbridge.features.schema import FeatureSchema


def calculate_sha256(file_path: Path) -> str:
    """Calculates SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_processed_dataset(
    processed_dir: Path | None = None,
    metadata_dir: Path | None = None,
    expected_dim: int = FeatureSchema.TOTAL_DIMENSIONS,
    config: DatasetConfig | None = None,
) -> dict[str, Any]:
    """Validates processed dataset files and generates the official dataset manifest."""
    cfg = config or default_dataset_config
    target_proc_dir = processed_dir or cfg.processed_dir
    target_meta_dir = metadata_dir or cfg.metadata_dir

    print("=" * 80)
    print(" SIGNBRIDGE — STAGE 5: DATASET VALIDATOR & MANIFEST GENERATOR")
    print(f" Dataset Name:  {cfg.dataset_name}")
    print(f" Processed Dir: {target_proc_dir}")
    print(f" Single Source: src/signbridge/dataset/config.py")
    print("=" * 80)

    features_path = target_proc_dir / "features.npy"
    labels_path = target_proc_dir / "labels.npy"
    classes_path = target_proc_dir / "classes.json"
    index_path = target_proc_dir / "sample_index.json"

    validation_checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, message: str) -> None:
        validation_checks.append({"check": name, "passed": passed, "message": message})
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {message}")
        if not passed:
            raise ValueError(f"Validation failure in check '{name}': {message}")

    # 1. Existence Check
    check("features_exists", features_path.exists(), f"features.npy located at {features_path}")
    check("labels_exists", labels_path.exists(), f"labels.npy located at {labels_path}")
    check("classes_exists", classes_path.exists(), f"classes.json located at {classes_path}")
    check("sample_index_exists", index_path.exists(), f"sample_index.json located at {index_path}")

    # 2. Load arrays
    features = np.load(str(features_path))
    labels = np.load(str(labels_path))

    with open(classes_path, "r", encoding="utf-8") as f:
        class_meta = json.load(f)

    with open(index_path, "r", encoding="utf-8") as f:
        sample_index = json.load(f)

    class_to_id = class_meta.get("class_to_id", {})
    id_to_class = {int(k): v for k, v in class_meta.get("id_to_class", {}).items()}
    num_classes = len(class_to_id)
    n_samples = len(features)

    # 3. Shape & Dimension Checks
    check(
        "features_ndim",
        features.ndim == 2,
        f"Feature array is 2D with shape {features.shape}",
    )
    check(
        "features_dimension",
        features.shape[1] == expected_dim,
        f"Feature dimension is {features.shape[1]} (matches FeatureSchema v1.0 target: {expected_dim})",
    )
    check(
        "labels_alignment",
        len(labels) == n_samples,
        f"Labels count ({len(labels)}) matches features count ({n_samples})",
    )
    check(
        "sample_index_alignment",
        len(sample_index) == n_samples,
        f"Sample index records ({len(sample_index)}) match sample count ({n_samples})",
    )

    # 4. Data Type Checks
    check(
        "features_dtype",
        features.dtype == np.float32,
        f"Feature array dtype is float32 (actual: {features.dtype})",
    )
    check(
        "labels_dtype",
        np.issubdtype(labels.dtype, np.integer),
        f"Labels array dtype is integer (actual: {labels.dtype})",
    )

    # 5. Numerical Integrity Checks
    nan_count = int(np.isnan(features).sum())
    inf_count = int(np.isinf(features).sum())
    check(
        "numerical_finite",
        nan_count == 0 and inf_count == 0,
        f"Zero non-finite values detected (NaN count: {nan_count}, Inf count: {inf_count})",
    )

    # 6. Label & Class Sanity
    min_label = int(np.min(labels)) if n_samples > 0 else 0
    max_label = int(np.max(labels)) if n_samples > 0 else 0
    check(
        "labels_range",
        min_label >= 0 and max_label < num_classes,
        f"Labels span range [{min_label}, {max_label}] within [0, {num_classes - 1}]",
    )

    # 7. Class Distribution & Balance
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_distribution = {id_to_class[int(l)]: int(c) for l, c in zip(unique_labels, counts)}
    check(
        "classes_represented",
        len(unique_labels) == num_classes,
        f"All {num_classes} classes have valid extracted samples (represented: {len(unique_labels)})",
    )

    min_class_samples = int(np.min(counts)) if len(counts) > 0 else 0
    max_class_samples = int(np.max(counts)) if len(counts) > 0 else 0
    imbalance_ratio = max_class_samples / max(1, min_class_samples)
    check(
        "class_balance",
        imbalance_ratio < 4.0,
        f"Imbalance ratio is {imbalance_ratio:.2f} (min: {min_class_samples}, max: {max_class_samples})",
    )

    # 8. Train/Test Split Consistency & Leakage Protection
    train_paths = {entry.get("relative_path") for entry in sample_index if entry.get("split") == "train"}
    test_paths = {entry.get("relative_path") for entry in sample_index if entry.get("split") == "test"}
    intersection = train_paths.intersection(test_paths)
    check(
        "split_disjointness",
        len(intersection) == 0,
        f"Train and test sets strictly disjoint (intersection count: {len(intersection)})",
    )
    train_count = sum(1 for entry in sample_index if entry.get("split") == "train")
    test_count = sum(1 for entry in sample_index if entry.get("split") == "test")
    print(f"  [SPLIT STATUS] Train samples: {train_count}, Test samples: {test_count}")

    # 9. Compute Checksums
    print("\n[CHECKSUMS] Computing SHA256 hashes of processed dataset artifacts...")
    checksums = {
        "features.npy": calculate_sha256(features_path),
        "labels.npy": calculate_sha256(labels_path),
        "classes.json": calculate_sha256(classes_path),
        "sample_index.json": calculate_sha256(index_path),
    }
    for fname, sha in checksums.items():
        print(f"  {fname}: {sha}")

    # 10. Generate Official Manifest
    target_meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_meta_dir / "dataset_manifest.json"

    manifest: dict[str, Any] = {
        "dataset_name": cfg.dataset_name,
        "dataset_version": cfg.dataset_version,
        "provenance": {
            "source_repo": cfg.source_repo,
            "source_url": cfg.source_url,
            "license": cfg.license,
            "modality": cfg.modality,
        },
        "feature_spec": {
            "schema_version": FeatureSchema.VERSION,
            "dimension": expected_dim,
            "dtype": "float32",
            "slices": {name: [sl.start, sl.end] for name, sl in FeatureSchema.SLICES.items()},
        },
        "dataset_metrics": {
            "total_processed_samples": n_samples,
            "train_samples": train_count,
            "test_samples": test_count,
            "class_count": num_classes,
            "min_class_samples": min_class_samples,
            "max_class_samples": max_class_samples,
            "imbalance_ratio": round(imbalance_ratio, 3),
            "class_distribution": class_distribution,
        },
        "checksums": checksums,
        "validation": {
            "status": "PASSED",
            "checks_passed": len(validation_checks),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[SUCCESS] All {len(validation_checks)} validation checks passed!")
    print(f"Manifest written to: {manifest_path}")
    print("=" * 80)

    return manifest


def main() -> None:
    validate_processed_dataset()


if __name__ == "__main__":
    main()
