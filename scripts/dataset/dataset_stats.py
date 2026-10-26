"""Stage 5: Dataset Statistical Analysis & Profiling.

Computes comprehensive statistical distributions for the processed SignBridge dataset:
  - Class distribution (frequencies, percentages, min, max, imbalance ratio)
  - Split distribution (train vs test counts, group independence)
  - Dimensional statistical summary (min, max, mean, std, L2 norms)
  - Writes datasets/reports/dataset_stats.json
"""

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
from signbridge.features.schema import FEATURE_NAMES, FeatureSchema


def compute_dataset_statistics(
    processed_dir: Path | None = None,
    reports_dir: Path | None = None,
    config: DatasetConfig | None = None,
) -> dict[str, Any]:
    cfg = config or default_dataset_config
    target_processed_dir = processed_dir or cfg.processed_dir
    target_reports_dir = reports_dir or cfg.reports_dir

    print("=" * 80)
    print(f" SIGNBRIDGE — STAGE 5: DATASET STATISTICAL ANALYSIS ({cfg.dataset_name})")
    print("=" * 80)

    features_path = target_processed_dir / "features.npy"
    labels_path = target_processed_dir / "labels.npy"
    classes_path = target_processed_dir / "classes.json"
    index_path = target_processed_dir / "sample_index.json"

    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Processed dataset not found in {target_processed_dir}. Run prepare_dataset.py first.")

    features = np.load(str(features_path))
    labels = np.load(str(labels_path))

    with open(classes_path, "r", encoding="utf-8") as f:
        class_meta = json.load(f)

    with open(index_path, "r", encoding="utf-8") as f:
        sample_index = json.load(f)

    id_to_class = {int(k): v for k, v in class_meta.get("id_to_class", {}).items()}
    num_samples = len(features)
    num_classes = len(id_to_class)

    # 1. Class Distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_stats: dict[str, dict[str, Any]] = {}
    for lbl, count in zip(unique_labels, counts):
        name = id_to_class[int(lbl)]
        pct = (count / num_samples) * 100.0
        class_stats[name] = {"count": int(count), "percentage": round(pct, 2)}

    min_samples = int(np.min(counts))
    max_samples = int(np.max(counts))
    mean_samples = float(np.mean(counts))
    std_samples = float(np.std(counts))
    imbalance_ratio = max_samples / max(1, min_samples)

    # 2. Split Distribution
    split_counts: dict[str, int] = {}
    class_split_matrix: dict[str, dict[str, int]] = {}
    for item in sample_index:
        s = item.get("split", "unspecified")
        c = item.get("class_name", "unknown")
        split_counts[s] = split_counts.get(s, 0) + 1

        if c not in class_split_matrix:
            class_split_matrix[c] = {}
        class_split_matrix[c][s] = class_split_matrix[c].get(s, 0) + 1

    # 3. Dimensional Statistics (across 94 features)
    dim_means = np.mean(features, axis=0)
    dim_stds = np.std(features, axis=0)
    dim_mins = np.min(features, axis=0)
    dim_maxs = np.max(features, axis=0)

    # L2 Norms across samples
    sample_norms = np.linalg.norm(features, axis=1)
    mean_norm = float(np.mean(sample_norms))
    std_norm = float(np.std(sample_norms))

    # Slices summary
    slice_stats: dict[str, dict[str, float]] = {}
    for slice_name, f_slice in FeatureSchema.SLICES.items():
        slice_data = features[:, f_slice.slice]
        slice_stats[slice_name] = {
            "mean": float(np.mean(slice_data)),
            "std": float(np.std(slice_data)),
            "min": float(np.min(slice_data)),
            "max": float(np.max(slice_data)),
        }

    # Print Report
    print(f"\n[SUMMARY] Total Valid Samples: {num_samples:,} across {num_classes} classes")
    print(f"  Class sizes: min={min_samples}, max={max_samples}, mean={mean_samples:.1f} (std={std_samples:.1f})")
    print(f"  Imbalance Ratio: {imbalance_ratio:.2f}")
    print(f"\n[SPLIT BREAKDOWN]")
    for split_name, s_count in split_counts.items():
        print(f"  {split_name:12s}: {s_count:,} ({s_count / num_samples * 100.0:.1f}%)")

    print(f"\n[CLASS DISTRIBUTION (Sample Table)]")
    for cls_name, c_info in sorted(class_stats.items())[:12]:
        print(f"  Class '{cls_name}': {c_info['count']} samples ({c_info['percentage']}%)")
    if len(class_stats) > 12:
        print(f"  ... ({len(class_stats) - 12} more classes)")

    print(f"\n[FEATURE SLICE METRICS (FeatureSchema v1.0)]")
    for s_name, s_met in slice_stats.items():
        print(f"  {s_name:22s} -> mean: {s_met['mean']:6.3f} | std: {s_met['std']:6.3f} | min: {s_met['min']:6.3f} | max: {s_met['max']:6.3f}")

    print(f"\n[GLOBAL FEATURE VECTOR L2 NORM]: {mean_norm:.3f} +/- {std_norm:.3f}")

    # Compile report dictionary
    report: dict[str, Any] = {
        "dataset_name": cfg.dataset_name,
        "sample_count": num_samples,
        "class_count": num_classes,
        "class_balance": {
            "min_class_samples": min_samples,
            "max_class_samples": max_samples,
            "mean_class_samples": round(mean_samples, 2),
            "std_class_samples": round(std_samples, 2),
            "imbalance_ratio": round(imbalance_ratio, 3),
        },
        "splits": split_counts,
        "class_distribution": class_stats,
        "class_split_matrix": class_split_matrix,
        "feature_slices": slice_stats,
        "global_l2_norm": {"mean": round(mean_norm, 4), "std": round(std_norm, 4)},
        "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    target_reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = target_reports_dir / "dataset_stats.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to: {report_file}")
    print("=" * 80)

    return report


def main() -> None:
    compute_dataset_statistics()


if __name__ == "__main__":
    main()
