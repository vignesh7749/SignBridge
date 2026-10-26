"""Stage 5: Preprocessing & Feature Engineering Pipeline.

Transforms raw dataset images into ML-ready 94D feature vectors:
  Image -> OpenCV -> MediaPipe Hand Detection -> DetectedHand ->
  Stage 4 HandFeatureExtractor -> FeatureSchema v1.0 (94D float32)

Features:
  - Supports `--subset N` to run a small verification batch first (Part 21).
  - Explicit quarantine logging for undetectable or corrupt samples.
  - Multi-hand arbitration: selects the highest confidence detection.
  - Preserves train/test partitions from raw directory structure to prevent data leakage.
  - Generates contiguous NumPy arrays (features.npy, labels.npy) and metadata.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
import cv2
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.core.config import VisionConfig
from signbridge.core.logger import get_logger, setup_logging
from signbridge.dataset import (
    DatasetConfig,
    DatasetIdentityError,
    DatasetNotFoundError,
    default_dataset_config,
)
from signbridge.features import FeatureSchema, HandFeatureExtractor
from signbridge.vision import MediaPipeHandDetector

setup_logging()
logger = get_logger("dataset.prepare")


def discover_dataset_files(raw_dir: Path) -> tuple[dict[str, list[tuple[Path, str]]], list[str]]:
    """Discovers images grouped by class and preserves train/test splits.

    Returns:
        tuple of (class_to_samples_dict, sorted_class_names)
        where each sample is (image_path, split_name).
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_extensions]

    class_samples: dict[str, list[tuple[Path, str]]] = {}

    for img_path in all_images:
        class_name = img_path.parent.name
        rel_parts = img_path.relative_to(raw_dir).parts

        split_name = "unspecified"
        for part in rel_parts[:-1]:
            if part.lower() in ("train", "training"):
                split_name = "train"
                break
            elif part.lower() in ("test", "testing"):
                split_name = "test"
                break
            elif part.lower() in ("val", "validation"):
                split_name = "val"
                break

        if class_name not in class_samples:
            class_samples[class_name] = []
        class_samples[class_name].append((img_path, split_name))

    sorted_classes = sorted(class_samples.keys())
    return class_samples, sorted_classes


def process_dataset(
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
    reports_dir: Path | None = None,
    subset_per_class: int | None = None,
    min_confidence: float = 0.4,
    config: DatasetConfig | None = None,
) -> dict[str, Any]:
    """Runs end-to-end vision and feature extraction across the dataset.

    Strictly validates dataset identity against authoritative config before processing.
    """
    cfg = config or default_dataset_config
    target_raw_dir = raw_dir or cfg.raw_dir
    target_out_dir = output_dir or cfg.processed_dir
    target_rep_dir = reports_dir or cfg.reports_dir

    target_out_dir.mkdir(parents=True, exist_ok=True)
    target_rep_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" SIGNBRIDGE — STAGE 5: DATASET PREPROCESSING & FEATURE EXTRACTION")
    print(f" Dataset Name:     {cfg.dataset_name}")
    print(f" Raw Directory:    {target_raw_dir}")
    print(f" Output Directory: {target_out_dir}")
    print(f" Single Source:    src/signbridge/dataset/config.py (Zero Silent Fallbacks)")
    print("=" * 80)

    # 1. Strict Identity Validation (Fails if directory missing or classes mismatch)
    discovered_classes = cfg.validate_identity(target_raw_dir)
    print(f"[IDENTITY OK] Verified {len(discovered_classes)} classes: {', '.join(discovered_classes)}")

    class_samples, sorted_classes = discover_dataset_files(target_raw_dir)
    total_raw_found = sum(len(samples) for samples in class_samples.values())

    print(f"Discovered {total_raw_found} raw images across {len(sorted_classes)} classes.")

    # Class mapping: stable integer index 0 .. C-1
    class_to_id = {cls_name: idx for idx, cls_name in enumerate(sorted_classes)}
    id_to_class = {idx: cls_name for idx, cls_name in enumerate(sorted_classes)}

    # Save stable class mapping
    classes_file = target_out_dir / "classes.json"
    with open(classes_file, "w", encoding="utf-8") as f:
        json.dump({"class_to_id": class_to_id, "id_to_class": id_to_class}, f, indent=2)

    # Initialize MediaPipe Detector and Stage 4 Feature Extractor
    vision_config = VisionConfig(
        max_num_hands=2,
        min_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
        static_image_mode=True,
    )
    detector = MediaPipeHandDetector(config=vision_config)
    extractor = HandFeatureExtractor(canonicalize_handedness=True)

    features_list: list[np.ndarray] = []
    labels_list: list[int] = []
    sample_index: list[dict[str, Any]] = []
    quarantine_list: list[dict[str, Any]] = []

    total_attempted = 0
    total_processed = 0
    total_rejected = 0

    detection_latencies: list[float] = []
    extraction_latencies: list[float] = []

    start_time = time.time()
    last_status_time = start_time

    try:
        for class_idx, class_name in enumerate(sorted_classes):
            samples = class_samples[class_name]
            if subset_per_class is not None:
                # Distribute subset across train and test splits proportionally
                train_s = [s for s in samples if s[1] == "train"]
                test_s = [s for s in samples if s[1] == "test"]
                other_s = [s for s in samples if s[1] not in ("train", "test")]

                if train_s and test_s:
                    n_test = max(1, round(subset_per_class * 0.2))
                    n_train = max(1, subset_per_class - n_test)
                    samples = train_s[:n_train] + test_s[:n_test]
                else:
                    samples = samples[:subset_per_class]

            for img_path, split_name in samples:
                total_attempted += 1

                # 1. Read image via OpenCV
                img = cv2.imread(str(img_path))
                if img is None or img.size == 0:
                    quarantine_list.append({
                        "sample_id": total_attempted,
                        "file_path": str(img_path),
                        "class_name": class_name,
                        "split": split_name,
                        "reason": "unreadable_image_or_empty",
                        "detection_status": "rejected",
                    })
                    total_rejected += 1
                    continue

                # 2. MediaPipe Hand Detection
                t_det0 = time.perf_counter()
                det_result = detector.detect(img)
                detection_latencies.append((time.perf_counter() - t_det0) * 1000.0)

                if not det_result.has_hands:
                    quarantine_list.append({
                        "sample_id": total_attempted,
                        "file_path": str(img_path),
                        "class_name": class_name,
                        "split": split_name,
                        "reason": "no_hand_detected",
                        "detection_status": "rejected",
                    })
                    total_rejected += 1
                    continue

                # 3. Multiple hand arbitration: select highest confidence hand
                best_hand = max(det_result.hands, key=lambda h: getattr(h, "score", 1.0))

                # 4. Stage 4 Feature Extraction
                try:
                    t_ext0 = time.perf_counter()
                    feat_vec = extractor.extract(best_hand)
                    extraction_latencies.append((time.perf_counter() - t_ext0) * 1000.0)
                except Exception as e:
                    quarantine_list.append({
                        "sample_id": total_attempted,
                        "file_path": str(img_path),
                        "class_name": class_name,
                        "split": split_name,
                        "reason": f"extraction_exception: {str(e)}",
                        "detection_status": "rejected",
                    })
                    total_rejected += 1
                    continue

                # Verify numerical sanity
                if not np.all(np.isfinite(feat_vec.features)):
                    quarantine_list.append({
                        "sample_id": total_attempted,
                        "file_path": str(img_path),
                        "class_name": class_name,
                        "split": split_name,
                        "reason": "non_finite_features",
                        "detection_status": "rejected",
                    })
                    total_rejected += 1
                    continue

                # 5. Append valid sample
                features_list.append(feat_vec.features)
                labels_list.append(class_idx)

                sample_index.append({
                    "sample_id": total_processed,
                    "class_name": class_name,
                    "class_id": class_idx,
                    "split": split_name,
                    "file_name": img_path.name,
                    "relative_path": str(img_path.relative_to(raw_dir)).replace("\\", "/"),
                    "handedness": feat_vec.handedness,
                    "is_canonicalized": feat_vec.is_canonicalized,
                    "scale_factor": float(feat_vec.scale_factor),
                    "detection_score": float(best_hand.score),
                })

                total_processed += 1

                # Periodic progress output
                now = time.time()
                if now - last_status_time >= 3.0:
                    elapsed = now - start_time
                    fps = total_attempted / elapsed
                    pct = (total_attempted / (len(sorted_classes) * (subset_per_class or len(samples)))) * 100.0 if subset_per_class else (total_attempted / total_raw_found) * 100.0
                    print(f"  [PROGRESS] {total_processed:,} processed, {total_rejected} quarantined ({total_attempted:,} attempted, {fps:.1f} imgs/sec, {pct:.1f}%)", end="\r")
                    last_status_time = now

    finally:
        detector.close()

    total_elapsed = time.time() - start_time
    print(f"\n[DONE] Processing complete in {total_elapsed:.1f} seconds ({total_processed / max(0.001, total_elapsed):.1f} samples/sec).")

    # Convert to contiguous NumPy arrays
    features_array = np.array(features_list, dtype=np.float32)
    labels_array = np.array(labels_list, dtype=np.int64)

    # Save binary feature arrays
    features_file = output_dir / "features.npy"
    labels_file = output_dir / "labels.npy"
    sample_index_file = output_dir / "sample_index.json"
    quarantine_file = reports_dir / "quarantine_log.json"

    np.save(str(features_file), features_array)
    np.save(str(labels_file), labels_array)

    with open(sample_index_file, "w", encoding="utf-8") as f:
        json.dump(sample_index, f, indent=2)

    with open(quarantine_file, "w", encoding="utf-8") as f:
        json.dump(quarantine_list, f, indent=2)

    success_rate = (total_processed / total_attempted * 100.0) if total_attempted else 0.0

    summary = {
        "raw_directory": str(raw_dir),
        "processed_directory": str(output_dir),
        "total_raw_found": total_raw_found,
        "total_attempted": total_attempted,
        "total_processed": total_processed,
        "total_rejected": total_rejected,
        "success_rate_pct": success_rate,
        "classes_count": len(sorted_classes),
        "feature_dimension": FeatureSchema.TOTAL_DIMENSIONS,
        "feature_dtype": str(features_array.dtype),
        "features_shape": list(features_array.shape),
        "labels_shape": list(labels_array.shape),
        "mean_detection_latency_ms": float(np.mean(detection_latencies)) if detection_latencies else 0.0,
        "mean_extraction_latency_ms": float(np.mean(extraction_latencies)) if extraction_latencies else 0.0,
        "total_processing_time_sec": total_elapsed,
        "throughput_samples_per_sec": float(total_processed / max(0.001, total_elapsed)),
    }

    print("-" * 80)
    print(f"PREPROCESSING SUMMARY:")
    print(f"  Attempted:        {total_attempted:,}")
    print(f"  Valid Processed:  {total_processed:,} ({success_rate:.2f}%)")
    print(f"  Quarantined:      {total_rejected} ({100.0 - success_rate:.2f}%)")
    print(f"  Feature Matrix:   {features_array.shape} (dtype: {features_array.dtype})")
    print(f"  Labels Vector:    {labels_array.shape} (dtype: {labels_array.dtype})")
    print(f"  Mean Det. Latency: {summary['mean_detection_latency_ms']:.2f} ms")
    print(f"  Mean Ext. Latency: {summary['mean_extraction_latency_ms']:.3f} ms")
    print(f"  Throughput:       {summary['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"  Quarantine Log:   {quarantine_file}")
    print("=" * 80)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="SignBridge Stage 5: Dataset Preprocessing & Feature Extraction")
    parser.add_argument(
        "--subset-per-class",
        type=int,
        default=None,
        help="Number of samples to process from EACH class (deterministic testing)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Alias for --subset-per-class (processes N samples from EACH class)",
    )
    parser.add_argument("--confidence", type=float, default=0.4, help="MediaPipe detection confidence threshold")
    parser.add_argument("--raw-dir", type=str, default=None, help="Custom raw dataset directory path (defaults to DatasetConfig)")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory for processed features")
    args = parser.parse_args()

    subset_count = args.subset_per_class if args.subset_per_class is not None else args.subset

    target_raw = Path(args.raw_dir) if args.raw_dir else default_dataset_config.raw_dir
    target_out = Path(args.output_dir) if args.output_dir else default_dataset_config.processed_dir
    target_rep = default_dataset_config.reports_dir

    process_dataset(
        raw_dir=target_raw,
        output_dir=target_out,
        reports_dir=target_rep,
        subset_per_class=subset_count,
        min_confidence=args.confidence,
    )


if __name__ == "__main__":
    main()
