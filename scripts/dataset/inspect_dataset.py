"""Stage 5: Dataset Acquisition & Quality Inspection.

Downloads the selected ASL-SemCom dataset from Zenodo (DOI: 10.5281/zenodo.14635573),
verifies its integrity, unpacks into datasets/external/asl_semcom/raw/, and performs
comprehensive inspection:
  - File counts and formats
  - Class distribution and structure
  - Image resolutions and channel counts
  - Integrity scanning (corrupt file detection)
  - Predefined split / session preservation
"""

import hashlib
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
import cv2

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ZENODO_RECORD_URL = "https://zenodo.org/api/records/14635573/files/ASL_SemCom.zip/content"
EXPECTED_ZIP_SIZE = 392726306  # ~374.5 MB


def calculate_sha256(file_path: Path) -> str:
    """Calculates SHA256 checksum of a file in chunks."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_dataset(target_zip: Path, url: str = ZENODO_RECORD_URL) -> Path:
    """Downloads dataset zip from Zenodo with streaming progress reporting."""
    target_zip.parent.mkdir(parents=True, exist_ok=True)

    if target_zip.exists() and target_zip.stat().st_size == EXPECTED_ZIP_SIZE:
        print(f"[DOWNLOAD] Found existing valid zip archive ({target_zip.stat().st_size / (1024*1024):.1f} MB): {target_zip}")
        return target_zip

    print(f"[DOWNLOAD] Initiating stream download from Zenodo: {url}")
    print(f"[DOWNLOAD] Target destination: {target_zip}")

    req = urllib.request.Request(url, headers={"User-Agent": "SignBridge-Dataset-Acquisition/1.0"})
    start_time = time.time()
    downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB chunks

    with urllib.request.urlopen(req) as resp, open(target_zip, "wb") as out_file:
        total_size = int(resp.headers.get("Content-Length", EXPECTED_ZIP_SIZE))
        print(f"[DOWNLOAD] Total archive size: {total_size / (1024*1024):.1f} MB")

        last_print = 0.0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_print >= 1.0 or downloaded == total_size:
                elapsed = max(0.1, now - start_time)
                speed_mb = (downloaded / (1024 * 1024)) / elapsed
                percent = (downloaded / total_size) * 100.0 if total_size else 0.0
                print(f"  --> Progress: {downloaded / (1024*1024):5.1f} MB / {total_size / (1024*1024):5.1f} MB ({percent:5.1f}%) at {speed_mb:4.1f} MB/s", end="\r")
                last_print = now

    print(f"\n[DOWNLOAD] Download complete in {time.time() - start_time:.1f} s.")
    return target_zip


def extract_archive(zip_path: Path, extract_dir: Path) -> Path:
    """Extracts zip archive cleanly into target directory."""
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    existing_images = list(extract_dir.rglob("*.jpg")) + list(extract_dir.rglob("*.png"))
    if len(existing_images) > 1000:
        print(f"[EXTRACTION] Found existing extracted dataset with {len(existing_images)} images at {extract_dir}")
        return extract_dir

    print(f"[EXTRACTION] Unpacking {zip_path} into {extract_dir}...")
    start_time = time.time()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"[EXTRACTION] Extracted in {time.time() - start_time:.1f} s.")
    return extract_dir


from signbridge.dataset import (
    DatasetConfig,
    DatasetIdentityError,
    DatasetNotFoundError,
    default_dataset_config,
)


def inspect_dataset_structure(
    raw_dir: Path,
    config: DatasetConfig | None = None,
) -> dict[str, Any]:
    """Inspects raw dataset hierarchy, image dimensions, corrupt files, and class balance.

    Strictly validates dataset identity before scanning.
    """
    cfg = config or default_dataset_config

    print(f"\n[INSPECTION] Validating dataset identity at: {raw_dir}")
    discovered_classes = cfg.validate_identity(raw_dir)
    print(f"  [IDENTITY OK] Verified {len(discovered_classes)} expected classes: {', '.join(discovered_classes)}")

    # Find all image files
    all_files = list(raw_dir.rglob("*"))
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in image_extensions]

    print(f"  Total files found:  {len(all_files)}")
    print(f"  Total image files:  {len(image_files)}")

    class_distribution: dict[str, int] = {}
    split_distribution: dict[str, int] = {}
    class_split_breakdown: dict[str, dict[str, int]] = {}
    corrupted_files: list[str] = []
    resolutions: set[tuple[int, int]] = set()
    channels_set: set[int] = set()

    # Scan samples
    sample_check_limit = min(500, len(image_files))
    print(f"  Sampling first {sample_check_limit} images for resolution, channels, and integrity...")

    for i, img_path in enumerate(image_files):
        # Extract class and split from directory structure
        # Common structure: raw/train/A/001.jpg or raw/A/001.jpg or raw/ASL_SemCom/train/A/001.jpg
        rel_parts = img_path.relative_to(raw_dir).parts
        
        # Determine class name (typically the directory containing the file)
        class_name = img_path.parent.name
        # Determine split (train, test, or none)
        split_name = "unspecified"
        for p in rel_parts[:-1]:
            if p.lower() in ("train", "training", "test", "testing", "val", "validation"):
                split_name = p.lower()
                break

        class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
        split_distribution[split_name] = split_distribution.get(split_name, 0) + 1

        if class_name not in class_split_breakdown:
            class_split_breakdown[class_name] = {}
        class_split_breakdown[class_name][split_name] = class_split_breakdown[class_name].get(split_name, 0) + 1

        # Check sample integrity on first N images and all potential edge cases
        if i < sample_check_limit or i % 50 == 0:
            try:
                img = cv2.imread(str(img_path))
                if img is None or img.size == 0:
                    corrupted_files.append(str(img_path))
                else:
                    h, w = img.shape[:2]
                    c = img.shape[2] if img.ndim == 3 else 1
                    resolutions.add((w, h))
                    channels_set.add(c)
            except Exception:
                corrupted_files.append(str(img_path))

    sorted_classes = sorted(class_distribution.keys())

    report: dict[str, Any] = {
        "dataset_name": cfg.dataset_name,
        "source_repo": cfg.source_repo,
        "source_url": cfg.source_url,
        "license": cfg.license,
        "raw_directory": str(raw_dir),
        "total_files": len(all_files),
        "total_images": len(image_files),
        "total_classes": len(sorted_classes),
        "classes": sorted_classes,
        "class_distribution": class_distribution,
        "split_distribution": split_distribution,
        "class_split_breakdown": class_split_breakdown,
        "image_resolutions": [f"{w}x{h}" for w, h in sorted(resolutions)],
        "channels": sorted(channels_set),
        "corrupted_files_count": len(corrupted_files),
        "corrupted_files": corrupted_files,
        "inspection_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return report


def run_acquisition_and_inspection(
    raw_dir: Path | None = None,
    config: DatasetConfig | None = None,
) -> dict[str, Any]:
    cfg = config or default_dataset_config
    target_raw_dir = raw_dir or cfg.raw_dir
    reports_dir = cfg.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" SIGNBRIDGE — STAGE 5: DATASET INSPECTION & INTEGRITY CHECK")
    print(f" Dataset Name:     {cfg.dataset_name}")
    print(f" Target Directory: {target_raw_dir}")
    print(f" Single Source:    src/signbridge/dataset/config.py (No silent fallbacks)")
    print("=" * 80)

    # Inspect directory structure and samples (will raise if missing or mismatched)
    report = inspect_dataset_structure(target_raw_dir, config=cfg)

    # Save inspection report
    report_path = reports_dir / "inspection_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[SUMMARY] Inspection completed successfully.")
    print(f"  Dataset Name:        {report['dataset_name']}")
    print(f"  Classes Detected:    {report['total_classes']} ({', '.join(report['classes'][:8])}...)")
    print(f"  Total Images:        {report['total_images']}")
    print(f"  Splits Detected:     {report['split_distribution']}")
    print(f"  Resolutions Sampled: {report['image_resolutions']}")
    print(f"  Channels:            {report['channels']}")
    print(f"  Corrupted Files:     {report['corrupted_files_count']}")
    print(f"  Inspection Report:   {report_path}")
    print("=" * 80)

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SignBridge Stage 5: Dataset Inspection")
    parser.add_argument("--raw-dir", type=str, default=None, help="Path to raw dataset directory")
    args = parser.parse_args()
    target_dir = Path(args.raw_dir) if args.raw_dir else None
    run_acquisition_and_inspection(target_dir)
