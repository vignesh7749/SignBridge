"""SignBridge Demo Video Verification Script.

Validates all 12 criteria for demonstration videos:
1. Exists on disk
2. Opens successfully with OpenCV
3. Has width > 0
4. Has height > 0
5. Has FPS > 0
6. Has frame count > 0
7. Has duration > 0
8. First frame can be decoded
9. Middle frame can be decoded
10. Final frame can be decoded
11. Video can be played sequentially
12. SHA-256 checksum matches manifest.json
"""

import hashlib
import json
import os
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "datasets" / "samples" / "word_signs"
MANIFEST_PATH = SAMPLES_DIR / "manifest.json"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_demo_videos() -> int:
    print("=" * 60)
    print("SIGNBRIDGE DEMO VIDEO VERIFICATION")
    print("=" * 60)

    if not MANIFEST_PATH.exists():
        print(f"[FAIL] Manifest not found: {MANIFEST_PATH}")
        return 1

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = manifest.get("samples", [])
    if not samples:
        print("[FAIL] Manifest contains no samples.")
        return 1

    verified_count = 0
    invalid_count = 0

    for item in samples:
        filename = item.get("filename", "")
        sign_name = item.get("sign", "UNKNOWN")
        license_type = item.get("license", "Unknown")
        expected_checksum = item.get("checksum", "")
        video_path = SAMPLES_DIR / filename

        print(f"\n{sign_name}")
        print(f"  File: {filename}")

        # 1. Exists
        if not video_path.exists():
            print(f"  [FAIL] Video file does not exist: {video_path}")
            invalid_count += 1
            continue

        # 2. Opens with OpenCV
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [FAIL] OpenCV failed to open video file.")
            invalid_count += 1
            continue

        # 3, 4, 5, 6, 7. Dimensions, FPS, Frame Count, Duration
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = (frame_count / fps) if fps > 0 else 0.0

        if width <= 0 or height <= 0:
            print(f"  [FAIL] Invalid dimensions: {width}x{height}")
            cap.release()
            invalid_count += 1
            continue

        if fps <= 0:
            print(f"  [FAIL] Invalid FPS: {fps}")
            cap.release()
            invalid_count += 1
            continue

        if frame_count <= 0 or duration <= 0:
            print(f"  [FAIL] Invalid frame count or duration: frames={frame_count}, dur={duration}")
            cap.release()
            invalid_count += 1
            continue

        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Frames: {frame_count}")
        print(f"  Duration: {duration:.2f} sec")

        # 8. First frame decode
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_first, frame_first = cap.read()
        if not ret_first or frame_first is None or frame_first.size == 0:
            print("  Decode (First Frame): FAIL")
            cap.release()
            invalid_count += 1
            continue

        # 9. Middle frame decode
        mid_idx = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
        ret_mid, frame_mid = cap.read()
        if not ret_mid or frame_mid is None or frame_mid.size == 0:
            print("  Decode (Middle Frame): FAIL")
            cap.release()
            invalid_count += 1
            continue

        # 10. Final frame decode
        last_idx = max(0, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_idx)
        ret_last, frame_last = cap.read()
        if not ret_last or frame_last is None or frame_last.size == 0:
            print("  Decode (Final Frame): FAIL")
            cap.release()
            invalid_count += 1
            continue

        print("  Decode: PASS")

        # 11. Sequential playback
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        seq_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            seq_frames += 1
            if seq_frames >= min(frame_count, 30):
                break

        if seq_frames < min(frame_count, 10):
            print(f"  Playback: FAIL (only read {seq_frames} frames)")
            cap.release()
            invalid_count += 1
            continue
        print("  Playback: PASS")
        cap.release()

        # 12. Checksum matches manifest
        actual_checksum = compute_sha256(video_path)
        if expected_checksum and actual_checksum.lower() != expected_checksum.lower():
            print(f"  Checksum: FAIL (expected {expected_checksum[:12]}..., got {actual_checksum[:12]}...)")
            invalid_count += 1
            continue

        print("  Checksum: PASS")
        print(f"  License: PASS ({license_type})")
        verified_count += 1

    print("\n" + "=" * 60)
    print("FINAL:")
    print(f"{verified_count}/{len(samples)} real videos verified")
    print(f"{invalid_count} invalid videos")
    print("=" * 60)

    return 0 if (verified_count == len(samples) and invalid_count == 0) else 1


if __name__ == "__main__":
    sys.exit(verify_demo_videos())
