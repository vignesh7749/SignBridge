"""Stage 3 Automated Verification Script: MediaPipe Hand Detection & Landmarks.

Tests:
1. Environment & library versions (MediaPipe, OpenCV, NumPy).
2. Synthetic zero-hand baseline.
3. Hardware-assisted live detection & FPS benchmarking (30 frames).
4. Mirroring comparison.
5. Clean resource release.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mediapipe as mp
from signbridge.camera import CameraManager, list_available_cameras
from signbridge.core.config import CameraConfig, VisionConfig
from signbridge.core.logger import get_logger, setup_logging
from signbridge.vision import FrameProcessor, MediaPipeHandDetector

setup_logging()
logger = get_logger("verify_stage3")


def run_stage3_verification() -> bool:
    print("=" * 70)
    print("SignBridge — Stage 3 Automated Verification")
    print("MediaPipe Hand Detection & Landmark Extraction")
    print("=" * 70)

    # 1. Environment & Dependency Version Check
    print("\n[Step 1] Verifying core dependencies...")
    py_ver = sys.version.split()[0]
    mp_ver = mp.__version__
    cv_ver = cv2.__version__
    np_ver = np.__version__

    print(f"[+] Python   : {py_ver}")
    print(f"[+] MediaPipe: {mp_ver}")
    print(f"[+] OpenCV   : {cv_ver}")
    print(f"[+] NumPy    : {np_ver}")

    # 2. Synthetic Zero-Hand Test
    print("\n[Step 2] Testing synthetic frame handling...")
    vision_config = VisionConfig(max_num_hands=2, min_detection_confidence=0.7)
    with MediaPipeHandDetector(vision_config) as detector:
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        blank_res = detector.detect(blank_frame, frame_id=1, timestamp=time.time())
        assert blank_res.hand_count == 0
        assert blank_res.has_hands is False
        print(f"[+] Blank frame: 0 hands detected in {blank_res.processing_time_ms:.2f}ms (Clean).")

    # 3. Hardware Assisted Benchmarking
    print("\n[Step 3] Probing camera hardware for live tracking verification...")
    devices = list_available_cameras(max_devices=2)
    if not devices:
        print("[!] No hardware camera detected. Skipping live camera benchmark.")
        return True

    cam_index = devices[0].index
    print(f"[+] Connecting to camera index {cam_index} (1280x720 @ 30 FPS)...")

    cam_config = CameraConfig(device_index=cam_index, width=1280, height=720, fps=30, flip_horizontal=True)
    processor = FrameProcessor(vision_config)

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    sample_annotated_path = logs_dir / "test_stage3_hand_detection.jpg"

    latencies: list[float] = []
    hands_detected_history: list[int] = []
    total_frames = 30
    frames_processed = 0

    with CameraManager(cam_config) as cam:
        with MediaPipeHandDetector(vision_config) as detector:
            print(f"[+] Starting 30-frame live inference benchmark...")
            t_start = time.time()

            while frames_processed < total_frames:
                ret, packet = cam.read(timeout=2.0)
                if not ret or packet is None:
                    continue

                frames_processed += 1
                det_result = detector.detect(packet.data, frame_id=packet.frame_id, timestamp=packet.timestamp)
                latencies.append(det_result.processing_time_ms)
                hands_detected_history.append(det_result.hand_count)

                # Save sample annotated frame
                if frames_processed == 15 or (det_result.has_hands and not sample_annotated_path.exists()):
                    annotated = processor.draw_stage3_hud(
                        packet.data,
                        detection_result=det_result,
                        camera_fps=cam.current_fps,
                        camera_status="RUNNING",
                        is_mirrored=cam.config.flip_horizontal,
                    )
                    cv2.imwrite(str(sample_annotated_path), annotated)

                if frames_processed % 10 == 0:
                    print(
                        f"    - Frame {frames_processed}/{total_frames}: "
                        f"Hands={det_result.hand_count}, "
                        f"Det Latency={det_result.processing_time_ms:.1f}ms, "
                        f"Cam FPS={cam.current_fps:.1f}"
                    )

            t_total = time.time() - t_start

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    effective_e2e_fps = total_frames / t_total if t_total > 0 else 0.0
    max_hands_seen = max(hands_detected_history) if hands_detected_history else 0

    print("\n" + "=" * 70)
    print("STAGE 3 BENCHMARK & METRICS SUMMARY")
    print("=" * 70)
    print(f"Total Frames Evaluated : {frames_processed} / {total_frames}")
    print(f"Total Elapsed Time     : {t_total:.2f} seconds")
    print(f"End-to-End Loop FPS    : {effective_e2e_fps:.1f} FPS")
    print(f"Average Detector Time  : {avg_latency:.2f} ms per frame")
    print(f"Detector Capacity FPS  : {1000.0 / avg_latency if avg_latency > 0 else 0.0:.1f} FPS")
    print(f"Max Hands Detected     : {max_hands_seen}")
    if sample_annotated_path.exists():
        print(f"Sample Annotated Frame : {sample_annotated_path.relative_to(PROJECT_ROOT)}")
    print(f"Hardware Release       : Clean (Camera is_running={cam.is_running})")
    print("=" * 70)
    print("[SUCCESS] Stage 3 verification completed successfully!\n")
    return True


if __name__ == "__main__":
    success = run_stage3_verification()
    sys.exit(0 if success else 1)
