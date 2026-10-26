"""Stage 2 Automated Verification Script: Camera & Capture Subsystem.

Tests camera discovery, initialization, continuous frame acquisition,
FPS monitoring, and clean hardware release.
"""

import sys
import time
from pathlib import Path
import cv2

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.camera import CameraManager, list_available_cameras
from signbridge.core.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("verify_camera")


def verify_camera_subsystem() -> bool:
    print("=" * 65)
    print("SignBridge Stage 2 — Camera Subsystem Automated Verification")
    print("=" * 65)

    # 1. Device Discovery
    print("\n[Step 1] Scanning for connected camera devices...")
    devices = list_available_cameras(max_devices=3)
    if not devices:
        print("[!] No camera devices detected. Cannot proceed with hardware test.")
        return False

    print(f"[+] Discovered {len(devices)} device(s):")
    for dev in devices:
        print(f"    - Index {dev.index}: {dev.name} ({dev.default_width}x{dev.default_height} @ {dev.default_fps:.1f} FPS)")

    target_index = devices[0].index
    print(f"\n[Step 2] Testing CameraManager lifecycle with device index {target_index}...")

    # 2. Context Manager & Initialization
    frames_collected = 0
    target_frames = 30
    saved_sample = False
    output_dir = PROJECT_ROOT / "logs"
    output_dir.mkdir(exist_ok=True)
    sample_image_path = output_dir / "test_camera_frame.jpg"

    start_time = time.time()
    with CameraManager(camera_index=target_index, width=1280, height=720, fps=30) as cam:
        print(f"[+] Camera opened. Status: {cam.status.name}")
        print(f"[+] Negotiated Resolution: {cam.actual_resolution[0]}x{cam.actual_resolution[1]}")

        print(f"\n[Step 3] Capturing {target_frames} frames continuously...")
        while frames_collected < target_frames:
            ret, packet = cam.read(timeout=2.0)
            if not ret or packet is None:
                print(f"[!] Frame read failed at frame #{frames_collected + 1}")
                break

            frames_collected += 1

            if not saved_sample:
                cv2.imwrite(str(sample_image_path), packet.data)
                saved_sample = True
                print(f"[+] Sample frame saved to: {sample_image_path.relative_to(PROJECT_ROOT)}")

            # Display progress every 10 frames
            if frames_collected % 10 == 0:
                print(
                    f"    - Frame {frames_collected}/{target_frames}: "
                    f"Resolution={packet.width}x{packet.height}, "
                    f"Live FPS={cam.current_fps:.1f}, "
                    f"Latency={packet.latency_ms:.2f}ms"
                )

    total_time = time.time() - start_time
    effective_fps = frames_collected / total_time if total_time > 0 else 0

    print(f"\n[Step 4] Checking teardown and resource release...")
    print(f"[+] Camera running status after exit: {cam.is_running} (Expected: False)")
    print(f"[+] Camera status: {cam.status.name} (Expected: CLOSED)")

    print("\n" + "=" * 65)
    print("VERIFICATION SUMMARY:")
    print(f"Total Frames Acquired : {frames_collected} / {target_frames}")
    print(f"Total Duration        : {total_time:.2f} seconds")
    print(f"Effective Capture FPS : {effective_fps:.1f}")
    print(f"Resource Teardown     : Clean (is_running={cam.is_running})")
    print("=" * 65)

    if frames_collected == target_frames and not cam.is_running:
        print("[SUCCESS] Camera subsystem verified successfully!\n")
        return True
    else:
        print("[FAILURE] Camera verification did not satisfy all criteria.\n")
        return False


if __name__ == "__main__":
    success = verify_camera_subsystem()
    sys.exit(0 if success else 1)
