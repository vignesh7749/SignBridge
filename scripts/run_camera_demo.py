"""Stage 2: Interactive Camera Demo & OpenCV Viewer.

Displays real-time video feed with FPS monitoring, camera device info,
robust pause/resume display state handling, and keyboard controls.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.camera import (
    CameraManager,
    DemoDisplayController,
    UserAction,
    list_available_cameras,
)
from signbridge.core.config import CameraConfig
from signbridge.core.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("camera.demo")


def render_running_hud(
    frame: np.ndarray,
    device_index: int,
    current_fps: float,
    status_name: str,
    frame_id: int,
) -> np.ndarray:
    """Renders the active running HUD overlay on the live frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Semi-transparent top & bottom bars
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, h - 35), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

    # Top HUD: Camera details & metrics
    top_info = (
        f"Camera: Dev {device_index} | "
        f"Res: {w}x{h} | "
        f"FPS: {current_fps:.1f} | "
        f"Frame: #{frame_id} | "
        f"State: {status_name}"
    )
    cv2.putText(annotated, top_info, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)

    # Bottom HUD: Controls guide
    bottom_info = "Keys: [P] Pause | [M] Toggle Mirror | [S] Snapshot | [Q/ESC] Quit"
    cv2.putText(annotated, bottom_info, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return annotated


def render_paused_hud(
    frame: np.ndarray,
    device_index: int,
    actual_res: tuple[int, int],
    frame_id: int,
) -> np.ndarray:
    """Renders a distinguished, non-frozen PAUSED overlay on the frozen frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Dim background slightly
    dim_overlay = np.zeros_like(annotated)
    cv2.addWeighted(annotated, 0.55, dim_overlay, 0.45, 0, annotated)

    # Top bar
    top_bar = annotated.copy()
    cv2.rectangle(top_bar, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.rectangle(top_bar, (0, h - 35), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(top_bar, 0.75, annotated, 0.25, 0, annotated)

    top_info = (
        f"Camera: Dev {device_index} | "
        f"Res: {actual_res[0]}x{actual_res[1]} | "
        f"Capture: ACTIVE (Thread running) | "
        f"Frame: #{frame_id}"
    )
    cv2.putText(annotated, top_info, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

    # Center Paused Modal Card
    card_w, card_h = 440, 180
    cx, cy = w // 2, h // 2
    x1, y1 = cx - card_w // 2, cy - card_h // 2
    x2, y2 = cx + card_w // 2, cy + card_h // 2

    # Draw rounded-like card with double border
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 3)  # Orange accent border

    # Card Title: PAUSED
    title_text = "|| PAUSED"
    cv2.putText(annotated, title_text, (cx - 110, y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 165, 255), 3)

    # Subtitle 1: Press P to Resume
    sub1 = "Press [ P ] to Resume Feed"
    cv2.putText(annotated, sub1, (cx - 150, y1 + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

    # Subtitle 2: Press Q / ESC to Quit
    sub2 = "Press [ Q / ESC ] to Quit"
    cv2.putText(annotated, sub2, (cx - 120, y1 + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Bottom bar
    bottom_info = "Status: PAUSED (Camera remains live in background) | [P] Resume | [Q/ESC] Quit"
    cv2.putText(annotated, bottom_info, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    return annotated


def run_camera_demo(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    flip: bool = True,
) -> None:
    """Runs interactive OpenCV window rendering the live webcam feed."""
    print("=" * 60)
    print("SignBridge — Stage 2: Camera & OpenCV Capture Subsystem Demo")
    print("=" * 60)
    print("Controls:")
    print("  [p]         - Pause / Resume display (camera stays active)")
    print("  [m]         - Toggle horizontal mirror mode")
    print("  [s]         - Save snapshot to logs/")
    print("  [q] / [ESC] - Quit")
    print("=" * 60)

    config = CameraConfig(
        device_index=camera_index,
        width=width,
        height=height,
        fps=fps,
        flip_horizontal=flip,
    )

    window_name = f"SignBridge Camera Demo [Device {camera_index}]"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    cam = CameraManager(config=config)
    controller = DemoDisplayController(debounce_interval_sec=0.20)

    try:
        cam.start()
        logger.info("Live feed active. Press 'q' in the window to quit.")

        snapshot_dir = PROJECT_ROOT / "logs"
        snapshot_dir.mkdir(exist_ok=True)

        latest_packet = None
        frozen_frame = None
        frozen_frame_id = 0

        while True:
            # 1. Continuous frame retrieval from CameraManager
            # Always query camera queue to drain old frames and keep latest frame fresh
            ret, packet = cam.read(timeout=0.03)
            if ret and packet is not None:
                latest_packet = packet
                # When running, update the frame cache
                if not controller.is_paused:
                    frozen_frame = packet.data
                    frozen_frame_id = packet.frame_id

            # Guard against uninitialized camera startup
            if frozen_frame is None and latest_packet is None:
                time.sleep(0.01)
                continue

            # 2. Render frame based on state
            if controller.is_paused:
                base = frozen_frame if frozen_frame is not None else latest_packet.data
                display_frame = render_paused_hud(
                    base,
                    device_index=cam.config.device_index,
                    actual_res=cam.actual_resolution,
                    frame_id=frozen_frame_id,
                )
            else:
                display_frame = render_running_hud(
                    latest_packet.data,
                    device_index=cam.config.device_index,
                    current_fps=cam.current_fps,
                    status_name="RUNNING",
                    frame_id=latest_packet.frame_id,
                )

            # 3. Display frame in window
            cv2.imshow(window_name, display_frame)

            # 4. Read keyboard event ONCE per loop
            raw_key = cv2.waitKey(1)

            # 5. Centralized single key processing via controller
            action = controller.handle_key(raw_key)

            if action == UserAction.QUIT:
                logger.info("Quit requested by user.")
                break
            elif action == UserAction.TOGGLE_PAUSE:
                if controller.is_paused:
                    logger.info("Display PAUSED. (Camera capture thread remains active in background).")
                    if latest_packet is not None:
                        frozen_frame = latest_packet.data.copy()
                        frozen_frame_id = latest_packet.frame_id
                else:
                    logger.info("Display RESUMED. Consuming fresh camera frames.")
            elif action == UserAction.TOGGLE_MIRROR:
                cam.config = CameraConfig(
                    device_index=cam.config.device_index,
                    width=cam.config.width,
                    height=cam.config.height,
                    fps=cam.config.fps,
                    flip_horizontal=not cam.config.flip_horizontal,
                )
                logger.info("Toggled mirror mode: %s", cam.config.flip_horizontal)
            elif action == UserAction.SNAPSHOT:
                snap_frame = frozen_frame if controller.is_paused else (latest_packet.data if latest_packet else None)
                if snap_frame is not None:
                    snap_path = snapshot_dir / f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(str(snap_path), snap_frame)
                    logger.info("Saved snapshot: %s", snap_path)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        logger.info("Demo exited cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SignBridge Camera & OpenCV Capture Subsystem Demo")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Target frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Target frame height (default: 720)")
    parser.add_argument("--fps", type=int, default=30, help="Target frame rate (default: 30)")
    parser.add_argument("--no-flip", action="store_true", help="Disable horizontal mirror flip")
    parser.add_argument("--list-cameras", action="store_true", help="List detected cameras and exit")
    args = parser.parse_args()

    if args.list_cameras:
        devices = list_available_cameras(max_devices=5)
        print(f"\nDetected {len(devices)} camera device(s):")
        for dev in devices:
            print(f"  [{dev.index}] {dev.name} - Native resolution: {dev.default_width}x{dev.default_height} @ {dev.default_fps:.1f} FPS")
        return

    run_camera_demo(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        flip=not args.no_flip,
    )


if __name__ == "__main__":
    main()
