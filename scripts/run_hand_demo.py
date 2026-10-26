"""Stage 3: Interactive Hand Detection & Landmark Extraction Demo.

Runs real-time camera capture, MediaPipe 21-point hand tracking,
skeletal visualization, and performance benchmarking without application bloat.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.camera import CameraManager, DemoDisplayController, UserAction
from signbridge.core.config import CameraConfig, VisionConfig
from signbridge.core.logger import get_logger, setup_logging
from signbridge.vision import FrameProcessor, MediaPipeHandDetector

setup_logging()
logger = get_logger("hand.demo")


def run_hand_demo(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    flip: bool = True,
    max_hands: int = 2,
    min_detection_confidence: float = 0.7,
    min_tracking_confidence: float = 0.6,
) -> None:
    """Runs interactive OpenCV window with real-time MediaPipe hand tracking."""
    print("=" * 65)
    print("SignBridge — Stage 3: MediaPipe Hand Detection & Landmarks Demo")
    print("=" * 65)
    print("Controls:")
    print("  [p]         - Pause / Resume display")
    print("  [m]         - Toggle horizontal mirror mode")
    print("  [i]         - Toggle landmark index labels (0-20)")
    print("  [s]         - Save snapshot to logs/")
    print("  [q] / [ESC] - Clean shutdown")
    print("=" * 65)

    cam_config = CameraConfig(
        device_index=camera_index,
        width=width,
        height=height,
        fps=fps,
        flip_horizontal=flip,
    )

    vision_config = VisionConfig(
        max_num_hands=max_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    window_name = f"SignBridge Stage 3 — Hand Tracking [Dev {camera_index}]"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    cam = CameraManager(config=cam_config)
    detector = MediaPipeHandDetector(config=vision_config)
    processor = FrameProcessor(config=vision_config)
    controller = DemoDisplayController(debounce_interval_sec=0.20)

    snapshot_dir = PROJECT_ROOT / "logs"
    snapshot_dir.mkdir(exist_ok=True)

    draw_indices = False
    latest_packet = None
    frozen_display = None

    # End-to-end FPS calculation
    loop_start_time = time.time()
    loop_frame_count = 0
    loop_fps = 0.0

    try:
        cam.start()
        logger.info("Camera and MediaPipe detector active. Press 'q' in window to quit.")

        while True:
            # 1. Retrieve latest frame from CameraManager
            ret, packet = cam.read(timeout=0.03)
            if ret and packet is not None:
                latest_packet = packet

            if latest_packet is None:
                time.sleep(0.01)
                continue

            # Rolling end-to-end loop FPS
            loop_frame_count += 1
            now = time.time()
            elapsed_loop = now - loop_start_time
            if elapsed_loop >= 0.5:
                loop_fps = loop_frame_count / elapsed_loop
                loop_frame_count = 0
                loop_start_time = now

            # 2. Run vision processing if not paused
            if not controller.is_paused:
                # Detect hands on current frame
                detection_result = detector.detect(
                    latest_packet.data,
                    frame_id=latest_packet.frame_id,
                    timestamp=latest_packet.timestamp,
                )

                # Render Stage 3 HUD with landmarks and metrics
                display_frame = processor.draw_stage3_hud(
                    latest_packet.data,
                    detection_result=detection_result,
                    camera_fps=loop_fps if loop_fps > 0 else cam.current_fps,
                    camera_status="RUNNING",
                    is_mirrored=cam.config.flip_horizontal,
                )
                frozen_display = display_frame.copy()
            else:
                # Paused: display frozen frame with paused overlay
                display_frame = frozen_display.copy() if frozen_display is not None else latest_packet.data.copy()
                h, w = display_frame.shape[:2]
                cv2.rectangle(display_frame, (w // 2 - 200, h // 2 - 50), (w // 2 + 200, h // 2 + 50), (30, 30, 30), -1)
                cv2.rectangle(display_frame, (w // 2 - 200, h // 2 - 50), (w // 2 + 200, h // 2 + 50), (0, 165, 255), 2)
                cv2.putText(display_frame, "|| PAUSED", (w // 2 - 90, h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 165, 255), 2)
                cv2.putText(display_frame, "Press [P] to Resume", (w // 2 - 110, h // 2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)

            # 3. Display in OpenCV GUI
            cv2.imshow(window_name, display_frame)

            # 4. Process keyboard events ONCE
            raw_key = cv2.waitKey(1)
            action = controller.handle_key(raw_key)

            if action == UserAction.QUIT:
                logger.info("Quit requested by user.")
                break
            elif action == UserAction.TOGGLE_PAUSE:
                logger.info("Toggled pause state -> %s", controller.state.name)
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
                snap_path = snapshot_dir / f"stage3_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(str(snap_path), display_frame)
                logger.info("Saved Stage 3 snapshot: %s", snap_path)
            else:
                # Custom key: 'i' toggles landmark indices
                if raw_key != -1 and (raw_key & 0xFF) in (ord("i"), ord("I")):
                    draw_indices = not draw_indices
                    logger.info("Toggled landmark index labels: %s", draw_indices)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        detector.close()
        cam.stop()
        cv2.destroyAllWindows()
        logger.info("Stage 3 demo shutdown complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SignBridge Stage 3: MediaPipe Hand Detection & Landmarks Demo")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Target frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Target frame height (default: 720)")
    parser.add_argument("--fps", type=int, default=30, help="Target frame rate (default: 30)")
    parser.add_argument("--max-hands", type=int, default=2, help="Max hands to detect (default: 2)")
    parser.add_argument("--min-detect", type=float, default=0.7, help="Min detection confidence (default: 0.7)")
    parser.add_argument("--min-track", type=float, default=0.6, help="Min tracking confidence (default: 0.6)")
    parser.add_argument("--no-flip", action="store_true", help="Disable horizontal mirror flip")
    args = parser.parse_args()

    run_hand_demo(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        flip=not args.no_flip,
        max_hands=args.max_hands,
        min_detection_confidence=args.min_detect,
        min_tracking_confidence=args.min_track,
    )


if __name__ == "__main__":
    main()
