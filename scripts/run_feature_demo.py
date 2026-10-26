"""Stage 4: Interactive Feature Engineering & Real-Time Visualization Demo.

Connects CameraManager -> MediaPipeHandDetector -> HandFeatureExtractor.
Renders real-time camera feed, skeletal tracking, and a comprehensive HUD
displaying the 94-dimensional feature vector, latency benchmarks, finger
extension gauges, and joint flexion statistics.
"""

import argparse
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.camera import CameraManager, DemoDisplayController, UserAction
from signbridge.core.config import CameraConfig, VisionConfig
from signbridge.core.logger import get_logger, setup_logging
from signbridge.features import FeatureSchema, HandFeatureExtractor
from signbridge.vision import FrameProcessor, MediaPipeHandDetector

setup_logging()
logger = get_logger("feature.demo")


def draw_feature_hud(
    frame: np.ndarray,
    detection_result,
    features_list: list,
    extractions_latency_us: float,
    camera_fps: float,
    canonicalize: bool,
) -> np.ndarray:
    """Draws Stage 4 Feature Engineering HUD overlay and metrics sidebar."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Semi-transparent top banner
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

    # Banner text
    cv2.putText(
        canvas,
        "SIGNBRIDGE  |  STAGE 4 — FEATURE ENGINEERING",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 240, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"FPS: {camera_fps:4.1f} | Schema: v{FeatureSchema.VERSION} ({FeatureSchema.TOTAL_DIMENSIONS}D)",
        (w - 380, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 240, 200),
        1,
        cv2.LINE_AA,
    )

    # Draw metrics panel on right side
    panel_w = 340
    panel_x0 = w - panel_w - 15
    panel_y0 = 65
    panel_h = min(480, h - 80)

    panel_overlay = canvas.copy()
    cv2.rectangle(panel_overlay, (panel_x0, panel_y0), (panel_x0 + panel_w, panel_y0 + panel_h), (24, 24, 28), -1)
    cv2.addWeighted(panel_overlay, 0.88, canvas, 0.12, 0, canvas)
    cv2.rectangle(canvas, (panel_x0, panel_y0), (panel_x0 + panel_w, panel_y0 + panel_h), (60, 60, 70), 1)

    # Panel header
    y = panel_y0 + 26
    cv2.putText(canvas, "FEATURE PIPELINE METRICS", (panel_x0 + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255), 2)

    y += 24
    cv2.putText(
        canvas,
        f"Ext. Latency:  {extractions_latency_us:5.1f} us ({extractions_latency_us / 1000.0:.3f} ms)",
        (panel_x0 + 15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 180),
        1,
    )

    y += 20
    canon_text = "ENABLED (x -> -x)" if canonicalize else "RAW (Disabled)"
    cv2.putText(
        canvas,
        f"Canonicalize:  {canon_text}",
        (panel_x0 + 15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 180, 100) if canonicalize else (180, 180, 180),
        1,
    )

    y += 20
    hands_detected = len(features_list)
    cv2.putText(
        canvas,
        f"Hands Detected: {hands_detected}",
        (panel_x0 + 15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    if not features_list:
        y += 40
        cv2.putText(
            canvas,
            "Place hand in view to extract features",
            (panel_x0 + 15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (140, 140, 150),
            1,
        )
    else:
        # Display first detected hand features
        feat = features_list[0]
        stats = feat.summary_stats()

        y += 24
        cv2.putText(canvas, f"Hand 1: {feat.handedness} (scale: {feat.scale_factor:.1f}px)", (panel_x0 + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 0), 1)

        y += 22
        cv2.putText(canvas, f"L2 Norm: {stats['l2_norm']:.3f} | Aspect: {feat.palm_aspect_ratio:.2f}", (panel_x0 + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        y += 26
        cv2.putText(canvas, "Fingertip Extension Ratios:", (panel_x0 + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        # Draw relative finger extension bars
        tip_dists = feat.wrist_tip_distances
        finger_labels = ["T", "I", "M", "R", "P"]
        bar_colors = [
            (255, 100, 100),  # Thumb (Blue)
            (100, 255, 100),  # Index (Green)
            (100, 220, 255),  # Middle (Yellow)
            (255, 160, 255),  # Ring (Pink)
            (200, 120, 255),  # Pinky (Purple)
        ]

        bar_x = panel_x0 + 35
        bar_max_w = 190
        for i, (label, dist, color) in enumerate(zip(finger_labels, tip_dists, bar_colors)):
            y += 20
            cv2.putText(canvas, f"{label}:", (panel_x0 + 15, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            # Normalize extension: typical open hand is ~1.2 - 2.2, curled is ~0.4 - 0.8
            ratio = float(np.clip((dist - 0.4) / 1.6, 0.0, 1.0))
            bar_len = int(ratio * bar_max_w)
            # Background slot
            cv2.rectangle(canvas, (bar_x, y), (bar_x + bar_max_w, y + 12), (45, 45, 52), -1)
            # Fill bar
            if bar_len > 0:
                cv2.rectangle(canvas, (bar_x, y), (bar_x + bar_len, y + 12), color, -1)
            cv2.putText(canvas, f"{dist:.2f}", (bar_x + bar_max_w + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        # Joint flexion degrees (MCP, PIP, DIP for Index and Middle)
        y += 26
        cv2.putText(canvas, "Joint Flexion Angles:", (panel_x0 + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        angles = feat.joint_angles
        deg = np.degrees(angles)
        y += 18
        cv2.putText(canvas, f"Idx MCP/PIP/DIP: {deg[3]:.0f}/{deg[4]:.0f}/{deg[5]:.0f} deg", (panel_x0 + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
        y += 16
        cv2.putText(canvas, f"Mid MCP/PIP/DIP: {deg[6]:.0f}/{deg[7]:.0f}/{deg[8]:.0f} deg", (panel_x0 + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

    # Bottom helper banner
    footer_overlay = canvas.copy()
    cv2.rectangle(footer_overlay, (0, h - 35), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(footer_overlay, 0.85, canvas, 0.15, 0, canvas)
    cv2.putText(
        canvas,
        "[P] Pause/Resume  |  [C] Toggle Canonicalization  |  [S] Print Feature JSON  |  [Q] Quit",
        (20, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 210, 220),
        1,
        cv2.LINE_AA,
    )

    return canvas


def run_feature_demo(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    canonicalize: bool = True,
) -> None:
    """Runs interactive Stage 4 feature engineering demo."""
    print("=" * 70)
    print("SignBridge — Stage 4: Feature Engineering & Preprocessing Demo")
    print("=" * 70)
    print("Controls:")
    print("  [p]         - Pause / Resume display")
    print("  [c]         - Toggle Left Hand Canonicalization")
    print("  [s]         - Print complete 94D feature JSON to terminal")
    print("  [m]         - Toggle mirror mode")
    print("  [q] / [ESC] - Clean shutdown")
    print("=" * 70)

    cam_config = CameraConfig(device_index=camera_index, width=width, height=height, fps=fps, flip_horizontal=True)
    vision_config = VisionConfig(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)

    window_name = f"SignBridge Stage 4 — Feature Engineering [Dev {camera_index}]"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    cam = CameraManager(config=cam_config)
    detector = MediaPipeHandDetector(config=vision_config)
    processor = FrameProcessor(config=vision_config)
    controller = DemoDisplayController(debounce_interval_sec=0.20)
    extractor = HandFeatureExtractor(canonicalize_handedness=canonicalize)

    latest_packet = None
    frozen_display = None
    canonicalize_state = canonicalize

    loop_start_time = time.time()
    loop_frame_count = 0
    loop_fps = 0.0

    try:
        cam.start()
        logger.info("Camera, MediaPipe, and Feature Extractor active.")

        while True:
            ret, packet = cam.read(timeout=0.03)
            if ret and packet is not None:
                latest_packet = packet

            if latest_packet is None:
                time.sleep(0.01)
                continue

            loop_frame_count += 1
            now = time.time()
            elapsed_loop = now - loop_start_time
            if elapsed_loop >= 0.5:
                loop_fps = loop_frame_count / elapsed_loop
                loop_frame_count = 0
                loop_start_time = now

            if not controller.is_paused:
                # 1. Vision Hand Detection
                detection_result = detector.detect(
                    latest_packet.data,
                    frame_id=latest_packet.frame_id,
                    timestamp=latest_packet.timestamp,
                )

                # 2. Stage 4 Feature Extraction
                features_list = []
                ext_latencies = []
                for hand in detection_result.hands:
                    t_ext0 = time.perf_counter()
                    feat = extractor.extract(hand)
                    ext_latencies.append((time.perf_counter() - t_ext0) * 1_000_000.0)  # microseconds
                    features_list.append(feat)

                mean_ext_lat_us = float(np.mean(ext_latencies)) if ext_latencies else 0.0

                # 3. Render Hand Skeleton & Stage 4 HUD
                drawn_frame = processor.draw_stage3_hud(
                    latest_packet.data,
                    detection_result=detection_result,
                    camera_fps=loop_fps if loop_fps > 0 else cam.current_fps,
                    camera_status="RUNNING",
                    is_mirrored=cam.config.flip_horizontal,
                )

                display_frame = draw_feature_hud(
                    drawn_frame,
                    detection_result=detection_result,
                    features_list=features_list,
                    extractions_latency_us=mean_ext_lat_us,
                    camera_fps=loop_fps if loop_fps > 0 else cam.current_fps,
                    canonicalize=canonicalize_state,
                )
                frozen_display = display_frame.copy()
            else:
                display_frame = frozen_display.copy() if frozen_display is not None else latest_packet.data.copy()
                h, w = display_frame.shape[:2]
                cv2.rectangle(display_frame, (w // 2 - 200, h // 2 - 50), (w // 2 + 200, h // 2 + 50), (30, 30, 30), -1)
                cv2.rectangle(display_frame, (w // 2 - 200, h // 2 - 50), (w // 2 + 200, h // 2 + 50), (0, 165, 255), 2)
                cv2.putText(display_frame, "|| PAUSED", (w // 2 - 90, h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 165, 255), 2)
                cv2.putText(display_frame, "Press [P] to Resume", (w // 2 - 110, h // 2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)

            cv2.imshow(window_name, display_frame)

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
            else:
                if raw_key != -1:
                    code = raw_key & 0xFF
                    # [C]: Toggle Handedness Canonicalization
                    if code in (ord("c"), ord("C")):
                        canonicalize_state = not canonicalize_state
                        extractor.canonicalize_handedness = canonicalize_state
                        logger.info("Toggled handedness canonicalization: %s", canonicalize_state)
                    # [S]: Print current feature vector JSON
                    elif code in (ord("s"), ord("S")):
                        if features_list:
                            feat_dict = features_list[0].to_dict()
                            print("\n" + "=" * 60)
                            print("FEATURE VECTOR SNAPSHOT (Hand 1):")
                            print(json.dumps(feat_dict, indent=2))
                            print("=" * 60)
                            logger.info("Printed feature vector snapshot (dim: %d)", len(feat_dict["features"]))
                        else:
                            logger.warning("No hand detected to snapshot.")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        detector.close()
        cam.stop()
        cv2.destroyAllWindows()
        logger.info("Stage 4 demo shutdown complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SignBridge Stage 4: Feature Engineering Demo")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Frame height (default: 720)")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate (default: 30)")
    parser.add_argument("--no-canonicalize", action="store_true", help="Disable left-hand canonicalization")
    args = parser.parse_args()

    run_feature_demo(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        canonicalize=not args.no_canonicalize,
    )


if __name__ == "__main__":
    main()
