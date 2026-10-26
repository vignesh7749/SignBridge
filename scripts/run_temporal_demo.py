"""Stage 7: Interactive Real-Time Temporal Recognition & Anti-Jitter Demo.

End-to-End Pipeline:
  Camera (Stage 2)
    ↓
  MediaPipe Hand Detector (Stage 3)
    ↓
  HandFeatureExtractor (Stage 4)
    ↓
  SignPredictor (Stage 6)
    ↓
  TemporalEngine (Stage 7)
    ↓
  Stable Sign Tokens & Diagnostic HUD

Controls:
  Q / ESC : Quit application
  P       : Pause / Resume video streaming
  R       : Reset temporal engine state and token history
  C       : Toggle confidence distribution display
  D       : Toggle detailed diagnostic telemetry overlay
"""

import argparse
from collections import deque
from pathlib import Path
import sys
import time
from typing import Optional
import cv2
import numpy as np

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.camera import CameraManager
from signbridge.core.config import CameraConfig, VisionConfig
from signbridge.core.logger import get_logger, setup_logging
from signbridge.features import HandFeatureExtractor
from signbridge.model.predictor import SignPredictor, get_default_predictor
from signbridge.temporal import (
    FramePrediction,
    StableSign,
    TemporalEngine,
    TemporalEngineConfig,
    TemporalFrameResult,
    TemporalState,
)
from signbridge.vision import MediaPipeHandDetector

setup_logging()
logger = get_logger("temporal.demo")


def draw_hud(
    frame: np.ndarray,
    temporal_result: TemporalFrameResult,
    raw_pred_label: str,
    raw_confidence: float,
    raw_margin: float,
    recent_tokens: deque[StableSign],
    fps: float,
    latencies: dict[str, float],
    show_confidence: bool = True,
    show_diagnostics: bool = True,
    is_paused: bool = False,
) -> np.ndarray:
    """Renders the comprehensive Stage 7 diagnostic HUD overlay on the live camera frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Semi-transparent top header banner
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (18, 18, 18), -1)
    # Bottom token ribbon
    cv2.rectangle(overlay, (0, h - 55), (w, h), (18, 18, 18), -1)
    # Left status telemetry card
    card_w = 340
    card_h = 320 if show_confidence else 260
    cv2.rectangle(overlay, (15, 70), (15 + card_w, 70 + card_h), (24, 24, 24), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

    # Top Header Text
    cv2.putText(
        canvas,
        "SIGNBRIDGE  |  STAGE 7 — TEMPORAL RECOGNITION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 240, 255),
        2,
        cv2.LINE_AA,
    )
    fps_color = (0, 255, 120) if fps >= 20.0 else (0, 165, 255)
    cv2.putText(
        canvas,
        f"{fps:.1f} FPS",
        (w - 120, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        fps_color,
        2,
        cv2.LINE_AA,
    )

    # Status Card Border
    state = temporal_result.state
    state_colors = {
        TemporalState.IDLE: (140, 140, 140),
        TemporalState.CANDIDATE: (0, 215, 255),
        TemporalState.STABLE: (0, 255, 120),
        TemporalState.EMITTED: (0, 255, 0),
        TemporalState.WAITING_FOR_CHANGE: (255, 180, 50),
    }
    state_color = state_colors.get(state, (200, 200, 200))
    cv2.rectangle(canvas, (15, 70), (15 + card_w, 70 + card_h), state_color, 2)

    # Card Title
    cv2.putText(
        canvas,
        "TEMPORAL STABILIZER",
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # State Badge
    state_text = f"STATE: {state.value}"
    cv2.putText(
        canvas,
        state_text,
        (30, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        state_color,
        2,
        cv2.LINE_AA,
    )

    # Active Candidate & Raw Prediction
    cand_label = temporal_result.current_candidate or "—"
    cv2.putText(
        canvas,
        f"Raw Prediction: {raw_pred_label} ({raw_confidence * 100:.1f}%)",
        (30, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Candidate:      {cand_label}",
        (30, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 240, 255),
        2,
        cv2.LINE_AA,
    )

    # Stability Progress Bar (0..100%)
    stability_pct = int(np.clip(temporal_result.stability_ratio * 100, 0, 100))
    cv2.putText(
        canvas,
        f"Consensus: {stability_pct}%",
        (30, 208),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    # Draw bar
    bar_x, bar_y, bar_w, bar_h = 140, 196, 190, 14
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_w = int(bar_w * (stability_pct / 100.0))
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), state_color, -1)

    # Dwell Time Bar
    dwell_ms = temporal_result.dwell_duration_ms
    target_dwell = 250.0
    dwell_pct = int(np.clip((dwell_ms / target_dwell) * 100, 0, 100))
    cv2.putText(
        canvas,
        f"Dwell: {dwell_ms:.0f} ms",
        (30, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    dbar_x, dbar_y, dbar_w, dbar_h = 140, 223, 190, 14
    cv2.rectangle(canvas, (dbar_x, dbar_y), (dbar_x + dbar_w, dbar_y + dbar_h), (60, 60, 60), -1)
    dfill_w = int(dbar_w * (dwell_pct / 100.0))
    dbar_color = (0, 255, 120) if dwell_pct >= 100 else (0, 215, 255)
    cv2.rectangle(canvas, (dbar_x, dbar_y), (dbar_x + dfill_w, dbar_y + dbar_h), dbar_color, -1)

    # Last Emitted Sign Banner
    last_label = recent_tokens[-1].label if recent_tokens else "—"
    cv2.putText(
        canvas,
        f"Last Emitted: [{last_label}]",
        (30, 265),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0) if last_label != "—" else (140, 140, 140),
        2,
        cv2.LINE_AA,
    )

    if show_confidence:
        cv2.putText(
            canvas,
            f"Margin: +{raw_margin * 100:.1f}% | Conf: {temporal_result.candidate_confidence * 100:.1f}%",
            (30, 292),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    # Right Diagnostic Telemetry Overlay
    if show_diagnostics:
        diag_x = w - 240
        diag_y = 70
        diag_h = 125
        overlay_diag = canvas.copy()
        cv2.rectangle(overlay_diag, (diag_x - 10, diag_y), (w - 15, diag_y + diag_h), (24, 24, 24), -1)
        cv2.addWeighted(overlay_diag, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (diag_x - 10, diag_y), (w - 15, diag_y + diag_h), (70, 70, 70), 1)

        cv2.putText(canvas, "LATENCY BREAKDOWN", (diag_x, diag_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1)
        cv2.putText(canvas, f"Detector: {latencies.get('detector', 0.0):.1f} ms", (diag_x, diag_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(canvas, f"Features: {latencies.get('features', 0.0):.2f} ms", (diag_x, diag_y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(canvas, f"Model:    {latencies.get('model', 0.0):.2f} ms", (diag_x, diag_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(canvas, f"Temporal: {latencies.get('temporal', 0.0):.3f} ms", (diag_x, diag_y + 102), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1)

    # Bottom Token History Ribbon
    tokens_str = "  ->  ".join(f"[{t.label}]" for t in list(recent_tokens)[-10:])
    history_display = f"EMITTED TOKENS: {tokens_str}" if tokens_str else "EMITTED TOKENS: (None yet - hold a sign steady for 250ms)"
    cv2.putText(
        canvas,
        history_display,
        (20, h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Bottom Controls Help
    help_info = "[Q/ESC] Quit | [P] Pause | [R] Reset | [C] Conf | [D] Diag"
    cv2.putText(canvas, help_info, (w - 420, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # Paused Watermark
    if is_paused:
        cv2.putText(
            canvas,
            "PAUSED",
            (w // 2 - 100, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="SignBridge Stage 7: Real-Time Temporal Recognition Demo")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--dwell-ms", type=float, default=250.0, help="Minimum dwell time in ms (default: 250.0)")
    parser.add_argument("--confidence-thresh", type=float, default=0.70, help="Confidence threshold (default: 0.70)")
    parser.add_argument("--rearm-policy", type=str, default="on_change", choices=["on_change", "timeout", "neutral_release"], help="Re-arm policy")
    parser.add_argument("--headless", action="store_true", help="Run without opening OpenCV GUI window (test mode)")
    args = parser.parse_args()

    print("=" * 70)
    print("SIGNBRIDGE — STAGE 7: REAL-TIME TEMPORAL RECOGNITION DEMO")
    print("=" * 70)
    print(f"Camera Device Index:   {args.camera_index}")
    print(f"Minimum Dwell:         {args.dwell_ms:.1f} ms")
    print(f"Confidence Threshold:  {args.confidence_thresh * 100:.1f}%")
    print(f"Re-arm Policy:         {args.rearm_policy}")
    print("=" * 70)

    # 1. Initialize Pipeline Subsystems
    print("[1/5] Initializing Camera Manager...")
    cam_config = CameraConfig(device_index=args.camera_index, width=1280, height=720, fps=30)
    camera = CameraManager(cam_config)
    using_synthetic_camera = False

    try:
        camera.start()
        # Test read
        ok, test_packet = camera.read(timeout=1.5)
        if not ok or test_packet is None:
            raise RuntimeError("Camera start timed out or returned no frames.")
    except Exception as exc:
        if args.headless:
            print(f"[INFO] Hardware camera unavailable ({exc}). Using synthetic test frames for headless verification.")
            using_synthetic_camera = True
        else:
            print(f"[ERROR] Could not open camera device {args.camera_index}: {exc}")
            return 1

    print("[2/5] Initializing MediaPipe Hand Detector (Stage 3)...")
    vision_config = VisionConfig(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)
    detector = MediaPipeHandDetector(vision_config)

    print("[3/5] Initializing Hand Feature Extractor (Stage 4)...")
    extractor = HandFeatureExtractor()

    print("[4/5] Loading Production Sign Predictor (Stage 6)...")
    try:
        predictor = get_default_predictor()
    except Exception as exc:
        print(f"[ERROR] Failed to load production sign classifier: {exc}")
        camera.release()
        return 1

    print("[5/5] Initializing Temporal Recognition Engine (Stage 7)...")
    temporal_config = TemporalEngineConfig(
        confidence_threshold=args.confidence_thresh,
        window_duration_ms=400.0,
        consensus_threshold=0.60,
        minimum_stable_frames=3,
        minimum_dwell_ms=args.dwell_ms,
        hand_loss_grace_ms=250.0,
        duplicate_suppression_ms=1200.0,
        rearm_policy=args.rearm_policy,
    )
    engine = TemporalEngine(temporal_config)

    recent_tokens: deque[StableSign] = deque(maxlen=20)
    show_confidence = True
    show_diagnostics = True
    is_paused = False

    fps_history = deque(maxlen=30)
    last_frame_time = time.time()
    latencies: dict[str, float] = {}

    window_name = "SignBridge — Stage 7 Temporal Recognition Demo"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    print("\n[READY] Starting real-time recognition loop. Press 'Q' or 'ESC' to exit.")
    frame_counter = 0

    try:
        while True:
            t0 = time.perf_counter()
            current_wall_ts = time.time()

            if not is_paused:
                if using_synthetic_camera:
                    # Synthetic 1280x720 frame with test pattern for headless verification
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(frame, "SignBridge Headless Test Pattern", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    time.sleep(0.033)
                else:
                    ok, packet = camera.read(timeout=0.5)
                    if not ok or packet is None or packet.data is None:
                        print("[WARN] Received empty camera frame.")
                        time.sleep(0.01)
                        continue
                    frame = packet.data

                frame_counter += 1

                # 1. Vision Detection (Stage 3)
                t_det_start = time.perf_counter()
                detection_result = detector.detect(frame, frame_id=frame_counter, timestamp=current_wall_ts)
                latencies["detector"] = (time.perf_counter() - t_det_start) * 1000.0

                raw_label = ""
                raw_conf = 0.0
                raw_margin = 0.0
                primary_hand = detection_result.primary_hand

                if primary_hand is not None:
                    # Draw landmarks on frame
                    annotated_frame = detection_result.annotated_frame if detection_result.annotated_frame is not None else frame

                    # 2. Feature Extraction (Stage 4)
                    t_feat_start = time.perf_counter()
                    feat_vector = extractor.extract(primary_hand, timestamp_ms=current_wall_ts * 1000.0)
                    latencies["features"] = (time.perf_counter() - t_feat_start) * 1000.0

                    if feat_vector is not None:
                        # 3. Model Classification (Stage 6)
                        t_mod_start = time.perf_counter()
                        pred_result = predictor.predict(feat_vector.features)
                        latencies["model"] = (time.perf_counter() - t_mod_start) * 1000.0

                        raw_label = pred_result.class_name
                        raw_conf = pred_result.confidence
                        raw_margin = pred_result.top_margin

                        frame_pred = FramePrediction.from_prediction_result(
                            pred_result,
                            timestamp=current_wall_ts,
                            has_hand=True,
                        )
                    else:
                        latencies["features"] = 0.0
                        latencies["model"] = 0.0
                        frame_pred = FramePrediction.no_hand(timestamp=current_wall_ts)
                else:
                    annotated_frame = frame
                    latencies["features"] = 0.0
                    latencies["model"] = 0.0
                    frame_pred = FramePrediction.no_hand(timestamp=current_wall_ts)

                # 4. Temporal Recognition & Stabilization (Stage 7)
                t_temp_start = time.perf_counter()
                temporal_result = engine.process_frame(frame_pred)
                latencies["temporal"] = (time.perf_counter() - t_temp_start) * 1000.0

                if temporal_result.has_emitted_sign:
                    assert temporal_result.stable_sign is not None
                    recent_tokens.append(temporal_result.stable_sign)
                    print(f"  >>> [STABLE SIGN EMITTED]: '{temporal_result.stable_sign.label}' "
                          f"(dwell: {temporal_result.stable_sign.duration_ms:.1f} ms, "
                          f"conf: {temporal_result.stable_sign.confidence * 100:.1f}%)")

                # FPS Calculation
                now = time.time()
                dt = now - last_frame_time
                last_frame_time = now
                if dt > 0.0:
                    fps_history.append(1.0 / dt)
                avg_fps = float(np.mean(fps_history)) if fps_history else 30.0

                # 5. Render HUD Overlay
                display_frame = draw_hud(
                    frame=annotated_frame,
                    temporal_result=temporal_result,
                    raw_pred_label=raw_label,
                    raw_confidence=raw_conf,
                    raw_margin=raw_margin,
                    recent_tokens=recent_tokens,
                    fps=avg_fps,
                    latencies=latencies,
                    show_confidence=show_confidence,
                    show_diagnostics=show_diagnostics,
                    is_paused=False,
                )
            else:
                # Paused frame
                display_frame = draw_hud(
                    frame=display_frame,
                    temporal_result=temporal_result,
                    raw_pred_label=raw_label,
                    raw_confidence=raw_conf,
                    raw_margin=raw_margin,
                    recent_tokens=recent_tokens,
                    fps=0.0,
                    latencies=latencies,
                    show_confidence=show_confidence,
                    show_diagnostics=show_diagnostics,
                    is_paused=True,
                )

            if args.headless:
                # Run for 30 frames in headless mode then finish
                if frame_counter >= 30:
                    print("[INFO] Headless test run completed successfully.")
                    break
                continue

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                print("[INFO] User requested exit.")
                break
            elif key in (ord("p"), ord("P")):
                is_paused = not is_paused
                print(f"[INFO] Streaming {'PAUSED' if is_paused else 'RESUMED'}.")
            elif key in (ord("r"), ord("R")):
                engine.reset()
                recent_tokens.clear()
                print("[INFO] Temporal engine state & token history reset.")
            elif key in (ord("c"), ord("C")):
                show_confidence = not show_confidence
            elif key in (ord("d"), ord("D")):
                show_diagnostics = not show_diagnostics

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()
        print("[INFO] Clean shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
