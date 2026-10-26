"""SignBridge Stage 9: End-to-End Real-Time Translation Runtime.

Unifies:
Camera (Stage 2) -> MediaPipe (Stage 3) -> 94D Features (Stage 4) ->
Static Classifier (Stage 6) -> Temporal Stabilization (Stage 7) ->
Translation Engine (Stage 8) -> Developer HUD & Adaptive Skeleton.

Usage:
  python scripts/run_signbridge.py
  python scripts/run_signbridge.py --camera-index 0 --width 1280 --height 720
  python scripts/run_signbridge.py --headless --max-frames 60

Controls:
  [Q] / [ESC]   : Quit application
  [P]           : Pause / Resume frame processing
  [M]           : Toggle horizontal mirror flipping
  [SPACE]       : Commit word boundary (Stage 8 space)
  [BACKSPACE]   : Delete last letter or re-open preceding completed word
  [C]           : Clear entire sentence
  [R]           : Reset all translation state
  [U]           : Undo last translation action
"""

import argparse
from pathlib import Path
import sys
import time
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.camera.base import CameraStatus
from signbridge.core.config import CameraConfig, VisionConfig
from signbridge.pipeline import PipelineConfig, SignBridgePipeline
from signbridge.temporal.config import TemporalEngineConfig
from signbridge.translation.config import TranslationConfig
from signbridge.tts.config import TTSConfig
from signbridge.ui.app import SignBridgeApp
from signbridge.ui.state import ActiveModal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SignBridge Professional Translation Suite")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Application window width in pixels (default: 1280)")
    parser.add_argument("--height", type=int, default=800, help="Application window height in pixels (default: 800)")
    parser.add_argument("--fps", type=int, default=30, help="Target camera FPS (default: 30)")
    parser.add_argument("--mirror", action="store_true", default=True, help="Enable horizontal mirror flipping")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false", help="Disable mirror flipping")
    parser.add_argument("--dwell-ms", type=float, default=250.0, help="Temporal dwell threshold in ms")
    parser.add_argument("--pause-timeout", type=float, default=2.0, help="Inactivity pause timeout in seconds for word boundary")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice synthesis")
    parser.add_argument("--speak-words", action="store_true", default=True, help="Speak each completed word at word boundary (default: True)")
    parser.add_argument("--no-speak-words", dest="speak_words", action="store_false", help="Do not speak completed words automatically")
    parser.add_argument("--speak-letters", action="store_true", default=False, help="Speak each confirmed fingerspelled letter (default: False)")
    parser.add_argument("--demo", action="store_true", help="Launch in automated presentation demonstration mode")
    parser.add_argument("--high-contrast", action="store_true", help="Launch in high-contrast accessibility theme")
    parser.add_argument("--reduced-motion", action="store_true", help="Launch in reduced-motion mode")
    parser.add_argument("--onboarding", action="store_true", help="Display onboarding tutorial on launch")
    parser.add_argument("--headless", action="store_true", help="Run in headless verification mode (no GUI window)")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (useful for verification)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print("SIGNBRIDGE — STAGE 11 PROFESSIONAL APPLICATION SUITE")
    print("=" * 70)
    print(f"Device: Camera {args.camera_index} ({args.width}x{args.height} @ {args.fps} FPS, Mirror: {args.mirror})")
    print(f"Temporal Dwell: {args.dwell_ms} ms | Pause Boundary: {args.pause_timeout} s")
    print(f"Voice Output: {'Disabled' if args.no_voice else 'Enabled'} (Words: {args.speak_words}, Letters: {args.speak_letters})")
    print(f"Mode: {'Headless verification' if args.headless else 'Interactive GUI'}")
    print("=" * 70)

    # Build Pipeline Configuration
    pipeline_config = PipelineConfig(
        camera=CameraConfig(
            device_index=args.camera_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
            flip_horizontal=args.mirror,
        ),
        vision=VisionConfig(
            max_num_hands=2,
            min_detection_confidence=0.70,
            min_tracking_confidence=0.65,
        ),
        temporal=TemporalEngineConfig(
            minimum_dwell_ms=args.dwell_ms,
        ),
        translation=TranslationConfig(
            word_boundary_pause_seconds=args.pause_timeout,
        ),
        tts=TTSConfig(
            enabled=not args.no_voice,
            speak_words=args.speak_words,
            speak_letters=args.speak_letters,
            speak_sentences=True,
        ),
        min_classification_confidence=0.65,
        draw_hud=True,
        draw_skeleton=True,
        auto_reconnect=True,
    )

    pipeline = SignBridgePipeline(config=pipeline_config)

    if not args.headless:
        # Launch Professional Desktop UI
        app = SignBridgeApp(
            pipeline=pipeline,
            window_title="SignBridge — Professional Translation Suite",
            width=args.width,
            height=args.height,
            high_contrast=args.high_contrast,
            reduced_motion=args.reduced_motion,
            demo_mode=args.demo,
        )
        if args.onboarding:
            app.ui_state.open_modal(ActiveModal.ONBOARDING)

        return app.run(max_frames=args.max_frames)

    # Headless verification execution loop
    try:
        pipeline.start()
    except Exception as exc:
        print(f"[ERROR] Failed to start pipeline: {exc}")
        return 1

    print("[READY] Real-time translation loop active in headless mode.")
    frame_counter = 0
    try:
        while True:
            success, output = pipeline.process_next_frame(timeout=1.0)
            if not success or output is None:
                state = pipeline.get_state()
                if state.camera_status == CameraStatus.DISCONNECTED:
                    print("[WARNING] Camera disconnected.")
                    time.sleep(0.5)
                    continue
                time.sleep(0.01)
                continue

            frame_counter += 1
            if args.max_frames and frame_counter >= args.max_frames:
                print(f"[INFO] Processed requested {frame_counter} frames. Exiting cleanly.")
                break

            if frame_counter % 30 == 0:
                state = pipeline.get_state()
                print(
                    f"  [Frame {frame_counter:04d}] "
                    f"FPS: {state.camera_fps:.1f} (Proc: {state.processing_fps:.1f}) | "
                    f"Lat: {state.pipeline_latency:.1f} ms | "
                    f"Word: '{state.current_word}' | Sentence: '{state.sentence}'"
                )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        pipeline.stop()
        print("[INFO] SignBridge pipeline shut down cleanly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
