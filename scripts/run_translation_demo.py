"""Stage 8: Interactive Translation Engine Demo.

Provides both:
1. Default Interactive Console / Keyboard Mode:
   Type characters, spaces, backspaces, undo, clear, and reset to experience
   the Stage 8 Translation Engine in a deterministic CLI interface.
2. Optional Real-Time Camera Mode (--camera):
   Connects Camera (Stage 2) -> MediaPipe (Stage 3) -> Features (Stage 4) ->
   Model (Stage 6) -> TemporalEngine (Stage 7) -> TranslationEngine (Stage 8) -> Live HUD.

Controls (Interactive Console Mode):
  [A-Z]       : Append letter to current word
  SPACE       : Word boundary (commit word to sentence)
  BACKSPACE   : Delete last letter or re-open preceding completed word
  W           : Clear active word
  C           : Clear entire sentence
  U           : Undo last action
  R           : Reset all state
  Q / ESC     : Quit application
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

from signbridge.translation.config import TranslationConfig
from signbridge.translation.contracts import (
    StableSignToken,
    TranslationCommand,
    TranslationResult,
    TranslationState,
)
from signbridge.translation.engine import TranslationEngine


def render_console_hud(res: TranslationResult) -> None:
    """Prints a clean ASCII diagnostic display of the translation state."""
    print("\n" + "=" * 50)
    print("      SIGNBRIDGE — TRANSLATION ENGINE (STAGE 8)")
    print("=" * 50)
    print(f"CURRENT WORD:  '{res.current_word}'")
    print(f"SENTENCE:      '{res.sentence_text}'")
    print(f"COMPLETED:     {res.completed_words}")
    print(f"LAST EVENT:    {res.event_type.value}")
    print(f"LAST TOKEN:    '{res.last_token.label if res.last_token else '—'}'")
    print(f"TOKEN COUNT:   {res.tokens_count}")
    print(f"STATE:         {res.state.value}")
    print(f"LATENCY:       {res.processing_latency_ms:.4f} ms")
    print("-" * 50)
    print("Controls: [A-Z] Type | [SPACE] Word | [BS] Delete | [W] Clear Word | [C] Clear | [U] Undo | [R] Reset | [Q] Quit")
    print("=" * 50)


def run_interactive_cli(engine: TranslationEngine) -> int:
    """Runs interactive terminal loop for testing translation engine."""
    print("=" * 70)
    print("SIGNBRIDGE — STAGE 8 INTERACTIVE TRANSLATION ENGINE")
    print("=" * 70)
    print("Type letters or words, or use special commands:")
    print("  'SPACE' or ' '      -> word boundary")
    print("  'BACKSPACE' or 'BS' -> delete character / re-open word")
    print("  'CLEAR_WORD' or 'W' -> clear active word")
    print("  'CLEAR' or 'C'      -> clear sentence")
    print("  'UNDO' or 'U'       -> undo previous action")
    print("  'RESET' or 'R'      -> reset engine")
    print("  'Q' or 'QUIT'       -> exit demo")
    print("=" * 70)

    # Initial display
    render_console_hud(
        TranslationResult(
            state=engine.current_state,
            current_word=engine.current_word,
            sentence_text=engine.sentence_text,
            completed_words=engine.completed_words,
            event_type=engine.process_char("H").event_type,
            tokens_count=0,
        )
    )
    engine.reset()

    while True:
        try:
            raw_in = input("\nEnter command or text: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Exiting interactive demo.")
            break

        if not raw_in:
            continue

        upper_in = raw_in.upper()
        if upper_in in ("Q", "QUIT", "EXIT"):
            print("[INFO] Quitting.")
            break
        elif upper_in in ("SPACE", " "):
            res = engine.space()
        elif upper_in in ("BACKSPACE", "BS", "DEL"):
            res = engine.backspace()
        elif upper_in in ("CLEAR_WORD", "W"):
            res = engine.clear_word()
        elif upper_in in ("CLEAR", "C"):
            res = engine.clear_sentence()
        elif upper_in in ("UNDO", "U"):
            res = engine.undo()
        elif upper_in in ("RESET", "R"):
            res = engine.reset()
        else:
            # Treat as sequence of characters or word
            for char in raw_in:
                if char == " ":
                    res = engine.space()
                elif char.isalpha():
                    res = engine.process_char(char)
                else:
                    # Non-alpha
                    res = engine.process_char(char)

        render_console_hud(res)

    return 0


def run_camera_pipeline(args: argparse.Namespace) -> int:
    """Optional end-to-end integration connecting Camera -> MediaPipe -> Features -> Model -> Temporal -> Translation."""
    from signbridge.camera import CameraManager
    from signbridge.core.config import CameraConfig, VisionConfig
    from signbridge.features import HandFeatureExtractor
    from signbridge.model.predictor import get_default_predictor
    from signbridge.temporal import (
        FramePrediction,
        TemporalEngine,
        TemporalEngineConfig,
    )
    from signbridge.vision import MediaPipeHandDetector

    print("=" * 70)
    print("SIGNBRIDGE — STAGE 8 REAL-TIME TRANSLATION PIPELINE")
    print("=" * 70)

    cam_config = CameraConfig(device_index=args.camera_index, width=1280, height=720, fps=30)
    camera = CameraManager(cam_config)
    using_synthetic_camera = False

    try:
        camera.start()
        ok, test_packet = camera.read(timeout=1.5)
        if not ok or test_packet is None:
            raise RuntimeError("Camera timed out.")
    except Exception as exc:
        if args.headless:
            print(f"[INFO] Hardware camera unavailable ({exc}). Using synthetic frames for test.")
            using_synthetic_camera = True
        else:
            print(f"[ERROR] Could not open camera {args.camera_index}: {exc}")
            return 1

    detector = MediaPipeHandDetector(VisionConfig(max_num_hands=2))
    extractor = HandFeatureExtractor()
    predictor = get_default_predictor()
    temporal_engine = TemporalEngine(TemporalEngineConfig(minimum_dwell_ms=args.dwell_ms))
    translation_engine = TranslationEngine(TranslationConfig())

    window_name = "SignBridge — Stage 8 Translation Pipeline"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    print("[READY] Real-time translation loop active. Press 'Q' or 'ESC' to exit.")
    frame_counter = 0

    try:
        while True:
            current_ts = time.time()
            if using_synthetic_camera:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame, "SignBridge Stage 8 Headless Feed", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                time.sleep(0.033)
            else:
                ok, packet = camera.read(timeout=0.5)
                if not ok or packet is None or packet.data is None:
                    time.sleep(0.01)
                    continue
                frame = packet.data

            frame_counter += 1

            # Vision & Classifier
            det_res = detector.detect(frame, frame_id=frame_counter, timestamp=current_ts)
            primary_hand = det_res.primary_hand

            if primary_hand is not None:
                annotated = det_res.annotated_frame if det_res.annotated_frame is not None else frame
                feat_vec = extractor.extract(primary_hand, timestamp_ms=current_ts * 1000.0)
                if feat_vec is not None:
                    pred_res = predictor.predict(feat_vec.features)
                    frame_pred = FramePrediction.from_prediction_result(pred_res, current_ts, has_hand=True)
                else:
                    frame_pred = FramePrediction.no_hand(current_ts)
            else:
                annotated = frame
                frame_pred = FramePrediction.no_hand(current_ts)

            # Temporal Engine (Stage 7)
            temp_res = temporal_engine.process_frame(frame_pred)

            # If a stable sign was committed by Stage 7, feed it to Stage 8 Translation Engine!
            if temp_res.has_emitted_sign:
                assert temp_res.stable_sign is not None
                trans_res = translation_engine.process_token(temp_res.stable_sign)
                print(f"  >>> TRANSLATION EVENT: [{temp_res.stable_sign.label}] -> Word: '{trans_res.current_word}', Sentence: '{trans_res.sentence_text}'")

            # Render HUD
            h, w = annotated.shape[:2]
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
            cv2.rectangle(overlay, (0, h - 85), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

            # Top bar: Header & Status
            cv2.putText(annotated, "SIGNBRIDGE  |  STAGE 8 — TRANSLATION ENGINE", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 240, 255), 2, cv2.LINE_AA)
            state_str = f"STATE: {translation_engine.current_state.value}"
            cv2.putText(annotated, state_str, (w - 320, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2, cv2.LINE_AA)

            # Bottom bar: Current Word & Full Sentence
            word_str = f"ACTIVE WORD:  '{translation_engine.current_word}'"
            sent_str = f"SENTENCE:     '{translation_engine.sentence_text}'"
            cv2.putText(annotated, word_str, (25, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, sent_str, (25, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

            help_text = "[SPACE] Word | [BS] Del | [C] Clear | [U] Undo | [Q] Quit"
            cv2.putText(annotated, help_text, (w - 450, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

            if args.headless:
                if frame_counter >= 30:
                    print("[INFO] Headless camera pipeline verification completed.")
                    break
                continue

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == ord(" "):
                translation_engine.space()
                print("  [KEYBOARD ACTION]: SPACE")
            elif key in (8, 127):  # Backspace
                translation_engine.backspace()
                print("  [KEYBOARD ACTION]: BACKSPACE")
            elif key in (ord("c"), ord("C")):
                translation_engine.clear_sentence()
                print("  [KEYBOARD ACTION]: CLEAR_SENTENCE")
            elif key in (ord("w"), ord("W")):
                translation_engine.clear_word()
                print("  [KEYBOARD ACTION]: CLEAR_WORD")
            elif key in (ord("u"), ord("U")):
                translation_engine.undo()
                print("  [KEYBOARD ACTION]: UNDO")
            elif key in (ord("r"), ord("R")):
                translation_engine.reset()
                temporal_engine.reset()
                print("  [KEYBOARD ACTION]: RESET")

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()
        print("[INFO] Pipeline closed cleanly.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SignBridge Stage 8: Translation Engine Interactive Demo")
    parser.add_argument("--camera", action="store_true", help="Launch real-time camera translation pipeline instead of CLI mode")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--dwell-ms", type=float, default=250.0, help="Minimum dwell time in ms (default: 250.0)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows (for testing)")
    args = parser.parse_args()

    if args.camera:
        return run_camera_pipeline(args)
    else:
        config = TranslationConfig()
        engine = TranslationEngine(config)
        return run_interactive_cli(engine)


if __name__ == "__main__":
    sys.exit(main())
