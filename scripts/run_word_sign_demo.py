"""Interactive and headless demonstration script for ASL Word-Sign Recognition.

Supports:
1. Running built-in demonstration samples (--sample hello/help/yes/no/stop/thank_you).
2. Running user-provided video files (--video <path>).
3. Running live camera input (--camera).
4. Running headless batch mode (--headless) for automated verification.
5. Testing recognition modes (--mode ALPHABET | WORD_SIGNS | AUTO).

Usage examples:
  python scripts/run_word_sign_demo.py --sample hello --headless
  python scripts/run_word_sign_demo.py --sample help --headless
  python scripts/run_word_sign_demo.py --sample hello
"""

import argparse
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.pipeline.orchestrator import SignBridgePipeline
from signbridge.video.reader import VideoFileReader
from signbridge.video.sample_manager import SampleVideoManager
from signbridge.translation.contracts import TranslationEventType
from signbridge.word_sign.router import RecognitionMode


def parse_args():
    parser = argparse.ArgumentParser(description="SignBridge Word-Sign Recognition Demo")
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Name of built-in sample (hello, help, yes, no, stop, thank_you)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to custom video file (.mp4, .avi, .mov, etc.)",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Use live camera capture instead of video file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="WORD_SIGNS",
        choices=["ALPHABET", "WORD_SIGNS", "AUTO"],
        help="Recognition mode (default: WORD_SIGNS)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless CLI mode without opening an OpenCV window",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Maximum frames to process in headless mode (default: 60)",
    )
    return parser.parse_args()


def run_demo():
    args = parse_args()
    print("=" * 65)
    print("SIGNBRIDGE ASL WORD-SIGN RECOGNITION DEMO")
    print("=" * 65)

    mgr = SampleVideoManager()
    target_video_path = None

    if args.sample:
        normalized_sign = args.sample.upper().replace("_", " ")
        sample_rec = mgr.get_sample_by_sign(normalized_sign)
        if not sample_rec:
            # Try matching filename substring
            for s in mgr.list_builtin_samples():
                if args.sample.lower() in s.file.lower():
                    sample_rec = s
                    break

        if not sample_rec:
            print(f"[ERROR] Sample '{args.sample}' not found.")
            print(f"Available samples: {[s.sign for s in mgr.list_builtin_samples()]}")
            sys.exit(1)

        target_video_path = Path(sample_rec.full_path)
        print(f"Loaded built-in demonstration sample: {sample_rec.sign} ({sample_rec.file})")

    elif args.video:
        target_video_path = Path(args.video).resolve()
        if not target_video_path.exists():
            print(f"[ERROR] Video file does not exist: {target_video_path}")
            sys.exit(1)
        print(f"Loaded video file: {target_video_path}")

    elif not args.camera:
        # Default to hello sample if neither is specified
        sample_rec = mgr.get_sample_by_sign("HELLO")
        if sample_rec:
            target_video_path = Path(sample_rec.full_path)
            print(f"No source specified; defaulting to sample: {sample_rec.sign} ({sample_rec.file})")
        else:
            args.camera = True

    # Initialize Pipeline Orchestrator
    orchestrator = SignBridgePipeline()
    mode = RecognitionMode.from_string(args.mode)
    orchestrator.set_recognition_mode(mode)
    print(f"Recognition Mode set to: {mode.value}")

    if target_video_path:
        reader = VideoFileReader(
            str(target_video_path),
            loop=not args.headless,
            realtime_pacing=not args.headless,
        )
        orchestrator.set_camera_source(reader)

    orchestrator.start()
    print(f"Pipeline started successfully. Headless mode: {args.headless}")

    if args.headless:
        print("\nProcessing frames in headless mode...")
        detected_events = []
        start_time = time.time()

        for frame_no in range(1, args.max_frames + 1):
            ok, out = orchestrator.process_next_frame(timeout=0.2)
            if not ok or out is None:
                continue

            if out.routed_result and out.routed_result.is_newly_emitted and out.routed_result.token:
                detected_events.append((frame_no, "ROUTED_SIGN", out.routed_result.token, out.routed_result.confidence))
                print(f"  [Frame {frame_no:02d}] Routed Sign Emitted: {out.routed_result.token} (conf: {out.routed_result.confidence:.2f}, is_word: {out.routed_result.is_word})")
            elif out.temporal_result and out.temporal_result.stable_sign:
                detected_events.append((frame_no, "STABLE_SIGN", out.temporal_result.stable_sign.sign, out.temporal_result.stable_sign.confidence))
                print(f"  [Frame {frame_no:02d}] Stable Sign Emitted: {out.temporal_result.stable_sign.sign} (conf: {out.temporal_result.stable_sign.confidence:.2f})")

            if out.translation_result and out.translation_result.event_type == TranslationEventType.WORD_COMPLETED:
                print(f"  [Frame {frame_no:02d}] Translation Word Completed: '{out.translation_result.completed_words[-1] if out.translation_result.completed_words else ''}' | Sentence: '{out.translation_result.sentence_text}'")

        elapsed = time.time() - start_time
        print(f"\nCompleted {args.max_frames} frames in {elapsed:.2f}s ({args.max_frames / max(elapsed, 0.001):.1f} FPS)")
        orchestrator.stop()

        state = orchestrator.get_state()
        print("\nFinal Pipeline State:")
        print(f"  Current Word:     '{state.current_word}'")
        print(f"  Current Sentence: '{state.sentence}'")
        print(f"  Active Word-Sign: '{state.active_word_sign}' (conf: {state.word_sign_confidence:.2f})")
        print(f"  Completed Words:  {state.completed_words}")
        print("=" * 65)
        print("DEMO RUN COMPLETE [SUCCESS]")
        print("=" * 65)

    else:
        import cv2

        print("\nPress 'Q' to quit, 'M' to toggle recognition mode, 'R' to reset.")
        window_name = "SignBridge Word-Sign Recognition Demo"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 640)

        try:
            while orchestrator.is_running:
                ok, out = orchestrator.process_next_frame(timeout=0.033)
                if not ok or out is None or out.annotated_frame is None:
                    time.sleep(0.01)
                    continue

                display_img = out.annotated_frame.copy()

                # Add demo information overlay
                st = orchestrator.get_state()
                mode_str = st.recognition_mode
                curr_sign = st.active_word_sign or "-"
                curr_conf = st.word_sign_confidence
                curr_word = st.current_word or "-"
                curr_sentence = st.sentence or "-"

                cv2.rectangle(display_img, (10, 10), (450, 100), (20, 20, 20), -1)
                cv2.rectangle(display_img, (10, 10), (450, 100), (0, 200, 100), 1)
                cv2.putText(display_img, f"Mode: {mode_str} | Sign: {curr_sign} ({curr_conf:.2f})", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(display_img, f"Word: {curr_word}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)
                cv2.putText(display_img, f"Sentence: {curr_sentence}", (20, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1)

                cv2.imshow(window_name, display_img)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key in (ord("m"), ord("M")):
                    modes = [RecognitionMode.ALPHABET, RecognitionMode.WORD_SIGNS, RecognitionMode.AUTO]
                    current_idx = modes.index(orchestrator.recognition_mode)
                    next_mode = modes[(current_idx + 1) % len(modes)]
                    orchestrator.set_recognition_mode(next_mode)
                    print(f"Switched mode to: {next_mode.value}")
                elif key in (ord("r"), ord("R")):
                    orchestrator.reset()
                    print("Reset pipeline state.")
        finally:
            orchestrator.stop()
            cv2.destroyAllWindows()
            print("Demo finished.")


if __name__ == "__main__":
    run_demo()
