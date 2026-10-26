"""Integration tests verifying demo video recognition accuracy and state isolation.

Asserts that demo videos predict their intended signs (HELLO, HELP, YES, NO, STOP, THANK YOU)
and do NOT collapse to BAD.
"""

from pathlib import Path
import cv2
import pytest

from tests.test_pipeline import MockCameraSource
from signbridge.pipeline.config import PipelineConfig
from signbridge.pipeline.orchestrator import SignBridgePipeline
from signbridge.pipeline.contracts import FrameSource
from signbridge.video.reader import VideoFileReader
from signbridge.word_sign.router import RecognitionMode

DEMO_DIR = Path(__file__).resolve().parents[1] / "datasets" / "samples" / "word_signs"


@pytest.fixture(scope="module")
def pipeline():
    config = PipelineConfig(draw_hud=False, draw_skeleton=False)
    mock_cam = MockCameraSource(fps=30.0)
    pipe = SignBridgePipeline(config=config, camera=mock_cam)
    pipe.start()
    yield pipe
    pipe.stop()


@pytest.mark.parametrize(
    "video_filename,expected_sign",
    [
        ("hello.mp4", "HELLO"),
        ("help.mp4", "HELP"),
        ("yes.mp4", "YES"),
        ("no.mp4", "NO"),
        ("stop.mp4", "STOP"),
        ("thank_you.mp4", "THANK YOU"),
    ],
)
def test_demo_video_recognition_predicts_correct_sign(pipeline, video_filename, expected_sign):
    video_path = DEMO_DIR / video_filename
    assert video_path.exists(), f"Demo video missing: {video_path}"

    reader = VideoFileReader(video_path, loop=False, realtime_pacing=False)
    pipeline.switch_to_demo_video(reader)

    assert pipeline._active_source == FrameSource.DEMO_VIDEO
    assert pipeline.router.mode == RecognitionMode.WORD_SIGNS

    observed_active_words = []
    emitted_words = []

    # Stream all frames
    while True:
        success, output = pipeline.process_next_frame(timeout=0.1)
        if not success or output is None:
            break

        state = pipeline.get_state()
        if state.active_word_sign:
            observed_active_words.append(state.active_word_sign)

        if pipeline.last_stable_token:
            emitted_words.append(pipeline.last_stable_token)

    # 1. Verification: The intended sign MUST be recognized
    assert expected_sign in observed_active_words, (
        f"Video {video_filename} failed to recognize {expected_sign}. Observed: {set(observed_active_words)}"
    )

    # 2. Verification: The prediction must NOT be BAD (unless expected is BAD)
    if expected_sign != "BAD":
        bad_count = observed_active_words.count("BAD")
        total_count = len(observed_active_words)
        bad_ratio = bad_count / total_count if total_count > 0 else 0.0
        assert bad_ratio < 0.20, (
            f"Video {video_filename} collapsed to BAD ({bad_count}/{total_count} frames). "
            f"Observed: {set(observed_active_words)}"
        )

    # 3. Verification: The most frequent prediction across active frames is the expected sign
    most_frequent = max(set(observed_active_words), key=observed_active_words.count)
    assert most_frequent == expected_sign, (
        f"Video {video_filename} most frequent word was {most_frequent}, expected {expected_sign}"
    )


def test_switch_back_to_live_camera_resets_demo_state(pipeline):
    # Switch to demo video
    reader = VideoFileReader(DEMO_DIR / "hello.mp4", loop=False, realtime_pacing=False)
    pipeline.switch_to_demo_video(reader)
    pipeline.process_next_frame(timeout=0.1)

    # Switch back to live camera
    pipeline.switch_to_live_camera()
    assert pipeline._active_source == FrameSource.LIVE_CAMERA
    assert pipeline.router.mode == RecognitionMode.AUTO

    state = pipeline.get_state()
    assert state.active_word_sign is None or state.active_word_sign == ""
    assert state.frame_source == FrameSource.LIVE_CAMERA.value
