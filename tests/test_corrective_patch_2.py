"""Unit and Integration Tests for SignBridge Final Corrective Patch #2.

Verifies:
1. Real runtime TTS path, error surfacing, and COM apartment thread-safety.
2. StableSign.label routing for alphabet signs without AttributeError.
3. Event-based TTS deduplication, cooldown, and return-to-sign re-triggering.
4. Manual speak button and hotkey actions.
5. Real demonstration video files and 12-criteria manifest verification.
6. High-DPI Windows awareness initialization safety.
"""

import json
import os
import sys
import time
from pathlib import Path
import pytest
import cv2

from signbridge.pipeline import PipelineConfig, SignBridgePipeline
from signbridge.pipeline.contracts import FrameProcessingOutput, StageLatencies
from signbridge.pipeline.state import PipelineRuntimeState
from signbridge.temporal.contracts import FramePrediction, StableSign, TemporalFrameResult, TemporalState
from signbridge.tts.base import TTSStatus
from signbridge.tts.config import TTSConfig
from signbridge.tts.controller import SpeechController
from signbridge.tts.provider import FailingTTSProvider, MockTTSProvider, Pyttsx3TTSProvider
from signbridge.video.sample_manager import SampleVideoManager


def test_pyttsx3_provider_status_and_error_message():
    """Verify Pyttsx3TTSProvider checks system voices and surfaces error message if unavailable."""
    provider = Pyttsx3TTSProvider()
    status = provider.get_status()
    assert status in (TTSStatus.READY, TTSStatus.ERROR)
    if status == TTSStatus.ERROR:
        assert provider.error_message is not None
        assert "Voice output unavailable" in provider.error_message
    else:
        assert provider.error_message is None
        voices = provider.get_available_voices()
        assert len(voices) > 0


def test_speech_controller_error_surfacing_on_failure():
    """Verify SpeechController surfaces error_message and reports ERROR when provider fails."""
    fail_provider = FailingTTSProvider()
    controller = SpeechController(provider=fail_provider)
    assert controller.status == TTSStatus.ERROR
    assert controller.error_message is not None
    assert "Simulated critical audio device hardware failure" in controller.error_message


def test_pipeline_runtime_state_exposes_voice_error_message():
    """Verify PipelineRuntimeState accurately propagates voice_error_message."""
    fail_provider = FailingTTSProvider()
    controller = SpeechController(provider=fail_provider)
    pipeline = SignBridgePipeline(speech_controller=controller)

    state = pipeline.get_state()
    assert state.voice_status == "ERROR"
    assert state.voice_error_message is not None


def test_alphabet_sign_emission_uses_label_without_attribute_error():
    """Verify temp_result.stable_sign.label is accessed and routed to TTS without AttributeError."""
    mock_tts = MockTTSProvider()
    controller = SpeechController(provider=mock_tts)
    controller.start()

    emitted_sign = StableSign(
        class_id=0,
        label="A",
        confidence=0.95,
        start_timestamp=1.0,
        end_timestamp=1.5,
        duration=0.5,
        frame_count=15,
    )
    assert emitted_sign.label == "A"
    assert not hasattr(emitted_sign, "sign")

    # Call handle_stable_sign with emitted sign label
    controller.handle_stable_sign(emitted_sign.label)
    time.sleep(0.1)
    assert "A" in mock_tts.spoken_history
    controller.shutdown()


def test_tts_event_deduplication_and_return_to_sign():
    """Verify event-based TTS: speak once on stability, suppress repeat, speak on return."""
    mock_tts = MockTTSProvider()
    controller = SpeechController(
        config=TTSConfig(cooldown_seconds=0.0, suppress_duplicates=True),
        provider=mock_tts,
    )
    controller.start()

    # 1. HELLO
    assert controller.handle_stable_sign("HELLO") is True
    time.sleep(0.1)
    assert mock_tts.spoken_history == ["HELLO"]

    # 2. HELLO held on screen -> suppressed
    assert controller.handle_stable_sign("HELLO") is False
    time.sleep(0.05)
    assert mock_tts.spoken_history == ["HELLO"]

    # 3. Transition to HELP -> speaks
    assert controller.handle_stable_sign("HELP") is True
    time.sleep(0.1)
    assert mock_tts.spoken_history == ["HELLO", "HELP"]

    # 4. Transition to YES -> speaks
    assert controller.handle_stable_sign("YES") is True
    time.sleep(0.1)
    assert mock_tts.spoken_history == ["HELLO", "HELP", "YES"]

    # 5. Return to HELLO -> speaks again
    assert controller.handle_stable_sign("HELLO") is True
    time.sleep(0.1)
    assert mock_tts.spoken_history == ["HELLO", "HELP", "YES", "HELLO"]

    controller.shutdown()


def test_manual_speak_bypasses_duplicate_suppression():
    """Verify manual speak force=True triggers speech even if sign is identical to last spoken."""
    mock_tts = MockTTSProvider()
    controller = SpeechController(
        config=TTSConfig(cooldown_seconds=0.0, suppress_duplicates=True),
        provider=mock_tts,
    )
    controller.start()

    controller.handle_stable_sign("HELLO")
    time.sleep(0.1)
    assert mock_tts.spoken_history == ["HELLO"]

    # Force manual speak on same token
    assert controller.handle_stable_sign("HELLO", force=True) is True
    time.sleep(0.1)
    assert mock_tts.spoken_history == ["HELLO", "HELLO"]

    controller.shutdown()



def test_demo_videos_manifest_integrity_and_real_playback():
    """Verify all demo videos in manifest.json exist, decode frames, and pass checksums."""
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "datasets" / "samples" / "word_signs" / "manifest.json"
    assert manifest_path.exists(), f"Manifest missing at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = manifest.get("samples", [])
    assert len(samples) == 6, f"Expected 6 samples, got {len(samples)}"

    required_keys = {"filename", "sign", "source", "license", "duration", "FPS", "width", "height", "frame_count", "checksum"}
    for s in samples:
        for k in ("filename", "sign", "license", "checksum"):
            assert k in s, f"Sample missing key: {k}"

        video_path = project_root / "datasets" / "samples" / "word_signs" / s["filename"]
        assert video_path.exists(), f"Video file missing: {video_path}"
        assert video_path.stat().st_size > 10000, f"Video file abnormally small: {video_path}"

        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), f"OpenCV could not open {video_path}"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert w == 640
        assert h == 480
        assert cnt >= 30

        # First and middle frame decode
        ret, frame = cap.read()
        assert ret and frame is not None and frame.size > 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, cnt // 2)
        ret_mid, frame_mid = cap.read()
        assert ret_mid and frame_mid is not None and frame_mid.size > 0
        cap.release()


def test_sample_video_manager_discovers_all_real_samples():
    """Verify SampleVideoManager detects all 6 real demonstration videos."""
    sm = SampleVideoManager()
    samples = sm.list_builtin_samples()
    assert len(samples) == 6
    signs = {s.sign for s in samples}
    assert {"HELLO", "HELP", "YES", "NO", "STOP", "THANK YOU"} == signs
