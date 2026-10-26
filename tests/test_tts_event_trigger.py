"""Unit and integration tests for event-based TTS triggering in SignBridge.

Validates:
1. Same sign does not repeat every frame while held.
2. Transitioning to a new sign triggers speech once confirmed.
3. Returning to an earlier sign triggers speech again.
4. Cooldown gating suppresses rapid spam but allows spaced transitions.
5. TTS provider failures do not crash the application or pipeline.
6. Camera and processing loop remain non-blocking while speech is active.
7. Manual speak button explicitly re-speaks the current confirmed sign.
"""

import time
import pytest

from signbridge.pipeline.config import PipelineConfig
from signbridge.pipeline.orchestrator import SignBridgePipeline
from signbridge.tts.base import TTSStatus
from signbridge.tts.config import TTSConfig
from signbridge.tts.controller import SpeechController
from signbridge.tts.provider import FailingTTSProvider, MockTTSProvider


def test_same_sign_does_not_repeat_every_frame():
    """Verifies that holding the same sign in camera view does not re-trigger speech."""
    mock_provider = MockTTSProvider()
    config = TTSConfig(enabled=True, speak_stable_signs=True, cooldown_seconds=0.1)
    controller = SpeechController(config=config, provider=mock_provider)

    # First frame with HELLO: confirmed and spoken
    ok = controller.handle_stable_sign("HELLO")
    assert ok is True
    assert controller.last_spoken_token == "HELLO"

    # Simulate 30 subsequent frames with HELLO still present
    for _ in range(30):
        suppressed = controller.handle_stable_sign("HELLO")
        assert suppressed is False

    # Wait for background worker
    time.sleep(0.05)
    assert mock_provider.spoken_history == ["HELLO"]
    assert mock_provider.call_count == 1
    controller.stop()


def test_new_sign_triggers_speech():
    """Verifies that transitioning between distinct signs triggers speech for each sign."""
    mock_provider = MockTTSProvider()
    config = TTSConfig(enabled=True, speak_stable_signs=True, cooldown_seconds=0.01)
    controller = SpeechController(config=config, provider=mock_provider)

    # HELLO -> HELP -> YES
    assert controller.handle_stable_sign("HELLO") is True
    time.sleep(0.02)
    assert controller.handle_stable_sign("HELP") is True
    time.sleep(0.02)
    assert controller.handle_stable_sign("YES") is True

    time.sleep(0.05)
    assert mock_provider.spoken_history == ["HELLO", "HELP", "YES"]
    controller.stop()


def test_returning_to_earlier_sign_triggers_speech_again():
    """Verifies that HELLO -> HELP -> HELLO speaks HELLO again upon return."""
    mock_provider = MockTTSProvider()
    config = TTSConfig(enabled=True, speak_stable_signs=True, cooldown_seconds=0.01)
    controller = SpeechController(config=config, provider=mock_provider)

    # Sequence: HELLO -> HELP -> HELLO
    assert controller.handle_stable_sign("HELLO") is True
    time.sleep(0.02)
    assert controller.handle_stable_sign("HELP") is True
    time.sleep(0.02)
    assert controller.handle_stable_sign("HELLO") is True

    time.sleep(0.05)
    assert mock_provider.spoken_history == ["HELLO", "HELP", "HELLO"]

    # Also verify hand exit / reset clears state so same sign can trigger anew
    controller.reset_spoken_state()
    assert controller.last_spoken_token is None
    assert controller.handle_stable_sign("HELLO") is True

    time.sleep(0.05)
    assert mock_provider.spoken_history == ["HELLO", "HELP", "HELLO", "HELLO"]
    controller.stop()


def test_cooldown_works_as_expected():
    """Verifies that transitions faster than cooldown_seconds are suppressed, unless forced."""
    mock_provider = MockTTSProvider()
    config = TTSConfig(enabled=True, speak_stable_signs=True, cooldown_seconds=0.3)
    controller = SpeechController(config=config, provider=mock_provider)

    # Initial sign spoken
    assert controller.handle_stable_sign("HELLO") is True

    # Immediate transition to HELP within cooldown period (<0.3s) is suppressed
    assert controller.handle_stable_sign("HELP") is False

    # Wait past cooldown
    time.sleep(0.35)
    assert controller.handle_stable_sign("HELP") is True

    # Force flag bypasses cooldown
    assert controller.handle_stable_sign("HELP", force=True) is True

    time.sleep(0.05)
    assert mock_provider.spoken_history == ["HELLO", "HELP", "HELP"]
    controller.stop()


def test_tts_failure_does_not_crash_application():
    """Verifies that an unhandled TTS provider failure is safely caught and does not crash."""
    failing_provider = FailingTTSProvider()
    config = TTSConfig(enabled=True, speak_stable_signs=True)
    controller = SpeechController(config=config, provider=failing_provider)

    # handle_stable_sign and speak_manual must execute without crashing
    try:
        ok = controller.handle_stable_sign("HELLO")
        assert ok is True  # Successfully accepted into event queue
        ok_manual = controller.speak_manual("EMERGENCY")
        assert ok_manual is True
        # Allow worker thread to encounter the FailingTTSProvider exception
        time.sleep(0.08)
        # Worker thread must catch the exception safely and continue operating
        assert not controller.is_speaking
    except Exception as exc:
        pytest.fail(f"TTS failure raised unexpected unhandled exception: {exc}")
    finally:
        controller.stop()


def test_camera_remains_responsive_while_speech_is_playing():
    """Verifies that handle_stable_sign returns immediately without blocking frame processing."""
    # Provider with deliberate speaking delay
    slow_provider = MockTTSProvider(simulated_delay=0.4)
    config = TTSConfig(enabled=True, speak_stable_signs=True)
    controller = SpeechController(config=config, provider=slow_provider)

    t0 = time.perf_counter()
    ok = controller.handle_stable_sign("HELLO")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Handler must return near-instantly (well below camera frame time of 33ms)
    assert ok is True
    assert elapsed_ms < 20.0, f"Speech dispatch blocked for {elapsed_ms:.1f}ms (>20ms)"

    # Simulate camera loop maintaining 30 FPS while speech thread is busy
    loop_t0 = time.perf_counter()
    for _ in range(10):
        _ = controller.is_speaking
        time.sleep(0.001)
    loop_elapsed = (time.perf_counter() - loop_t0) * 1000.0
    assert loop_elapsed < 50.0

    controller.stop()


def test_manual_speak_button_speaks_current_prediction_again():
    """Verifies that manual speak actions re-speak the confirmed sign on demand."""
    mock_provider = MockTTSProvider()
    cfg = PipelineConfig()
    pipeline = SignBridgePipeline(config=cfg)
    pipeline.speech_controller.stop()
    pipeline.speech_controller.provider = mock_provider
    pipeline.speech_controller.config = TTSConfig(enabled=True, speak_stable_signs=True, cooldown_seconds=5.0)

    # Set active confirmed sign
    pipeline.last_stable_token = "HELLO"

    # 1. First manual trigger
    ok1 = pipeline.speak_current_stable_sign()
    assert ok1 is True

    # 2. Second manual trigger within cooldown (force=True must bypass cooldown)
    ok2 = pipeline.speak_current_stable_sign()
    assert ok2 is True

    # 3. Via speak_current_sentence fallback
    ok3 = pipeline.speak_current_sentence()
    assert ok3 is True

    time.sleep(0.1)
    assert mock_provider.spoken_history == ["HELLO", "HELLO", "HELLO"]
    pipeline.stop()
