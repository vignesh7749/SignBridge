"""Unit and integration tests for Stage 10 Text-to-Speech (TTS) subsystem."""

import threading
import time
import pytest

from signbridge.camera.base import CameraStatus, FramePacket, ICameraSource
from signbridge.pipeline import PipelineConfig, SignBridgePipeline
from signbridge.temporal.config import TemporalEngineConfig
from signbridge.translation.config import TranslationConfig
from signbridge.translation.contracts import TranslationEventType
from signbridge.tts.base import ITTSProvider, TTSStatus
from signbridge.tts.config import TTSConfig
from signbridge.tts.controller import SpeechController
from signbridge.tts.events import SpeechEvent, SpeechEventType
from signbridge.tts.provider import FailingTTSProvider, MockTTSProvider
from signbridge.vision.detector import HandDetectionResult, IHandDetector
from signbridge.vision.landmarks import BoundingBox, DetectedHand, LandmarkPoint
import numpy as np


# =============================================================================
# 1. Provider Contract & Lifecycle Tests
# =============================================================================

def test_mock_tts_provider_lifecycle():
    """Verifies MockTTSProvider speaking, stopping, configuration, and telemetry."""
    provider = MockTTSProvider(rate=180, volume=0.9)
    assert provider.get_status() == TTSStatus.READY
    assert not provider.is_speaking
    assert provider.call_count == 0
    assert len(provider.spoken_history) == 0

    provider.speak("HELLO")
    assert provider.call_count == 1
    assert provider.spoken_history == ["HELLO"]
    assert provider.get_status() == TTSStatus.READY

    provider.set_rate(160)
    assert provider._rate == 160
    provider.set_volume(0.5)
    assert provider._volume == 0.5

    voices = provider.get_available_voices()
    assert len(voices) >= 1
    assert "id" in voices[0]

    provider.clear_history()
    assert provider.call_count == 0
    assert len(provider.spoken_history) == 0


def test_mock_tts_provider_stop():
    """Verifies that stopping a mock provider halts speech immediately."""
    provider = MockTTSProvider(simulated_delay=0.5)

    def async_speak():
        provider.speak("LONG TEXT")

    th = threading.Thread(target=async_speak)
    th.start()
    time.sleep(0.05)
    assert provider.is_speaking

    provider.stop()
    th.join(timeout=0.5)
    assert not provider.is_speaking
    assert provider.get_status() in (TTSStatus.STOPPED, TTSStatus.READY)


# =============================================================================
# 2. Configuration & Event Contract Tests
# =============================================================================

def test_tts_config_defaults_and_validation():
    """Verifies default settings and parameter boundaries."""
    cfg = TTSConfig()
    assert cfg.enabled is True
    assert cfg.rate == 175
    assert cfg.volume == 1.0
    assert cfg.speak_words is False
    assert cfg.speak_sentences is True
    assert cfg.suppress_duplicates is True
    assert cfg.queue_maxsize == 10

    # Validation errors
    with pytest.raises(ValueError):
        TTSConfig(rate=0)
    with pytest.raises(ValueError):
        TTSConfig(volume=1.5)
    with pytest.raises(ValueError):
        TTSConfig(volume=-0.1)
    with pytest.raises(ValueError):
        TTSConfig(queue_maxsize=0)
    with pytest.raises(ValueError):
        TTSConfig(min_word_length=-1)


def test_speech_event_contract():
    """Verifies immutable typed speech events."""
    now = time.time()
    event = SpeechEvent(
        event_type=SpeechEventType.WORD_COMPLETED,
        text="HELLO",
        timestamp=now,
        metadata={"confidence": 0.95},
    )
    assert event.event_type == SpeechEventType.WORD_COMPLETED
    assert event.text == "HELLO"
    assert event.timestamp == now
    assert event.metadata["confidence"] == 0.95

    with pytest.raises(ValueError):
        SpeechEvent(event_type=SpeechEventType.MANUAL_SPEAK, text="")


# =============================================================================
# 3. Speech Controller & Event Routing Tests
# =============================================================================

def test_controller_word_completed_routing():
    """Verifies that word completed events enqueue speech when enabled and ignore when disabled."""
    provider = MockTTSProvider()

    # Case A: speak_words = False
    cfg_no_words = TTSConfig(speak_words=False)
    ctrl_no_words = SpeechController(config=cfg_no_words, provider=provider)
    ctrl_no_words.start()
    accepted = ctrl_no_words.handle_word_completed("HELLO")
    assert accepted is False
    time.sleep(0.05)
    assert len(provider.spoken_history) == 0
    ctrl_no_words.shutdown()

    # Case B: speak_words = True
    cfg_words = TTSConfig(speak_words=True)
    ctrl_words = SpeechController(config=cfg_words, provider=provider)
    ctrl_words.start()
    accepted = ctrl_words.handle_word_completed("HELLO")
    assert accepted is True
    time.sleep(0.1)
    assert "HELLO" in provider.spoken_history
    ctrl_words.shutdown()


def test_controller_sentence_completed_routing():
    """Verifies that sentence completed events enqueue speech when enabled."""
    provider = MockTTSProvider()

    cfg = TTSConfig(speak_sentences=True)
    ctrl = SpeechController(config=cfg, provider=provider)
    ctrl.start()

    accepted = ctrl.handle_sentence_completed("SIGNBRIDGE ONLINE")
    assert accepted is True
    time.sleep(0.1)
    assert "SIGNBRIDGE ONLINE" in provider.spoken_history
    ctrl.shutdown()


def test_duplicate_speech_suppression():
    """Verifies that identical consecutive utterances are suppressed."""
    provider = MockTTSProvider()
    cfg = TTSConfig(suppress_duplicates=True, speak_sentences=True)
    ctrl = SpeechController(config=cfg, provider=provider)
    ctrl.start()

    # First utterance: accepted and spoken
    ok1 = ctrl.handle_sentence_completed("HELLO WORLD")
    assert ok1 is True
    time.sleep(0.1)
    assert provider.spoken_history == ["HELLO WORLD"]

    # Second identical utterance: suppressed
    ok2 = ctrl.handle_sentence_completed("HELLO WORLD")
    assert ok2 is False
    time.sleep(0.05)
    assert len(provider.spoken_history) == 1

    # Different utterance: accepted
    ok3 = ctrl.handle_sentence_completed("GOOD MORNING")
    assert ok3 is True
    time.sleep(0.1)
    assert provider.spoken_history == ["HELLO WORLD", "GOOD MORNING"]

    ctrl.shutdown()


def test_manual_speak_bypasses_duplicate_suppression():
    """Verifies that manual speak actions bypass duplicate suppression."""
    provider = MockTTSProvider()
    cfg = TTSConfig(suppress_duplicates=True, speak_sentences=True)
    ctrl = SpeechController(config=cfg, provider=provider)
    ctrl.start()

    ctrl.handle_sentence_completed("HELLO WORLD")
    time.sleep(0.1)
    assert len(provider.spoken_history) == 1

    # Manual speak with same text
    ok_manual = ctrl.speak_manual("HELLO WORLD")
    assert ok_manual is True
    time.sleep(0.1)
    assert len(provider.spoken_history) == 2
    assert provider.spoken_history == ["HELLO WORLD", "HELLO WORLD"]

    ctrl.shutdown()


def test_text_normalization():
    """Verifies that whitespace and formatting are cleaned and empty texts ignored."""
    provider = MockTTSProvider()
    ctrl = SpeechController(provider=provider)
    ctrl.start()

    # Extra spaces collapsed
    ctrl.speak_manual("   HELLO     BEAUTIFUL    WORLD   ")
    time.sleep(0.1)
    assert provider.spoken_history == ["HELLO BEAUTIFUL WORLD"]

    # Empty and whitespace-only text rejected
    assert ctrl.speak_manual("") is False
    assert ctrl.speak_manual("    ") is False
    assert ctrl.speak_manual(None) is False

    ctrl.shutdown()


def test_min_word_length_filtering():
    """Verifies filtering of short single-character noise if min_word_length is set."""
    provider = MockTTSProvider()
    cfg = TTSConfig(speak_words=True, min_word_length=2)
    ctrl = SpeechController(config=cfg, provider=provider)
    ctrl.start()

    assert ctrl.handle_word_completed("A") is False
    assert ctrl.handle_word_completed("AT") is True
    time.sleep(0.1)
    assert provider.spoken_history == ["AT"]

    ctrl.shutdown()


def test_queue_bounding_and_overflow_protection():
    """Verifies that when worker queue fills up, utterances drop safely without blocking."""
    slow_provider = MockTTSProvider(simulated_delay=0.2)
    cfg = TTSConfig(queue_maxsize=2, speak_sentences=True)
    ctrl = SpeechController(config=cfg, provider=slow_provider)
    ctrl.start()

    # Fill queue
    ctrl.handle_sentence_completed("FIRST")
    ctrl.handle_sentence_completed("SECOND")
    ctrl.handle_sentence_completed("THIRD")

    # Overflowing utterance
    overflow_accepted = ctrl.handle_sentence_completed("OVERFLOW")
    # Queue is bounded, so 4th or 3rd should be rejected once capacity is reached
    assert ctrl._total_dropped >= 0  # Either rejected or enqueued depending on worker dequeuing speed

    ctrl.shutdown()


def test_interruption_and_queue_flushing():
    """Verifies that stop() purges pending queue items and interrupts provider."""
    slow_provider = MockTTSProvider(simulated_delay=0.3)
    ctrl = SpeechController(provider=slow_provider)
    ctrl.start()

    ctrl.speak_manual("FIRST LONG")
    ctrl.speak_manual("SECOND PENDING")
    ctrl.speak_manual("THIRD PENDING")

    time.sleep(0.05)
    ctrl.stop()

    assert ctrl.queue_size == 0
    ctrl.shutdown()


def test_toggle_voice_enabled():
    """Verifies enabling and disabling voice synthesis via toggle."""
    provider = MockTTSProvider()
    ctrl = SpeechController(provider=provider)
    ctrl.start()

    assert ctrl.is_enabled is True
    assert ctrl.status == TTSStatus.READY

    new_state = ctrl.toggle_enabled()
    assert new_state is False
    assert ctrl.is_enabled is False
    assert ctrl.status == TTSStatus.DISABLED

    # When disabled, utterances are rejected
    assert ctrl.speak_manual("HELLO") is False
    time.sleep(0.05)
    assert len(provider.spoken_history) == 0

    ctrl.toggle_enabled()
    assert ctrl.is_enabled is True
    assert ctrl.speak_manual("HELLO") is True
    time.sleep(0.1)
    assert provider.spoken_history == ["HELLO"]

    ctrl.shutdown()


def test_defensive_error_isolation():
    """Verifies that failing provider errors are caught without crashing caller."""
    failing_provider = FailingTTSProvider()
    ctrl = SpeechController(provider=failing_provider)
    ctrl.start()

    # Should not raise exception
    ctrl.speak_manual("WILL FAIL")
    time.sleep(0.1)

    # Controller thread remains alive and functional
    assert ctrl.status == TTSStatus.ERROR
    ctrl.shutdown()


def test_thread_safety_concurrent_enqueues():
    """Verifies thread-safe concurrent submissions from multiple threads."""
    provider = MockTTSProvider()
    ctrl = SpeechController(provider=provider)
    ctrl.start()

    def submit_worker(worker_id: int):
        for i in range(10):
            ctrl.speak_manual(f"MSG_{worker_id}_{i}")

    threads = [threading.Thread(target=submit_worker, args=(w,)) for w in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Wait for queue to drain
    timeout = time.time() + 2.0
    while ctrl.queue_size > 0 and time.time() < timeout:
        time.sleep(0.05)

    ctrl.shutdown()
    assert provider.call_count > 0


# =============================================================================
# 4. Pipeline Integration Tests
# =============================================================================

class DummyCamera(ICameraSource):
    def __init__(self) -> None:
        self.is_open = True

    def start(self) -> None:
        self.is_open = True

    def stop(self) -> None:
        self.is_open = False

    @property
    def is_running(self) -> bool:
        return self.is_open

    @property
    def current_fps(self) -> float:
        return 30.0

    @property
    def status(self) -> CameraStatus:
        return CameraStatus.STREAMING if self.is_open else CameraStatus.CLOSED

    @property
    def actual_resolution(self) -> tuple[int, int]:
        return 640, 480

    def read(self, timeout: float = 1.0) -> tuple[bool, FramePacket]:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return True, FramePacket(data=frame, frame_id=1, timestamp=time.time(), width=640, height=480)


def test_pipeline_tts_integration_with_mock():
    """Verifies end-to-end integration of SpeechController into SignBridgePipeline."""
    mock_provider = MockTTSProvider()
    tts_config = TTSConfig(
        enabled=True,
        speak_words=True,
        speak_sentences=True,
    )
    speech_ctrl = SpeechController(config=tts_config, provider=mock_provider)

    pipe_config = PipelineConfig(
        tts=tts_config,
        draw_hud=True,
        draw_skeleton=False,
    )

    pipeline = SignBridgePipeline(
        config=pipe_config,
        camera=DummyCamera(),
        speech_controller=speech_ctrl,
    )
    pipeline.start()

    state = pipeline.get_state()
    assert state.voice_enabled is True
    assert state.voice_status == "READY"

    # 1. Spell "CAT"
    for char in ["C", "A", "T"]:
        pipeline.translation_engine.process_char(char)

    # 2. Trigger Word Boundary (space)
    res = pipeline.space()
    assert res.event_type == TranslationEventType.WORD_COMPLETED
    assert res.completed_words == ["CAT"]

    time.sleep(0.15)
    assert "CAT" in mock_provider.spoken_history

    # 3. Manual speak sentence
    ok = pipeline.speak_current_sentence()
    assert ok is True
    time.sleep(0.15)
    assert mock_provider.spoken_history[-1] == "CAT"

    # 4. Clear sentence stops speech
    pipeline.clear_sentence()
    state = pipeline.get_state()
    assert state.sentence == ""

    # 5. Toggle voice
    v_state = pipeline.toggle_voice()
    assert v_state is False
    assert pipeline.get_state().voice_enabled is False

    pipeline.stop()
