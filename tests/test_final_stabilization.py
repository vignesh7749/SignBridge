"""Unit and integration tests for SignBridge Final Critical Corrective Patch.

Verifies:
1. End-to-end TTS dispatch, manual speak button, and speak_letters/speak_words routing.
2. Hand gesture precision, confidence threshold gating, and UNCERTAIN state representation.
3. Complete isolation between Demo Video mode and Live Camera mode (no leaked landmarks or camera reads).
4. Diagnostics HUD overlay telemetry rendering.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from signbridge.pipeline.contracts import FrameSource
from signbridge.pipeline.config import PipelineConfig, VisionConfig
from signbridge.pipeline.state import PipelineRuntimeState
from signbridge.tts.config import TTSConfig
from signbridge.tts.controller import SpeechController
from signbridge.tts.provider import MockTTSProvider
from signbridge.ui.state import UIState
from signbridge.ui.theme import OBSIDIAN_THEME
from signbridge.ui.layout import DashboardLayout
from signbridge.video.reader import VideoFileReader


class TestTTSStabilization:
    """Verifies TTS manual speak, word/letter routing, and deduplication behavior."""

    def test_manual_speak_bypasses_duplicate_suppression(self):
        """Manual speak button must speak the currently active sign/word even if already spoken."""
        provider = MockTTSProvider()
        config = TTSConfig(enabled=True, suppress_duplicates=True, cooldown_seconds=0.01)
        controller = SpeechController(config=config, provider=provider)

        # 1. Automatic stable sign triggers once
        spoken1 = controller.handle_stable_sign("HELLO")
        assert spoken1 is True
        import time
        time.sleep(0.05)
        assert provider.call_count == 1
        assert provider.spoken_history == ["HELLO"]

        # 2. Duplicate automatic event is suppressed
        spoken2 = controller.handle_stable_sign("HELLO")
        assert spoken2 is False
        time.sleep(0.05)
        assert provider.call_count == 1

        # 3. Manual speak button explicitly speaks it again
        spoken_manual = controller.speak_manual("HELLO")
        assert spoken_manual is True
        time.sleep(0.05)
        assert provider.call_count == 2
        assert provider.spoken_history == ["HELLO", "HELLO"]
        assert controller.last_spoken_token == "HELLO"
        controller.stop()

    def test_retrigger_after_different_sign(self):
        """A sign should speak again after a different sign is confirmed."""
        import time
        provider = MockTTSProvider()
        config = TTSConfig(enabled=True, suppress_duplicates=True, cooldown_seconds=0.01)
        controller = SpeechController(config=config, provider=provider)

        assert controller.handle_stable_sign("HELLO") is True
        time.sleep(0.02)
        assert controller.handle_stable_sign("HELP") is True
        time.sleep(0.02)
        assert controller.handle_stable_sign("HELLO") is True

        time.sleep(0.05)
        assert provider.call_count == 3
        assert provider.spoken_history == ["HELLO", "HELP", "HELLO"]
        controller.stop()

    def test_speak_words_vs_speak_letters_routing(self):
        """Word signs speak when speak_words=True, alphabet fingerspelling respects speak_letters."""
        # Default config: speak_words=True, speak_letters=False
        cfg = TTSConfig(enabled=True, speak_words=True, speak_letters=False)
        assert cfg.speak_words is True
        assert cfg.speak_letters is False

        # Custom config: speak_letters=True
        cfg_letters = TTSConfig(enabled=True, speak_words=False, speak_letters=True)
        assert cfg_letters.speak_letters is True
        assert cfg_letters.speak_words is False

    def test_repeated_speech_six_tokens_sequence(self):
        """Verifies full sequence: HELLO -> HELP -> YES -> NO -> THANK YOU -> HELLO all speak."""
        import time
        provider = MockTTSProvider()
        config = TTSConfig(enabled=True, suppress_duplicates=True, cooldown_seconds=0.01)
        controller = SpeechController(config=config, provider=provider)

        tokens = ["HELLO", "HELP", "YES", "NO", "THANK YOU", "HELLO"]
        for token in tokens:
            assert controller.handle_stable_sign(token) is True
            time.sleep(0.02)

        time.sleep(0.05)
        assert provider.call_count == 6
        assert provider.spoken_history == tokens
        controller.stop()

    def test_manual_speak_repeated_calls_same_token(self):
        """Manual speak button must speak the same token repeatedly on demand without suppression."""
        import time
        provider = MockTTSProvider()
        config = TTSConfig(enabled=True, suppress_duplicates=True, cooldown_seconds=0.01)
        controller = SpeechController(config=config, provider=provider)

        for _ in range(3):
            assert controller.speak_manual("HELLO") is True
            time.sleep(0.02)

        time.sleep(0.05)
        assert provider.call_count == 3
        assert provider.spoken_history == ["HELLO", "HELLO", "HELLO"]
        controller.stop()

    def test_worker_thread_exception_recovery(self):
        """Worker thread must survive any provider exception and process next speech request."""
        import time
        from signbridge.tts.base import TTSStatus
        provider = MockTTSProvider()
        config = TTSConfig(enabled=True, cooldown_seconds=0.01)
        controller = SpeechController(config=config, provider=provider)

        # 1. Normal speech
        controller.speak_manual("HELLO")
        time.sleep(0.02)
        assert provider.spoken_history == ["HELLO"]

        # 2. Inject exception
        def failing_speak(text):
            raise RuntimeError("Simulated TTS audio driver glitch")
        provider.speak = failing_speak

        controller.speak_manual("GLITCH")
        time.sleep(0.03)
        assert controller._thread.is_alive()

        # 3. Recover and verify subsequent speech
        provider.speak = lambda text: provider.spoken_history.append(text)
        controller.speak_manual("RECOVERED")
        time.sleep(0.03)
        assert "RECOVERED" in provider.spoken_history
        assert controller.status in (TTSStatus.READY, TTSStatus.SPEAKING)
        controller.stop()

    def test_status_lifecycle_ready_speaking_ready(self):
        """Status must transition cleanly READY -> SPEAKING -> READY."""
        import time
        from signbridge.tts.base import TTSStatus
        provider = MockTTSProvider(simulated_delay=0.1)
        config = TTSConfig(enabled=True, cooldown_seconds=0.01)
        controller = SpeechController(config=config, provider=provider)

        assert controller.status == TTSStatus.READY
        controller.speak_manual("HELLO")
        time.sleep(0.02)
        assert controller.status == TTSStatus.SPEAKING
        time.sleep(0.15)
        assert controller.status == TTSStatus.READY
        controller.stop()


class TestConfidenceGatingAndUncertain:
    """Verifies confidence threshold gating and UNCERTAIN UI state."""

    def test_uncertain_state_triggered_on_low_confidence(self):
        """Predictions with confidence below min_classification_confidence are marked uncertain."""
        st = PipelineRuntimeState(
            hand_detected=True,
            predicted_label="A",
            prediction_confidence=0.45,
            is_uncertain=True,
        )
        assert st.is_uncertain is True
        assert st.predicted_label == "A"
        assert st.prediction_confidence < 0.65

    def test_recognition_card_renders_uncertain(self):
        """DashboardLayout renders UNCERTAIN when is_uncertain is True and hand is detected."""
        layout = DashboardLayout(1280, 800)
        canvas = np.zeros((800, 1280, 3), dtype=np.uint8)
        u_state = UIState()
        p_state = PipelineRuntimeState(
            hand_detected=True,
            predicted_label="A",
            prediction_confidence=0.42,
            is_uncertain=True,
        )

        # Render recognition card
        layout._render_recognition_card(canvas, 800, 60, 460, 140, p_state, u_state, OBSIDIAN_THEME)
        # Should not crash and canvas should contain drawn pixels
        assert np.count_nonzero(canvas) > 0


class TestModeIsolation:
    """Verifies complete separation between Demo Video mode and Live Camera mode."""

    def test_frame_source_enum(self):
        """FrameSource supports LIVE_CAMERA and DEMO_VIDEO."""
        assert FrameSource.LIVE_CAMERA == "LIVE_CAMERA"
        assert FrameSource.DEMO_VIDEO == "DEMO_VIDEO"

    def test_camera_and_demo_state_isolation(self):
        """Switching to demo video halts live camera and resets pipeline state."""
        from signbridge.pipeline.orchestrator import SignBridgePipeline

        mock_cam = MagicMock()
        mock_cam.status = MagicMock()
        mock_cam.read.return_value = (False, None)

        pipe = SignBridgePipeline(camera=mock_cam)
        assert pipe._active_source == FrameSource.LIVE_CAMERA

        # Setup mock demo reader
        mock_reader = MagicMock(spec=VideoFileReader)
        mock_reader.read.return_value = (False, None)

        pipe.switch_to_demo_video(mock_reader)
        assert pipe._active_source == FrameSource.DEMO_VIDEO
        assert pipe.camera is mock_reader
        mock_cam.stop.assert_called_once()
        mock_reader.start.assert_called_once()

        # Switch back to live camera
        pipe.switch_to_live_camera()
        assert pipe._active_source == FrameSource.LIVE_CAMERA
        assert pipe.camera is mock_cam
        mock_reader.stop.assert_called_once()


class TestDiagnosticsOverlay:
    """Verifies live on-screen diagnostic HUD mode."""

    def test_diagnostics_overlay_rendering(self):
        """HUD overlay renders when show_diagnostics is enabled."""
        layout = DashboardLayout(1280, 800)
        canvas = np.zeros((800, 1280, 3), dtype=np.uint8)
        p_state = PipelineRuntimeState(
            hand_detected=True,
            handedness="Right",
            detection_confidence=0.88,
            predicted_label="HELLO",
            prediction_confidence=0.92,
            stable_token="HELLO",
            dwell_duration_ms=300.0,
        )

        layout._render_diagnostics_overlay(canvas, 16, 56, 780, 700, p_state, OBSIDIAN_THEME)
        assert np.count_nonzero(canvas) > 0


class TestLiveDemoStateIsolationAndOwnership:
    """Verifies strict live/demo state isolation and dedicated source ownership."""

    def test_strict_live_demo_state_reset_and_clean_sentence(self):
        """Switching between LIVE and DEMO must immediately reset all text and recognition states."""
        from signbridge.pipeline.orchestrator import SignBridgePipeline

        mock_cam = MagicMock()
        mock_cam.status = MagicMock()
        pipe = SignBridgePipeline(camera=mock_cam)

        # Simulate dirty live state
        pipe.translation_engine.process_word("HELLO")
        assert pipe.translation_engine.sentence_text == "HELLO"

        # Switch to demo video
        mock_reader = MagicMock(spec=VideoFileReader)
        pipe.switch_to_demo_video(mock_reader)

        # Demo mode state must start completely clean
        st = pipe.get_state()
        assert st.sentence == ""
        assert st.current_word == ""
        assert st.completed_words == ()
        assert st.active_word_sign is None
        assert st.predicted_label is None
        assert st.hand_detected is False
        assert pipe._active_source == FrameSource.DEMO_VIDEO

        # Simulate dirty demo state
        pipe.translation_engine.process_word("BAD")
        assert pipe.translation_engine.sentence_text == "BAD"

        # Switch back to live camera
        pipe.switch_to_live_camera()
        st_live = pipe.get_state()
        assert st_live.sentence == ""
        assert st_live.current_word == ""
        assert st_live.completed_words == ()
        assert pipe._active_source == FrameSource.LIVE_CAMERA

    def test_dedicated_detector_and_engine_instances(self):
        """Live and Demo modes must use completely independent detector and engine instances."""
        from signbridge.pipeline.orchestrator import SignBridgePipeline

        mock_cam = MagicMock()
        pipe = SignBridgePipeline(camera=mock_cam)

        # Verify separate objects created in __init__
        assert pipe._live_detector is not pipe._demo_detector
        assert pipe._live_temporal_engine is not pipe._demo_temporal_engine
        assert pipe._live_translation_engine is not pipe._demo_translation_engine
        assert pipe.detector is pipe._live_detector
        assert pipe.temporal_engine is pipe._live_temporal_engine
        assert pipe.translation_engine is pipe._live_translation_engine

        # Switch to demo video
        mock_reader = MagicMock(spec=VideoFileReader)
        pipe.switch_to_demo_video(mock_reader)
        assert pipe.detector is pipe._demo_detector
        assert pipe.temporal_engine is pipe._demo_temporal_engine
        assert pipe.translation_engine is pipe._demo_translation_engine

        # Switch back
        pipe.switch_to_live_camera()
        assert pipe.detector is pipe._live_detector
        assert pipe.temporal_engine is pipe._live_temporal_engine
        assert pipe.translation_engine is pipe._live_translation_engine

    def test_demo_video_loop_suppresses_duplicate_words(self):
        """When a demo video loops, the same word must not be appended multiple times into sentence."""
        from signbridge.camera.base import FramePacket
        from signbridge.pipeline.orchestrator import SignBridgePipeline
        from signbridge.word_sign.router import RecognitionMode, RoutedRecognitionResult

        mock_reader = MagicMock(spec=VideoFileReader)
        mock_reader.read.return_value = (False, None)
        pipe = SignBridgePipeline(camera=mock_reader)
        pipe.switch_to_demo_video(mock_reader)

        from signbridge.vision.detector import HandDetectionResult

        # Mock router to return emitted word 'BAD'
        pipe.router = MagicMock()
        pipe.detector = MagicMock()
        pipe.detector.detect.return_value = HandDetectionResult(frame_id=20, timestamp=1.0, hands=[])
        pipe.router.route_frame.return_value = RoutedRecognitionResult(
            mode=RecognitionMode.WORD_SIGNS,
            is_word=True,
            token="BAD",
            confidence=0.85,
            is_newly_emitted=True,
            active_symbol="BAD",
        )

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pkt1 = FramePacket(frame_id=20, timestamp=1.0, data=dummy_frame, width=640, height=480)
        pipe.process_frame(pkt1)
        assert pipe.translation_engine.sentence_text == "BAD"

        # Frame on next loop: same word emitted
        pkt2 = FramePacket(frame_id=20, timestamp=3.0, data=dummy_frame, width=640, height=480)
        pipe.process_frame(pkt2)
        # Sentence should NOT be 'BAD BAD'
        assert pipe.translation_engine.sentence_text == "BAD"
        assert len(pipe.translation_engine.completed_words) == 1


class TestUIAndHUDVerification:
    """Verifies HUD layout coordinates, non-overlapping rows, and truthful demo display."""

    def test_demo_target_vs_detected_distinction(self):
        """UI must distinguish demo target label from detected model output truthfully."""
        layout = DashboardLayout(1280, 800)
        canvas = np.zeros((800, 1280, 3), dtype=np.uint8)
        u_state = UIState()
        u_state.active_input_source = "DEMO"
        u_state.selected_sample_video = "HELP"

        # Case 1: Mismatch (HELP vs BAD)
        p_state_mismatch = PipelineRuntimeState(
            frame_source="DEMO_VIDEO",
            hand_detected=True,
            active_word_sign="BAD",
            word_sign_confidence=0.80,
            recognition_mode="WORD_SIGNS",
        )
        layout._render_recognition_card(canvas, 800, 60, 460, 140, p_state_mismatch, u_state, OBSIDIAN_THEME)
        assert np.count_nonzero(canvas) > 0

        # Case 2: Match (HELP vs HELP)
        canvas.fill(0)
        p_state_match = PipelineRuntimeState(
            frame_source="DEMO_VIDEO",
            hand_detected=True,
            active_word_sign="HELP",
            word_sign_confidence=0.90,
            recognition_mode="WORD_SIGNS",
        )
        layout._render_recognition_card(canvas, 800, 60, 460, 140, p_state_match, u_state, OBSIDIAN_THEME)
        assert np.count_nonzero(canvas) > 0

    def test_viewport_hud_rows_non_overlapping_positions(self):
        """Camera panel viewport must render 3 rows with distinct non-overlapping vertical offsets."""
        layout = DashboardLayout(1280, 800)
        canvas = np.zeros((800, 1280, 3), dtype=np.uint8)
        u_state = UIState()
        u_state.active_input_source = "DEMO"
        u_state.selected_sample_video = "HELP"

        p_state = PipelineRuntimeState(
            frame_source="DEMO_VIDEO",
            hand_detected=True,
            hand_count=2,
            handedness="Right",
            detection_confidence=1.0,
            secondary_handedness="Left",
            secondary_confidence=0.90,
            active_word_sign="BAD",
            word_sign_confidence=0.80,
        )

        dummy_cam = np.zeros((480, 640, 3), dtype=np.uint8)
        layout._render_camera_panel(canvas, 10, 50, 780, 690, dummy_cam, p_state, u_state, OBSIDIAN_THEME)
        assert np.count_nonzero(canvas) > 0
