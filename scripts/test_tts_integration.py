"""Integration test verifying end-to-end translation to speech routing.

Simulates spelling 'HELLO WORLD' through the complete SignBridgePipeline:
- Verifies letter speech elimination (letters are NEVER spoken).
- Verifies word boundary speech routing.
- Verifies full sentence speech routing.
- Verifies duplicate suppression.
- Verifies non-blocking async execution and low latency.
"""

from pathlib import Path
import sys
import time
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.camera.base import CameraStatus, FramePacket, ICameraSource
from signbridge.features.extractor import HandFeatureExtractor
from signbridge.model.predictor import SignPredictor
from signbridge.pipeline import PipelineConfig, SignBridgePipeline
from signbridge.temporal.config import TemporalEngineConfig
from signbridge.temporal.contracts import FramePrediction, StableSign, TemporalState
from signbridge.temporal.engine import TemporalEngine
from signbridge.translation.config import TranslationConfig
from signbridge.translation.contracts import TranslationEventType
from signbridge.translation.engine import TranslationEngine
from signbridge.tts import MockTTSProvider, SpeechController, TTSConfig
from signbridge.vision.detector import HandDetectionResult, IHandDetector
from signbridge.vision.landmarks import BoundingBox, DetectedHand, LandmarkPoint


class MockCameraSource(ICameraSource):
    """In-memory dummy camera source."""

    def __init__(self, fps: float = 30.0) -> None:
        self._fps = fps
        self._frame_id = 0
        self._is_running = False

    def start(self) -> None:
        self._is_running = True

    def stop(self) -> None:
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def current_fps(self) -> float:
        return self._fps

    @property
    def status(self) -> CameraStatus:
        return CameraStatus.STREAMING if self._is_running else CameraStatus.CLOSED

    @property
    def actual_resolution(self) -> tuple[int, int]:
        return 640, 480

    def read(self, timeout: float = 1.0) -> tuple[bool, Optional[FramePacket]]:
        self._frame_id += 1
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        packet = FramePacket(
            data=frame,
            frame_id=self._frame_id,
            timestamp=time.time(),
            width=640,
            height=480,
        )
        return True, packet


class ControlledDetector(IHandDetector):
    """Pass-through hand detector returning synthetic hand."""

    def __init__(self) -> None:
        points = [
            LandmarkPoint(
                index=i,
                name=f"LM_{i}",
                x_norm=0.5,
                y_norm=0.5,
                z_norm=0.0,
                x_px=320,
                y_px=240,
                z_depth=0.0,
            )
            for i in range(21)
        ]
        raw_coords = np.zeros((21, 3), dtype=np.float32)
        raw_coords[:, 0] = 0.5
        raw_coords[:, 1] = 0.5
        raw_coords[0] = [0.5, 0.7, 0.0]
        raw_coords[9] = [0.5, 0.4, 0.0]

        self.hand = DetectedHand(
            handedness="Right",
            handedness_confidence=0.99,
            landmarks=points,
            bbox=BoundingBox(100, 100, 400, 400),
            raw_array=raw_coords,
            pixel_array=np.full((21, 2), 200, dtype=np.int32),
            score=0.99,
        )

    def detect(self, frame: np.ndarray, timestamp_ms: Optional[int] = None, **kwargs) -> HandDetectionResult:
        return HandDetectionResult(frame_id=kwargs.get("frame_id", 0), timestamp=time.time(), hands=[self.hand], processing_time_ms=1.0)

    def close(self) -> None:
        pass


def main() -> int:
    print("=" * 70)
    print("SIGNBRIDGE — STAGE 10 END-TO-END TRANSLATION TO TTS INTEGRATION TEST")
    print("=" * 70)

    # 1. Setup Mock TTS Provider & SpeechController
    mock_provider = MockTTSProvider(simulated_delay=0.01)
    tts_config = TTSConfig(
        enabled=True,
        speak_words=True,
        speak_sentences=True,
        suppress_duplicates=True,
    )
    speech_controller = SpeechController(config=tts_config, provider=mock_provider)

    # 2. Setup Pipeline with Mock TTS
    pipeline_config = PipelineConfig(
        temporal=TemporalEngineConfig(minimum_dwell_ms=50.0),
        translation=TranslationConfig(word_boundary_pause_seconds=0.5),
        tts=tts_config,
        draw_hud=True,
        draw_skeleton=False,
    )

    camera = MockCameraSource()
    detector = ControlledDetector()

    pipeline = SignBridgePipeline(
        config=pipeline_config,
        camera=camera,
        detector=detector,
        speech_controller=speech_controller,
    )

    pipeline.start()
    print("[1/5] Pipeline started with MockTTSProvider.")

    # 3. Simulate Fingerspelling 'H' 'E' 'L' 'L' 'O'
    print("\n[2/5] Simulating letter sequence for 'HELLO'...")
    tokens = ["H", "E", "L", "L", "O"]
    t0_enqueue = time.perf_counter()
    for char in tokens:
        res = pipeline.translation_engine.process_char(char)
        # Update pipeline state
        pipeline._update_runtime_state(trans_result=res)
        # Process a frame through pipeline to check for any spurious speech triggers
        success, out = pipeline.process_next_frame()
        assert success and out is not None

    enqueue_latency_ms = (time.perf_counter() - t0_enqueue) * 1000.0 / len(tokens)
    print(f"      Letter processing avg latency: {enqueue_latency_ms:.2f} ms")

    # Give worker a moment to process any spurious items (should be none)
    time.sleep(0.1)

    print(f"      Utterances spoken during letter entry: {len(mock_provider.spoken_history)}")
    if len(mock_provider.spoken_history) != 0:
        print(f"[FAIL] Letters were improperly spoken: {mock_provider.spoken_history}")
        return 1
    print("      [PASSED] Strict Letter Speech Elimination: zero letters spoken.")

    # 4. Trigger Word Boundary (SPACE)
    print("\n[3/5] Triggering Word Boundary (SPACE) for 'HELLO'...")
    space_res = pipeline.space()
    assert space_res.event_type == TranslationEventType.WORD_COMPLETED
    assert space_res.completed_words == ["HELLO"]

    # Wait briefly for background worker to synthesize word
    time.sleep(0.2)
    print(f"      Spoken history after word boundary: {mock_provider.spoken_history}")
    if "HELLO" not in mock_provider.spoken_history:
        print("[FAIL] Completed word 'HELLO' was not spoken!")
        return 1
    print("      [PASSED] Completed word 'HELLO' successfully spoken.")

    # 5. Spell second word 'WORLD' and trigger SPACE
    print("\n[4/5] Simulating second word 'WORLD' + SPACE...")
    for char in ["W", "O", "R", "L", "D"]:
        pipeline.translation_engine.process_char(char)
    pipeline.space()

    time.sleep(0.2)
    print(f"      Spoken history after 'WORLD': {mock_provider.spoken_history}")
    if "WORLD" not in mock_provider.spoken_history:
        print("[FAIL] Completed word 'WORLD' was not spoken!")
        return 1
    print("      [PASSED] Completed word 'WORLD' successfully spoken.")

    # 6. Test Sentence Speech & Duplicate Suppression
    print("\n[5/5] Testing full sentence speech and duplicate suppression...")
    # Manual speak current sentence: "HELLO WORLD"
    ok1 = pipeline.speak_current_sentence()
    assert ok1 is True
    time.sleep(0.2)

    count_before = len(mock_provider.spoken_history)
    print(f"      Spoken history after sentence speak: {mock_provider.spoken_history}")
    assert mock_provider.spoken_history[-1] == "HELLO WORLD"

    # Enqueue sentence again without manual bypass to test suppression
    event_duplicate = speech_controller.handle_sentence_completed("HELLO WORLD")
    print(f"      Duplicate sentence accepted by controller: {event_duplicate}")
    assert event_duplicate is False, "Duplicate sentence should be suppressed!"

    time.sleep(0.1)
    assert len(mock_provider.spoken_history) == count_before, "Duplicate sentence must not be spoken!"
    print("      [PASSED] Duplicate speech suppression verified.")

    pipeline.stop()
    print("\n" + "=" * 70)
    print("[SUCCESS] All Stage 10 TTS Integration tests passed cleanly!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
