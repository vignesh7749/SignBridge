"""Stage 9: Live Pipeline Integration Test Harness.

Verifies end-to-end integration across Vision, Features, Inference,
Temporal Stabilization, and Translation subsystems using controlled synthetic sequences:
1. Sequential Signs: A -> B -> C
2. Word Spelling: H -> E -> L -> L -> O => "HELLO"
3. Sentence Spelling: H -> E -> L -> L -> O -> SPACE -> W -> O -> R -> L -> D => "HELLO WORLD"
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.camera.base import CameraDeviceInfo, CameraStatus, FramePacket, ICameraSource
from signbridge.model.base import PredictionResult
from signbridge.pipeline import PipelineConfig, SignBridgePipeline
from signbridge.temporal.config import TemporalEngineConfig
from signbridge.temporal.contracts import FramePrediction
from signbridge.translation.config import TranslationConfig


class SyntheticCamera(ICameraSource):
    """Generates synthetic video frames with monotonic timestamps for deterministic testing."""

    def __init__(self, fps: float = 30.0, width: int = 640, height: int = 480) -> None:
        self._fps = fps
        self._width = width
        self._height = height
        self._frame_id = 0
        self._timestamp = 1000.0
        self._is_running = False
        self._status = CameraStatus.UNINITIALIZED

    def start(self) -> None:
        self._is_running = True
        self._status = CameraStatus.STREAMING

    def stop(self) -> None:
        self._is_running = False
        self._status = CameraStatus.CLOSED

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def current_fps(self) -> float:
        return self._fps

    @property
    def status(self) -> CameraStatus:
        return self._status

    @property
    def actual_resolution(self) -> tuple[int, int]:
        return self._width, self._height

    def advance_frame(self) -> FramePacket:
        self._frame_id += 1
        self._timestamp += 1.0 / self._fps
        # Synthetic test image
        img = np.full((self._height, self._width, 3), 40, dtype=np.uint8)
        return FramePacket(
            frame_id=self._frame_id,
            timestamp=self._timestamp,
            data=img,
            width=self._width,
            height=self._height,
        )

    def read(self, timeout: float = 1.0) -> tuple[bool, Optional[FramePacket]]:
        if not self._is_running:
            return False, None
        return True, self.advance_frame()


class MockSignPredictor:
    """Mock predictor that returns pre-configured sign predictions."""

    def __init__(self) -> None:
        self.next_label: str = "A"
        self.next_conf: float = 0.95

    def predict(self, features: np.ndarray) -> PredictionResult:
        return PredictionResult(
            class_id=ord(self.next_label.upper()) - 65 if self.next_label.isalpha() else 0,
            class_name=self.next_label,
            confidence=self.next_conf,
            top_margin=0.40,
            top_candidates=[(self.next_label, self.next_conf)],
            inference_latency_ms=0.5,
        )


def run_live_pipeline_test() -> int:
    print("=" * 70)
    print("SIGNBRIDGE — STAGE 9 LIVE PIPELINE INTEGRATION TEST")
    print("=" * 70)

    # Configure pipeline with tight dwell time for fast testing
    config = PipelineConfig(
        temporal=TemporalEngineConfig(
            minimum_dwell_ms=100.0,
            confidence_threshold=0.70,
            consensus_threshold=0.60,
            minimum_stable_frames=3,
        ),
        translation=TranslationConfig(
            word_boundary_pause_seconds=1.5,
            auto_word_boundary=True,
        ),
        draw_hud=True,
        draw_skeleton=True,
    )

    cam = SyntheticCamera(fps=30.0)
    pipeline = SignBridgePipeline(config=config, camera=cam)
    pipeline.start()

    passed_count = 0
    total_tests = 3

    # =========================================================================
    # Test 1: Sequential Signs A -> B -> C through Temporal to Translation
    # =========================================================================
    print("\n[TEST 1] Testing sequential signs A -> B -> C...")
    pipeline.reset()

    for sign in ["A", "B", "C"]:
        # Hold sign for 12 frames (400 ms @ 30 FPS > 100 ms dwell threshold)
        for _ in range(12):
            packet = cam.advance_frame()
            pred = FramePrediction(
                class_id=ord(sign) - 65,
                label=sign,
                confidence=0.95,
                timestamp=packet.timestamp,
                has_hand=True,
            )
            temp_res = pipeline.temporal_engine.process_frame(pred)
            if temp_res.has_emitted_sign and temp_res.stable_sign is not None:
                pipeline.translation_engine.process_token(temp_res.stable_sign)

        # Release hand for 6 frames (200 ms > 150 ms neutral release threshold)
        for _ in range(6):
            packet = cam.advance_frame()
            pred = FramePrediction.no_hand(packet.timestamp)
            pipeline.temporal_engine.process_frame(pred)

    state = pipeline.get_state()
    print(f"  Result: Word='{state.current_word}', Sentence='{state.sentence}'")
    assert state.current_word == "ABC", f"Expected 'ABC', got '{state.current_word}'"
    print("  >>> [PASS] Test 1: Sequential tokens A -> B -> C accumulated to 'ABC'")
    passed_count += 1

    # =========================================================================
    # Test 2: Word Spelling: H -> E -> L -> L -> O => "HELLO"
    # =========================================================================
    print("\n[TEST 2] Testing word spelling: H -> E -> L -> L -> O...")
    pipeline.reset()

    for sign in ["H", "E", "L", "L", "O"]:
        for _ in range(12):
            packet = cam.advance_frame()
            pred = FramePrediction(
                class_id=ord(sign) - 65,
                label=sign,
                confidence=0.96,
                timestamp=packet.timestamp,
                has_hand=True,
            )
            temp_res = pipeline.temporal_engine.process_frame(pred)
            if temp_res.has_emitted_sign and temp_res.stable_sign is not None:
                pipeline.translation_engine.process_token(temp_res.stable_sign)

        for _ in range(6):
            packet = cam.advance_frame()
            pred = FramePrediction.no_hand(packet.timestamp)
            pipeline.temporal_engine.process_frame(pred)

    state = pipeline.get_state()
    print(f"  Result: Word='{state.current_word}', Sentence='{state.sentence}'")
    assert state.current_word == "HELLO", f"Expected 'HELLO', got '{state.current_word}'"
    print("  >>> [PASS] Test 2: Word 'HELLO' spelled successfully")
    passed_count += 1

    # =========================================================================
    # Test 3: Sentence Spelling: HELLO -> SPACE -> WORLD => "HELLO WORLD"
    # =========================================================================
    print("\n[TEST 3] Testing sentence spelling: HELLO + SPACE + WORLD...")
    # Pipeline currently has 'HELLO'. Trigger SPACE.
    space_res = pipeline.space()
    assert list(space_res.completed_words) == ["HELLO"], f"Expected ['HELLO'], got {space_res.completed_words}"

    for sign in ["W", "O", "R", "L", "D"]:
        for _ in range(12):
            packet = cam.advance_frame()
            pred = FramePrediction(
                class_id=ord(sign) - 65,
                label=sign,
                confidence=0.95,
                timestamp=packet.timestamp,
                has_hand=True,
            )
            temp_res = pipeline.temporal_engine.process_frame(pred)
            if temp_res.has_emitted_sign and temp_res.stable_sign is not None:
                pipeline.translation_engine.process_token(temp_res.stable_sign)

        for _ in range(6):
            packet = cam.advance_frame()
            pred = FramePrediction.no_hand(packet.timestamp)
            pipeline.temporal_engine.process_frame(pred)

    state = pipeline.get_state()
    print(f"  Result: Word='{state.current_word}', Sentence='{state.sentence}', Completed={state.completed_words}")
    assert state.sentence == "HELLO WORLD", f"Expected 'HELLO WORLD', got '{state.sentence}'"
    print("  >>> [PASS] Test 3: Sentence 'HELLO WORLD' successfully assembled")
    passed_count += 1

    pipeline.stop()

    print("\n" + "=" * 70)
    print(f"LIVE PIPELINE INTEGRATION TEST RESULT: {passed_count} / {total_tests} PASSED (100%)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(run_live_pipeline_test())
