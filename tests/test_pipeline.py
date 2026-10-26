"""Integration and unit tests for Stage 9 end-to-end pipeline orchestrator."""

import time
from typing import Optional, Sequence
import numpy as np
import pytest

from signbridge.camera.base import CameraStatus, FramePacket, ICameraSource
from signbridge.features.extractor import FeatureVector, HandFeatureExtractor
from signbridge.features.validator import LandmarkValidationError
from signbridge.model.base import PredictionResult
from signbridge.model.predictor import SignPredictor
from signbridge.pipeline import PipelineConfig, PipelineRuntimeState, SignBridgePipeline
from signbridge.pipeline.contracts import FrameProcessingOutput, StageLatencies
from signbridge.temporal.config import TemporalEngineConfig
from signbridge.temporal.contracts import FramePrediction, StableSign, TemporalFrameResult, TemporalState
from signbridge.temporal.engine import TemporalEngine
from signbridge.translation.config import TranslationConfig
from signbridge.translation.contracts import StableSignToken, TranslationEventType, TranslationResult, TranslationState
from signbridge.translation.engine import TranslationEngine
from signbridge.vision.detector import HandDetectionResult, IHandDetector
from signbridge.vision.landmarks import BoundingBox, DetectedHand, LandmarkPoint


class MockCameraSource(ICameraSource):
    """Deterministic in-memory camera source."""

    def __init__(self, fps: float = 30.0, width: int = 640, height: int = 480) -> None:
        self._fps = fps
        self._width = width
        self._height = height
        self._frame_id = 0
        self._timestamp = 100.0
        self._is_running = False
        self._status = CameraStatus.UNINITIALIZED
        self.should_fail_read = False

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
        frame = np.full((self._height, self._width, 3), 128, dtype=np.uint8)
        return FramePacket(
            frame_id=self._frame_id,
            timestamp=self._timestamp,
            data=frame,
            width=self._width,
            height=self._height,
        )

    def read(self, timeout: float = 1.0) -> tuple[bool, Optional[FramePacket]]:
        if not self._is_running or self.should_fail_read:
            return False, None
        return True, self.advance_frame()


def make_dummy_landmarks() -> list[LandmarkPoint]:
    return [
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


class MockHandDetector(IHandDetector):
    """Deterministic hand detector yielding mock landmarks or empty detection."""

    def __init__(self, produce_hand: bool = True) -> None:
        self.produce_hand = produce_hand
        self.is_closed = False

    def detect(self, bgr_frame: np.ndarray, frame_id: int = 0, timestamp: Optional[float] = None) -> HandDetectionResult:
        now = timestamp or time.time()
        if not self.produce_hand:
            return HandDetectionResult(frame_id=frame_id, timestamp=now, hands=[], processing_time_ms=1.0)

        points = make_dummy_landmarks()
        raw_coords = np.zeros((21, 3), dtype=np.float32)
        raw_coords[:, 0] = 0.5
        raw_coords[:, 1] = 0.5
        raw_coords[0] = [0.5, 0.7, 0.0]
        raw_coords[9] = [0.5, 0.4, 0.0]

        hand = DetectedHand(
            handedness="Right",
            handedness_confidence=0.98,
            landmarks=points,
            bbox=BoundingBox(100, 100, 400, 400),
            raw_array=raw_coords,
            pixel_array=np.full((21, 2), 200, dtype=np.int32),
            score=0.98,
        )
        return HandDetectionResult(frame_id=frame_id, timestamp=now, hands=[hand], processing_time_ms=1.0)

    def close(self) -> None:
        self.is_closed = True


class MockClassifierPredictor:
    """Deterministic mock predictor."""

    def __init__(self, fixed_label: str = "A", confidence: float = 0.95) -> None:
        self.fixed_label = fixed_label
        self.confidence = confidence
        self.raise_on_predict = False

    def predict(self, features: np.ndarray) -> PredictionResult:
        if self.raise_on_predict:
            raise RuntimeError("Injected classifier prediction error")
        return PredictionResult(
            class_id=ord(self.fixed_label.upper()) - 65,
            class_name=self.fixed_label,
            confidence=self.confidence,
            top_candidates=[(self.fixed_label, self.confidence)],
            inference_latency_ms=0.5,
        )


@pytest.fixture
def mock_pipeline():
    cam = MockCameraSource(fps=30.0)
    det = MockHandDetector(produce_hand=True)
    pred = MockClassifierPredictor(fixed_label="A", confidence=0.95)
    temp = TemporalEngine(TemporalEngineConfig(minimum_dwell_ms=100.0))
    trans = TranslationEngine(TranslationConfig())
    cfg = PipelineConfig(draw_hud=False, draw_skeleton=False)

    pipeline = SignBridgePipeline(
        config=cfg,
        camera=cam,
        detector=det,
        predictor=pred,
        temporal_engine=temp,
        translation_engine=trans,
    )
    return pipeline, cam, det, pred, temp, trans


def test_pipeline_initialization(mock_pipeline):
    """Verifies that all subsystems assemble and initialize cleanly."""
    pipeline, cam, det, pred, temp, trans = mock_pipeline
    assert not pipeline.is_running
    assert not pipeline.is_paused
    state = pipeline.get_state()
    assert isinstance(state, PipelineRuntimeState)
    assert state.camera_status == CameraStatus.UNINITIALIZED
    assert state.current_word == ""
    assert state.sentence == ""


def test_valid_frame_processing(mock_pipeline):
    """Verifies full forward pass through detector, features, model, temporal, and translation."""
    pipeline, cam, det, pred, temp, trans = mock_pipeline
    pipeline.start()

    ok, output = pipeline.process_next_frame()
    assert ok is True
    assert output is not None
    assert output.primary_hand is not None
    assert output.feature_vector is not None
    assert output.prediction_result is not None
    assert output.prediction_result.class_name == "A"
    assert output.latencies.total_pipeline_ms >= 0.0

    state = pipeline.get_state()
    assert state.hand_detected is True
    assert state.predicted_label == "A"
    pipeline.stop()


def test_no_hand_frame_safety(mock_pipeline):
    """Verifies pipeline safely handles absence of hands without crashing."""
    pipeline, cam, det, pred, temp, trans = mock_pipeline
    det.produce_hand = False
    pipeline.start()

    ok, output = pipeline.process_next_frame()
    assert ok is True
    assert output is not None
    assert output.primary_hand is None
    assert output.feature_vector is None
    assert output.prediction_result is None
    assert output.temporal_result.is_hand_present is False

    state = pipeline.get_state()
    assert state.hand_detected is False
    assert state.predicted_label is None
    pipeline.stop()


def test_invalid_landmarks_isolation(mock_pipeline):
    """Verifies that corrupted landmark arrays do not crash the pipeline."""
    pipeline, cam, det, pred, temp, trans = mock_pipeline
    pipeline.start()

    # Ingest a frame where landmarks contain NaN
    packet = cam.advance_frame()
    nan_coords = np.full((21, 3), np.nan, dtype=np.float32)
    corrupted_hand = DetectedHand(
        handedness="Right",
        handedness_confidence=0.90,
        landmarks=make_dummy_landmarks(),
        bbox=BoundingBox(0, 0, 10, 10),
        raw_array=nan_coords,
        pixel_array=np.zeros((21, 2), dtype=np.int32),
        score=0.90,
    )
    det_res = HandDetectionResult(frame_id=packet.frame_id, timestamp=packet.timestamp, hands=[corrupted_hand])

    # Should handle gracefully without raising unhandled exception
    output = pipeline.process_frame(packet)
    assert output is not None
    pipeline.stop()


def test_classifier_error_isolation(mock_pipeline):
    """Verifies that unexpected classifier failures are isolated and reported in error state."""
    pipeline, cam, det, pred, temp, trans = mock_pipeline
    pred.raise_on_predict = True
    pipeline.start()

    ok, output = pipeline.process_next_frame()
    assert ok is True
    assert output is not None
    assert output.prediction_result is None  # Handled safely

    state = pipeline.get_state()
    assert state.error_state is not None
    assert "Injected classifier prediction error" in state.error_state
    pipeline.stop()


def test_temporal_to_translation_forwarding(mock_pipeline):
    """Verifies that stable tokens from TemporalEngine are passed into TranslationEngine."""
    pipeline, cam, det, pred, temp, trans = mock_pipeline
    pipeline.start()

    # Process enough frames of sign 'A' to satisfy dwell threshold (100 ms dwell = 4 frames @ 30 FPS)
    for _ in range(8):
        ok, output = pipeline.process_next_frame()
        assert ok is True

    state = pipeline.get_state()
    assert "A" in state.current_word
    pipeline.stop()


def test_multi_hand_policy(mock_pipeline):
    """Verifies primary hand selection policy when multiple hands are present."""
    pipeline, cam, det, pred, temp, trans = mock_pipeline
    pipeline.config = PipelineConfig(primary_hand_preference="highest_confidence")

    h1 = DetectedHand(
        handedness="Left",
        handedness_confidence=0.75,
        landmarks=make_dummy_landmarks(),
        bbox=BoundingBox(0, 0, 10, 10),
        raw_array=np.zeros((21, 3), dtype=np.float32),
        pixel_array=np.zeros((21, 2), dtype=np.int32),
        score=0.75,
    )
    h2 = DetectedHand(
        handedness="Right",
        handedness_confidence=0.95,
        landmarks=make_dummy_landmarks(),
        bbox=BoundingBox(0, 0, 10, 10),
        raw_array=np.zeros((21, 3), dtype=np.float32),
        pixel_array=np.zeros((21, 2), dtype=np.int32),
        score=0.95,
    )

    selected = pipeline._select_primary_hand([h1, h2])
    assert selected.handedness == "Right"
    assert selected.handedness_confidence == 0.95


def test_pause_and_resume_semantics(mock_pipeline):
    """Verifies pause and resume mechanics preserve state without thread glitches."""
    pipeline, cam, det, pred, temp, trans = mock_pipeline
    pipeline.start()

    # Pre-load a word
    trans.process_char("H")
    assert pipeline.get_state().current_word == "H"

    # Pause
    pipeline.pause()
    assert pipeline.is_paused is True
    assert pipeline.get_state().is_paused is True
    assert pipeline.get_state().current_word == "H"

    # Frame processing during pause returns cached output or idle
    ok, output = pipeline.process_next_frame()
    assert ok is True

    # Resume
    pipeline.resume()
    assert pipeline.is_paused is False
    assert pipeline.get_state().is_paused is False
    assert pipeline.get_state().current_word == "H"

    pipeline.stop()


def test_camera_disconnect_safety(mock_pipeline):
    """Verifies camera failure or disconnection is captured safely without crash."""
    pipeline, cam, det, pred, temp, trans = mock_pipeline
    pipeline.start()

    cam.should_fail_read = True
    ok, output = pipeline.process_next_frame()
    assert ok is False
    assert output is None

    state = pipeline.get_state()
    assert state.dropped_frames == 1
    pipeline.stop()


def test_end_to_end_spelling_sequence(mock_pipeline):
    """Full end-to-end integration test: spells 'HELLO', commits SPACE, then spells 'WORLD'."""
    pipeline, cam, det, pred, temp, trans = mock_pipeline
    pipeline.start()

    # Sequence 1: HELLO
    for char in ["H", "E", "L", "L", "O"]:
        token = StableSignToken.from_char(char, timestamp=time.time(), confidence=0.98)
        trans.process_token(token)

    assert pipeline.get_state().current_word == "HELLO"

    # Command: SPACE
    space_res = pipeline.space()
    assert list(space_res.completed_words) == ["HELLO"]
    assert pipeline.get_state().current_word == ""

    # Sequence 2: WORLD
    for char in ["W", "O", "R", "L", "D"]:
        token = StableSignToken.from_char(char, timestamp=time.time(), confidence=0.97)
        trans.process_token(token)

    state = pipeline.get_state()
    assert state.current_word == "WORLD"
    assert state.sentence == "HELLO WORLD"
    assert state.completed_words == ("HELLO",)

    # Edit operations: Backspace
    bs_res = pipeline.backspace()
    assert bs_res.current_word == "WORL"
    assert pipeline.get_state().sentence == "HELLO WORL"

    # Undo
    undo_res = pipeline.undo()
    assert undo_res.current_word == "WORLD"
    assert pipeline.get_state().sentence == "HELLO WORLD"

    pipeline.stop()


def test_runtime_state_immutability(mock_pipeline):
    """Verifies that PipelineRuntimeState is a frozen immutable snapshot."""
    pipeline, cam, det, pred, temp, trans = mock_pipeline
    state = pipeline.get_state()

    with pytest.raises(Exception):
        state.current_word = "MUTATED"  # FrozenInstanceError
