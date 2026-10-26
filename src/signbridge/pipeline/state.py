"""Thread-safe, read-only runtime state representation for SignBridge."""

from dataclasses import dataclass, field
from typing import Optional, Sequence

from signbridge.camera.base import CameraStatus
from signbridge.pipeline.contracts import StageLatencies
from signbridge.temporal.contracts import TemporalState


@dataclass(frozen=True)
class PipelineRuntimeState:
    """Immutable snapshot of the end-to-end pipeline's operational state."""
    # Camera telemetry
    camera_status: CameraStatus = CameraStatus.UNINITIALIZED
    camera_fps: float = 0.0
    processing_fps: float = 0.0
    is_paused: bool = False
    frame_count: int = 0
    dropped_frames: int = 0
    uptime_seconds: float = 0.0

    # Latencies
    pipeline_latency: float = 0.0  # Total end-to-end ms
    stage_latencies: StageLatencies = field(default_factory=StageLatencies)

    # Hand detection (Vision)
    hand_detected: bool = False
    handedness: Optional[str] = None
    detection_confidence: float = 0.0
    hand_count: int = 0
    secondary_handedness: Optional[str] = None
    secondary_confidence: float = 0.0
    all_hands: tuple = ()

    # Static Recognition (Model)
    predicted_label: Optional[str] = None
    prediction_confidence: float = 0.0
    top_margin: float = 0.0

    # Temporal Stabilization
    temporal_state: TemporalState = TemporalState.IDLE
    temporal_candidate: Optional[str] = None
    dwell_progress: float = 0.0  # Normalized [0.0, 1.0] towards minimum dwell
    stable_token: Optional[str] = None  # Most recently emitted stable sign

    # Translation
    current_word: str = ""
    sentence: str = ""
    completed_words: tuple[str, ...] = ()
    translation_state: str = "EMPTY"
    last_event: Optional[str] = None

    # Error isolation
    error_state: Optional[str] = None

    # Text-to-Speech (Audio)
    voice_enabled: bool = False
    voice_status: str = "READY"
    is_speaking: bool = False
    last_spoken: Optional[str] = None
    voice_error_message: Optional[str] = None

    # Word-Sign & Mode Telemetry
    recognition_mode: str = "AUTO"
    active_word_sign: Optional[str] = None
    word_sign_confidence: float = 0.0
    frame_source: str = "LIVE_CAMERA"
    dwell_duration_ms: float = 0.0
    is_uncertain: bool = False

    @classmethod
    def initial(cls) -> "PipelineRuntimeState":
        """Creates an initial idle runtime state."""
        return cls()
