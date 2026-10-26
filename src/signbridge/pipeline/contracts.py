"""Pipeline contracts, latency models, and execution outputs."""

from dataclasses import dataclass, field
import time
from typing import Any, Optional
import numpy as np

from enum import Enum
from signbridge.camera.base import FramePacket
from signbridge.features.extractor import FeatureVector
from signbridge.model.base import PredictionResult
from signbridge.temporal.contracts import FramePrediction, TemporalFrameResult
from signbridge.translation.contracts import TranslationResult
from signbridge.vision.detector import HandDetectionResult
from signbridge.vision.landmarks import DetectedHand


class FrameSource(str, Enum):
    """Mutually exclusive input origin for video frame processing."""
    LIVE_CAMERA = "LIVE_CAMERA"
    DEMO_VIDEO = "DEMO_VIDEO"


@dataclass
class StageLatencies:
    """Breakdown of processing durations (in milliseconds) across pipeline stages."""
    camera_capture_ms: float = 0.0
    hand_detection_ms: float = 0.0
    feature_extraction_ms: float = 0.0
    classifier_ms: float = 0.0
    temporal_ms: float = 0.0
    translation_ms: float = 0.0
    hud_render_ms: float = 0.0
    total_pipeline_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "camera_capture_ms": self.camera_capture_ms,
            "hand_detection_ms": self.hand_detection_ms,
            "feature_extraction_ms": self.feature_extraction_ms,
            "classifier_ms": self.classifier_ms,
            "temporal_ms": self.temporal_ms,
            "translation_ms": self.translation_ms,
            "hud_render_ms": self.hud_render_ms,
            "total_pipeline_ms": self.total_pipeline_ms,
        }


@dataclass(frozen=True)
class PipelineError:
    """Record of an isolated exception during frame processing."""
    stage: str
    message: str
    timestamp: float = field(default_factory=time.time)
    recoverable: bool = True


@dataclass(frozen=True)
class FrameProcessingOutput:
    """Immutable data container returned for every processed video frame."""
    frame_packet: FramePacket
    detection_result: HandDetectionResult
    primary_hand: Optional[DetectedHand]
    feature_vector: Optional[FeatureVector]
    prediction_result: Optional[PredictionResult]
    frame_prediction: Optional[FramePrediction]
    temporal_result: TemporalFrameResult
    translation_result: TranslationResult
    latencies: StageLatencies
    annotated_frame: Optional[np.ndarray] = None
    routed_result: Optional[Any] = None
