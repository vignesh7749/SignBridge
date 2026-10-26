"""Temporal Recognition Data Contracts for Stage 7.

Defines immutable domain models for frame-level predictions, stabilized sign tokens,
state machine enumeration, and real-time temporal status telemetry.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
from typing import Any, Optional


class TemporalState(Enum):
    """Lifecycle states of the temporal sign stabilization finite state machine."""

    IDLE = "IDLE"                          # No active candidate or hand absent beyond grace period
    CANDIDATE = "CANDIDATE"                # Candidate sign observed; accumulating consensus and dwell time
    STABLE = "STABLE"                      # Candidate has met consensus and dwell duration requirements
    EMITTED = "EMITTED"                    # Stable sign token committed and emitted on current frame
    WAITING_FOR_CHANGE = "WAITING_FOR_CHANGE"  # Token emitted; suppressing duplicate emissions until change


@dataclass(frozen=True)
class FramePrediction:
    """Immutable single-frame prediction received from the ML classifier or vision detector.

    Attributes:
        class_id: Integer identifier of the predicted sign (e.g. 0..25), or -1 if no hand.
        label: Text string label of the sign (e.g. "A".."Z"), or "" if no hand.
        confidence: Posterior probability/confidence score in [0.0, 1.0].
        timestamp: Monotonic wall-clock timestamp in seconds.
        has_hand: Whether a hand was detected in this frame.
        top_margin: Probability difference between top-1 and top-2 candidates.
    """

    class_id: int
    label: str
    confidence: float
    timestamp: float
    has_hand: bool = True
    top_margin: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp < 0.0:
            raise ValueError(f"Timestamp must be non-negative, got {self.timestamp}")
        if math.isnan(self.confidence) or math.isinf(self.confidence):
            raise ValueError(f"Confidence must be finite, got {self.confidence}")
        if not (0.0 <= self.confidence <= 1.0001):
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")
        if self.has_hand and self.class_id < 0:
            raise ValueError(f"Valid hand prediction must have class_id >= 0, got {self.class_id}")

    @classmethod
    def no_hand(cls, timestamp: float) -> "FramePrediction":
        """Factory constructor for frames where no hand was detected."""
        return cls(
            class_id=-1,
            label="",
            confidence=0.0,
            timestamp=timestamp,
            has_hand=False,
            top_margin=0.0,
        )

    @classmethod
    def from_prediction_result(
        cls,
        result: Any,
        timestamp: float,
        has_hand: bool = True,
    ) -> "FramePrediction":
        """Factory constructor bridging Stage 6 PredictionResult into FramePrediction."""
        if not has_hand or result is None:
            return cls.no_hand(timestamp)
        return cls(
            class_id=int(result.class_id),
            label=str(result.class_name),
            confidence=float(result.confidence),
            timestamp=float(timestamp),
            has_hand=True,
            top_margin=float(getattr(result, "top_margin", 0.0)),
        )


@dataclass(frozen=True)
class StableSign:
    """Immutable discrete sign token emitted by the temporal stabilization layer.

    Attributes:
        class_id: Integer identifier of the stabilized sign.
        label: Text string label of the stabilized sign (e.g. "A").
        confidence: Mean calibrated confidence across the stabilization window.
        start_timestamp: Wall-clock timestamp (seconds) when candidate was first established.
        end_timestamp: Wall-clock timestamp (seconds) when candidate reached stable commitment.
        duration: Elapsed dwell duration in seconds (end_timestamp - start_timestamp).
        frame_count: Number of contributing frames in the stabilization sequence.
        top_margin: Mean top-candidate margin across the stabilization window.
    """

    class_id: int
    label: str
    confidence: float
    start_timestamp: float
    end_timestamp: float
    duration: float
    frame_count: int
    top_margin: float = 0.0

    @property
    def duration_ms(self) -> float:
        """Elapsed dwell duration in milliseconds."""
        return self.duration * 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Serializes stable sign token to a standard dictionary."""
        return {
            "class_id": self.class_id,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "start_timestamp": round(float(self.start_timestamp), 4),
            "end_timestamp": round(float(self.end_timestamp), 4),
            "duration_ms": round(float(self.duration_ms), 2),
            "frame_count": int(self.frame_count),
            "top_margin": round(float(self.top_margin), 4),
        }


@dataclass(frozen=True)
class TemporalFrameResult:
    """Diagnostic telemetry and emitted sign token returned on each processed video frame.

    Attributes:
        state: Current FSM state of the temporal engine.
        current_candidate: Active candidate label under evaluation, or None.
        candidate_confidence: Smoothed confidence of the active candidate.
        dwell_duration_ms: Current accumulated dwell time for active candidate in ms.
        stability_ratio: Agreement ratio of the active candidate in rolling window [0.0, 1.0].
        is_hand_present: Whether a hand was observed in the current frame.
        stable_sign: Emitted StableSign token if commitment occurred on this frame, else None.
        processing_latency_ms: Processing latency of the temporal update step in ms.
        frame_prediction: The input frame prediction processed in this step.
    """

    state: TemporalState
    current_candidate: Optional[str]
    candidate_confidence: float
    dwell_duration_ms: float
    stability_ratio: float
    is_hand_present: bool
    stable_sign: Optional[StableSign] = None
    processing_latency_ms: float = 0.0
    frame_prediction: Optional[FramePrediction] = None

    @property
    def has_emitted_sign(self) -> bool:
        """Returns True if a stable sign was emitted on this frame."""
        return self.stable_sign is not None
