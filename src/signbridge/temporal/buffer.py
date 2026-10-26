"""Bounded Temporal Prediction Buffer for Stage 7.

Maintains a bounded sliding window of recent predictions, pruning expired frames
and calculating majority and confidence-weighted consensus without unbounded memory growth.
"""

from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional, Sequence

from signbridge.temporal.config import TemporalEngineConfig
from signbridge.temporal.contracts import FramePrediction


@dataclass(frozen=True)
class ConsensusResult:
    """Consensus metrics computed across the active temporal buffer window."""

    top_label: Optional[str]
    top_class_id: Optional[int]
    agreement_ratio: float
    weighted_ratio: float
    mean_confidence: float
    mean_top_margin: float
    valid_frames: int
    total_frames: int
    earliest_timestamp: float = 0.0


class TemporalPredictionBuffer:
    """Bounded, time-windowed buffer for frame-level predictions."""

    def __init__(self, config: TemporalEngineConfig) -> None:
        self.config = config
        self._max_frames = config.max_buffer_frames
        self._window_sec = config.window_duration_ms / 1000.0
        self._buffer: deque[FramePrediction] = deque(maxlen=self._max_frames)

    def append(self, prediction: FramePrediction) -> None:
        """Appends a new prediction and prunes expired frames outside window."""
        self._buffer.append(prediction)
        self.prune(prediction.timestamp)

    def prune(self, current_timestamp: float) -> None:
        """Removes predictions older than current_timestamp - window_sec."""
        cutoff = current_timestamp - self._window_sec
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    def get_consensus(self, current_timestamp: float) -> ConsensusResult:
        """Calculates consensus metrics over valid, confident predictions in the window.

        Args:
            current_timestamp: Current reference timestamp in seconds.

        Returns:
            ConsensusResult containing top label, agreement ratio, and metrics.
        """
        self.prune(current_timestamp)

        if not self._buffer:
            return ConsensusResult(
                top_label=None,
                top_class_id=None,
                agreement_ratio=0.0,
                weighted_ratio=0.0,
                mean_confidence=0.0,
                mean_top_margin=0.0,
                valid_frames=0,
                total_frames=0,
            )

        total_frames = len(self._buffer)
        # Filter for frames where hand is present and confidence meets minimum threshold
        valid_predictions = [
            p
            for p in self._buffer
            if p.has_hand and p.label and p.confidence >= self.config.confidence_threshold
        ]

        if not valid_predictions:
            return ConsensusResult(
                top_label=None,
                top_class_id=None,
                agreement_ratio=0.0,
                weighted_ratio=0.0,
                mean_confidence=0.0,
                mean_top_margin=0.0,
                valid_frames=0,
                total_frames=total_frames,
            )

        # Count frequencies of labels
        label_counts = Counter(p.label for p in valid_predictions)
        top_label, top_count = label_counts.most_common(1)[0]

        # Find matching predictions for top_label
        top_preds = [p for p in valid_predictions if p.label == top_label]
        top_class_id = top_preds[0].class_id
        earliest_timestamp = top_preds[0].timestamp

        # Agreement ratio relative to valid confident frames
        agreement_ratio = top_count / len(valid_predictions)

        # Confidence-weighted consensus ratio
        total_conf = sum(p.confidence for p in valid_predictions)
        top_conf = sum(p.confidence for p in top_preds)
        weighted_ratio = (top_conf / total_conf) if total_conf > 0.0 else agreement_ratio

        # Averages for the winning class
        mean_confidence = top_conf / len(top_preds)
        mean_top_margin = sum(p.top_margin for p in top_preds) / len(top_preds)

        return ConsensusResult(
            top_label=top_label,
            top_class_id=top_class_id,
            agreement_ratio=agreement_ratio,
            weighted_ratio=weighted_ratio,
            mean_confidence=mean_confidence,
            mean_top_margin=mean_top_margin,
            valid_frames=len(valid_predictions),
            total_frames=total_frames,
            earliest_timestamp=earliest_timestamp,
        )

    def clear(self) -> None:
        """Clears all buffered predictions."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    @property
    def history(self) -> Sequence[FramePrediction]:
        return list(self._buffer)
