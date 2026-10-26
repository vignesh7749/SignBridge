"""Module 9: Temporal Smoothing & Hysteresis Filtering."""

from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional

from signbridge.core.config import TemporalConfig
from signbridge.recognition.confidence import ScoredPrediction


@dataclass(frozen=True)
class StabilizedSign:
    label: str
    consensus_ratio: float
    duration_frames: int
    is_stable: bool


class TemporalSmoother:
    """Sliding-window majority voter with hysteresis thresholding."""

    def __init__(self, config: TemporalConfig) -> None:
        self.config = config
        self._history: deque[str] = deque(maxlen=config.window_size)
        self._current_stable_label: Optional[str] = None
        self._stable_frame_count = 0

    def update(self, prediction: ScoredPrediction) -> StabilizedSign:
        """Appends new prediction to history and computes consensus."""
        if prediction.is_valid:
            self._history.append(prediction.label)
        else:
            # If hand is invalid or confidence too low, clear or decay history
            if self._history:
                self._history.popleft()

        if not self._history:
            self._current_stable_label = None
            self._stable_frame_count = 0
            return StabilizedSign(label="", consensus_ratio=0.0, duration_frames=0, is_stable=False)

        counts = Counter(self._history)
        top_label, top_count = counts.most_common(1)[0]
        consensus = top_count / len(self._history)

        is_stable = (
            len(self._history) >= (self.config.window_size // 2)
            and consensus >= self.config.consensus_threshold
        )

        if is_stable:
            if top_label == self._current_stable_label:
                self._stable_frame_count += 1
            else:
                self._current_stable_label = top_label
                self._stable_frame_count = 1
        else:
            self._stable_frame_count = 0

        return StabilizedSign(
            label=top_label if is_stable else "",
            consensus_ratio=consensus,
            duration_frames=self._stable_frame_count,
            is_stable=is_stable,
        )

    def reset(self) -> None:
        """Clears rolling buffer and reset counters."""
        self._history.clear()
        self._current_stable_label = None
        self._stable_frame_count = 0
