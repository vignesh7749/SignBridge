"""Temporal confidence filtering, rolling window stabilization, and duplicate suppression.

Converts frame-level sequence predictions into stable, reliable word tokens.
Prevents flickering and redundant emissions via confidence thresholding and cooldown timers.
"""

from collections import deque
from dataclasses import dataclass
import time
from typing import Optional, Sequence
import numpy as np

from signbridge.core.logger import get_logger
from signbridge.vision.detector import HandDetectionResult
from signbridge.word_sign.features import (
    FRAME_FEATURE_DIM,
    TARGET_SEQUENCE_LENGTH,
    extract_frame_features,
    resample_sequence_features,
)
from signbridge.word_sign.model import WordPredictionOutput, WordSignClassifier

logger = get_logger("word_sign.temporal")

DEFAULT_MIN_CONFIDENCE: float = 0.75
DEFAULT_STABLE_COUNT: int = 3
DEFAULT_COOLDOWN_SECONDS: float = 2.0
DEFAULT_WINDOW_FRAMES: int = 30
DEFAULT_NEUTRAL_DWELL_SECONDS: float = 0.3


@dataclass(frozen=True)
class WordSignPrediction:
    """State of word-sign prediction emitted by the temporal stabilization system."""
    word: str
    confidence: float
    is_stable: bool
    is_newly_emitted: bool
    timestamp: float
    top_k: list[tuple[str, float]]


class WordSignTemporalSmoother:
    """Temporal buffer and stabilization engine for word-level ASL recognition.

    Maintains a rolling window of multi-hand frame features, triggers sequence inference,
    accumulates consecutive agreement, and enforces active gesture hold tracking and
    neutral-release re-arming duplicate suppression.
    """

    def __init__(
        self,
        classifier: WordSignClassifier,
        window_size: int = DEFAULT_WINDOW_FRAMES,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        stable_count: int = DEFAULT_STABLE_COUNT,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        neutral_dwell_seconds: float = DEFAULT_NEUTRAL_DWELL_SECONDS,
    ) -> None:
        self.classifier = classifier
        self.window_size = max(TARGET_SEQUENCE_LENGTH, window_size)
        self.min_confidence = min_confidence
        self.stable_count = stable_count
        self.cooldown_seconds = cooldown_seconds
        self.neutral_dwell_seconds = neutral_dwell_seconds

        self._frame_buffer: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._consecutive_word: Optional[str] = None
        self._consecutive_count: int = 0
        self._last_emitted_word: Optional[str] = None
        self._last_emission_time: float = 0.0
        self._holding_emitted_word: Optional[str] = None
        self._neutral_since: Optional[float] = None

    def reset(self) -> None:
        """Clears the temporal window and stabilization counters."""
        self._frame_buffer.clear()
        self._consecutive_word = None
        self._consecutive_count = 0
        self._last_emitted_word = None
        self._last_emission_time = 0.0
        self._holding_emitted_word = None
        self._neutral_since = None

    def process_frame(
        self,
        detection: Optional[HandDetectionResult],
        current_time: Optional[float] = None,
    ) -> WordSignPrediction:
        """Appends current frame detection, evaluates sequence trajectory, and returns prediction state."""
        now = current_time if current_time is not None else time.time()

        # Extract 128D multi-hand frame vector
        feat = extract_frame_features(detection)
        self._frame_buffer.append(feat)

        # Require at least TARGET_SEQUENCE_LENGTH frames before running sequence inference
        if len(self._frame_buffer) < TARGET_SEQUENCE_LENGTH:
            return WordSignPrediction(
                word="",
                confidence=0.0,
                is_stable=False,
                is_newly_emitted=False,
                timestamp=now,
                top_k=[],
            )

        # Check if any hands are present in the recent buffer (last 5 frames ~150ms)
        recent_has_hands = any(f[0] > 0.5 or f[64] > 0.5 for f in list(self._frame_buffer)[-5:])
        if not recent_has_hands:
            self._consecutive_count = 0
            self._consecutive_word = None
            self._holding_emitted_word = None
            self._neutral_since = None
            return WordSignPrediction(
                word="",
                confidence=0.0,
                is_stable=False,
                is_newly_emitted=False,
                timestamp=now,
                top_k=[],
            )

        # Resample buffer to 20 timesteps and flatten to 2,560D
        seq_array = np.vstack(list(self._frame_buffer))
        resampled = resample_sequence_features(seq_array, target_length=TARGET_SEQUENCE_LENGTH)
        trajectory_vec = resampled.flatten()

        # Run inference
        pred: WordPredictionOutput = self.classifier.predict(trajectory_vec)

        # Evaluate confidence threshold
        is_above_conf = pred.confidence >= self.min_confidence
        candidate_word = pred.predicted_word if is_above_conf else ""

        # Track neutral dwell for re-arming held gestures
        if not candidate_word:
            if self._neutral_since is None:
                self._neutral_since = now
            elif (now - self._neutral_since) >= self.neutral_dwell_seconds:
                self._holding_emitted_word = None
        else:
            self._neutral_since = None
            if self._holding_emitted_word and candidate_word != self._holding_emitted_word:
                self._holding_emitted_word = None

        if candidate_word and candidate_word == self._consecutive_word:
            self._consecutive_count += 1
        else:
            self._consecutive_word = candidate_word
            self._consecutive_count = 1 if candidate_word else 0

        is_stable = (
            self._consecutive_count >= self.stable_count
            and bool(self._consecutive_word)
        )

        is_newly_emitted = False
        if is_stable and self._consecutive_word:
            is_holding = (self._consecutive_word == self._holding_emitted_word)
            in_cooldown = (
                self._consecutive_word == self._last_emitted_word
                and (now - self._last_emission_time) < self.cooldown_seconds
            )
            if not is_holding and not in_cooldown:
                is_newly_emitted = True
                self._holding_emitted_word = self._consecutive_word
                self._last_emitted_word = self._consecutive_word
                self._last_emission_time = now
                logger.info(
                    "Emitted stable word sign: '%s' (conf=%.2f, time=%.2f)",
                    self._consecutive_word,
                    pred.confidence,
                    now,
                )

        return WordSignPrediction(
            word=self._consecutive_word or (pred.predicted_word if is_above_conf else ""),
            confidence=pred.confidence,
            is_stable=is_stable,
            is_newly_emitted=is_newly_emitted,
            timestamp=now,
            top_k=pred.top_k,
        )
