"""Recognition Router: Orchestrates routing between Static Alphabet and Word-Sign modules.

Provides 3 user-selectable operational modes:
1. ALPHABET: Routes exclusively to Stage 6 static fingerspelling classifier and Stage 7 dwell-time engine.
2. WORD_SIGNS: Routes exclusively to temporal trajectory sequence model.
3. AUTO: Intelligently arbitrates between word signs and alphabet fingerspelling without thrashing.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Optional

from signbridge.core.logger import get_logger
from signbridge.vision.detector import HandDetectionResult
from signbridge.word_sign.temporal import WordSignPrediction, WordSignTemporalSmoother

logger = get_logger("word_sign.router")


class RecognitionMode(str, Enum):
    """Operational mode for sign language recognition pipeline."""
    ALPHABET = "ALPHABET"
    WORD_SIGNS = "WORD_SIGNS"
    AUTO = "AUTO"

    @classmethod
    def from_string(cls, value: str) -> "RecognitionMode":
        val = value.strip().upper().replace(" ", "_")
        for mode in cls:
            if mode.value == val or mode.name == val:
                return mode
        return cls.AUTO


@dataclass(frozen=True)
class RoutedRecognitionResult:
    """Consolidated recognition outcome across active recognition pathways."""
    mode: RecognitionMode
    token: Optional[str] = None
    is_word: bool = False
    confidence: float = 0.0
    is_newly_emitted: bool = False
    active_symbol: str = ""
    word_prediction: Optional[WordSignPrediction] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RecognitionRouter:
    """Manages routing of live frames to alphabet and/or word-sign classification pipelines."""

    def __init__(
        self,
        word_smoother: Optional[WordSignTemporalSmoother] = None,
        initial_mode: RecognitionMode = RecognitionMode.AUTO,
    ) -> None:
        self.mode: RecognitionMode = initial_mode
        self.word_smoother = word_smoother
        self._last_token: Optional[str] = None
        self._last_token_time: float = 0.0

    def set_mode(self, mode: RecognitionMode | str) -> None:
        """Switches the active operational mode."""
        if isinstance(mode, str):
            self.mode = RecognitionMode.from_string(mode)
        else:
            self.mode = mode
        logger.info("Recognition router mode changed to: %s", self.mode.value)

    def reset(self) -> None:
        """Resets internal state buffers."""
        if self.word_smoother is not None:
            self.word_smoother.reset()
        self._last_token = None
        self._last_token_time = 0.0

    def route_frame(
        self,
        detection: Optional[HandDetectionResult],
        alphabet_token_candidate: Optional[str] = None,
        alphabet_confidence: float = 0.0,
        alphabet_is_emitted: bool = False,
        current_time: Optional[float] = None,
    ) -> RoutedRecognitionResult:
        """Processes a frame through the appropriate recognizer based on active mode."""
        now = current_time if current_time is not None else time.time()

        # 1. ALPHABET MODE: Bypass word-sign sequence model entirely
        if self.mode == RecognitionMode.ALPHABET:
            return RoutedRecognitionResult(
                mode=RecognitionMode.ALPHABET,
                token=alphabet_token_candidate if alphabet_is_emitted else None,
                is_word=False,
                confidence=alphabet_confidence,
                is_newly_emitted=alphabet_is_emitted,
                active_symbol=alphabet_token_candidate or "",
                word_prediction=None,
                metadata={"engine": "alphabet"},
            )

        # 2. WORD_SIGNS MODE: Process only sequence model
        word_pred: Optional[WordSignPrediction] = None
        if self.word_smoother is not None:
            word_pred = self.word_smoother.process_frame(detection, current_time=now)

        if self.mode == RecognitionMode.WORD_SIGNS:
            if word_pred is None:
                return RoutedRecognitionResult(
                    mode=RecognitionMode.WORD_SIGNS,
                    active_symbol="",
                )
            return RoutedRecognitionResult(
                mode=RecognitionMode.WORD_SIGNS,
                token=word_pred.word if word_pred.is_newly_emitted else None,
                is_word=True,
                confidence=word_pred.confidence,
                is_newly_emitted=word_pred.is_newly_emitted,
                active_symbol=word_pred.word,
                word_prediction=word_pred,
                metadata={"engine": "word_signs"},
            )

        # 3. AUTO HYBRID MODE: Arbitrates between word-signs and fingerspelling
        # Priority rule: If a word sign has high confidence (>= 0.75) and stability, favor word sign.
        # Otherwise, fall back to alphabet fingerspelling.
        if word_pred is not None and word_pred.confidence >= 0.75 and (word_pred.is_stable or word_pred.is_newly_emitted):
            return RoutedRecognitionResult(
                mode=RecognitionMode.AUTO,
                token=word_pred.word if word_pred.is_newly_emitted else None,
                is_word=True,
                confidence=word_pred.confidence,
                is_newly_emitted=word_pred.is_newly_emitted,
                active_symbol=word_pred.word,
                word_prediction=word_pred,
                metadata={"engine": "word_signs_auto_favored"},
            )

        # Fallback to alphabet in AUTO mode
        return RoutedRecognitionResult(
            mode=RecognitionMode.AUTO,
            token=alphabet_token_candidate if alphabet_is_emitted else None,
            is_word=False,
            confidence=alphabet_confidence,
            is_newly_emitted=alphabet_is_emitted,
            active_symbol=alphabet_token_candidate or (word_pred.word if word_pred and word_pred.confidence > 0.4 else ""),
            word_prediction=word_pred,
            metadata={"engine": "alphabet_auto_fallback"},
        )
