"""Stage 8 Translation Engine Data Contracts.

Defines immutable domain models for stabilized sign tokens, completed words,
assembled sentences, translation events, and state machine lifecycle states.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import time
from typing import Any, Optional, Sequence


class TranslationState(Enum):
    """Lifecycle states of the translation finite state machine."""

    EMPTY = "EMPTY"                          # No completed words and no active characters
    BUILDING_WORD = "BUILDING_WORD"          # Actively accumulating letters for the first word
    WORD_COMPLETED = "WORD_COMPLETED"        # A word boundary occurred; awaiting next word
    BUILDING_SENTENCE = "BUILDING_SENTENCE"  # Multiple words present; actively accumulating next word


class TranslationCommand(Enum):
    """Explicit semantic control commands accepted by the translation engine."""

    LETTER = "LETTER"
    SPACE = "SPACE"
    BACKSPACE = "BACKSPACE"
    CLEAR_WORD = "CLEAR_WORD"
    CLEAR_SENTENCE = "CLEAR_SENTENCE"
    RESET = "RESET"
    UNDO = "UNDO"


class TranslationEventType(Enum):
    """Classification of translation engine event outcomes."""

    TOKEN_ACCEPTED = "TOKEN_ACCEPTED"
    WORD_COMPLETED = "WORD_COMPLETED"
    SPACE_INSERTED = "SPACE_INSERTED"
    BACKSPACE = "BACKSPACE"
    CLEAR_WORD = "CLEAR_WORD"
    CLEAR_SENTENCE = "CLEAR_SENTENCE"
    RESET = "RESET"
    UNDO = "UNDO"
    SENTENCE_UPDATED = "SENTENCE_UPDATED"
    IGNORED = "IGNORED"


@dataclass(frozen=True)
class StableSignToken:
    """Immutable representation of a recognized sign token entering the translation engine.

    Attributes:
        label: Text string of the sign (e.g. "A".."Z", "SPACE", "DEL").
        class_id: Numeric class identifier from classifier, or -1 for custom tokens.
        confidence: Posterior probability score in [0.0, 1.0].
        timestamp: Wall-clock timestamp in seconds when token was stabilized.
        duration: Dwell duration in seconds during temporal stabilization.
    """

    label: str
    class_id: int
    confidence: float
    timestamp: float
    duration: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp < 0.0:
            raise ValueError(f"Timestamp must be non-negative, got {self.timestamp}")
        if math.isnan(self.confidence) or math.isinf(self.confidence):
            raise ValueError(f"Confidence must be finite, got {self.confidence}")
        if not (0.0 <= self.confidence <= 1.0001):
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")
        if not self.label:
            raise ValueError("Token label cannot be empty.")

    @classmethod
    def from_stable_sign(cls, stable_sign: Any) -> "StableSignToken":
        """Constructs a StableSignToken from a Stage 7 StableSign object."""
        return cls(
            label=str(stable_sign.label),
            class_id=int(stable_sign.class_id),
            confidence=float(stable_sign.confidence),
            timestamp=float(stable_sign.end_timestamp),
            duration=float(getattr(stable_sign, "duration", 0.0)),
        )

    @classmethod
    def from_char(
        cls,
        char: str,
        timestamp: Optional[float] = None,
        confidence: float = 1.0,
        class_id: int = -1,
    ) -> "StableSignToken":
        """Factory constructor for character input (e.g. interactive simulation / keyboard)."""
        clean_char = str(char).strip().upper()
        if not clean_char:
            raise ValueError("Character label cannot be empty or whitespace.")
        ts = time.time() if timestamp is None else float(timestamp)
        return cls(
            label=clean_char,
            class_id=class_id,
            confidence=float(confidence),
            timestamp=ts,
            duration=0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "class_id": self.class_id,
            "confidence": round(float(self.confidence), 4),
            "timestamp": round(float(self.timestamp), 4),
            "duration": round(float(self.duration), 4),
        }


@dataclass(frozen=True)
class Word:
    """Immutable completed word constructed from a sequence of tokens.

    Attributes:
        text: String representation of the word (e.g. "HELLO").
        tokens: Tuple of contributing StableSignToken instances.
        start_timestamp: Timestamp of the first token in the word.
        end_timestamp: Timestamp of the final token in the word.
        confidence: Mean confidence across all contributing tokens.
    """

    text: str
    tokens: tuple[StableSignToken, ...] = field(default_factory=tuple)
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    confidence: float = 1.0

    @classmethod
    def from_tokens(cls, tokens: Sequence[StableSignToken]) -> "Word":
        """Builds a Word from an ordered sequence of tokens."""
        if not tokens:
            raise ValueError("Cannot construct a Word from an empty sequence of tokens.")
        text = "".join(t.label for t in tokens)
        mean_conf = sum(t.confidence for t in tokens) / len(tokens)
        return cls(
            text=text,
            tokens=tuple(tokens),
            start_timestamp=tokens[0].timestamp,
            end_timestamp=tokens[-1].timestamp,
            confidence=round(mean_conf, 4),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tokens_count": len(self.tokens),
            "start_timestamp": round(float(self.start_timestamp), 4),
            "end_timestamp": round(float(self.end_timestamp), 4),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass(frozen=True)
class Sentence:
    """Immutable sentence representation constructed from a sequence of words.

    Attributes:
        words: Tuple of Word objects comprising the sentence.
        text: Normalized, space-delimited string representation.
        start_timestamp: Timestamp of the first token of the first word.
        end_timestamp: Timestamp of the final token of the final word.
    """

    words: tuple[Word, ...] = field(default_factory=tuple)
    text: str = ""
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0

    @classmethod
    def from_words(cls, words: Sequence[Word]) -> "Sentence":
        """Builds a Sentence from an ordered sequence of Word objects."""
        if not words:
            return cls(words=(), text="", start_timestamp=0.0, end_timestamp=0.0)
        text = " ".join(w.text for w in words)
        return cls(
            words=tuple(words),
            text=text,
            start_timestamp=words[0].start_timestamp,
            end_timestamp=words[-1].end_timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "word_count": len(self.words),
            "words": [w.to_dict() for w in self.words],
            "start_timestamp": round(float(self.start_timestamp), 4),
            "end_timestamp": round(float(self.end_timestamp), 4),
        }


@dataclass(frozen=True)
class TranslationResult:
    """Telemetry and outcome returned by TranslationEngine upon processing an event.

    Attributes:
        state: Current FSM state.
        current_word: In-progress word text currently in the letter buffer.
        sentence_text: Full active sentence text (including completed words + current word).
        completed_words: List of string representations of all finalized words.
        event_type: The specific event executed during this update.
        last_token: The most recently processed token, if applicable.
        tokens_count: Total accepted tokens accumulated across current sentence.
        processing_latency_ms: Processing latency of the translation step in milliseconds.
    """

    state: TranslationState
    current_word: str
    sentence_text: str
    completed_words: list[str]
    event_type: TranslationEventType
    last_token: Optional[StableSignToken] = None
    tokens_count: int = 0
    processing_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "current_word": self.current_word,
            "sentence_text": self.sentence_text,
            "completed_words": self.completed_words,
            "event_type": self.event_type.value,
            "last_token": self.last_token.label if self.last_token else None,
            "tokens_count": self.tokens_count,
            "processing_latency_ms": round(float(self.processing_latency_ms), 4),
        }
