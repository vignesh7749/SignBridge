"""Stage 8 Translation Engine.

Converts sequences of stable sign tokens into letters, words, and sentences
via a deterministic finite state machine with full support for word boundaries,
duplicate space suppression, deterministic backspace, clear, reset, and multi-step undo.
"""

from collections import deque
from dataclasses import dataclass
import time
from typing import Any, Optional, Sequence

from signbridge.core.logger import get_logger
from signbridge.translation.config import TranslationConfig
from signbridge.translation.contracts import (
    Sentence,
    StableSignToken,
    TranslationCommand,
    TranslationEventType,
    TranslationResult,
    TranslationState,
    Word,
)
from signbridge.translation.normalizer import format_sentence_text

logger = get_logger("translation.engine")


@dataclass(frozen=True)
class _TranslationSnapshot:
    """Internal immutable snapshot for deterministic multi-step undo."""

    completed_words: tuple[Word, ...]
    current_tokens: tuple[StableSignToken, ...]
    state: TranslationState


class TranslationEngine:
    """Deterministic token-to-word-to-sentence translation engine."""

    def __init__(self, config: Optional[TranslationConfig] = None) -> None:
        self.config = config or TranslationConfig()

        self._state: TranslationState = TranslationState.EMPTY
        self._completed_words: list[Word] = []
        self._current_tokens: list[StableSignToken] = []

        self._token_history: deque[StableSignToken] = deque(
            maxlen=self.config.max_token_history
        )
        self._undo_stack: deque[_TranslationSnapshot] = deque(
            maxlen=self.config.max_undo_depth
        )

        self._last_token_timestamp: float = 0.0

    @property
    def current_state(self) -> TranslationState:
        """Current lifecycle state of the translation state machine."""
        return self._state

    @property
    def current_word(self) -> str:
        """String representation of the active in-progress word."""
        return "".join(t.label for t in self._current_tokens)

    @property
    def completed_words(self) -> list[str]:
        """List of string representations of all completed words in current sentence."""
        return [w.text for w in self._completed_words]

    @property
    def sentence_text(self) -> str:
        """Current formatted sentence text (completed words + active word)."""
        return format_sentence_text(
            self.completed_words,
            self.current_word,
            auto_capitalize=self.config.auto_capitalize,
        )

    @property
    def tokens_count(self) -> int:
        """Total accepted tokens accumulated across current sentence."""
        completed_tokens_count = sum(len(w.tokens) for w in self._completed_words)
        return completed_tokens_count + len(self._current_tokens)

    @property
    def token_history(self) -> Sequence[StableSignToken]:
        """Read-only sequence of recent accepted tokens."""
        return list(self._token_history)

    def get_sentence(self) -> Sentence:
        """Constructs an immutable Sentence object of all finalized words."""
        return Sentence.from_words(self._completed_words)

    # ----------------------------------------------------------------------
    # Core Processing Methods
    # ----------------------------------------------------------------------

    def process_token(self, token: Any) -> TranslationResult:
        """Ingests a sign token (StableSignToken or Stage 7 StableSign).

        Args:
            token: StableSignToken, Stage 7 StableSign, or object convertible to token.

        Returns:
            TranslationResult containing updated sentence, word, state, and event telemetry.
        """
        start_t = time.perf_counter()

        if isinstance(token, StableSignToken):
            st_token = token
        elif hasattr(token, "label") and hasattr(token, "confidence"):
            st_token = StableSignToken.from_stable_sign(token)
        else:
            raise TypeError(
                f"Expected StableSignToken or StableSign, got {type(token).__name__}"
            )

        # Check for embedded special commands
        normalized_label = st_token.label.strip().upper()
        if normalized_label in ("SPACE", " "):
            res = self.space()
            return self._with_latency(res, start_t, st_token)
        elif normalized_label in ("BACKSPACE", "DEL", "DELETE"):
            res = self.backspace()
            return self._with_latency(res, start_t, st_token)
        elif normalized_label in ("CLEAR", "CLEAR_SENTENCE"):
            res = self.clear_sentence()
            return self._with_latency(res, start_t, st_token)
        elif normalized_label == "CLEAR_WORD":
            res = self.clear_word()
            return self._with_latency(res, start_t, st_token)
        elif normalized_label == "RESET":
            res = self.reset()
            return self._with_latency(res, start_t, st_token)
        elif normalized_label == "UNDO":
            res = self.undo()
            return self._with_latency(res, start_t, st_token)

        # Automatic pause-based word boundary detection
        if self.config.auto_word_boundary and self._current_tokens and self._last_token_timestamp > 0.0:
            pause_duration = st_token.timestamp - self._last_token_timestamp
            if pause_duration >= self.config.word_boundary_pause_seconds:
                # Commit previous word before accepting new letter
                self.space()

        # Push undo snapshot before state modification
        self._push_undo_snapshot()

        # Append letter to current word
        self._current_tokens.append(st_token)
        self._token_history.append(st_token)
        self._last_token_timestamp = st_token.timestamp

        # Update state
        if len(self._completed_words) == 0:
            self._state = TranslationState.BUILDING_WORD
        else:
            self._state = TranslationState.BUILDING_SENTENCE

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return self._make_result(TranslationEventType.TOKEN_ACCEPTED, elapsed_ms, st_token)

    def process_char(
        self,
        char: str,
        timestamp: Optional[float] = None,
        confidence: float = 1.0,
    ) -> TranslationResult:
        """Convenience method for accepting a character string directly."""
        token = StableSignToken.from_char(char, timestamp=timestamp, confidence=confidence)
        return self.process_token(token)

    def process_word(
        self,
        word: str,
        timestamp: Optional[float] = None,
        confidence: float = 1.0,
    ) -> TranslationResult:
        """Directly accepts an entire word (e.g. from word-level ASL sign recognition).

        Commits any pending in-progress letters, pushes an undo snapshot,
        and appends the whole word directly into completed words.
        """
        start_t = time.perf_counter()
        clean_word = str(word).strip().upper()
        if not clean_word:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return self._make_result(TranslationEventType.IGNORED, elapsed_ms, None)

        ts = time.time() if timestamp is None else float(timestamp)

        # If there are active uncommitted letters from fingerspelling, commit them first
        if self._current_tokens:
            self.space()

        self._push_undo_snapshot()

        token = StableSignToken(
            label=clean_word,
            class_id=-1,
            confidence=float(confidence),
            timestamp=ts,
            duration=0.0,
        )
        word_obj = Word(
            text=clean_word,
            tokens=(token,),
            start_timestamp=ts,
            end_timestamp=ts,
            confidence=float(confidence),
        )
        self._completed_words.append(word_obj)
        self._token_history.append(token)
        self._last_token_timestamp = ts

        if len(self._completed_words) == 1:
            self._state = TranslationState.WORD_COMPLETED
        else:
            self._state = TranslationState.BUILDING_SENTENCE

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return self._make_result(TranslationEventType.WORD_COMPLETED, elapsed_ms, token)

    def process_command(
        self,
        command: TranslationCommand,
        timestamp: Optional[float] = None,
    ) -> TranslationResult:
        """Executes an explicit semantic translation control command."""
        start_t = time.perf_counter()
        if command == TranslationCommand.SPACE:
            res = self.space()
        elif command == TranslationCommand.BACKSPACE:
            res = self.backspace()
        elif command == TranslationCommand.CLEAR_WORD:
            res = self.clear_word()
        elif command == TranslationCommand.CLEAR_SENTENCE:
            res = self.clear_sentence()
        elif command == TranslationCommand.RESET:
            res = self.reset()
        elif command == TranslationCommand.UNDO:
            res = self.undo()
        else:
            res = self._make_result(TranslationEventType.IGNORED, 0.0, None)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return self._with_latency(res, start_t)

    # ----------------------------------------------------------------------
    # Control Actions
    # ----------------------------------------------------------------------

    def space(self) -> TranslationResult:
        """Inserts a word boundary, committing active word to completed words.

        Guarantees idempotency: multiple consecutive space calls do not insert
        empty words or duplicate spaces.
        """
        start_t = time.perf_counter()

        if not self._current_tokens:
            # Idempotent: No active word to commit
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return self._make_result(TranslationEventType.IGNORED, elapsed_ms, None)

        # Save snapshot
        self._push_undo_snapshot()

        # Commit word
        committed_word = Word.from_tokens(self._current_tokens)
        self._completed_words.append(committed_word)
        self._current_tokens.clear()
        self._state = TranslationState.WORD_COMPLETED

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return self._make_result(TranslationEventType.WORD_COMPLETED, elapsed_ms, None)

    def backspace(self) -> TranslationResult:
        """Deletes the most recent letter or re-opens the preceding completed word.

        Deterministic Behavior:
        - If current_word is non-empty: removes the last token.
        - If current_word is empty and completed words exist: pops the last completed
          word back into current_word (re-opening it for editing).
        - If completely empty: safe no-op.
        """
        start_t = time.perf_counter()

        if self._current_tokens:
            self._push_undo_snapshot()
            popped = self._current_tokens.pop()

            # Update state
            if not self._current_tokens:
                if not self._completed_words:
                    self._state = TranslationState.EMPTY
                else:
                    self._state = TranslationState.WORD_COMPLETED

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return self._make_result(TranslationEventType.BACKSPACE, elapsed_ms, popped)

        if self._completed_words:
            self._push_undo_snapshot()
            # Re-open preceding word
            last_word = self._completed_words.pop()
            self._current_tokens = list(last_word.tokens)

            if not self._completed_words:
                self._state = TranslationState.BUILDING_WORD
            else:
                self._state = TranslationState.BUILDING_SENTENCE

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return self._make_result(TranslationEventType.BACKSPACE, elapsed_ms, None)

        # Completely empty: safe no-op
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return self._make_result(TranslationEventType.IGNORED, elapsed_ms, None)

    def clear_word(self) -> TranslationResult:
        """Clears the active word in the letter buffer without modifying completed words."""
        start_t = time.perf_counter()

        if not self._current_tokens:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return self._make_result(TranslationEventType.IGNORED, elapsed_ms, None)

        self._push_undo_snapshot()
        self._current_tokens.clear()

        if not self._completed_words:
            self._state = TranslationState.EMPTY
        else:
            self._state = TranslationState.WORD_COMPLETED

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return self._make_result(TranslationEventType.CLEAR_WORD, elapsed_ms, None)

    def clear_sentence(self) -> TranslationResult:
        """Clears all completed words and active in-progress characters."""
        start_t = time.perf_counter()

        if not self._completed_words and not self._current_tokens:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return self._make_result(TranslationEventType.IGNORED, elapsed_ms, None)

        self._push_undo_snapshot()
        self._completed_words.clear()
        self._current_tokens.clear()
        self._state = TranslationState.EMPTY

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return self._make_result(TranslationEventType.CLEAR_SENTENCE, elapsed_ms, None)

    def reset(self) -> TranslationResult:
        """Completely clears all translation state, words, history, and undo stack."""
        start_t = time.perf_counter()

        self._completed_words.clear()
        self._current_tokens.clear()
        self._token_history.clear()
        self._undo_stack.clear()
        self._state = TranslationState.EMPTY
        self._last_token_timestamp = 0.0

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return self._make_result(TranslationEventType.RESET, elapsed_ms, None)

    def undo(self) -> TranslationResult:
        """Restores the exact prior state snapshot from the undo stack."""
        start_t = time.perf_counter()

        if not self._undo_stack:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return self._make_result(TranslationEventType.IGNORED, elapsed_ms, None)

        snapshot = self._undo_stack.pop()
        self._completed_words = list(snapshot.completed_words)
        self._current_tokens = list(snapshot.current_tokens)
        self._state = snapshot.state

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return self._make_result(TranslationEventType.UNDO, elapsed_ms, None)

    # ----------------------------------------------------------------------
    # Internal Helpers
    # ----------------------------------------------------------------------

    def _push_undo_snapshot(self) -> None:
        """Saves current state snapshot onto the bounded undo stack."""
        snapshot = _TranslationSnapshot(
            completed_words=tuple(self._completed_words),
            current_tokens=tuple(self._current_tokens),
            state=self._state,
        )
        self._undo_stack.append(snapshot)

    def _make_result(
        self,
        event_type: TranslationEventType,
        latency_ms: float,
        last_token: Optional[StableSignToken],
    ) -> TranslationResult:
        """Constructs a standardized TranslationResult."""
        return TranslationResult(
            state=self._state,
            current_word=self.current_word,
            sentence_text=self.sentence_text,
            completed_words=self.completed_words,
            event_type=event_type,
            last_token=last_token,
            tokens_count=self.tokens_count,
            processing_latency_ms=round(latency_ms, 4),
        )

    def _with_latency(
        self,
        res: TranslationResult,
        start_t: float,
        token: Optional[StableSignToken] = None,
    ) -> TranslationResult:
        """Enriches result with total latency and token reference."""
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return TranslationResult(
            state=res.state,
            current_word=res.current_word,
            sentence_text=res.sentence_text,
            completed_words=res.completed_words,
            event_type=res.event_type,
            last_token=token or res.last_token,
            tokens_count=res.tokens_count,
            processing_latency_ms=round(elapsed_ms, 4),
        )
