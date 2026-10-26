"""Module 11: Word Formation & Dwell-Time Debounce."""

import time
from dataclasses import dataclass
from typing import Optional

from signbridge.nlp.mapping import LinguisticToken, TokenType


@dataclass(frozen=True)
class WordEvent:
    current_word: str
    committed_word: Optional[str] = None
    character_appended: Optional[str] = None


class WordBuilder:
    """Accumulates characters into words using dwell-time debouncing."""

    def __init__(self, dwell_seconds: float = 0.5) -> None:
        self.dwell_seconds = dwell_seconds
        self._current_word: list[str] = []
        self._active_token: Optional[LinguisticToken] = None
        self._token_first_seen: float = 0.0
        self._token_committed = False

    def process_token(self, token: LinguisticToken, now: Optional[float] = None) -> WordEvent:
        """Processes a token and checks dwell duration."""
        current_time = now if now is not None else time.time()

        # If token changed, reset dwell timer
        if self._active_token != token:
            self._active_token = token
            self._token_first_seen = current_time
            self._token_committed = False
            return WordEvent(current_word="".join(self._current_word))

        # If token is held and not yet committed, check elapsed time
        elapsed = current_time - self._token_first_seen
        if elapsed >= self.dwell_seconds and not self._token_committed:
            self._token_committed = True
            return self._commit_token(token)

        return WordEvent(current_word="".join(self._current_word))

    def _commit_token(self, token: LinguisticToken) -> WordEvent:
        """Commits action based on token type."""
        if token.token_type == TokenType.CHARACTER:
            self._current_word.append(token.value)
            return WordEvent(
                current_word="".join(self._current_word),
                character_appended=token.value,
            )
        elif token.token_type == TokenType.BACKSPACE:
            if self._current_word:
                self._current_word.pop()
            return WordEvent(current_word="".join(self._current_word))
        elif token.token_type in (TokenType.SPACE, TokenType.WORD):
            completed = "".join(self._current_word)
            self._current_word.clear()
            word_to_commit = token.value if token.token_type == TokenType.WORD else completed
            return WordEvent(
                current_word="",
                committed_word=word_to_commit if word_to_commit else None,
            )
        elif token.token_type == TokenType.CLEAR:
            self._current_word.clear()
            return WordEvent(current_word="")

        return WordEvent(current_word="".join(self._current_word))

    def force_commit(self) -> Optional[str]:
        """Forces commitment of active word (e.g., when signing stops)."""
        if self._current_word:
            word = "".join(self._current_word)
            self._current_word.clear()
            return word
        return None

    @property
    def current_word(self) -> str:
        return "".join(self._current_word)
