"""Module 12: Sentence Formation & Punctuation Assembly."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SentenceState:
    text: str
    completed_sentence: Optional[str] = None


class SentenceBuilder:
    """Assembles committed words into grammatically capitalized sentences."""

    def __init__(self) -> None:
        self._words: list[str] = []

    def add_word(self, word: str) -> SentenceState:
        """Appends a word to the active sentence."""
        clean_word = word.strip()
        if not clean_word:
            return SentenceState(text=self.current_text)

        self._words.append(clean_word)
        return SentenceState(text=self.current_text)

    def add_punctuation(self, mark: str) -> SentenceState:
        """Appends punctuation (e.g. '.', '?') and marks sentence as complete."""
        if not self._words:
            return SentenceState(text="")

        sentence = self.current_text + mark
        self._words.clear()
        return SentenceState(text="", completed_sentence=sentence)

    def finalize(self) -> Optional[str]:
        """Finalizes and returns current sentence, then clears buffer."""
        if not self._words:
            return None
        sentence = self.current_text
        if not sentence.endswith((".", "?", "!")):
            sentence += "."
        self._words.clear()
        return sentence

    def clear(self) -> None:
        """Clears accumulated sentence buffer."""
        self._words.clear()

    @property
    def current_text(self) -> str:
        """Joins words with spacing and capitalizes first character."""
        if not self._words:
            return ""
        raw = " ".join(self._words)
        return raw[0].upper() + raw[1:] if raw else ""
