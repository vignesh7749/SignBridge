"""Module 10: Sign-to-Character & Linguistic Token Mapping."""

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    CHARACTER = auto()
    WORD = auto()
    SPACE = auto()
    BACKSPACE = auto()
    CLEAR = auto()
    PUNCTUATION = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class LinguisticToken:
    token_type: TokenType
    value: str


class SignTokenMapper:
    """Translates sign classifier labels into semantic linguistic tokens."""

    _CUSTOM_MAP = {
        "space": LinguisticToken(TokenType.SPACE, " "),
        "del": LinguisticToken(TokenType.BACKSPACE, ""),
        "backspace": LinguisticToken(TokenType.BACKSPACE, ""),
        "clear": LinguisticToken(TokenType.CLEAR, ""),
        "period": LinguisticToken(TokenType.PUNCTUATION, "."),
        "question": LinguisticToken(TokenType.PUNCTUATION, "?"),
    }

    @classmethod
    def map_sign(cls, sign_label: str) -> LinguisticToken:
        """Converts a sign string (e.g., 'A', 'B', 'space') to a LinguisticToken."""
        normalized = sign_label.strip().lower()
        if not normalized:
            return LinguisticToken(TokenType.UNKNOWN, "")

        if normalized in cls._CUSTOM_MAP:
            return cls._CUSTOM_MAP[normalized]

        if len(sign_label) == 1 and sign_label.isalpha():
            return LinguisticToken(TokenType.CHARACTER, sign_label.upper())

        return LinguisticToken(TokenType.WORD, sign_label)
