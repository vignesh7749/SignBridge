"""Stage 8 Translation Package.

Provides token-to-word-to-sentence translation, word boundary management,
duplicate space suppression, backspace, clear, reset, undo, and state management.
"""

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
from signbridge.translation.engine import TranslationEngine
from signbridge.translation.normalizer import format_sentence_text, normalize_whitespace

__all__ = [
    "Sentence",
    "StableSignToken",
    "TranslationCommand",
    "TranslationConfig",
    "TranslationEngine",
    "TranslationEventType",
    "TranslationResult",
    "TranslationState",
    "Word",
    "format_sentence_text",
    "normalize_whitespace",
]
