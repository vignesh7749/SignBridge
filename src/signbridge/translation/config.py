"""Stage 8 Translation Configuration.

Defines centralized, type-safe settings for word boundary timeouts,
token history memory limits, undo stack depth, and text normalization policies.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TranslationConfig:
    """Configuration parameters for Stage 8 Translation Engine.

    Attributes:
        word_boundary_pause_seconds: Time in seconds of inactive signing after which a word
            is automatically completed (if auto_word_boundary is enabled).
        auto_word_boundary: If True, pauses in signing automatically commit words;
            if False, words require explicit SPACE events or boundary triggers.
        max_token_history: Maximum capacity of the bounded recent token audit deque.
        max_undo_depth: Maximum number of previous snapshots preserved on the undo stack.
        auto_capitalize: If True, capitalizes the first letter of sentences and words.
        trim_whitespace: If True, trims extraneous whitespace and collapses duplicate spaces.
    """

    word_boundary_pause_seconds: float = 1.2
    auto_word_boundary: bool = False
    max_token_history: int = 200
    max_undo_depth: int = 50
    auto_capitalize: bool = False
    trim_whitespace: bool = True

    def __post_init__(self) -> None:
        if self.word_boundary_pause_seconds <= 0.0:
            raise ValueError(
                f"word_boundary_pause_seconds must be positive, got {self.word_boundary_pause_seconds}"
            )
        if self.max_token_history <= 0:
            raise ValueError(
                f"max_token_history must be positive, got {self.max_token_history}"
            )
        if self.max_undo_depth <= 0:
            raise ValueError(
                f"max_undo_depth must be positive, got {self.max_undo_depth}"
            )
