"""Stage 10: Speech Event Contracts.

Defines domain event models that trigger voice synthesis actions.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any


class SpeechEventType(Enum):
    """Semantic origin of a speech synthesis request."""

    WORD_COMPLETED = "WORD_COMPLETED"
    SENTENCE_COMPLETED = "SENTENCE_COMPLETED"
    MANUAL_SPEAK = "MANUAL_SPEAK"


@dataclass(frozen=True)
class SpeechEvent:
    """Immutable record representing an utterance to be synthesized.

    Attributes:
        event_type: Semantic categorization of speech trigger.
        text: Utterance text to be spoken.
        timestamp: Creation timestamp in seconds.
        metadata: Optional metadata dictionary.
    """

    event_type: SpeechEventType
    text: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("SpeechEvent text cannot be empty.")
