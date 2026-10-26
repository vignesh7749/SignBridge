"""Stage 10: TTS Configuration dataclass.

Configures offline voice synthesis parameters, event routing, and queue constraints.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TTSConfig:
    """Type-safe runtime configuration for speech synthesis."""

    enabled: bool = True
    rate: int = 175                      # Words per minute (100 - 300)
    volume: float = 1.0                  # Volume scaling [0.0, 1.0]
    voice_id: Optional[str] = None       # Specific voice identifier (None uses default)
    speak_words: bool = False            # Speak each completed word at word boundary / recognized word signs
    speak_sentences: bool = True         # Speak full sentence on completion/manual trigger
    speak_stable_signs: bool = True      # Speak stably confirmed signs (event-based)
    speak_letters: bool = False          # Speak individual fingerspelled letters in alphabet mode
    suppress_duplicates: bool = True     # Ignore consecutive identical utterances
    cooldown_seconds: float = 1.0        # Minimum elapsed seconds between speech triggers
    min_word_length: int = 1             # Ignore single-character noise if > 1
    queue_maxsize: int = 10              # Bounded worker queue capacity
    interruption_enabled: bool = True    # Allow stop() to halt playback immediately
    worker_timeout: float = 0.2          # Queue polling timeout in seconds

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError(f"Speech rate must be positive, got {self.rate}")
        if not (0.0 <= self.volume <= 1.0):
            raise ValueError(f"Volume must be in [0.0, 1.0], got {self.volume}")
        if self.queue_maxsize <= 0:
            raise ValueError(f"queue_maxsize must be positive, got {self.queue_maxsize}")
        if self.min_word_length < 0:
            raise ValueError(f"min_word_length must be non-negative, got {self.min_word_length}")
        if self.cooldown_seconds < 0:
            raise ValueError(f"cooldown_seconds must be non-negative, got {self.cooldown_seconds}")
