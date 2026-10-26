"""Stage 10: Text-to-Speech (TTS) Base Interfaces and Data Contracts.

Defines the abstract contract for offline voice synthesis providers and status states.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class TTSStatus(Enum):
    """Lifecycle and operational status of the TTS subsystem."""

    READY = "READY"
    SPEAKING = "SPEAKING"
    STOPPED = "STOPPED"
    DISABLED = "DISABLED"
    ERROR = "ERROR"


class ITTSProvider(ABC):
    """Abstract interface for offline voice synthesis backends."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Synthesizes and plays speech for the provided text.

        Must be callable synchronously within the dedicated worker thread.

        Args:
            text: Normalized text string to speak.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Immediately interrupts and halts any active speech synthesis/playback."""
        pass

    @property
    @abstractmethod
    def is_speaking(self) -> bool:
        """Returns True if speech playback is actively occurring."""
        pass

    @abstractmethod
    def get_status(self) -> TTSStatus:
        """Returns current operational status."""
        pass

    @abstractmethod
    def set_rate(self, rate: int) -> None:
        """Configures speaking rate in words per minute.

        Args:
            rate: Integer words-per-minute (e.g. 150-200).
        """
        pass

    @abstractmethod
    def set_volume(self, volume: float) -> None:
        """Configures audio playback volume.

        Args:
            volume: Float in range [0.0, 1.0].
        """
        pass

    @abstractmethod
    def get_available_voices(self) -> list[dict[str, Any]]:
        """Enumerates installed system voice profiles.

        Returns:
            List of dictionaries containing 'id', 'name', 'languages', etc.
        """
        pass

    @abstractmethod
    def set_voice(self, voice_id: str) -> None:
        """Selects the active voice profile by ID.

        Args:
            voice_id: System voice identifier string.
        """
        pass
