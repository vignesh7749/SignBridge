"""Module 14: Text-To-Speech Contract."""

from abc import ABC, abstractmethod


class ITextToSpeech(ABC):
    """Abstract interface for voice synthesis backends."""

    @abstractmethod
    def speak(self, text: str, non_blocking: bool = True) -> None:
        """Speaks the text without blocking real-time video processing."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Interrupts and stops ongoing speech playback."""
        pass

    @property
    @abstractmethod
    def is_speaking(self) -> bool:
        """Returns True if speech synthesis is currently active."""
        pass
