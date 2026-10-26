"""Audio synthesis and Text-To-Speech pipeline."""

from signbridge.audio.base import ITextToSpeech
from signbridge.audio.tts_engine import AsyncTTSManager
from signbridge.tts import (
    ITTSProvider,
    MockTTSProvider,
    Pyttsx3TTSProvider,
    SpeechController,
    TTSConfig,
    TTSStatus,
)

__all__ = [
    "ITextToSpeech",
    "AsyncTTSManager",
    "SpeechController",
    "TTSConfig",
    "TTSStatus",
    "ITTSProvider",
    "Pyttsx3TTSProvider",
    "MockTTSProvider",
]
