"""Stage 10: Text-to-Speech (TTS) Subsystem Public Interface."""

from signbridge.tts.base import ITTSProvider, TTSStatus
from signbridge.tts.config import TTSConfig
from signbridge.tts.controller import SpeechController
from signbridge.tts.events import SpeechEvent, SpeechEventType
from signbridge.tts.provider import FailingTTSProvider, MockTTSProvider, Pyttsx3TTSProvider

__all__ = [
    "ITTSProvider",
    "TTSStatus",
    "TTSConfig",
    "SpeechEvent",
    "SpeechEventType",
    "Pyttsx3TTSProvider",
    "MockTTSProvider",
    "FailingTTSProvider",
    "SpeechController",
]
