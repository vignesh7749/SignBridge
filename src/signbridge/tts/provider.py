"""Stage 10: Concrete TTS Providers.

Implements offline Pyttsx3 (SAPI5) synthesizer, fast in-memory Mock provider,
and simulated failure provider for defensive testing.
"""

import sys
import threading
import time
from typing import Any, Optional

from signbridge.core.logger import get_logger
from signbridge.tts.base import ITTSProvider, TTSStatus

logger = get_logger("tts.provider")


class Pyttsx3TTSProvider(ITTSProvider):
    """Offline voice synthesis provider using native Windows SAPI5 with pyttsx3 fallback.

    Guarantees:
    - Never permanently locks into an ERROR state.
    - Synchronous, non-blocking execution inside worker thread.
    - Full thread isolation via COM CoInitialize per calling thread.
    - Instant interruption via SVSFPurgeBeforeSpeak without state corruption.
    - Multiple sequential utterances speak cleanly and repeatedly.
    """

    def __init__(
        self,
        rate: int = 175,
        volume: float = 1.0,
        voice_id: Optional[str] = None,
    ) -> None:
        self._rate = rate
        self._volume = volume
        self._voice_id = voice_id
        self._local = threading.local()
        self._lock = threading.Lock()
        self._is_speaking = False
        self._status = TTSStatus.STOPPED
        self._error_message: Optional[str] = None
        self._active_engine: Optional[Any] = None
        self._check_system_availability()

    @staticmethod
    def _is_sapi5(engine: Any) -> bool:
        """Returns True if engine instance is native Windows SAPI5 SpVoice."""
        return hasattr(engine, "GetVoices") and not hasattr(engine, "say")

    def _create_engine_instance(self) -> Optional[Any]:
        """Creates an independent TTS engine instance (native SAPI5 on Windows, pyttsx3 fallback)."""
        if sys.platform == "win32":
            try:
                import pythoncom
                pythoncom.CoInitialize()
                import win32com.client
                sp = win32com.client.Dispatch("SAPI.SpVoice")
                # SAPI Rate is integer [-10, 10], default 0 (~175 wpm)
                sapi_rate = max(-10, min(10, int(round((self._rate - 175) / 15.0))))
                sp.Rate = sapi_rate
                sp.Volume = int(max(0.0, min(1.0, self._volume)) * 100)
                if self._voice_id:
                    voices = sp.GetVoices()
                    for i in range(voices.Count):
                        v = voices.Item(i)
                        if v.Id == self._voice_id or self._voice_id in v.Id or self._voice_id in v.GetDescription():
                            sp.Voice = v
                            break
                return sp
            except Exception as e:
                logger.warning("Native Windows SAPI5 dispatch failed (%s), falling back to pyttsx3", e)

        try:
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass

            import pyttsx3
            if hasattr(pyttsx3, "_activeEngines"):
                pyttsx3._activeEngines.clear()

            from pyttsx3.engine import Engine
            engine = Engine()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            if self._voice_id:
                engine.setProperty("voice", self._voice_id)
            return engine
        except Exception as e:
            logger.error("Error creating pyttsx3 Engine: %s", e)
            return None

    def _check_system_availability(self) -> None:
        """Verifies TTS backend and system voice viability on init."""
        try:
            test_engine = self._create_engine_instance()
            if test_engine is None:
                self._status = TTSStatus.ERROR
                self._error_message = "Voice output unavailable — no compatible Windows TTS voice was found."
                logger.error(self._error_message)
                return

            if self._is_sapi5(test_engine):
                voices_count = test_engine.GetVoices().Count
            else:
                voices = test_engine.getProperty("voices")
                voices_count = len(voices) if voices else 0

            if voices_count == 0:
                self._status = TTSStatus.ERROR
                self._error_message = "Voice output unavailable — no compatible Windows TTS voice was found."
                logger.error(self._error_message)
            else:
                self._status = TTSStatus.READY
                self._error_message = None
                backend_name = "Native SAPI5" if self._is_sapi5(test_engine) else getattr(test_engine, "driver_name", "sapi5")
                logger.info(
                    "Pyttsx3TTSProvider verified with %d system voice(s). Backend: %s, Rate: %d, Volume: %.2f",
                    voices_count,
                    backend_name,
                    self._rate,
                    self._volume,
                )
        except Exception as e:
            self._status = TTSStatus.ERROR
            self._error_message = "Voice output unavailable — no compatible Windows TTS voice was found."
            logger.error("Failed to initialize Pyttsx3TTSProvider: %s", e)

    def _get_engine(self) -> Optional[Any]:
        """Retrieves or creates thread-local TTS engine for the active calling thread."""
        engine = getattr(self._local, "engine", None)
        if engine is not None:
            return engine

        engine = self._create_engine_instance()
        if engine is not None:
            self._local.engine = engine
            return engine
        else:
            self._status = TTSStatus.ERROR
            self._error_message = "Voice output unavailable — no compatible Windows TTS voice was found."
            return None

    def speak(self, text: str) -> None:
        """Synthesizes speech and plays audio synchronously on caller thread.

        Guarantees:
        - Auto-recovers from any prior errors.
        - Transitions READY -> SPEAKING -> READY.
        - Never latches into a permanent muted ERROR state.
        """
        if not text:
            return

        # Auto-recover if previously halted or in transient error
        if self._status in (TTSStatus.ERROR, TTSStatus.STOPPED):
            self._status = TTSStatus.READY
            self._error_message = None

        engine = self._get_engine()
        if engine is None:
            engine = self._create_engine_instance()
            if engine is not None:
                self._local.engine = engine
            else:
                self._status = TTSStatus.ERROR
                return

        with self._lock:
            self._active_engine = engine

        try:
            self._is_speaking = True
            self._status = TTSStatus.SPEAKING
            logger.debug("Pyttsx3TTSProvider synthesizing utterance: '%s'", text)
            if self._is_sapi5(engine):
                # Synchronous native Windows speech (0 = SVSFDefault)
                engine.Speak(text, 0)
            else:
                engine.say(text)
                engine.runAndWait()
            self._error_message = None
        except Exception as e:
            logger.error("Pyttsx3TTSProvider error during speak('%s'): %s", text, e)
            self._error_message = f"Voice output unavailable — speech error: {e}"
            # Invalidate thread-local engine so next call gets a clean instance
            self._local.engine = None
        finally:
            with self._lock:
                self._active_engine = None
            self._is_speaking = False
            # Never stay permanently stuck in ERROR or SPEAKING
            self._status = TTSStatus.READY

    def stop(self) -> None:
        """Interrupts ongoing speech synthesis cleanly."""
        with self._lock:
            engine = self._active_engine
        if engine is not None:
            try:
                if self._is_sapi5(engine):
                    # Flag 2 = SVSFPurgeBeforeSpeak halts active playback immediately
                    engine.Speak("", 2)
                else:
                    engine.stop()
                logger.debug("Pyttsx3TTSProvider stop called.")
            except Exception as e:
                logger.warning("Error stopping TTS engine: %s", e)
        self._is_speaking = False
        if self._status != TTSStatus.ERROR:
            self._status = TTSStatus.READY

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def get_status(self) -> TTSStatus:
        return self._status

    @property
    def error_message(self) -> Optional[str]:
        """Returns descriptive error message if voice engine is unviable."""
        return self._error_message

    def set_rate(self, rate: int) -> None:
        self._rate = rate
        engine = getattr(self._local, "engine", None)
        if engine is not None:
            try:
                if self._is_sapi5(engine):
                    sapi_rate = max(-10, min(10, int(round((rate - 175) / 15.0))))
                    engine.Rate = sapi_rate
                else:
                    engine.setProperty("rate", rate)
            except Exception as e:
                logger.warning("Failed to set rate: %s", e)

    def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(1.0, volume))
        engine = getattr(self._local, "engine", None)
        if engine is not None:
            try:
                if self._is_sapi5(engine):
                    engine.Volume = int(self._volume * 100)
                else:
                    engine.setProperty("volume", self._volume)
            except Exception as e:
                logger.warning("Failed to set volume: %s", e)

    def get_available_voices(self) -> list[dict[str, Any]]:
        engine = self._get_engine()
        if engine is None:
            return []
        try:
            if self._is_sapi5(engine):
                voices = engine.GetVoices()
                result = []
                for i in range(voices.Count):
                    v = voices.Item(i)
                    result.append({
                        "id": v.Id,
                        "name": v.GetDescription(),
                        "languages": [],
                    })
                return result
            else:
                voices = engine.getProperty("voices")
                return [
                    {
                        "id": getattr(v, "id", str(v)),
                        "name": getattr(v, "name", "unknown"),
                        "languages": getattr(v, "languages", []),
                    }
                    for v in voices
                ]
        except Exception as e:
            logger.warning("Failed to get voices: %s", e)
            return []

    def set_voice(self, voice_id: str) -> None:
        self._voice_id = voice_id
        engine = getattr(self._local, "engine", None)
        if engine is not None:
            try:
                if self._is_sapi5(engine):
                    voices = engine.GetVoices()
                    for i in range(voices.Count):
                        v = voices.Item(i)
                        if v.Id == voice_id or voice_id in v.Id or voice_id in v.GetDescription():
                            engine.Voice = v
                            break
                else:
                    engine.setProperty("voice", voice_id)
            except Exception as e:
                logger.warning("Failed to set voice: %s", e)


class MockTTSProvider(ITTSProvider):
    """In-memory mock TTS provider for deterministic testing without physical audio devices."""

    def __init__(
        self,
        rate: int = 175,
        volume: float = 1.0,
        simulated_delay: float = 0.0,
    ) -> None:
        self._rate = rate
        self._volume = volume
        self.simulated_delay = simulated_delay
        self.spoken_history: list[str] = []
        self.call_count = 0
        self._is_speaking = False
        self._status = TTSStatus.READY
        self._lock = threading.Lock()
        self._interrupted = False

    def speak(self, text: str) -> None:
        clean = text.strip()
        if not clean:
            return

        with self._lock:
            self._is_speaking = True
            self._status = TTSStatus.SPEAKING
            self.call_count += 1
            self.spoken_history.append(clean)
            self._interrupted = False

        if self.simulated_delay > 0.0:
            elapsed = 0.0
            step = 0.01
            while elapsed < self.simulated_delay and not self._interrupted:
                time.sleep(step)
                elapsed += step

        with self._lock:
            self._is_speaking = False
            self._status = TTSStatus.READY

    def stop(self) -> None:
        with self._lock:
            self._interrupted = True
            self._is_speaking = False
            self._status = TTSStatus.STOPPED

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._is_speaking

    def get_status(self) -> TTSStatus:
        with self._lock:
            return self._status

    def set_rate(self, rate: int) -> None:
        self._rate = rate

    def set_volume(self, volume: float) -> None:
        self._volume = volume

    def get_available_voices(self) -> list[dict[str, Any]]:
        return [
            {"id": "mock_voice_1", "name": "Mock Male Voice", "languages": ["en_US"]},
            {"id": "mock_voice_2", "name": "Mock Female Voice", "languages": ["en_US"]},
        ]

    def set_voice(self, voice_id: str) -> None:
        pass

    def clear_history(self) -> None:
        with self._lock:
            self.spoken_history.clear()
            self.call_count = 0

    @property
    def error_message(self) -> Optional[str]:
        return None



class FailingTTSProvider(ITTSProvider):
    """Simulates hard engine failures to verify fault isolation."""

    def __init__(self) -> None:
        self._status = TTSStatus.ERROR
        self._is_speaking = False
        self._error_message = "Simulated critical audio device hardware failure."

    def speak(self, text: str) -> None:
        self._status = TTSStatus.ERROR
        raise RuntimeError("Simulated critical audio device hardware failure.")

    def stop(self) -> None:
        self._status = TTSStatus.ERROR

    @property
    def is_speaking(self) -> bool:
        return False

    def get_status(self) -> TTSStatus:
        return self._status

    @property
    def error_message(self) -> Optional[str]:
        return self._error_message

    def set_rate(self, rate: int) -> None:
        pass

    def set_volume(self, volume: float) -> None:
        pass

    def get_available_voices(self) -> list[dict[str, Any]]:
        return []

    def set_voice(self, voice_id: str) -> None:
        pass

