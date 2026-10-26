"""Stage 10: Speech Controller.

Central thread-safe coordinator managing asynchronous voice synthesis,
bounded queuing, text normalization, duplicate suppression, and interruption mechanics.
"""

import queue
import re
import threading
import time
from typing import Optional

from signbridge.core.logger import get_logger
from signbridge.tts.base import ITTSProvider, TTSStatus
from signbridge.tts.config import TTSConfig
from signbridge.tts.events import SpeechEvent, SpeechEventType
from signbridge.tts.provider import MockTTSProvider, Pyttsx3TTSProvider

logger = get_logger("tts.controller")


class SpeechController:
    """Orchestrates asynchronous voice output with zero camera capture loop blocking."""

    def __init__(
        self,
        config: Optional[TTSConfig] = None,
        provider: Optional[ITTSProvider] = None,
    ) -> None:
        self.config = config or TTSConfig()
        self._lock = threading.RLock()
        self._is_enabled = self.config.enabled

        # Resolve provider
        self._error_message: Optional[str] = None
        if provider is not None:
            self.provider = provider
            self._error_message = getattr(provider, "error_message", None)
        else:
            try:
                self.provider = Pyttsx3TTSProvider(
                    rate=self.config.rate,
                    volume=self.config.volume,
                    voice_id=self.config.voice_id,
                )
                if self.provider.get_status() == TTSStatus.ERROR:
                    self._error_message = (
                        getattr(self.provider, "error_message", None)
                        or "Voice output unavailable — no compatible Windows TTS voice was found."
                    )
                    logger.error("Pyttsx3TTSProvider reported error: %s", self._error_message)
            except Exception as e:
                self._error_message = "Voice output unavailable — no compatible Windows TTS voice was found."
                logger.error("Failed to construct Pyttsx3TTSProvider: %s", e)
                self.provider = MockTTSProvider(rate=self.config.rate, volume=self.config.volume)

        # Worker thread and bounded queue
        self._queue: queue.Queue[SpeechEvent] = queue.Queue(maxsize=self.config.queue_maxsize)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False

        # Telemetry & State
        self._last_spoken_text: Optional[str] = None
        self._last_spoken_token: Optional[str] = None
        self._last_spoken_timestamp: float = 0.0
        self._total_enqueued = 0
        self._total_spoken = 0
        self._total_dropped = 0

    def start(self) -> None:
        """Starts the dedicated background worker thread."""
        with self._lock:
            if self._is_running:
                return

            self._stop_event.clear()
            self._is_running = True
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="SignBridge-TTS-Worker",
                daemon=True,
            )
            self._thread.start()
            logger.info("SpeechController background worker thread started.")

    def shutdown(self, timeout: float = 1.0) -> None:
        """Cleanly terminates the worker thread and releases resources."""
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False
            self._stop_event.set()

        # Stop active playback and drain queue
        self.stop()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("SpeechController background worker thread shut down.")

    def stop(self) -> None:
        """Immediately interrupts active playback and flushes any pending queued utterances."""
        with self._lock:
            # Drain queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._total_dropped += 1
                except queue.Empty:
                    break

            # Interrupt provider playback
            if self.config.interruption_enabled:
                try:
                    self.provider.stop()
                except Exception as e:
                    logger.warning("Error stopping TTS provider: %s", e)

            logger.info("SpeechController stopped: queue purged, active playback interrupted.")

    # -------------------------------------------------------------------------
    # Public Speech Submission Methods
    # -------------------------------------------------------------------------

    def handle_event(self, event: SpeechEvent) -> bool:
        """Processes and enqueues a SpeechEvent.

        Returns:
            True if utterance was successfully accepted into queue, False otherwise.
        """
        with self._lock:
            if not self._is_enabled or not self.config.enabled:
                return False

            if not self._is_running or self._thread is None or not self._thread.is_alive():
                # Auto-start / resurrect worker if not already running
                self._is_running = False
                self.start()

            # 1. Text normalization & cleaning
            clean_text = self._normalize_text(event.text)
            if not clean_text:
                return False

            # Check minimum word length
            if event.event_type == SpeechEventType.WORD_COMPLETED and len(clean_text) < self.config.min_word_length:
                return False

            # 2. Strict Duplicate Speech Suppression
            if self.config.suppress_duplicates and event.event_type != SpeechEventType.MANUAL_SPEAK:
                if clean_text == self._last_spoken_text:
                    logger.debug("Duplicate speech suppressed: '%s'", clean_text)
                    return False

            # 3. Non-blocking enqueue with bounded capacity
            normalized_event = SpeechEvent(
                event_type=event.event_type,
                text=clean_text,
                timestamp=event.timestamp,
                metadata=event.metadata,
            )

            try:
                self._queue.put_nowait(normalized_event)
                self._total_enqueued += 1
                logger.debug("Enqueued TTS event [%s]: '%s'", event.event_type.value, clean_text)
                return True
            except queue.Full:
                self._total_dropped += 1
                logger.warning(
                    "TTS speech queue full (%d items). Dropping utterance: '%s'",
                    self.config.queue_maxsize,
                    clean_text,
                )
                return False

    def handle_word_completed(self, word: Optional[str]) -> bool:
        """Submits a completed word utterance if word speaking is enabled."""
        if not word or not self.config.speak_words:
            return False
        clean = self._normalize_text(word)
        if not clean:
            return False
        event = SpeechEvent(event_type=SpeechEventType.WORD_COMPLETED, text=clean)
        return self.handle_event(event)

    def handle_sentence_completed(self, sentence: Optional[str]) -> bool:
        """Submits a finalized sentence utterance if sentence speaking is enabled."""
        if not sentence or not self.config.speak_sentences:
            return False
        clean = self._normalize_text(sentence)
        if not clean:
            return False
        event = SpeechEvent(event_type=SpeechEventType.SENTENCE_COMPLETED, text=clean)
        return self.handle_event(event)

    def handle_stable_sign(
        self,
        sign_token: Optional[str],
        timestamp: Optional[float] = None,
        force: bool = False,
    ) -> bool:
        """Processes event-based TTS triggering for stably confirmed signs and words.

        Guarantees:
        1. Emits speech when a newly confirmed sign arrives.
        2. Same sign held on screen does NOT repeatedly trigger speech.
        3. Transitioning to a new sign (e.g. HELLO -> HELP -> YES) triggers speech.
        4. Returning to an earlier sign (e.g. YES -> HELLO) triggers speech again.
        5. Enforces configurable cooldown between speech triggers.
        6. Never blocks the camera or UI thread (runs asynchronously).
        """
        if not sign_token:
            return False
        if not self.config.speak_stable_signs and not self.config.speak_words:
            return False

        clean = self._normalize_text(sign_token)
        if not clean or len(clean) < self.config.min_word_length:
            return False

        now = timestamp if timestamp is not None else time.time()

        with self._lock:
            # 1. Check duplicate suppression on identical sign held continuously
            if not force and self.config.suppress_duplicates:
                if clean == self._last_spoken_token:
                    logger.debug("Stable sign speech suppressed (same sign held): '%s'", clean)
                    return False

            # 2. Check configurable cooldown
            if not force and self.config.cooldown_seconds > 0.0:
                elapsed = now - self._last_spoken_timestamp
                if elapsed < self.config.cooldown_seconds and self._last_spoken_timestamp > 0.0:
                    logger.debug(
                        "Stable sign speech suppressed by cooldown (%.2fs < %.2fs): '%s'",
                        elapsed,
                        self.config.cooldown_seconds,
                        clean,
                    )
                    return False

            # 3. Update token and timestamp
            self._last_spoken_token = clean
            self._last_spoken_timestamp = now

        # Enqueue event asynchronously
        event_type = SpeechEventType.MANUAL_SPEAK if force else SpeechEventType.WORD_COMPLETED
        event = SpeechEvent(event_type=event_type, text=clean, timestamp=now)
        return self.handle_event(event)

    def reset_spoken_state(self) -> None:
        """Resets last spoken token tracking to allow re-triggering upon new gestures."""
        with self._lock:
            self._last_spoken_token = None
            self._last_spoken_timestamp = 0.0
            self._last_spoken_text = None

    def speak_manual(self, text: Optional[str]) -> bool:
        """Explicitly speaks text, bypassing duplicate suppression."""
        if not text:
            return False
        clean = self._normalize_text(text)
        if not clean:
            return False
        with self._lock:
            self._last_spoken_token = clean
            self._last_spoken_timestamp = time.time()
        event = SpeechEvent(event_type=SpeechEventType.MANUAL_SPEAK, text=clean)
        return self.handle_event(event)

    # -------------------------------------------------------------------------
    # Control API
    # -------------------------------------------------------------------------

    def toggle_enabled(self) -> bool:
        """Toggles voice synthesis on/off. Returns new state."""
        with self._lock:
            self._is_enabled = not self._is_enabled
            if not self._is_enabled:
                self.stop()
            logger.info("Voice synthesis toggled: %s", "ENABLED" if self._is_enabled else "DISABLED")
            return self._is_enabled

    def set_enabled(self, enabled: bool) -> None:
        """Sets voice synthesis enabled state."""
        with self._lock:
            self._is_enabled = bool(enabled)
            if not self._is_enabled:
                self.stop()

    @property
    def is_enabled(self) -> bool:
        with self._lock:
            return self._is_enabled and self.config.enabled

    @property
    def is_speaking(self) -> bool:
        return self.provider.is_speaking

    @property
    def status(self) -> TTSStatus:
        if not self.is_enabled:
            return TTSStatus.DISABLED
        if self.is_speaking:
            return TTSStatus.SPEAKING
        if self.provider.get_status() == TTSStatus.ERROR:
            return TTSStatus.ERROR
        return TTSStatus.READY

    @property
    def error_message(self) -> Optional[str]:
        with self._lock:
            return self._error_message or getattr(self.provider, "error_message", None)

    @property
    def last_spoken_text(self) -> Optional[str]:
        with self._lock:
            return self._last_spoken_text

    @property
    def last_spoken_token(self) -> Optional[str]:
        with self._lock:
            return self._last_spoken_token

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    # -------------------------------------------------------------------------
    # Background Worker Execution
    # -------------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """Continuous execution loop running on the dedicated background thread."""
        logger.debug("SpeechController worker loop running.")
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

        try:
            while not self._stop_event.is_set():
                try:
                    event = self._queue.get(timeout=self.config.worker_timeout)
                except queue.Empty:
                    continue

                if self._stop_event.is_set():
                    break

                # Synthesize and play audio
                try:
                    self.provider.speak(event.text)
                    with self._lock:
                        self._last_spoken_text = event.text
                        self._total_spoken += 1
                        self._error_message = None
                    logger.info("Synthesized speech [%s]: '%s'", event.event_type.value, event.text)
                except Exception as e:
                    logger.error("TTS Provider error synthesizing '%s': %s", event.text, e)
                finally:
                    self._queue.task_done()
        finally:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass
            logger.debug("SpeechController worker loop exited.")

    @staticmethod
    def _normalize_text(text: Optional[str]) -> str:
        """Normalizes and cleans input text."""
        if not text:
            return ""
        # Strip extraneous whitespace
        clean = text.strip()
        # Collapse multiple whitespaces
        clean = re.sub(r"\s+", " ", clean)
        return clean
