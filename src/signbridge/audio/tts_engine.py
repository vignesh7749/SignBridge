"""Module 14: Non-Blocking Asynchronous Text-To-Speech Manager."""

import queue
import threading
from typing import Optional

from signbridge.audio.base import ITextToSpeech
from signbridge.core.config import AudioConfig
from signbridge.core.logger import get_logger

logger = get_logger("audio.tts")


class AsyncTTSManager(ITextToSpeech):
    """Executes voice synthesis inside a detached worker thread to avoid UI lag."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._speech_queue: queue.Queue[str] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_speaking = False

    def start(self) -> None:
        """Starts background worker thread."""
        if not self.config.enabled:
            logger.info("Text-To-Speech is disabled via configuration.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("Async TTS worker thread started.")

    def speak(self, text: str, non_blocking: bool = True) -> None:
        """Enqueues utterance for speech playback."""
        clean_text = text.strip()
        if not clean_text or not self.config.enabled:
            return

        logger.info("Enqueuing TTS utterance: '%s'", clean_text)
        self._speech_queue.put(clean_text)

    def _worker(self) -> None:
        """Worker loop processing speech queue."""
        # Note: Concrete engine initialization (e.g. pyttsx3.init()) occurs here in worker
        while not self._stop_event.is_set():
            try:
                text = self._speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            self._is_speaking = True
            logger.debug("Synthesizing: %s", text)
            # Placeholder for pyttsx3 or OS synthesizer engine call
            self._is_speaking = False

    def stop(self) -> None:
        """Stops ongoing speech and releases worker thread."""
        self._stop_event.set()
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        logger.info("Async TTS manager stopped.")

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
