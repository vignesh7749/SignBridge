"""Stage 11: Demonstration Driver.

Provides a smooth, timed simulation sequence for faculty evaluations, portfolio
showcases, and automated demonstrations without requiring camera or lighting setup.
"""

import time
from typing import Optional, Tuple
import numpy as np

from signbridge.pipeline.contracts import StageLatencies
from signbridge.pipeline.orchestrator import SignBridgePipeline
from signbridge.temporal.contracts import TemporalFrameResult, TemporalState
from signbridge.translation.contracts import TranslationResult


# Pre-scripted timed sequence for demonstration
DEMO_SEQUENCE = [
    ("CHAR", "H", 0.6),
    ("CHAR", "E", 0.6),
    ("CHAR", "L", 0.6),
    ("CHAR", "L", 0.6),
    ("CHAR", "O", 0.6),
    ("SPACE", " ", 0.8),
    ("CHAR", "W", 0.6),
    ("CHAR", "O", 0.6),
    ("CHAR", "R", 0.6),
    ("CHAR", "L", 0.6),
    ("CHAR", "D", 0.6),
    ("SPACE", " ", 0.8),
    ("SPEAK", "HELLO WORLD", 1.5),
    ("PAUSE", "DONE", 4.0),
]


class SignBridgeDemoDriver:
    """Feeds authentic timed sign sequences into the pipeline for faculty presentations."""

    def __init__(self, pipeline: SignBridgePipeline) -> None:
        self.pipeline = pipeline
        self.is_active = False
        self._start_time = 0.0
        self._step_index = 0
        self._last_step_time = 0.0
        self.script = list(DEMO_SEQUENCE)

    def start(self) -> None:
        """Begins automated demonstration loop."""
        self.is_active = True
        self._start_time = time.time()
        self._step_index = 0
        self._last_step_time = time.time()
        self.pipeline.reset()

    def stop(self) -> None:
        """Halts demonstration mode."""
        self.is_active = False

    def update(self, step_duration: Optional[float] = None) -> Optional[str]:
        """Advances timed script. Returns status string if an event occurred."""
        if not self.is_active or not self.script:
            return None

        now = time.time()
        action_type, payload, default_duration = self.script[self._step_index]
        duration = step_duration if step_duration is not None else default_duration

        if (now - self._last_step_time) >= duration:
            self._last_step_time = now
            self._step_index = (self._step_index + 1) % len(self.script)

            if action_type == "CHAR":
                self.pipeline.translation_engine.process_char(payload)
                self.pipeline._update_runtime_state()
                return f"Simulated Sign: [{payload}]"
            elif action_type == "SPACE":
                self.pipeline.space()
                return "Committed Word Break"
            elif action_type == "SPEAK":
                self.pipeline.speak_current_sentence()
                return "Voicing Sentence: 'HELLO WORLD'"
            elif action_type == "PAUSE" and payload == "DONE":
                self.pipeline.reset()
                return "Demo sequence restarting..."

        return None
