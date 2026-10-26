"""Module 2: Camera Demo State and Keyboard Input Controller.

Implements deterministic state transitions for camera playback, edge-triggered
debouncing for OpenCV cv2.waitKey(), and clean action dispatching.
"""

from enum import Enum, auto
import time
from typing import Optional


class DemoState(Enum):
    """Operational display state for interactive camera demos."""
    RUNNING = auto()
    PAUSED = auto()


class UserAction(Enum):
    """Actions emitted from keyboard input processing."""
    NONE = auto()
    TOGGLE_PAUSE = auto()
    TOGGLE_MIRROR = auto()
    SNAPSHOT = auto()
    QUIT = auto()


class DemoDisplayController:
    """Manages display state transitions and debounced keyboard inputs.

    Guarantees that one physical key press produces exactly ONE state transition,
    preventing double-toggles caused by OS key repeat or rapid loop iterations.
    """

    def __init__(self, debounce_interval_sec: float = 0.20) -> None:
        self.state = DemoState.RUNNING
        self.debounce_interval = debounce_interval_sec
        self._last_toggle_time: float = 0.0
        self._p_key_released: bool = True

    @property
    def is_paused(self) -> bool:
        """Returns True if the display/processing is in PAUSED state."""
        return self.state == DemoState.PAUSED

    @property
    def is_running(self) -> bool:
        """Returns True if the display/processing is in RUNNING state."""
        return self.state == DemoState.RUNNING

    def handle_key(self, raw_key: int, now: Optional[float] = None) -> UserAction:
        """Processes a raw key code returned from cv2.waitKey().

        Args:
            raw_key: Raw integer returned by cv2.waitKey(). When no key is pressed,
                     OpenCV returns -1.
            now: Timestamp in seconds (defaults to time.time()).

        Returns:
            UserAction representing the validated user intent.
        """
        current_time = now if now is not None else time.time()

        # No key pressed (cv2.waitKey returns -1)
        if raw_key == -1:
            self._p_key_released = True
            return UserAction.NONE

        key = raw_key & 0xFF

        # Quit on 'q', 'Q', or ESC (code 27)
        if key in (ord("q"), ord("Q"), 27):
            return UserAction.QUIT

        # Toggle pause on 'p' or 'P'
        if key in (ord("p"), ord("P")):
            time_since_toggle = current_time - self._last_toggle_time
            if self._p_key_released and time_since_toggle >= self.debounce_interval:
                # Valid physical keypress: perform single state transition
                self.state = DemoState.PAUSED if self.state == DemoState.RUNNING else DemoState.RUNNING
                self._last_toggle_time = current_time
                self._p_key_released = False
                return UserAction.TOGGLE_PAUSE
            else:
                # Key is still held down from the initial press (key repeat/rapid poll)
                return UserAction.NONE

        # Other keys: reset P release flag in case another key was pressed
        self._p_key_released = True

        # Toggle mirror mode on 'm' or 'M'
        if key in (ord("m"), ord("M")):
            return UserAction.TOGGLE_MIRROR

        # Take snapshot on 's' or 'S'
        if key in (ord("s"), ord("S")):
            return UserAction.SNAPSHOT

        return UserAction.NONE
