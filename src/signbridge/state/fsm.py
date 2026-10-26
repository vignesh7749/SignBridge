"""Module 13: Translation State Management (Finite State Machine)."""

from dataclasses import dataclass
from enum import Enum, auto
import time
from typing import Optional

from signbridge.core.logger import get_logger

logger = get_logger("state.fsm")


class SystemState(Enum):
    NO_HAND = auto()        # Camera streaming, no hand detected
    TRACKING = auto()       # Hand detected, stabilizing landmarks
    SIGNING = auto()        # Active valid sign detected, accumulating dwell time
    COMMITTED = auto()      # Character/sign was committed, waiting for token release
    PAUSED = auto()         # System paused by user


@dataclass(frozen=True)
class StateTransition:
    previous_state: SystemState
    new_state: SystemState
    timestamp: float
    trigger: str


class TranslationFSM:
    """Finite State Machine orchestrating translation flow."""

    def __init__(self) -> None:
        self._state: SystemState = SystemState.NO_HAND
        self._last_state_change: float = time.time()

    @property
    def current_state(self) -> SystemState:
        return self._state

    def transition_to(self, new_state: SystemState, trigger: str = "") -> Optional[StateTransition]:
        """Performs a validated state transition."""
        if new_state == self._state:
            return None

        previous = self._state
        self._state = new_state
        now = time.time()
        self._last_state_change = now

        logger.debug("State transition: %s -> %s (trigger: %s)", previous.name, new_state.name, trigger)
        return StateTransition(previous_state=previous, new_state=new_state, timestamp=now, trigger=trigger)

    def update(self, has_hand: bool, is_stable_sign: bool, sign_committed: bool) -> SystemState:
        """Determines state based on detector and recognizer signals."""
        if self._state == SystemState.PAUSED:
            return self._state

        if not has_hand:
            self.transition_to(SystemState.NO_HAND, "no_hand_detected")
            return self._state

        if sign_committed:
            self.transition_to(SystemState.COMMITTED, "token_committed")
            return self._state

        if is_stable_sign:
            self.transition_to(SystemState.SIGNING, "stable_sign_active")
            return self._state

        self.transition_to(SystemState.TRACKING, "hand_moving_or_unstable")
        return self._state
