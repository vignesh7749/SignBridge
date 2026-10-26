"""Safe Academic Emergency Simulation Workflow.

Provides an interactive safety confirmation modal and event state machine when the 'HELP' sign is detected.
CRITICAL SAFETY BOUNDARY:
This is an isolated academic demonstration component. It does NOT make any real
external network requests, telephony calls, or contacts to real emergency services (e.g. 911).
All assistance protocols are strictly local simulated demonstrations.
"""

from dataclasses import dataclass
from enum import Enum
import time
from typing import Callable, Optional

from signbridge.core.logger import get_logger

logger = get_logger("word_sign.emergency")


class EmergencyState(str, Enum):
    """Lifecycle states of the simulated emergency assistance system."""
    IDLE = "IDLE"
    CONFIRMATION_REQUESTED = "CONFIRMATION_REQUESTED"
    ASSISTANCE_ACTIVE = "ASSISTANCE_ACTIVE"
    DISMISSED = "DISMISSED"


@dataclass(frozen=True)
class EmergencyEvent:
    """Telemetry payload for simulated emergency events."""
    state: EmergencyState
    sign_trigger: str
    confidence: float
    timestamp: float
    message: str


class EmergencyModalManager:
    """Manages the lifecycle, user interaction, and safe simulation of emergency assistance."""

    def __init__(
        self,
        on_confirmed: Optional[Callable[[EmergencyEvent], None]] = None,
        on_cancelled: Optional[Callable[[EmergencyEvent], None]] = None,
    ) -> None:
        self.state: EmergencyState = EmergencyState.IDLE
        self.last_trigger_sign: str = ""
        self.last_trigger_confidence: float = 0.0
        self.last_event_time: float = 0.0
        self.status_message: str = "Ready"

        self._on_confirmed = on_confirmed
        self._on_cancelled = on_cancelled
        self._history: list[EmergencyEvent] = []

    @property
    def is_modal_active(self) -> bool:
        """Returns True if the user confirmation dialog should be displayed."""
        return self.state == EmergencyState.CONFIRMATION_REQUESTED

    @property
    def is_assistance_active(self) -> bool:
        """Returns True if simulated emergency assistance is currently engaged."""
        return self.state == EmergencyState.ASSISTANCE_ACTIVE

    def trigger_sign_detected(
        self,
        sign: str,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Evaluates detected sign and triggers confirmation if sign is an emergency sign.

        Returns True if emergency confirmation was newly triggered.
        """
        now = timestamp if timestamp is not None else time.time()
        normalized = sign.strip().upper()

        if normalized in ("HELP", "DANGER", "EMERGENCY"):
            if self.state in (EmergencyState.IDLE, EmergencyState.DISMISSED):
                self.state = EmergencyState.CONFIRMATION_REQUESTED
                self.last_trigger_sign = normalized
                self.last_trigger_confidence = confidence
                self.last_event_time = now
                self.status_message = f"Simulated Emergency: '{normalized}' sign detected. Awaiting user response."

                event = EmergencyEvent(
                    state=self.state,
                    sign_trigger=normalized,
                    confidence=confidence,
                    timestamp=now,
                    message=self.status_message,
                )
                self._history.append(event)
                logger.info(
                    "Simulated emergency confirmation opened for '%s' (conf=%.2f)",
                    normalized,
                    confidence,
                )
                return True

        return False

    def confirm_assistance(self) -> EmergencyEvent:
        """User clicked [GET HELP]: Engages local simulated assistance protocol."""
        self.state = EmergencyState.ASSISTANCE_ACTIVE
        self.status_message = "SIMULATED DISPATCH: Emergency contacts alerted locally (Academic Demo Mode)."
        event = EmergencyEvent(
            state=self.state,
            sign_trigger=self.last_trigger_sign,
            confidence=self.last_trigger_confidence,
            timestamp=time.time(),
            message=self.status_message,
        )
        self._history.append(event)
        logger.info("Simulated emergency assistance confirmed by user.")

        if self._on_confirmed is not None:
            try:
                self._on_confirmed(event)
            except Exception as e:
                logger.error("Error in on_confirmed callback: %s", e)

        return event

    def cancel(self) -> EmergencyEvent:
        """User clicked [CANCEL]: Dismisses confirmation dialog and returns to normal operation."""
        self.state = EmergencyState.DISMISSED
        self.status_message = "Emergency alert cancelled by user."
        event = EmergencyEvent(
            state=self.state,
            sign_trigger=self.last_trigger_sign,
            confidence=self.last_trigger_confidence,
            timestamp=time.time(),
            message=self.status_message,
        )
        self._history.append(event)
        logger.info("Simulated emergency dialog dismissed by user.")

        if self._on_cancelled is not None:
            try:
                self._on_cancelled(event)
            except Exception as e:
                logger.error("Error in on_cancelled callback: %s", e)

        return event

    def reset(self) -> None:
        """Resets the emergency manager state to IDLE."""
        self.state = EmergencyState.IDLE
        self.status_message = "Ready"
        self.last_trigger_sign = ""
        self.last_trigger_confidence = 0.0
