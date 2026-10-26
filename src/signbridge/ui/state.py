"""Stage 11: UI State & Interactive Runtime Model.

Manages presentation-layer state: modals, mouse interaction, toasts,
accessibility preferences, and view parameters.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Optional


class AppStatus(Enum):
    """High-level application operational status displayed in the header badge."""

    INITIALIZING = "INITIALIZING"
    CAMERA_READY = "CAMERA READY"
    HAND_DETECTED = "HAND DETECTED"
    NO_HAND = "NO HAND"
    RECOGNIZING = "RECOGNIZING"
    DWELLING = "DWELLING..."
    STABLE_EMITTED = "STABLE EMITTED"
    WORD_COMPLETED = "WORD COMPLETED"
    SPEAKING = "SPEAKING..."
    PAUSED = "PAUSED"
    CAMERA_ERROR = "CAMERA ERROR"
    MODEL_ERROR = "MODEL ERROR"
    VOICE_ERROR = "VOICE ERROR"


class ActiveModal(Enum):
    """Enumerates mutually exclusive dialog overlays."""

    NONE = "NONE"
    SETTINGS = "SETTINGS"
    HELP = "HELP"
    ONBOARDING = "ONBOARDING"
    EMERGENCY = "EMERGENCY"
    VIDEO_DEMO = "VIDEO_DEMO"


@dataclass
class NotificationToast:
    """Transient user notification toast displayed in the status area."""

    message: str
    level: str = "info"  # 'info', 'success', 'warning', 'error'
    created_at: float = field(default_factory=time.time)
    duration: float = 2.5

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.duration

    @property
    def progress(self) -> float:
        """Returns remaining life fraction [0.0, 1.0]."""
        elapsed = time.time() - self.created_at
        return max(0.0, min(1.0, 1.0 - (elapsed / self.duration)))


@dataclass
class UIState:
    """Encapsulates interactive presentation state without touching ML or CV state."""

    # Modal Dialog State
    active_modal: ActiveModal = ActiveModal.NONE
    help_tab: int = 0                  # 0: Overview, 1: Controls, 2: Hand Guide, 3: ASL Notice
    onboarding_step: int = 0           # 0 to 4 tutorial steps

    # Mouse & Interactive Pointer State
    mouse_x: int = -1
    mouse_y: int = -1
    mouse_down: bool = False
    clicked_id: Optional[str] = None
    hovered_id: Optional[str] = None

    # Accessibility Preferences
    high_contrast: bool = False
    reduced_motion: bool = False

    # Overlay Toggles
    show_skeleton: bool = True
    show_bbox: bool = True
    demo_mode: bool = False
    show_diagnostics: bool = False

    # Video Source & Demo State
    active_input_source: str = "CAMERA"  # 'CAMERA', 'DEMO', 'UPLOAD'
    selected_sample_video: Optional[str] = None
    video_is_playing: bool = True
    video_current_frame: int = 0
    video_total_frames: int = 0
    video_current_time: float = 0.0
    video_duration: float = 0.0
    video_filename: Optional[str] = None
    video_recognition_active: bool = True

    # Notification Banner
    toast: Optional[NotificationToast] = None

    # Transient Action Timers for subtle feedback
    last_action_text: str = "Ready. Hold sign steady to begin."
    last_action_time: float = field(default_factory=time.time)

    def set_toast(self, message: str, level: str = "info", duration: float = 2.5) -> None:
        """Triggers a transient visual notification toast."""
        self.toast = NotificationToast(message=message, level=level, duration=duration)
        self.last_action_text = message
        self.last_action_time = time.time()

    def open_modal(self, modal: ActiveModal) -> None:
        """Displays requested modal dialog and resets sub-tabs."""
        self.active_modal = modal
        if modal == ActiveModal.HELP:
            self.help_tab = 0
        elif modal == ActiveModal.ONBOARDING:
            self.onboarding_step = 0

    def close_modal(self) -> None:
        """Closes active modal dialog."""
        self.active_modal = ActiveModal.NONE

    def toggle_high_contrast(self) -> bool:
        """Toggles high-contrast accessibility theme."""
        self.high_contrast = not self.high_contrast
        mode_str = "ENABLED" if self.high_contrast else "DISABLED"
        self.set_toast(f"High Contrast Mode: {mode_str}", level="info")
        return self.high_contrast

    def toggle_reduced_motion(self) -> bool:
        """Toggles reduced-motion preference."""
        self.reduced_motion = not self.reduced_motion
        mode_str = "ENABLED" if self.reduced_motion else "DISABLED"
        self.set_toast(f"Reduced Motion: {mode_str}", level="info")
        return self.reduced_motion
