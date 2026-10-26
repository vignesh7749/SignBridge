"""Contracts and data models for video file sources and playback."""

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from signbridge.camera.base import ICameraSource


class VideoPlaybackState(Enum):
    """Lifecycle states of a video file player."""
    UNINITIALIZED = auto()
    READY = auto()
    PLAYING = auto()
    PAUSED = auto()
    ENDED = auto()
    ERROR = auto()
    CLOSED = auto()


@dataclass(frozen=True)
class VideoMetadata:
    """Technical metadata describing a video file."""
    path: Path
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str

    @property
    def resolution_str(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(frozen=True)
class VideoValidationResult:
    """Outcome of video file integrity and decodability validation."""
    is_valid: bool
    error_message: Optional[str] = None
    metadata: Optional[VideoMetadata] = None


class IVideoSource(ICameraSource):
    """Abstract interface extending ICameraSource for file-based video playback."""

    @abstractmethod
    def play(self) -> None:
        """Starts or resumes playback."""
        pass

    @abstractmethod
    def pause(self) -> None:
        """Pauses video playback at the current frame."""
        pass

    @abstractmethod
    def restart(self) -> None:
        """Rewinds playback to frame 0 and resumes."""
        pass

    @abstractmethod
    def seek(self, frame_index: int) -> bool:
        """Positions playback at the target frame index."""
        pass

    @property
    @abstractmethod
    def playback_state(self) -> VideoPlaybackState:
        """Returns the current playback state."""
        pass

    @property
    @abstractmethod
    def current_frame_index(self) -> int:
        """Returns the 0-indexed position of the last emitted frame."""
        pass

    @property
    @abstractmethod
    def total_frames(self) -> int:
        """Returns total frame count of the video file."""
        pass

    @property
    @abstractmethod
    def metadata(self) -> Optional[VideoMetadata]:
        """Returns validated video metadata if loaded."""
        pass
