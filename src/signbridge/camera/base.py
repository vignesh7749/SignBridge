"""Module 2: Camera Contracts, Data Models, and Abstract Interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import numpy as np


class CameraStatus(Enum):
    """Lifecycle states of the camera hardware source."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    STREAMING = auto()
    PAUSED = auto()
    DISCONNECTED = auto()
    ERROR = auto()
    CLOSED = auto()


@dataclass(frozen=True)
class CameraDeviceInfo:
    """Metadata describing a detected camera device."""
    index: int
    name: str
    is_available: bool
    default_width: int
    default_height: int
    default_fps: float
    backend_name: str


@dataclass(frozen=True)
class FramePacket:
    """A single captured video frame bundled with precise telemetry."""
    frame_id: int
    timestamp: float
    data: np.ndarray  # Clean BGR frame array
    width: int
    height: int
    latency_ms: float = 0.0

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape


class ICameraSource(ABC):
    """Abstract interface for video frame acquisition sources."""

    @abstractmethod
    def start(self) -> None:
        """Starts video capture and background acquisition thread."""
        pass

    @abstractmethod
    def read(self, timeout: float = 1.0) -> tuple[bool, Optional[FramePacket]]:
        """Retrieves the latest available frame.

        Args:
            timeout: Maximum seconds to wait for a frame.

        Returns:
            Tuple of (success_flag, FramePacket_or_None).
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops capture thread and releases camera hardware."""
        pass

    def release(self) -> None:
        """Alias for stop() to support standard OpenCV idiom."""
        self.stop()

    def __enter__(self) -> "ICameraSource":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Returns True if the camera is actively capturing."""
        pass

    @property
    @abstractmethod
    def current_fps(self) -> float:
        """Returns live measured capture frames per second."""
        pass

    @property
    @abstractmethod
    def status(self) -> CameraStatus:
        """Returns the current operational status of the camera."""
        pass

    @property
    @abstractmethod
    def actual_resolution(self) -> tuple[int, int]:
        """Returns (width, height) negotiated with hardware."""
        pass
