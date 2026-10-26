"""Camera subsystem for SignBridge: discovery, capture, and contracts."""

from signbridge.camera.base import (
    CameraDeviceInfo,
    CameraStatus,
    FramePacket,
    ICameraSource,
)
from signbridge.camera.capture import CameraManager
from signbridge.camera.controller import DemoDisplayController, DemoState, UserAction
from signbridge.camera.discovery import list_available_cameras

__all__ = [
    "CameraDeviceInfo",
    "CameraStatus",
    "FramePacket",
    "ICameraSource",
    "CameraManager",
    "DemoDisplayController",
    "DemoState",
    "UserAction",
    "list_available_cameras",
]
