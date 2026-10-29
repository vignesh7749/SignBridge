"""Module 2: Camera Device Detection & Enumeration."""

import os
import sys
from typing import Optional
import cv2

from signbridge.camera.base import CameraDeviceInfo
from signbridge.core.logger import get_logger

logger = get_logger("camera.discovery")


def get_default_backend() -> int:
    """Returns DirectShow on Windows for fast device enumeration, else CAP_ANY."""
    return cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY


def list_available_cameras(
    max_devices: int = 5,
    backend: Optional[int] = None,
    consecutive_fail_limit: int = 2,
) -> list[CameraDeviceInfo]:
    """Scans system video indices to detect connected webcams.

    Args:
        max_devices: Maximum index count to probe (default: 5).
        backend: OpenCV capture API backend (e.g. cv2.CAP_DSHOW, cv2.CAP_ANY).
        consecutive_fail_limit: Stop scanning after N consecutive unopenable indices.

    Returns:
        List of CameraDeviceInfo for all detected, working camera devices.
    """
    selected_backend = get_default_backend() if backend is None else backend
    available: list[CameraDeviceInfo] = []
    consecutive_fails = 0

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.ole32.CoInitialize(None)
        except Exception:
            pass

    logger.info("Scanning for camera devices (probing up to index %d)...", max_devices - 1)

    try:
        for index in range(max_devices):
            cap = None
            try:
                cap = cv2.VideoCapture(index, selected_backend)
                if cap is not None and cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = float(cap.get(cv2.CAP_PROP_FPS))
                        backend_name = cap.getBackendName()

                        info = CameraDeviceInfo(
                            index=index,
                            name=f"Camera Device {index} ({backend_name})",
                            is_available=True,
                            default_width=w if w > 0 else frame.shape[1],
                            default_height=h if h > 0 else frame.shape[0],
                            default_fps=fps if fps > 0 else 30.0,
                            backend_name=backend_name,
                        )
                        available.append(info)
                        logger.info("Found active camera: %s [%dx%d @ %.1f FPS]", info.name, info.default_width, info.default_height, info.default_fps)
                        consecutive_fails = 0
                    else:
                        consecutive_fails += 1
                else:
                    consecutive_fails += 1

                if consecutive_fails >= consecutive_fail_limit:
                    logger.debug("Stopping scan after %d consecutive inactive camera indices.", consecutive_fails)
                    break

            except Exception as err:
                logger.warning("Error probing camera index %d: %s", index, err)
                consecutive_fails += 1
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
    finally:
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.ole32.CoUninitialize()
            except Exception:
                pass

    if not available:
        logger.warning("No functional camera devices detected on this system.")
    else:
        logger.info("Camera discovery complete: %d device(s) ready.", len(available))

    return available
