"""Module 2: Production Reusable Camera Manager.

Implements threaded producer-consumer frame acquisition with zero buffer latency,
FPS estimation, robust disconnection handling, and clean hardware release.
"""

import os
import queue
import threading
import time
from typing import Optional
import cv2
import numpy as np

from signbridge.camera.base import CameraStatus, FramePacket, ICameraSource
from signbridge.core.config import CameraConfig
from signbridge.core.exceptions import CameraError, CameraNotFoundError
from signbridge.core.logger import get_logger

logger = get_logger("camera.manager")


class CameraManager(ICameraSource):
    """Threaded OpenCV video capture manager.

    Features:
    - Runs frame acquisition on a dedicated background thread.
    - Uses a bounded queue (maxsize=1) to prevent video buffer lag.
    - Rolling window FPS measurement.
    - Automatic horizontal mirror flipping for intuitive user interaction.
    - Disconnection detection and graceful error reporting.
    - Context manager support (`with CameraManager(...) as cam:`).
    """

    def __init__(
        self,
        config: Optional[CameraConfig] = None,
        camera_index: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        flip_horizontal: Optional[bool] = None,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 3,
    ) -> None:
        base_config = config or CameraConfig()
        self.config = CameraConfig(
            device_index=camera_index if camera_index is not None else base_config.device_index,
            width=width if width is not None else base_config.width,
            height=height if height is not None else base_config.height,
            fps=fps if fps is not None else base_config.fps,
            flip_horizontal=flip_horizontal if flip_horizontal is not None else base_config.flip_horizontal,
        )
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._queue: queue.Queue[FramePacket] = queue.Queue(maxsize=1)
        self._status: CameraStatus = CameraStatus.UNINITIALIZED

        self._frame_id = 0
        self._fps: float = 0.0
        self._actual_width: int = 0
        self._actual_height: int = 0
        self._lock = threading.Lock()

    @property
    def status(self) -> CameraStatus:
        return self._status

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    @property
    def current_fps(self) -> float:
        return self._fps

    @property
    def actual_resolution(self) -> tuple[int, int]:
        return self._actual_width, self._actual_height

    def start(self) -> None:
        """Opens camera and begins asynchronous frame capture thread."""
        with self._lock:
            if self.is_running:
                logger.warning("CameraManager is already running.")
                return

            self._status = CameraStatus.INITIALIZING
            self._open_device()

            self._stop_event.clear()
            self._pause_event.clear()
            self._thread = threading.Thread(
                target=self._capture_worker,
                name=f"SignBridge-CameraCapture-{self.config.device_index}",
                daemon=True,
            )
            self._thread.start()
            self._status = CameraStatus.STREAMING
            logger.info("Camera capture worker thread started successfully.")

    def _open_device(self) -> None:
        """Safely opens cv2.VideoCapture with backend fallbacks."""
        logger.info(
            "Opening camera index %d (requested resolution: %dx%d @ %d FPS)...",
            self.config.device_index,
            self.config.width,
            self.config.height,
            self.config.fps,
        )

        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.config.device_index, backend)

        if not cap.isOpened() and backend != cv2.CAP_ANY:
            logger.warning("Failed opening with CAP_DSHOW; attempting fallback with CAP_ANY...")
            cap.release()
            cap = cv2.VideoCapture(self.config.device_index, cv2.CAP_ANY)

        if not cap.isOpened():
            self._status = CameraStatus.ERROR
            error_msg = (
                f"Unable to open camera index {self.config.device_index}. "
                "Ensure device is connected and not in use by another program."
            )
            logger.error(error_msg)
            raise CameraNotFoundError(error_msg)

        # Configure hardware resolution and FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.config.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.height))
        cap.set(cv2.CAP_PROP_FPS, float(self.config.fps))

        # Query actual negotiated settings
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS))

        self._actual_width = actual_w if actual_w > 0 else self.config.width
        self._actual_height = actual_h if actual_h > 0 else self.config.height
        self._cap = cap

        logger.info(
            "Camera opened successfully. Actual negotiated format: %dx%d (reported FPS: %.1f, backend: %s)",
            self._actual_width,
            self._actual_height,
            actual_fps,
            cap.getBackendName(),
        )

    def _capture_worker(self) -> None:
        """Producer thread: continuously acquires frames into the queue."""
        consecutive_failures = 0
        max_failure_threshold = 30  # ~1 second of missing frames at 30 FPS
        last_fps_time = time.time()
        frame_counter = 0

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.05)
                continue

            if self._cap is None or not self._cap.isOpened():
                self._status = CameraStatus.DISCONNECTED
                logger.error("Camera hardware disconnected.")
                if self.auto_reconnect:
                    if self._attempt_reconnect():
                        consecutive_failures = 0
                        continue
                break

            read_start = time.time()
            ret, frame = self._cap.read()

            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures % 15 == 0:
                    logger.warning("Dropped frame read (consecutive failures: %d)", consecutive_failures)

                if consecutive_failures >= max_failure_threshold:
                    self._status = CameraStatus.DISCONNECTED
                    logger.error("Exceeded failure threshold (%d frames). Connection lost.", consecutive_failures)
                    if self.auto_reconnect:
                        if self._attempt_reconnect():
                            consecutive_failures = 0
                            continue
                    break
                time.sleep(0.01)
                continue

            # Successful frame read
            consecutive_failures = 0
            self._frame_id += 1
            now = time.time()
            latency_ms = (now - read_start) * 1000.0

            # Update actual resolution dynamically if changed
            h, w = frame.shape[:2]
            self._actual_width, self._actual_height = w, h

            # Mirror mode
            if self.config.flip_horizontal:
                frame = cv2.flip(frame, 1)

            # Rolling FPS calculation
            frame_counter += 1
            elapsed = now - last_fps_time
            if elapsed >= 0.5:
                self._fps = frame_counter / elapsed
                frame_counter = 0
                last_fps_time = now

            packet = FramePacket(
                frame_id=self._frame_id,
                timestamp=now,
                data=frame,
                width=w,
                height=h,
                latency_ms=latency_ms,
            )

            # Single-item queue: discard stale unread frame to eliminate display/pipeline lag
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

            self._queue.put(packet)

        logger.debug("Camera capture worker thread terminating.")

    def _attempt_reconnect(self) -> bool:
        """Attempts to reinitialize the camera connection after failure."""
        logger.info("Attempting automatic camera reconnection...")
        for attempt in range(1, self.max_reconnect_attempts + 1):
            if self._stop_event.is_set():
                return False
            logger.info("Reconnection attempt %d of %d in 1.0s...", attempt, self.max_reconnect_attempts)
            time.sleep(1.0)
            try:
                if self._cap:
                    self._cap.release()
                self._open_device()
                self._status = CameraStatus.STREAMING
                logger.info("Camera reconnected successfully!")
                return True
            except Exception as e:
                logger.warning("Reconnection attempt %d failed: %s", attempt, e)

        self._status = CameraStatus.ERROR
        logger.error("Camera reconnection exhausted all %d attempts.", self.max_reconnect_attempts)
        return False

    def read(self, timeout: float = 1.0) -> tuple[bool, Optional[FramePacket]]:
        """Retrieves the latest available frame from the queue.

        Args:
            timeout: Maximum seconds to wait for frame.

        Returns:
            (True, FramePacket) on success, (False, None) on timeout or error.
        """
        if not self.is_running and self._queue.empty():
            return False, None

        try:
            packet = self._queue.get(timeout=timeout)
            return True, packet
        except queue.Empty:
            return False, None

    def pause(self) -> None:
        """Pauses frame acquisition worker without closing camera."""
        self._pause_event.set()
        self._status = CameraStatus.PAUSED
        logger.info("Camera acquisition paused.")

    def resume(self) -> None:
        """Resumes frame acquisition worker."""
        self._pause_event.clear()
        self._status = CameraStatus.STREAMING
        logger.info("Camera acquisition resumed.")

    def stop(self) -> None:
        """Gracefully stops acquisition thread and releases hardware."""
        with self._lock:
            self._status = CameraStatus.CLOSED
            self._stop_event.set()
            self._pause_event.clear()

            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
                self._thread = None

            if self._cap:
                self._cap.release()
                self._cap = None

            # Empty queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

            logger.info("Camera hardware released and manager stopped.")
