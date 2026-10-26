"""Thread-safe video file frame player and source implementation."""

from pathlib import Path
import threading
import time
from typing import Optional, Union
import cv2
import numpy as np

from signbridge.camera.base import CameraStatus, FramePacket
from signbridge.core.logger import get_logger
from signbridge.video.base import IVideoSource, VideoMetadata, VideoPlaybackState
from signbridge.video.validator import validate_video_file

logger = get_logger("video.reader")


class VideoFileReader(IVideoSource):
    """File-based video source implementing ICameraSource and IVideoSource.

    Enables SignBridge to process recorded demonstration videos seamlessly
    as drop-in replacements for live webcam hardware.
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        loop: bool = True,
        realtime_pacing: bool = True,
    ) -> None:
        self.video_path = Path(video_path).resolve()
        self.loop = loop
        self.realtime_pacing = realtime_pacing

        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        self._playback_state = VideoPlaybackState.UNINITIALIZED
        self._camera_status = CameraStatus.UNINITIALIZED

        self._current_frame_idx = 0
        self._last_frame_time = 0.0
        self._lock = threading.RLock()

        # Validate file upfront
        val_res = validate_video_file(self.video_path)
        if not val_res.is_valid or val_res.metadata is None:
            self._playback_state = VideoPlaybackState.ERROR
            self._camera_status = CameraStatus.ERROR
            raise ValueError(f"Invalid video file '{self.video_path.name}': {val_res.error_message}")

        self._metadata = val_res.metadata
        self._playback_state = VideoPlaybackState.READY
        self._camera_status = CameraStatus.INITIALIZING

    @property
    def metadata(self) -> Optional[VideoMetadata]:
        return self._metadata

    @property
    def playback_state(self) -> VideoPlaybackState:
        return self._playback_state

    @property
    def current_frame_index(self) -> int:
        return self._current_frame_idx

    @property
    def total_frames(self) -> int:
        return self._metadata.frame_count if self._metadata else 0

    # -------------------------------------------------------------------------
    # ICameraSource Protocol Properties
    # -------------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._playback_state in (VideoPlaybackState.PLAYING, VideoPlaybackState.PAUSED)

    @property
    def current_fps(self) -> float:
        return self._metadata.fps if self._metadata else 30.0

    @property
    def status(self) -> CameraStatus:
        return self._camera_status

    @property
    def actual_resolution(self) -> tuple[int, int]:
        if self._metadata:
            return self._metadata.width, self._metadata.height
        return 640, 480

    # -------------------------------------------------------------------------
    # Lifecycle & Playback Controls
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Opens capture stream and begins video playback."""
        with self._lock:
            if self._cap is None:
                self._cap = cv2.VideoCapture(str(self.video_path))
                self._current_frame_idx = 0

            self._playback_state = VideoPlaybackState.PLAYING
            self._camera_status = CameraStatus.STREAMING
            self._last_frame_time = time.time()
            logger.info("VideoFileReader started: %s (%s)", self.video_path.name, self._metadata.resolution_str)

    def stop(self) -> None:
        """Releases video capture and resets player."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._playback_state = VideoPlaybackState.CLOSED
            self._camera_status = CameraStatus.CLOSED
            logger.info("VideoFileReader stopped: %s", self.video_path.name)

    def play(self) -> None:
        with self._lock:
            if self._playback_state in (VideoPlaybackState.READY, VideoPlaybackState.PAUSED, VideoPlaybackState.ENDED):
                if self._cap is None:
                    self.start()
                else:
                    self._playback_state = VideoPlaybackState.PLAYING
                    self._camera_status = CameraStatus.STREAMING
                    self._last_frame_time = time.time()

    def pause(self) -> None:
        with self._lock:
            if self._playback_state == VideoPlaybackState.PLAYING:
                self._playback_state = VideoPlaybackState.PAUSED
                self._camera_status = CameraStatus.PAUSED
                logger.debug("Video playback paused at frame %d", self._current_frame_idx)

    def resume(self) -> None:
        self.play()

    def restart(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._current_frame_idx = 0
            self._playback_state = VideoPlaybackState.PLAYING
            self._camera_status = CameraStatus.STREAMING
            self._last_frame_time = time.time()
            logger.info("Video playback restarted from frame 0")

    def seek(self, frame_index: int) -> bool:
        with self._lock:
            if self._cap is None:
                return False
            max_frame = max(0, self.total_frames - 1)
            target = max(0, min(frame_index, max_frame))
            success = self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(target))
            if success:
                self._current_frame_idx = target
                if self._playback_state == VideoPlaybackState.ENDED:
                    self._playback_state = VideoPlaybackState.PAUSED
                    self._camera_status = CameraStatus.PAUSED
            return success

    def set_loop(self, loop: bool) -> None:
        self.loop = loop

    # -------------------------------------------------------------------------
    # Frame Acquisition
    # -------------------------------------------------------------------------

    def read(self, timeout: float = 1.0) -> tuple[bool, Optional[FramePacket]]:
        """Retrieves the next frame in the video file sequence."""
        with self._lock:
            if self._playback_state != VideoPlaybackState.PLAYING or self._cap is None:
                return False, None

            # Real-time frame rate pacing if requested
            if self.realtime_pacing and self._metadata and self._metadata.fps > 0:
                target_interval = 1.0 / self._metadata.fps
                elapsed = time.time() - self._last_frame_time
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)

            ret, frame = self._cap.read()

            if not ret or frame is None:
                # Reached end of video stream
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._current_frame_idx = 0
                    ret, frame = self._cap.read()
                    if not ret or frame is None:
                        self._playback_state = VideoPlaybackState.ENDED
                        return False, None
                else:
                    self._playback_state = VideoPlaybackState.ENDED
                    self._camera_status = CameraStatus.PAUSED
                    return False, None

            now = time.time()
            self._last_frame_time = now
            self._current_frame_idx += 1

            h, w = frame.shape[:2]
            packet = FramePacket(
                frame_id=self._current_frame_idx,
                timestamp=now,
                data=frame,
                width=w,
                height=h,
                latency_ms=0.0,
            )
            return True, packet
