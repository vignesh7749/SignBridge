"""Defensive validation of video files prior to playback or processing."""

from pathlib import Path
from typing import Union
import cv2

from signbridge.core.logger import get_logger
from signbridge.video.base import VideoMetadata, VideoValidationResult

logger = get_logger("video.validator")

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SUPPORTED_EXTENSIONS = SUPPORTED_VIDEO_EXTENSIONS


def validate_video_file(video_path: Union[str, Path]) -> VideoValidationResult:
    """Validates video file existence, format support, integrity, and decodability.

    Args:
        video_path: Path to video file.

    Returns:
        VideoValidationResult indicating validity, descriptive error message if any,
        and technical VideoMetadata.
    """
    path = Path(video_path).resolve()

    # 1. Existence and file check
    if not path.exists():
        msg = f"Video file not found: {path.name}"
        logger.warning(msg)
        return VideoValidationResult(is_valid=False, error_message=msg)

    if not path.is_file():
        msg = f"Specified path is not a file: {path.name}"
        logger.warning(msg)
        return VideoValidationResult(is_valid=False, error_message=msg)

    # 2. File extension check
    ext = path.suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        msg = (
            f"Unsupported video container format '{ext}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
        )
        logger.warning(msg)
        return VideoValidationResult(is_valid=False, error_message=msg)

    # 3. File size check
    try:
        size_bytes = path.stat().st_size
        if size_bytes == 0:
            msg = f"Video file is empty (0 bytes): {path.name}"
            logger.warning(msg)
            return VideoValidationResult(is_valid=False, error_message=msg)
    except Exception as e:
        msg = f"Unable to read file status: {e}"
        logger.warning(msg)
        return VideoValidationResult(is_valid=False, error_message=msg)

    # 4. OpenCV VideoCapture decodability test
    cap = None
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            msg = f"Video could not be decoded. OpenCV cannot open {path.name}."
            logger.warning(msg)
            return VideoValidationResult(is_valid=False, error_message=msg)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()

        if width <= 0 or height <= 0:
            msg = f"Video has invalid dimensions ({width}x{height}): {path.name}"
            logger.warning(msg)
            return VideoValidationResult(is_valid=False, error_message=msg)

        # Fallback for FPS if missing or non-positive
        effective_fps = fps if fps > 0 else 30.0
        duration_sec = (frame_count / effective_fps) if frame_count > 0 else 0.0

        # 5. Read test: Verify that at least one frame can actually be decoded
        ret, test_frame = cap.read()
        if not ret or test_frame is None or test_frame.size == 0:
            msg = f"Video could not be decoded. Frame stream could not be read from {path.name}."
            logger.warning(msg)
            return VideoValidationResult(is_valid=False, error_message=msg)

        metadata = VideoMetadata(
            path=path,
            filename=path.name,
            width=width,
            height=height,
            fps=effective_fps,
            frame_count=frame_count,
            duration_seconds=duration_sec,
            codec=codec_str or "UNKNOWN",
        )

        return VideoValidationResult(is_valid=True, metadata=metadata)

    except Exception as e:
        msg = f"Video decoding failed with unexpected error: {e}"
        logger.error(msg)
        return VideoValidationResult(is_valid=False, error_message=msg)
    finally:
        if cap is not None:
            cap.release()
