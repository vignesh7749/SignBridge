"""Tests for video input validation, sample management, and VideoFileReader playback."""

from pathlib import Path
import tempfile
import pytest

from signbridge.camera.base import CameraStatus
from signbridge.video.base import VideoPlaybackState
from signbridge.video.reader import VideoFileReader
from signbridge.video.sample_manager import SampleVideoManager
from signbridge.video.validator import validate_video_file

ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_HELLO = ROOT_DIR / "datasets" / "samples" / "word_signs" / "hello_sample.mp4"


def test_validate_video_file_errors():
    """Video validator catches non-existent and unsupported file types defensively."""
    # 1. Non-existent file
    res_missing = validate_video_file("non_existent_file.mp4")
    assert not res_missing.is_valid
    assert "not found" in res_missing.error_message.lower()

    # 2. Unsupported extension
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        res_ext = validate_video_file(tmp.name)
        assert not res_ext.is_valid
        assert "unsupported" in res_ext.error_message.lower()


def test_validate_video_file_valid_sample():
    """Built-in demonstration video passes full integrity and decodability validation."""
    assert SAMPLE_HELLO.exists(), f"Sample video missing at {SAMPLE_HELLO}"
    res = validate_video_file(SAMPLE_HELLO)

    assert res.is_valid
    assert res.metadata is not None
    assert res.metadata.width == 640
    assert res.metadata.height == 480
    assert res.metadata.fps > 0
    assert res.metadata.frame_count > 0
    assert res.metadata.duration_seconds > 0


def test_sample_video_manager_builtin_samples():
    """SampleVideoManager discovers and registers all verified built-in samples."""
    mgr = SampleVideoManager()
    samples = mgr.list_builtin_samples()

    assert len(samples) >= 6
    signs = [s.sign for s in samples]
    assert "HELLO" in signs
    assert "HELP" in signs
    assert "YES" in signs
    assert "STOP" in signs

    sample_hello = mgr.get_sample_by_sign("HELLO")
    assert sample_hello is not None
    assert sample_hello.sign == "HELLO"
    assert sample_hello.full_path is not None


def test_video_file_reader_lifecycle_and_frames():
    """VideoFileReader initializes, plays, reads frames, and stops safely."""
    reader = VideoFileReader(SAMPLE_HELLO, loop=True, realtime_pacing=False)
    assert reader.playback_state == VideoPlaybackState.READY

    reader.start()
    assert reader.playback_state == VideoPlaybackState.PLAYING

    frames_read = 0
    for _ in range(10):
        ok, packet = reader.read(timeout=0.5)
        if ok and packet is not None:
            frames_read += 1
            assert packet.width == 640
            assert packet.height == 480
            assert packet.data.shape == (480, 640, 3)

    assert frames_read == 10
    reader.stop()
    assert reader.playback_state == VideoPlaybackState.CLOSED
    assert reader.status == CameraStatus.CLOSED
