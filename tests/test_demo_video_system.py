"""Unit and integration tests for Demo Video Subsystem and Video Uploads.

Validates:
1. Machine-readable demo manifest loading and schema integrity.
2. Discovery of curated demonstration samples and metadata extraction.
3. Video decoding and frame-by-frame streaming via VideoFileReader.
4. Playback transport controls (start, play, pause, resume, restart, stop).
5. Error handling for non-existent, unsupported, or corrupted video files.
6. User video upload validation, isolated segregation, cataloging, and removal.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from signbridge.video.base import VideoPlaybackState, VideoValidationResult
from signbridge.video.reader import VideoFileReader
from signbridge.video.sample_manager import SampleVideoManager
from signbridge.video.validator import validate_video_file


@pytest.fixture
def sample_manager() -> SampleVideoManager:
    """Fixture providing initialized SampleVideoManager."""
    return SampleVideoManager()


def test_demo_manifest_loading(sample_manager: SampleVideoManager):
    """Verifies that manifest.json exists, adheres to schema, and contains required 6 signs."""
    manifest_path = sample_manager.samples_dir / "manifest.json"
    assert manifest_path.exists(), f"manifest.json not found at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "manifest_version" in data or "version" in data
    assert "samples" in data
    samples = data["samples"]
    assert len(samples) >= 6

    expected_signs = {"HELLO", "YES", "NO", "HELP", "STOP", "THANK_YOU"}
    found_signs = set()

    for s in samples:
        filename = s.get("filename") or s.get("file")
        assert filename, "Sample record missing filename"
        sign = (s.get("sign") or s.get("gloss") or "").upper().replace(" ", "_")
        found_signs.add(sign)

        assert s.get("source"), f"Sample {sign} missing source"
        assert s.get("license"), f"Sample {sign} missing license"
        assert s.get("source_url"), f"Sample {sign} missing source_url"
        assert s.get("duration", 0) > 0, f"Sample {sign} invalid duration"
        assert s.get("verification_status") == "VERIFIED", f"Sample {sign} not VERIFIED"

    missing = expected_signs - found_signs
    assert not missing, f"Missing required signs in manifest: {missing}"


def test_demo_video_discovery_and_metadata(sample_manager: SampleVideoManager):
    """Verifies discovery of physical sample videos and extraction of valid metadata."""
    samples = sample_manager.list_builtin_samples()
    assert len(samples) >= 6

    for rec in samples:
        assert rec.full_path and Path(rec.full_path).exists()
        assert rec.width > 0, f"Invalid width for {rec.sign}"
        assert rec.height > 0, f"Invalid height for {rec.sign}"
        assert rec.fps > 0, f"Invalid fps for {rec.sign}"
        assert rec.frame_count > 0, f"Invalid frame count for {rec.sign}"
        assert rec.duration_seconds > 0, f"Invalid duration for {rec.sign}"
        assert rec.verified is True
        assert rec.is_user_sample is False
        assert rec.verification_status == "VERIFIED"

    # Test case-insensitive retrieval
    hello = sample_manager.get_sample_by_sign("hello")
    assert hello is not None
    assert hello.sign == "HELLO"

    help_sign = sample_manager.get_sample_by_sign("HELP")
    assert help_sign is not None
    assert help_sign.sign == "HELP"

    non_existent = sample_manager.get_sample_by_sign("NON_EXISTENT_SIGN_XYZ")
    assert non_existent is None


def test_demo_video_decoding_and_playback(sample_manager: SampleVideoManager):
    """Verifies VideoFileReader decodes frames correctly from a sample video."""
    sample = sample_manager.get_sample_by_sign("HELLO")
    assert sample and sample.full_path

    reader = VideoFileReader(sample.full_path, loop=False, realtime_pacing=False)
    assert not reader.is_running

    reader.start()
    assert reader.is_running
    assert reader.playback_state == VideoPlaybackState.PLAYING

    frames_read = 0
    for _ in range(10):
        ret, packet = reader.read()
        if not ret or packet is None:
            break
        frames_read += 1
        assert packet.data is not None
        assert isinstance(packet.data, np.ndarray)
        assert packet.data.dtype == np.uint8
        assert packet.data.shape[0] == sample.height
        assert packet.data.shape[1] == sample.width
        assert packet.data.shape[2] == 3

    assert frames_read == 10
    assert reader.current_frame_index >= 10
    reader.stop()
    assert not reader.is_running
    assert reader.playback_state == VideoPlaybackState.CLOSED


def test_playback_controls(sample_manager: SampleVideoManager):
    """Verifies transport controls: start, pause, resume, restart, stop."""
    sample = sample_manager.get_sample_by_sign("HELLO")
    assert sample and sample.full_path

    reader = VideoFileReader(sample.full_path, loop=True, realtime_pacing=False)

    # 1. Start / Play
    reader.start()
    assert reader.playback_state == VideoPlaybackState.PLAYING
    assert reader.is_running

    # Read 5 frames
    for _ in range(5):
        ret, packet = reader.read()
        assert ret and packet is not None
    assert reader.current_frame_index == 5

    # 2. Pause
    reader.pause()
    assert reader.playback_state == VideoPlaybackState.PAUSED

    # Reading while paused returns False without advancing index
    ret, packet = reader.read()
    assert not ret
    assert packet is None
    assert reader.current_frame_index == 5

    # 3. Resume / Play
    reader.play()
    assert reader.playback_state == VideoPlaybackState.PLAYING

    # 4. Restart
    reader.restart()
    assert reader.current_frame_index == 0
    assert reader.playback_state == VideoPlaybackState.PLAYING

    # 5. Stop
    reader.stop()
    assert reader.playback_state == VideoPlaybackState.CLOSED
    assert not reader.is_running


def test_invalid_video_handling(tmp_path: Path):
    """Verifies that non-existent, corrupt, or non-video files are rejected safely."""
    # 1. Non-existent file
    res = validate_video_file(tmp_path / "ghost_file.mp4")
    assert not res.is_valid
    assert res.error_message is not None

    # 2. Non-video file (text file)
    txt_file = tmp_path / "document.txt"
    txt_file.write_text("not a video")
    res_txt = validate_video_file(txt_file)
    assert not res_txt.is_valid
    assert "unsupported" in res_txt.error_message.lower()

    # 3. 0-byte corrupt video file
    empty_mp4 = tmp_path / "corrupt.mp4"
    empty_mp4.touch()
    res_empty = validate_video_file(empty_mp4)
    assert not res_empty.is_valid
    assert "empty" in res_empty.error_message.lower()

    # 4. VideoFileReader with non-existent file raises descriptive ValueError
    with pytest.raises(ValueError, match="Invalid video file"):
        _ = VideoFileReader(tmp_path / "ghost.mp4")


def test_upload_video_handling(tmp_path: Path, sample_manager: SampleVideoManager):
    """Verifies adding, listing, and removing user-uploaded sample videos in isolation."""
    user_dir = tmp_path / "user_samples"
    mgr = SampleVideoManager(
        samples_dir=sample_manager.samples_dir,
        user_samples_dir=user_dir,
    )

    # Initially empty
    assert len(mgr.list_user_samples()) == 0

    # Source video to upload
    source_sample = sample_manager.get_sample_by_sign("HELLO")
    assert source_sample and source_sample.full_path

    # Import user sample
    val = mgr.add_user_sample(source_sample.full_path, sign_label="MY_HELLO")
    assert val.is_valid
    assert val.metadata is not None

    # Verify user sample list
    user_samples = mgr.list_user_samples()
    assert len(user_samples) == 1
    rec = user_samples[0]
    assert rec.sign == "MY_HELLO"
    assert rec.is_user_sample is True
    assert Path(rec.full_path).exists()
    assert Path(rec.full_path).parent.resolve() == user_dir.resolve()
    assert rec.duration_seconds > 0

    # Test removal
    removed = mgr.remove_user_sample(rec.file)
    assert removed is True
    assert len(mgr.list_user_samples()) == 0
    assert not Path(rec.full_path).exists()

    # Test invalid upload rejection
    fake_file = tmp_path / "bad.txt"
    fake_file.write_text("garbage")
    bad_val = mgr.add_user_sample(fake_file, sign_label="BAD")
    assert not bad_val.is_valid
    assert len(mgr.list_user_samples()) == 0
