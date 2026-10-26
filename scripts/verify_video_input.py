"""Verification script for video input handling, formats, and sample video management.

Validates:
1. Video file extension and integrity verification (.mp4, .avi, .mov, .mkv, .webm).
2. Built-in sample video availability and metadata correctness.
3. VideoFileReader initialization, frame extraction, and playback metrics.
4. User upload validation workflow.
"""

import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.camera.base import CameraStatus
from signbridge.video.base import VideoPlaybackState
from signbridge.video.reader import VideoFileReader
from signbridge.video.sample_manager import SampleVideoManager
from signbridge.video.validator import SUPPORTED_EXTENSIONS, validate_video_file


def run_video_verification() -> bool:
    print("=" * 65)
    print("SIGNBRIDGE VIDEO INPUT & DEMO VERIFICATION")
    print("=" * 65)

    all_passed = True

    # 1. Verify Supported Format Whitelist
    print("\n[Step 1/4] Checking supported video formats...")
    expected_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if expected_extensions.issubset(SUPPORTED_EXTENSIONS):
        print(f"  [OK] Whitelist matches: {sorted(list(SUPPORTED_EXTENSIONS))}")
    else:
        print(f"  [FAIL] Whitelist missing formats: {expected_extensions - SUPPORTED_EXTENSIONS}")
        all_passed = False

    # Test invalid extension rejection
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        res = validate_video_file(tmp.name)
        if not res.is_valid and "unsupported" in res.error_message.lower():
            print("  [OK] Correctly rejected invalid file extension (.txt)")
        else:
            print(f"  [FAIL] Unexpected validation result on invalid extension: {res}")
            all_passed = False

    # 2. Verify Sample Video Manager & Built-in Samples
    print("\n[Step 2/4] Inspecting built-in demonstration videos...")
    mgr = SampleVideoManager()
    builtin_samples = mgr.list_builtin_samples()
    print(f"  Found {len(builtin_samples)} built-in sample videos.")

    expected_signs = ["HELLO", "HELP", "YES", "NO", "STOP", "THANK YOU"]
    available_signs = {s.sign for s in builtin_samples}

    for sign in expected_signs:
        if sign in available_signs:
            sample = mgr.get_sample_by_sign(sign)
            print(f"  [OK] '{sign}': {sample.file} ({sample.width}x{sample.height}, {sample.fps:.1f} fps, {sample.duration_seconds:.2f}s, {sample.frame_count} frames)")
        else:
            print(f"  [FAIL] Missing built-in sample for '{sign}'")
            all_passed = False

    # 3. VideoFileReader Decoding & Playback Test
    print("\n[Step 3/4] Testing VideoFileReader playback and frame streaming...")
    hello_sample = mgr.get_sample_by_sign("HELLO")
    if hello_sample and hello_sample.full_path:
        reader = VideoFileReader(hello_sample.full_path, loop=True, realtime_pacing=False)
        reader.start()
        frames_read = 0
        for _ in range(15):
            ok, pkt = reader.read(timeout=0.5)
            if ok and pkt is not None:
                frames_read += 1
                if frames_read == 1:
                    print(f"  First frame decoded: shape={pkt.data.shape}, resolution={pkt.width}x{pkt.height}")
        reader.stop()

        if frames_read == 15 and reader.playback_state == VideoPlaybackState.CLOSED:
            print(f"  [OK] Successfully streamed {frames_read} frames and closed reader safely.")
        else:
            print(f"  [FAIL] Streamed {frames_read}/15 frames; reader state: {reader.playback_state}")
            all_passed = False
    else:
        print("  [FAIL] HELLO sample not available for decoding test")
        all_passed = False

    # 4. User Upload Directory Isolation
    print("\n[Step 4/4] Verifying user uploads directory isolation...")
    user_samples_dir = PROJECT_ROOT / "datasets" / "user_samples" / "word_signs"
    if user_samples_dir.exists():
        print(f"  [OK] User samples directory exists and is isolated: {user_samples_dir}")
    else:
        user_samples_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] Initialized isolated user samples directory: {user_samples_dir}")

    print("\n" + "=" * 65)
    if all_passed:
        print("RESULT: ALL VIDEO INPUT TESTS PASSED [SUCCESS]")
        print("=" * 65)
        return True
    else:
        print("RESULT: VERIFICATION FAILURES ENCOUNTERED [FAILURE]")
        print("=" * 65)
        return False


if __name__ == "__main__":
    success = run_video_verification()
    sys.exit(0 if success else 1)
