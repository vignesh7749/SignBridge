"""Automated validation of the 8 Pause/Resume test cases.

Verifies:
- TEST 1: Press P once -> PAUSED remains paused.
- TEST 2: Wait 5 seconds while paused -> Still PAUSED.
- TEST 3: Press P once -> RUNNING.
- TEST 4: Rapidly press P P P P -> Exactly four state transitions.
- TEST 5: Pause -> wait 10 seconds -> resume -> camera functional, no stale-frame buildup.
- TEST 6: Pause -> Q -> Clean shutdown.
- TEST 7: Pause -> ESC -> Clean shutdown.
- TEST 8: Reopen camera after shutdown -> Opens normally.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.camera import (
    CameraManager,
    CameraStatus,
    DemoDisplayController,
    DemoState,
    UserAction,
    list_available_cameras,
)
from signbridge.core.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("test_pause_semantics")


def run_all_tests() -> bool:
    print("=" * 70)
    print("SignBridge — Stage 2 Targeted Verification: Pause / Resume Semantics")
    print("=" * 70)

    devices = list_available_cameras(max_devices=2)
    if not devices:
        print("[!] No hardware camera detected. Aborting test.")
        return False
    cam_index = devices[0].index

    # -------------------------------------------------------------
    # TESTS 1, 2, 3: Single P -> PAUSED -> Wait 5s -> Single P -> RUNNING
    # -------------------------------------------------------------
    print("\n--- Running TEST 1, TEST 2, TEST 3 ---")
    ctrl = DemoDisplayController(debounce_interval_sec=0.20)
    assert ctrl.state == DemoState.RUNNING

    # TEST 1: Press P once
    a1 = ctrl.handle_key(ord("p"), now=1.0)
    ctrl.handle_key(-1, now=1.05)
    assert a1 == UserAction.TOGGLE_PAUSE
    assert ctrl.state == DemoState.PAUSED
    print("[PASS] TEST 1: Pressed P once -> Transitioned to PAUSED.")

    # TEST 2: Wait 5 seconds while paused
    print("Waiting 5 seconds in PAUSED state (feeding idle key -1)...")
    for sec in range(1, 6):
        time.sleep(1.0)
        action = ctrl.handle_key(-1, now=1.05 + sec)
        assert action == UserAction.NONE
        assert ctrl.state == DemoState.PAUSED
        print(f"       Elapsed: {sec}s -> State is still {ctrl.state.name}")
    assert ctrl.state == DemoState.PAUSED
    print("[PASS] TEST 2: 5 seconds elapsed -> Maintained PAUSED state.")

    # TEST 3: Press P once -> RUNNING
    a3 = ctrl.handle_key(ord("p"), now=7.0)
    ctrl.handle_key(-1, now=7.05)
    assert a3 == UserAction.TOGGLE_PAUSE
    assert ctrl.state == DemoState.RUNNING
    print("[PASS] TEST 3: Pressed P once -> Transitioned back to RUNNING.")

    # -------------------------------------------------------------
    # TEST 4: Rapidly press P P P P -> Exactly four state transitions
    # -------------------------------------------------------------
    print("\n--- Running TEST 4: Rapid P Presses ---")
    ctrl4 = DemoDisplayController(debounce_interval_sec=0.15)
    assert ctrl4.state == DemoState.RUNNING
    transitions = []

    # Rapid press 1: P
    if ctrl4.handle_key(ord("p"), now=10.0) == UserAction.TOGGLE_PAUSE:
        transitions.append(ctrl4.state.name)
    ctrl4.handle_key(-1, now=10.08)

    # Rapid press 2: P
    if ctrl4.handle_key(ord("p"), now=10.20) == UserAction.TOGGLE_PAUSE:
        transitions.append(ctrl4.state.name)
    ctrl4.handle_key(-1, now=10.28)

    # Rapid press 3: P
    if ctrl4.handle_key(ord("p"), now=10.40) == UserAction.TOGGLE_PAUSE:
        transitions.append(ctrl4.state.name)
    ctrl4.handle_key(-1, now=10.48)

    # Rapid press 4: P
    if ctrl4.handle_key(ord("p"), now=10.60) == UserAction.TOGGLE_PAUSE:
        transitions.append(ctrl4.state.name)
    ctrl4.handle_key(-1, now=10.68)

    print(f"Recorded state transitions: {transitions}")
    assert len(transitions) == 4
    assert transitions == ["PAUSED", "RUNNING", "PAUSED", "RUNNING"]
    print("[PASS] TEST 4: Exactly 4 rapid physical presses produced exactly 4 state transitions.")

    # -------------------------------------------------------------
    # TEST 5: Hardware pause -> wait 10 seconds -> resume
    # -------------------------------------------------------------
    print("\n--- Running TEST 5: 10-Second Hardware Pause & Fresh Frame Validation ---")
    with CameraManager(camera_index=cam_index, width=640, height=480, fps=30) as cam:
        assert cam.is_running is True
        ret_initial, p_initial = cam.read(timeout=2.0)
        assert ret_initial is True and p_initial is not None
        id_before_pause = p_initial.frame_id
        print(f"[+] Initial frame acquired (ID #{id_before_pause}). Entering 10-second pause...")

        # Display paused for 10 seconds (cam.read() is either queried or queued)
        for s in range(1, 11):
            time.sleep(1.0)
            if s % 2 == 0:
                print(f"       Waiting... {s}s / 10s (Camera hardware stream active)")

        # Verify camera worker remained alive
        assert cam.is_running is True
        assert cam.status == CameraStatus.STREAMING

        # Resume and read latest frame
        ret_resumed, p_resumed = cam.read(timeout=1.0)
        assert ret_resumed is True and p_resumed is not None
        id_after_pause = p_resumed.frame_id
        frame_age = time.time() - p_resumed.timestamp

        print(f"[+] Resumed frame acquired (ID #{id_after_pause}).")
        print(f"    - Frames elapsed in background capture thread: {id_after_pause - id_before_pause}")
        print(f"    - Age of newly delivered frame: {frame_age * 1000.0:.2f} ms")

        assert id_after_pause > id_before_pause
        assert frame_age < 0.25  # Frame must be fresh (< 250ms), proving zero buffer buildup!
        print("[PASS] TEST 5: Camera remained alive during 10s pause; delivered fresh frame immediately upon resume.")

    # -------------------------------------------------------------
    # TEST 6 & 7: Pause -> Q / ESC -> Clean Shutdown
    # -------------------------------------------------------------
    print("\n--- Running TEST 6 & TEST 7: Shutdown from Paused State ---")
    ctrl6 = DemoDisplayController()
    ctrl6.handle_key(ord("p"))
    ctrl6.handle_key(-1)
    assert ctrl6.is_paused is True
    # Press 'q'
    assert ctrl6.handle_key(ord("q")) == UserAction.QUIT
    print("[PASS] TEST 6: Pause -> 'Q' emits UserAction.QUIT cleanly.")

    ctrl7 = DemoDisplayController()
    ctrl7.handle_key(ord("p"))
    ctrl7.handle_key(-1)
    assert ctrl7.is_paused is True
    # Press ESC (code 27)
    assert ctrl7.handle_key(27) == UserAction.QUIT
    print("[PASS] TEST 7: Pause -> ESC emits UserAction.QUIT cleanly.")

    # -------------------------------------------------------------
    # TEST 8: Reopen camera demo after shutdown
    # -------------------------------------------------------------
    print("\n--- Running TEST 8: Reopening Camera Hardware After Shutdown ---")
    cam_reopen = CameraManager(camera_index=cam_index, width=640, height=480, fps=30)
    cam_reopen.start()
    assert cam_reopen.is_running is True
    ret_re, p_re = cam_reopen.read(timeout=2.0)
    assert ret_re is True and p_re is not None
    print(f"[+] Reopened camera successfully. Acquired frame #{p_re.frame_id} ({p_re.width}x{p_re.height}).")
    cam_reopen.stop()
    assert cam_reopen.is_running is False
    assert cam_reopen.status == CameraStatus.CLOSED
    print("[PASS] TEST 8: Camera reopened normally and released cleanly after shutdown.")

    print("\n" + "=" * 70)
    print("[ALL 8 TEST CASES PASSED SUCCESSFULLY]")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
