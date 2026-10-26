"""Unit tests for Stage 2: Camera and OpenCV capture subsystem."""

import time
import numpy as np
import pytest

from signbridge.camera import (
    CameraDeviceInfo,
    CameraManager,
    CameraStatus,
    DemoDisplayController,
    DemoState,
    FramePacket,
    UserAction,
    list_available_cameras,
)
from signbridge.core.config import CameraConfig
from signbridge.core.exceptions import CameraNotFoundError


def test_frame_packet_creation() -> None:
    """Verifies FramePacket telemetry and shape properties."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    packet = FramePacket(
        frame_id=1,
        timestamp=123456.78,
        data=fake_frame,
        width=640,
        height=480,
        latency_ms=5.2,
    )
    assert packet.frame_id == 1
    assert packet.shape == (480, 640, 3)
    assert packet.width == 640
    assert packet.height == 480
    assert packet.latency_ms == 5.2


def test_camera_manager_config_initialization() -> None:
    """Verifies CameraManager handles custom and default parameters correctly."""
    custom_config = CameraConfig(device_index=1, width=1920, height=1080, fps=60)
    cam = CameraManager(config=custom_config)

    assert cam.config.device_index == 1
    assert cam.config.width == 1920
    assert cam.config.height == 1080
    assert cam.config.fps == 60
    assert cam.status == CameraStatus.UNINITIALIZED
    assert cam.is_running is False
    assert cam.current_fps == 0.0


def test_camera_manager_invalid_device_index() -> None:
    """Verifies that attempting to open an invalid camera index raises CameraNotFoundError."""
    invalid_index = 999
    cam = CameraManager(camera_index=invalid_index)

    with pytest.raises(CameraNotFoundError):
        cam.start()

    assert cam.status == CameraStatus.ERROR
    assert cam.is_running is False
    cam.stop()


def test_camera_discovery_executes_safely() -> None:
    """Verifies that device discovery executes without crashing and returns typed metadata."""
    devices = list_available_cameras(max_devices=2, consecutive_fail_limit=2)
    assert isinstance(devices, list)
    for dev in devices:
        assert isinstance(dev, CameraDeviceInfo)
        assert dev.index >= 0
        assert dev.default_width > 0
        assert dev.default_height > 0
        assert dev.default_fps > 0


def test_pause_state_transitions() -> None:
    """Verifies: initial state = RUNNING -> P event -> PAUSED -> no key -> remains PAUSED -> P event -> RUNNING.

    Does not require a physical webcam.
    """
    ctrl = DemoDisplayController(debounce_interval_sec=0.20)

    # Initial state must be RUNNING
    assert ctrl.state == DemoState.RUNNING
    assert ctrl.is_running is True
    assert ctrl.is_paused is False

    # 1. P event -> transitions to PAUSED
    action1 = ctrl.handle_key(ord("p"), now=1.0)
    assert action1 == UserAction.TOGGLE_PAUSE
    assert ctrl.state == DemoState.PAUSED
    assert ctrl.is_paused is True
    assert ctrl.is_running is False

    # 2. No additional key event (-1) over multiple ticks -> remains PAUSED
    for tick in [1.05, 1.10, 1.50, 5.0]:
        action_idle = ctrl.handle_key(-1, now=tick)
        assert action_idle == UserAction.NONE
        assert ctrl.state == DemoState.PAUSED
        assert ctrl.is_paused is True

    # 3. P event -> transitions back to RUNNING
    action2 = ctrl.handle_key(ord("p"), now=5.25)
    assert action2 == UserAction.TOGGLE_PAUSE
    assert ctrl.state == DemoState.RUNNING
    assert ctrl.is_running is True
    assert ctrl.is_paused is False


def test_pause_key_held_down_prevents_double_toggle() -> None:
    """Verifies that physically holding down P across multiple frames only triggers one toggle."""
    ctrl = DemoDisplayController(debounce_interval_sec=0.20)
    assert ctrl.state == DemoState.RUNNING

    # Frame 1: P pressed at t=1.0
    a1 = ctrl.handle_key(ord("p"), now=1.0)
    assert a1 == UserAction.TOGGLE_PAUSE
    assert ctrl.state == DemoState.PAUSED

    # Frame 2-5: P key still held down (typematic repeat / rapid loop)
    for t in [1.02, 1.05, 1.08, 1.12]:
        a = ctrl.handle_key(ord("p"), now=t)
        assert a == UserAction.NONE  # Ignored!
        assert ctrl.state == DemoState.PAUSED  # Does NOT toggle back to RUNNING!

    # Key released (no key event)
    ctrl.handle_key(-1, now=1.15)
    assert ctrl.state == DemoState.PAUSED

    # Next distinct press resumes
    a_resume = ctrl.handle_key(ord("p"), now=1.40)
    assert a_resume == UserAction.TOGGLE_PAUSE
    assert ctrl.state == DemoState.RUNNING


def test_rapid_p_presses_four_transitions() -> None:
    """Verifies that 4 deliberate rapid physical presses (P, release, P, release, P, release, P)

    produce exactly four state transitions: RUNNING -> PAUSED -> RUNNING -> PAUSED -> RUNNING.
    """
    ctrl = DemoDisplayController(debounce_interval_sec=0.15)
    assert ctrl.state == DemoState.RUNNING

    # Press 1
    assert ctrl.handle_key(ord("p"), now=1.0) == UserAction.TOGGLE_PAUSE
    assert ctrl.state == DemoState.PAUSED
    ctrl.handle_key(-1, now=1.08)

    # Press 2
    assert ctrl.handle_key(ord("p"), now=1.20) == UserAction.TOGGLE_PAUSE
    assert ctrl.state == DemoState.RUNNING
    ctrl.handle_key(-1, now=1.28)

    # Press 3
    assert ctrl.handle_key(ord("p"), now=1.40) == UserAction.TOGGLE_PAUSE
    assert ctrl.state == DemoState.PAUSED
    ctrl.handle_key(-1, now=1.48)

    # Press 4
    assert ctrl.handle_key(ord("p"), now=1.60) == UserAction.TOGGLE_PAUSE
    assert ctrl.state == DemoState.RUNNING


def test_camera_manager_real_hardware_lifecycle() -> None:
    """Verifies start, read, pause, resume, and release with available hardware."""
    devices = list_available_cameras(max_devices=2)
    if not devices:
        pytest.skip("No physical camera detected on host.")

    target_index = devices[0].index
    with CameraManager(camera_index=target_index, width=640, height=480, fps=30) as cam:
        assert cam.is_running is True
        assert cam.status == CameraStatus.STREAMING

        # Acquire at least one valid frame
        ret, packet = cam.read(timeout=3.0)
        assert ret is True
        assert packet is not None
        assert isinstance(packet.data, np.ndarray)
        assert len(packet.shape) == 3
        assert packet.width > 0
        assert packet.height > 0

    # Verify context manager release
    assert cam.is_running is False
    assert cam.status == CameraStatus.CLOSED


def test_camera_continues_capturing_during_display_pause() -> None:
    """Verifies that while display state is PAUSED, CameraManager keeps streaming,

    frame IDs progress continuously, and fresh frames are available upon resume.
    """
    devices = list_available_cameras(max_devices=2)
    if not devices:
        pytest.skip("No physical camera detected on host.")

    target_index = devices[0].index
    ctrl = DemoDisplayController()

    with CameraManager(camera_index=target_index, width=640, height=480, fps=30) as cam:
        # 1. Capture first frame in RUNNING state
        ret1, p1 = cam.read(timeout=2.0)
        assert ret1 is True and p1 is not None
        initial_frame_id = p1.frame_id

        # 2. Transition display controller to PAUSED (without pausing CameraManager)
        ctrl.handle_key(ord("p"))
        ctrl.handle_key(-1)  # Physical key released
        assert ctrl.is_paused is True

        # Wait 1.5 seconds while display is paused
        time.sleep(1.5)

        # 3. Verify camera capture worker remained active and progressed frame IDs
        assert cam.is_running is True
        ret2, p2 = cam.read(timeout=1.0)
        assert ret2 is True and p2 is not None
        assert p2.frame_id > initial_frame_id

        # 4. Verify frame latency is fresh (< 1.0 s, verifying frame wasn't frozen from 1.5s ago)
        age = time.time() - p2.timestamp
        assert age < 1.0  # Frame was captured freshly during pause, not 1.5s ago!

        # 5. Resume display controller
        ctrl.handle_key(ord("p"), now=time.time() + 1.0)
        assert ctrl.is_running is True
