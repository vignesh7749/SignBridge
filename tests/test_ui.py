"""Stage 11: Comprehensive Test Suite for SignBridge UI/UX & Application Layer.

Tests:
1. Themes: Color palettes, contrast, and font scales.
2. UI State: Modals, toasts, pointer events, and accessibility toggles.
3. Vector Components: Card, button, progress bar, badge, switch, and hit testing.
4. Dashboard Layout: Header, Camera viewport, Recognition panel, Translation panel,
   Voice panel, Toolbar, and Footer rendering under multiple operational states.
5. Modal Dialogs: Settings, Help, Onboarding dialog rendering and tab switches.
6. Demo Driver: Timed sequence playback, pipeline injection, and reset.
7. App Orchestrator: Event dispatch, keyboard handling, and clean shutdown.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock
import numpy as np
import pytest

from signbridge.camera.base import CameraStatus
from signbridge.pipeline.contracts import StageLatencies
from signbridge.pipeline.state import PipelineRuntimeState
from signbridge.temporal.contracts import TemporalState
from signbridge.ui.app import SignBridgeApp
from signbridge.ui.components import (
    draw_badge,
    draw_button,
    draw_card,
    draw_card_header,
    draw_metric_pill,
    draw_progress_bar,
    draw_status_indicator,
    draw_switch,
    hit_test,
)
from signbridge.ui.demo import DEMO_SEQUENCE, SignBridgeDemoDriver
from signbridge.ui.layout import DashboardLayout
from signbridge.ui.overlays import ModalOverlayManager
from signbridge.ui.state import ActiveModal, AppStatus, NotificationToast, UIState
from signbridge.ui.theme import HIGH_CONTRAST_THEME, OBSIDIAN_THEME, UITheme


# ============================================================================
# 1. THEME TESTS
# ============================================================================

def test_theme_properties() -> None:
    """Verifies that themes define all required color tuples and styles."""
    for theme in (OBSIDIAN_THEME, HIGH_CONTRAST_THEME):
        assert len(theme.bg_canvas) == 3
        assert len(theme.card_bg) == 3
        assert len(theme.text_primary) == 3
        assert len(theme.primary_accent) == 3
        assert len(theme.secondary_accent) == 3
        assert len(theme.warning_accent) == 3
        assert len(theme.error_accent) == 3
        assert theme.border_width >= 1
        assert theme.card_radius >= 0


def test_high_contrast_theme_distinction() -> None:
    """Verifies high contrast theme uses pure black/white/yellow WCAG AAA colors."""
    assert HIGH_CONTRAST_THEME.bg_canvas == (0, 0, 0)
    assert HIGH_CONTRAST_THEME.text_primary == (255, 255, 255)
    assert HIGH_CONTRAST_THEME.primary_accent == (0, 255, 255)  # High-visibility yellow
    assert HIGH_CONTRAST_THEME.border_width == 2


# ============================================================================
# 2. UI STATE TESTS
# ============================================================================

def test_ui_state_defaults() -> None:
    """Tests default state initialization."""
    state = UIState()
    assert state.active_modal == ActiveModal.NONE
    assert not state.high_contrast
    assert not state.reduced_motion
    assert state.toast is None
    assert state.mouse_x == -1
    assert state.mouse_y == -1


def test_ui_state_modal_lifecycle() -> None:
    """Tests opening, closing, and toggling modals."""
    state = UIState()
    state.open_modal(ActiveModal.SETTINGS)
    assert state.active_modal == ActiveModal.SETTINGS

    state.close_modal()
    assert state.active_modal == ActiveModal.NONE

    state.open_modal(ActiveModal.HELP)
    assert state.active_modal == ActiveModal.HELP
    assert state.help_tab == 0

    state.close_modal()
    assert state.active_modal == ActiveModal.NONE


def test_ui_state_accessibility_toggles() -> None:
    """Tests toggling high contrast and reduced motion flags."""
    state = UIState()
    assert not state.high_contrast
    hc = state.toggle_high_contrast()
    assert hc and state.high_contrast

    assert not state.reduced_motion
    rm = state.toggle_reduced_motion()
    assert rm and state.reduced_motion


def test_ui_state_toast_lifecycle() -> None:
    """Tests setting, checking progress, and expiration of toasts."""
    toast = NotificationToast(message="Test notification", duration=0.1)
    assert not toast.is_expired
    assert 0.0 <= toast.progress <= 1.0

    time.sleep(0.12)
    assert toast.is_expired
    assert toast.progress == 0.0

    state = UIState()
    state.set_toast("Saved successfully", level="success", duration=1.0)
    assert state.toast is not None
    assert state.toast.message == "Saved successfully"
    assert state.toast.level == "success"


# ============================================================================
# 3. COMPONENT RENDERING & HIT-TESTING TESTS
# ============================================================================

def test_hit_test() -> None:
    """Tests point-in-rect geometric calculation."""
    # bx=100, by=200, bw=50, bh=30
    assert hit_test(100, 200, 100, 200, 50, 30)
    assert hit_test(125, 215, 100, 200, 50, 30)
    assert hit_test(150, 230, 100, 200, 50, 30)
    assert not hit_test(99, 200, 100, 200, 50, 30)
    assert not hit_test(151, 200, 100, 200, 50, 30)
    assert not hit_test(125, 199, 100, 200, 50, 30)
    assert not hit_test(125, 231, 100, 200, 50, 30)


def test_component_drawing_primitives() -> None:
    """Verifies that vector drawing primitives render into a canvas without exceptions."""
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)

    # draw_card
    draw_card(canvas, 10, 10, 200, 150, OBSIDIAN_THEME.card_bg, OBSIDIAN_THEME.card_border)

    # draw_card_header
    draw_card_header(canvas, 10, 10, 200, "RECOGNITION", OBSIDIAN_THEME)

    # draw_button
    draw_button(canvas, 20, 50, 80, 30, "TEST", "T", is_hovered=True, is_active=False, theme=OBSIDIAN_THEME)

    # draw_progress_bar
    draw_progress_bar(canvas, 20, 90, 150, 12, 0.75, (0, 255, 0), (30, 30, 30), (50, 50, 50))

    # draw_badge
    draw_badge(canvas, 20, 110, "LIVE", (0, 200, 0), (20, 40, 20), (0, 200, 0))

    # draw_status_indicator
    draw_status_indicator(canvas, 300, 50, AppStatus.HAND_DETECTED, OBSIDIAN_THEME)

    # draw_metric_pill
    draw_metric_pill(canvas, 300, 100, "FPS", "30.0", OBSIDIAN_THEME)

    # draw_switch
    draw_switch(canvas, 300, 150, 200, 30, is_on=True, label="Speech", is_hovered=False, theme=OBSIDIAN_THEME)

    # Canvas should contain drawn pixels
    assert np.any(canvas > 0)


# ============================================================================
# 4. DASHBOARD LAYOUT TESTS
# ============================================================================

def test_dashboard_layout_render_idle() -> None:
    """Tests layout rendering in clean initial/idle state."""
    layout = DashboardLayout(1280, 800)
    canvas = np.zeros((800, 1280, 3), dtype=np.uint8)
    state = PipelineRuntimeState.initial()
    ui_state = UIState()

    layout.render(canvas, state, ui_state, OBSIDIAN_THEME)

    # Verify canvas was filled
    assert np.any(canvas > 0)

    # Verify registered interactive buttons
    buttons = layout.get_registered_buttons()
    assert "btn_help" in buttons
    assert "btn_settings" in buttons
    assert "btn_speak" in buttons
    assert "btn_stop" in buttons
    assert "btn_voice_toggle" in buttons
    assert "btn_space" in buttons
    assert "btn_backspace" in buttons
    assert "btn_clear" in buttons
    assert "btn_undo" in buttons
    assert "btn_pause" in buttons
    assert "btn_reset" in buttons


def test_dashboard_layout_render_active_stream() -> None:
    """Tests layout rendering with simulated active hand recognition."""
    layout = DashboardLayout(1280, 800)
    canvas = np.zeros((800, 1280, 3), dtype=np.uint8)
    camera_frame = np.full((480, 640, 3), 50, dtype=np.uint8)

    state = PipelineRuntimeState(
        camera_status=CameraStatus.STREAMING,
        camera_fps=30.0,
        processing_fps=29.5,
        frame_count=200,
        pipeline_latency=14.2,
        stage_latencies=StageLatencies(camera_capture_ms=2.1, hand_detection_ms=7.0, classifier_ms=1.2),
        hand_detected=True,
        handedness="Right",
        detection_confidence=0.98,
        hand_count=1,
        predicted_label="A",
        prediction_confidence=0.94,
        top_margin=0.88,
        temporal_state=TemporalState.CANDIDATE,
        temporal_candidate="A",
        dwell_progress=0.70,
        stable_token="A",
        current_word="HELL",
        sentence="HELLO WORLD",
        completed_words=("HELLO", "WORLD"),
        translation_state="APPENDING",
        voice_enabled=True,
        voice_status="READY",
        is_speaking=False,
    )
    ui_state = UIState(high_contrast=True)

    layout.render(canvas, state, ui_state, HIGH_CONTRAST_THEME, camera_frame=camera_frame)
    assert np.any(canvas > 0)


# ============================================================================
# 5. MODAL OVERLAYS TESTS
# ============================================================================

def test_modal_overlay_manager() -> None:
    """Tests rendering all three modal dialogs and their controls."""
    manager = ModalOverlayManager()
    canvas = np.zeros((800, 1280, 3), dtype=np.uint8)
    ui_state = UIState()
    app_settings = {
        "voice_enabled": True,
        "speak_words": True,
        "speech_rate": 150,
        "speech_volume": 0.9,
        "show_skeleton": True,
    }

    # NONE modal
    ui_state.active_modal = ActiveModal.NONE
    manager.render(canvas, ui_state, OBSIDIAN_THEME, app_settings)
    assert len(manager.get_registered_buttons()) == 0

    # SETTINGS modal
    ui_state.active_modal = ActiveModal.SETTINGS
    manager.render(canvas, ui_state, OBSIDIAN_THEME, app_settings)
    buttons = manager.get_registered_buttons()
    assert "modal_close" in buttons
    assert "set_toggle_voice" in buttons
    assert "set_toggle_contrast" in buttons
    assert "set_toggle_motion" in buttons

    # HELP modal
    ui_state.active_modal = ActiveModal.HELP
    manager.render(canvas, ui_state, OBSIDIAN_THEME, app_settings)
    buttons = manager.get_registered_buttons()
    assert "modal_close" in buttons
    assert "help_tab_0" in buttons
    assert "help_tab_1" in buttons
    assert "help_tab_2" in buttons
    assert "help_tab_3" in buttons

    # ONBOARDING modal
    ui_state.active_modal = ActiveModal.ONBOARDING
    ui_state.onboarding_step = 0
    manager.render(canvas, ui_state, OBSIDIAN_THEME, app_settings)
    buttons = manager.get_registered_buttons()
    assert "modal_close" in buttons
    assert "onboarding_done" in buttons
    assert "onboarding_skip" in buttons


# ============================================================================
# 6. DEMO DRIVER TESTS
# ============================================================================

def test_demo_driver_execution() -> None:
    """Tests demo script step progression and pipeline simulation."""
    mock_pipeline = MagicMock()
    driver = SignBridgeDemoDriver(mock_pipeline)

    driver.start()
    assert driver.is_active
    assert driver._step_index == 0

    # Advance one step with zero duration override
    msg = driver.update(step_duration=0.0)
    assert msg is not None
    assert driver._step_index > 0
    mock_pipeline.translation_engine.process_char.assert_called_once_with("H")

    # Simulate running through all steps
    for _ in range(len(DEMO_SEQUENCE) + 5):
        driver.update(step_duration=0.0)

    driver.stop()
    assert not driver.is_active


# ============================================================================
# 7. APP ORCHESTRATOR & DISPATCH TESTS
# ============================================================================

def test_app_action_dispatch() -> None:
    """Tests button action dispatch handlers."""
    mock_pipeline = MagicMock()
    mock_pipeline.space.return_value = MagicMock(sentence_text="HELLO")
    mock_pipeline.backspace.return_value = MagicMock(current_word="HE")
    mock_pipeline.toggle_voice.return_value = True
    mock_pipeline.speak_current_sentence.return_value = True

    app = SignBridgeApp(pipeline=mock_pipeline)

    # Test SPACE
    app._dispatch_button_action("btn_space")
    mock_pipeline.space.assert_called_once()

    # Test BACKSPACE
    app._dispatch_button_action("btn_backspace")
    mock_pipeline.backspace.assert_called_once()

    # Test CLEAR
    app._dispatch_button_action("btn_clear")
    mock_pipeline.clear_sentence.assert_called_once()

    # Test SPEAK
    app._dispatch_button_action("btn_speak")
    mock_pipeline.speak_current_sentence.assert_called_once()

    # Test VOICE TOGGLE
    app._dispatch_button_action("btn_voice_toggle")
    mock_pipeline.toggle_voice.assert_called_once()

    # Test PAUSE/RESUME
    mock_pipeline.is_paused = False
    app._dispatch_button_action("btn_pause")
    mock_pipeline.pause.assert_called_once()

    # Test RESET
    app._dispatch_button_action("btn_reset")
    mock_pipeline.reset.assert_called_once()

    # Test SETTINGS button
    app._dispatch_button_action("btn_settings")
    assert app.ui_state.active_modal == ActiveModal.SETTINGS

    # Test HELP button
    app._dispatch_button_action("btn_help")
    assert app.ui_state.active_modal == ActiveModal.HELP

    # Test MODAL CLOSE
    app._dispatch_button_action("modal_close")
    assert app.ui_state.active_modal == ActiveModal.NONE


def test_app_keyboard_shortcuts() -> None:
    """Tests keyboard shortcut event translation."""
    mock_pipeline = MagicMock()
    mock_pipeline.space.return_value = MagicMock(sentence_text="HELLO")
    mock_pipeline.backspace.return_value = MagicMock(current_word="HE")
    mock_pipeline.toggle_voice.return_value = True
    mock_pipeline.speak_current_sentence.return_value = True

    app = SignBridgeApp(pipeline=mock_pipeline)

    # Test Space key (32)
    app._handle_key(32)
    mock_pipeline.space.assert_called_once()

    # Test 'c' key
    app._handle_key(ord("c"))
    mock_pipeline.clear_sentence.assert_called_once()

    # Test 't' key -> Toggle Voice
    app._handle_key(ord("t"))
    mock_pipeline.toggle_voice.assert_called_once()

    # Test 'p' key -> Pause
    mock_pipeline.is_paused = False
    app._handle_key(ord("p"))
    mock_pipeline.pause.assert_called_once()

    # Test 's' key -> Speak
    app._handle_key(ord("s"))
    mock_pipeline.speak_current_sentence.assert_called_once()

    # Test 'x' key -> Stop Speech
    app._handle_key(ord("x"))
    mock_pipeline.stop_speech.assert_called_once()

    # Test 'o' key -> Open Settings
    app._handle_key(ord("o"))
    assert app.ui_state.active_modal == ActiveModal.SETTINGS

    # Test Escape key (27) -> Close Modal
    app._handle_key(27)
    assert app.ui_state.active_modal == ActiveModal.NONE

    # Test Escape key (27) when no modal -> Quit
    should_continue = app._handle_key(27)
    assert not should_continue


def test_app_mouse_callback() -> None:
    """Tests OpenCV mouse callback translation into UI state and actions."""
    import cv2
    mock_pipeline = MagicMock()
    mock_pipeline.space.return_value = MagicMock(sentence_text="HELLO")
    app = SignBridgeApp(pipeline=mock_pipeline)

    # Register buttons via a test layout render
    state = PipelineRuntimeState.initial()
    app.layout.render(app._canvas, state, app.ui_state, OBSIDIAN_THEME)

    # Find position of btn_space
    buttons = app.layout.get_registered_buttons()
    assert "btn_space" in buttons
    x, y, w, h = buttons["btn_space"]

    # Simulate mouse move over button
    app._on_mouse(cv2.EVENT_MOUSEMOVE, x + w // 2, y + h // 2, 0, None)
    assert app.ui_state.hovered_id == "btn_space"

    # Simulate mouse click on button
    app._on_mouse(cv2.EVENT_LBUTTONDOWN, x + w // 2, y + h // 2, 0, None)
    app._on_mouse(cv2.EVENT_LBUTTONUP, x + w // 2, y + h // 2, 0, None)
    mock_pipeline.space.assert_called_once()
