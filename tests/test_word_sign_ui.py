"""Tests for safe emergency simulation modal, UI overlays, and demo controls."""

from unittest.mock import MagicMock
import numpy as np
import pytest

from signbridge.pipeline.state import PipelineRuntimeState
from signbridge.ui.layout import DashboardLayout
from signbridge.ui.overlays import ModalOverlayManager
from signbridge.ui.state import ActiveModal, UIState
from signbridge.ui.theme import OBSIDIAN_THEME
from signbridge.word_sign.emergency import (
    EmergencyEvent,
    EmergencyModalManager,
    EmergencyState,
)


def test_emergency_modal_manager_safety_and_lifecycle():
    """Emergency workflow triggers strictly in simulation mode without external requests."""
    confirmed_cb = MagicMock()
    cancelled_cb = MagicMock()
    manager = EmergencyModalManager(on_confirmed=confirmed_cb, on_cancelled=cancelled_cb)

    assert manager.state == EmergencyState.IDLE
    assert not manager.is_modal_active

    # Non-emergency sign does not trigger modal
    assert not manager.trigger_sign_detected("HELLO", confidence=0.95)
    assert not manager.is_modal_active

    # Emergency sign triggers modal confirmation
    assert manager.trigger_sign_detected("HELP", confidence=0.92)
    assert manager.is_modal_active
    assert manager.state == EmergencyState.CONFIRMATION_REQUESTED

    # Confirm action
    event = manager.confirm_assistance()
    assert manager.is_assistance_active
    assert event.state == EmergencyState.ASSISTANCE_ACTIVE
    assert "simulated" in event.message.lower()
    confirmed_cb.assert_called_once()

    # Reset and test cancel
    manager.reset()
    manager.trigger_sign_detected("HELP", confidence=0.90)
    cancel_event = manager.cancel()
    assert manager.state == EmergencyState.DISMISSED
    cancelled_cb.assert_called_once()


def test_emergency_modal_render_and_hit_test():
    """ModalOverlayManager renders emergency modal and exposes interactive action buttons."""
    overlay = ModalOverlayManager()
    canvas = np.zeros((800, 1280, 3), dtype=np.uint8)
    ui_state = UIState()
    ui_state.active_modal = ActiveModal.EMERGENCY

    overlay.render(canvas, ui_state, OBSIDIAN_THEME, {})
    buttons = overlay.get_registered_buttons()

    assert "btn_emergency_confirm" in buttons
    assert "btn_emergency_cancel" in buttons
    assert "modal_close" in buttons


def test_video_demo_modal_render_and_buttons():
    """ModalOverlayManager renders video demo hub with sample buttons."""
    overlay = ModalOverlayManager()
    canvas = np.zeros((800, 1280, 3), dtype=np.uint8)
    ui_state = UIState()
    ui_state.active_modal = ActiveModal.VIDEO_DEMO

    overlay.render(canvas, ui_state, OBSIDIAN_THEME, {})
    buttons = overlay.get_registered_buttons()

    assert "btn_sample_hello" in buttons
    assert "btn_sample_help" in buttons
    assert "btn_source_camera" in buttons
    assert "modal_close_btn" in buttons


def test_dashboard_layout_renders_mode_toggles():
    """DashboardLayout renders recognition mode selector pills and registers click areas."""
    layout = DashboardLayout()
    canvas = np.zeros((800, 1280, 3), dtype=np.uint8)
    p_state = PipelineRuntimeState(recognition_mode="AUTO", active_word_sign="HELP", word_sign_confidence=0.92)
    ui_state = UIState()

    layout.render(canvas, p_state, ui_state, OBSIDIAN_THEME)
    buttons = layout.get_registered_buttons()

    assert "btn_mode_alpha" in buttons
    assert "btn_mode_word" in buttons
    assert "btn_mode_auto" in buttons
    assert "btn_video_demo" in buttons
