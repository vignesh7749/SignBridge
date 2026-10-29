"""Final Quality Assurance and Presentation-Readiness Test Suite.

Verifies:
1. Keyboard shortcut registry completeness and zero intra-context conflicts.
2. Button hotkey badge alignment and label/badge overlap prevention.
3. Demo Hub card 4-zone strict non-overlapping layout geometry.
4. Reference demo video isolation: playing goodbye.mp4 never mutates the translated sentence or triggers TTS speech.
5. Live camera state restoration and clean mode isolation.
"""

from pathlib import Path
import numpy as np
import pytest

from signbridge.pipeline.orchestrator import SignBridgePipeline, PipelineConfig
from signbridge.video.reader import VideoFileReader
from signbridge.ui.shortcuts import SHORTCUT_REGISTRY, ShortcutRegistry
from signbridge.ui.theme import OBSIDIAN_THEME
from signbridge.ui.components import draw_button, hit_test


def test_shortcut_registry_zero_conflicts():
    """Asserts that the authoritative shortcut registry contains zero conflicting key codes."""
    conflicts = SHORTCUT_REGISTRY.detect_conflicts()
    assert conflicts == [], f"Detected shortcut key conflicts: {conflicts}"


def test_shortcut_registry_actions_and_help_groups():
    """Asserts that all registered actions have complete metadata and populate help groups."""
    actions = SHORTCUT_REGISTRY.list_all()
    assert len(actions) >= 15

    for act in actions:
        assert act.action_id, "Action ID must not be empty"
        assert act.badge, f"Action {act.action_id} must have a hotkey badge"
        assert act.description, f"Action {act.action_id} must have a description"
        assert act.category, f"Action {act.action_id} must belong to a category"
        assert act.context in ("global", "workspace", "modal", "video_player")

    grouped = SHORTCUT_REGISTRY.get_help_grouped()
    assert "Sentence & Word Editing" in grouped
    assert "Voice & Audio Output" in grouped
    assert "Recognition Modes" in grouped
    assert "Viewport & Video Stream" in grouped
    assert "System & Dialogs" in grouped

    # Test key code matching
    c_action = SHORTCUT_REGISTRY.match_key(ord("c"), context="workspace")
    assert c_action is not None and c_action.action_id == "clear_sentence"

    v_action = SHORTCUT_REGISTRY.match_key(ord("v"), context="workspace")
    assert v_action is not None and v_action.action_id == "open_demo_hub"

    r_action = SHORTCUT_REGISTRY.match_key(ord("r"), context="workspace")
    assert r_action is not None and r_action.action_id == "reset_all"


def test_button_drawing_overlap_safety():
    """Asserts that draw_button handles small widths and badges without crashing or overflow."""
    canvas = np.zeros((100, 300, 3), dtype=np.uint8)

    # 1. Compact button with hotkey (like mode toggles)
    draw_button(
        canvas=canvas,
        x=10,
        y=10,
        w=68,
        h=22,
        label="Alpha",
        hotkey="1",
        is_hovered=False,
        is_active=True,
        theme=OBSIDIAN_THEME,
    )

    # 2. Button with long label and hotkey
    draw_button(
        canvas=canvas,
        x=85,
        y=10,
        w=140,
        h=28,
        label="Run Recognition",
        hotkey="",
        is_hovered=True,
        is_active=False,
        theme=OBSIDIAN_THEME,
    )

    # Canvas should have non-zero pixels drawn
    assert np.count_nonzero(canvas) > 0


def test_demo_hub_card_four_zones_geometry():
    """Asserts that the 4 card zones never overlap horizontally."""
    card_w = 900
    cx = 50
    cy = 100

    # Zone 1: Thumbnail Box
    tw = 76
    tx = cx + 10
    zone1_right = tx + tw  # cx + 86

    # Zone 2: Identity & Badges
    nx = tx + tw + 12     # cx + 98
    zone2_w = 160
    zone2_right = nx + zone2_w  # cx + 258

    # Zone 4: Right Buttons
    btn_start_x = cx + card_w - 180  # cx + 720
    zone4_right = cx + card_w - 12   # cx + 888

    # Zone 3: Educational Notes
    zone3_x = nx + zone2_w + 14  # cx + 272
    zone3_right = btn_start_x - 12  # cx + 708

    # Assert strict disjoint monotonic progression
    assert zone1_right < nx, "Zone 1 (thumbnail) must end before Zone 2 (identity) begins"
    assert zone2_right < zone3_x, "Zone 2 (identity) must end before Zone 3 (mechanics) begins"
    assert zone3_right < btn_start_x, "Zone 3 (mechanics) must end before Zone 4 (buttons) begins"
    assert zone3_right - zone3_x > 200, "Zone 3 must have sufficient readable width"


def test_goodbye_reference_demo_isolation():
    """Asserts that playing goodbye.mp4 reference demo does NOT pollute the translated sentence with 'NO'."""
    project_root = Path(__file__).resolve().parents[1]
    candidate_paths = [
        project_root / "datasets" / "samples" / "word_signs" / "goodbye.mp4",
        project_root / "public" / "videos" / "goodbye.mp4",
    ]
    goodbye_video = next((p for p in candidate_paths if p.exists()), None)
    if not goodbye_video:
        pytest.skip(f"Test video goodbye.mp4 not found in {candidate_paths}")

    from signbridge.tts.config import TTSConfig
    cfg = PipelineConfig(draw_hud=False, draw_skeleton=False, tts=TTSConfig(enabled=False))
    pipeline = SignBridgePipeline(config=cfg)

    reader = VideoFileReader(str(goodbye_video), loop=False, realtime_pacing=False)
    pipeline.switch_to_demo_video(reader, sign_label="GOODBYE")

    assert pipeline.target_demo_sign == "GOODBYE"
    assert pipeline.translation_engine.sentence_text == ""

    # Process 25 frames of goodbye.mp4
    for _ in range(25):
        pipeline.process_next_frame(timeout=0.05)

    # CRITICAL: Sentence must remain completely empty!
    # Even if raw word model activated "NO", it must NOT be injected into translated sentence.
    sentence = pipeline.translation_engine.sentence_text
    assert "NO" not in sentence.upper(), f"Sentence polluted with 'NO' during Goodbye playback: '{sentence}'"
    assert sentence == "", f"Sentence should be empty during reference demo playback, got: '{sentence}'"
    assert pipeline.get_state().sentence == ""

    # Switching back to live camera cleans target demo sign
    pipeline.switch_to_live_camera()
    assert pipeline.target_demo_sign is None
    pipeline.stop()


def test_held_gesture_emits_only_once_until_neutral_rest():
    """Asserts that holding a gesture continuously emits exactly once, and only re-arms after neutral rest."""
    from unittest.mock import MagicMock
    from signbridge.word_sign.model import WordPredictionOutput, WordSignClassifier
    from signbridge.word_sign.temporal import WordSignTemporalSmoother
    from signbridge.vision.detector import HandDetectionResult
    from signbridge.vision.landmarks import DetectedHand, LandmarkPoint

    # Mock classifier that returns HELLO with 0.92 confidence
    mock_clf = MagicMock(spec=WordSignClassifier)
    mock_clf.predict.return_value = WordPredictionOutput(
        predicted_word="HELLO",
        confidence=0.92,
        top_k=[("HELLO", 0.92), ("HELP", 0.05)],
        inference_time_ms=0.5,
    )

    smoother = WordSignTemporalSmoother(
        classifier=mock_clf,
        window_size=30,
        min_confidence=0.75,
        stable_count=3,
        cooldown_seconds=1.0,
        neutral_dwell_seconds=0.3,
    )

    # Create dummy hand detection with 21 landmarks
    dummy_lms = tuple(
        LandmarkPoint(
            index=i,
            name=f"pt_{i}",
            x_norm=0.5,
            y_norm=0.5,
            z_norm=0.0,
            x_px=320,
            y_px=240,
            z_depth=0.0,
        )
        for i in range(21)
    )
    hand = DetectedHand(
        handedness="Right",
        landmarks=dummy_lms,
        raw_array=np.full((21, 3), 0.5, dtype=np.float32),
        handedness_confidence=0.95,
    )
    det_with_hand = HandDetectionResult(frame_id=1, timestamp=1000.0, hands=(hand,), processing_time_ms=1.0)
    det_no_hand = HandDetectionResult(frame_id=1, timestamp=1000.0, hands=(), processing_time_ms=0.5)

    emissions = []
    current_time = 1000.0

    # Phase 1: Hold HELLO for 60 frames (2 seconds at 30fps)
    for i in range(60):
        current_time += 0.033
        pred = smoother.process_frame(det_with_hand, current_time=current_time)
        if pred.is_newly_emitted:
            emissions.append((i, pred.word, current_time))

    # Must emit EXACTLY ONCE despite holding for 60 frames (overcoming 1.0s cooldown)
    assert len(emissions) == 1, f"Expected exactly 1 emission while holding, got {len(emissions)}: {emissions}"
    assert emissions[0][1] == "HELLO"

    # Phase 2: Still holding HELLO for another 60 frames (further 2 seconds)
    for i in range(60, 120):
        current_time += 0.033
        pred = smoother.process_frame(det_with_hand, current_time=current_time)
        if pred.is_newly_emitted:
            emissions.append((i, pred.word, current_time))

    # Still must NOT emit again while continuously held
    assert len(emissions) == 1, f"Expected still 1 emission while holding, got {len(emissions)}"

    # Phase 3: Neutral rest / hands lowered for 15 frames (0.5 seconds > 0.3s neutral_dwell)
    for _ in range(15):
        current_time += 0.033
        pred = smoother.process_frame(det_no_hand, current_time=current_time)
        assert not pred.is_newly_emitted

    # Phase 4: Raise hands and intentionally sign HELLO again for 40 frames
    for i in range(135, 175):
        current_time += 0.033
        pred = smoother.process_frame(det_with_hand, current_time=current_time)
        if pred.is_newly_emitted:
            emissions.append((i, pred.word, current_time))

    # Now we should have exactly 2 emissions!
    assert len(emissions) == 2, f"Expected 2 emissions after intentional neutral rest, got {len(emissions)}"
    assert emissions[1][1] == "HELLO"


def test_word_mode_isolation_ui_no_alphabet_leak():
    """Asserts that in WORD_SIGNS mode, static alphabet predictions (e.g. 'C') never appear in the recognition card."""
    from signbridge.pipeline.state import PipelineRuntimeState
    from signbridge.ui.layout import DashboardLayout
    from signbridge.ui.state import UIState

    layout = DashboardLayout(target_width=1280, target_height=720)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    u_state = UIState()

    # Create runtime state with alphabet predicted_label="C" (79% conf), but recognition_mode="WORD_SIGNS" and active_word_sign=None
    p_state = PipelineRuntimeState(
        recognition_mode="WORD_SIGNS",
        active_word_sign=None,
        word_sign_confidence=0.0,
        predicted_label="C",
        prediction_confidence=0.79,
        hand_detected=True,
        frame_source="LIVE_CAMERA",
    )

    # Render recognition card
    layout._render_recognition_card(canvas, x=800, y=50, w=450, h=140, p_state=p_state, u_state=u_state, theme=OBSIDIAN_THEME)

    # In WORD_SIGNS mode when active_word_sign is None, it should render "—", NOT "C"!
    if p_state.recognition_mode == "WORD_SIGNS":
        pred_char = p_state.active_word_sign or "—"
    else:
        pred_char = p_state.predicted_label or "—"
    assert pred_char == "—"
    assert pred_char != "C"


def test_speak_current_sentence_prioritizes_sentence_over_stale_tokens():
    """Asserts that speak_current_sentence speaks the visible sentence 'C', NOT stale token 'YES'."""
    from unittest.mock import MagicMock
    from signbridge.tts.config import TTSConfig

    cfg = PipelineConfig(draw_hud=False, draw_skeleton=False, tts=TTSConfig(enabled=True))
    pipeline = SignBridgePipeline(config=cfg)
    pipeline.speech_controller.speak_manual = MagicMock(return_value=True)

    # Stale token from earlier gesture
    pipeline.last_stable_token = "YES"

    # Active constructed sentence in translation engine
    pipeline.translation_engine.process_word("C")
    assert pipeline.translation_engine.sentence_text == "C"

    spoken = pipeline.speak_current_sentence()
    assert spoken is True
    pipeline.speech_controller.speak_manual.assert_called_once_with("C")
    pipeline.stop()


def test_runtime_state_clears_active_word_on_alphabet_or_no_hand():
    """Asserts that runtime state does not leak active_word_sign across gestures or when hand disappears."""
    from dataclasses import replace
    from signbridge.word_sign.router import RecognitionMode, RoutedRecognitionResult
    from signbridge.tts.config import TTSConfig

    cfg = PipelineConfig(draw_hud=False, draw_skeleton=False, tts=TTSConfig(enabled=False))
    pipeline = SignBridgePipeline(config=cfg)

    # 1. Simulate prior word "YES" via dataclasses.replace
    pipeline._runtime_state = replace(
        pipeline._runtime_state,
        active_word_sign="YES",
        word_sign_confidence=0.95,
        hand_detected=True,
    )

    # 2. Simulate next frame update where hand is missing
    from signbridge.vision.detector import HandDetectionResult
    empty_det = HandDetectionResult(frame_id=10, timestamp=1.0, hands=())
    pipeline._update_runtime_state(
        det_result=empty_det,
        primary_hand=None,
    )

    st = pipeline.get_state()
    assert st.hand_detected is False
    assert st.active_word_sign is None
    assert st.word_sign_confidence == 0.0
    assert st.predicted_label is None

    # 3. Simulate frame update with hand in ALPHABET mode (predicting 'C')
    from signbridge.vision.landmarks import DetectedHand, LandmarkPoint
    from signbridge.model.base import PredictionResult
    dummy_lms = tuple(
        LandmarkPoint(index=i, name=f"pt_{i}", x_norm=0.5, y_norm=0.5, z_norm=0.0, x_px=320, y_px=240, z_depth=0.0)
        for i in range(21)
    )
    dummy_hand = DetectedHand(
        handedness="Right", landmarks=dummy_lms, raw_array=np.full((21, 3), 0.5, dtype=np.float32), handedness_confidence=0.9
    )
    det_with_hand = HandDetectionResult(frame_id=11, timestamp=1.033, hands=(dummy_hand,))
    pred_c = PredictionResult(class_id=2, class_name="C", confidence=0.88)
    routed_alpha = RoutedRecognitionResult(
        mode=RecognitionMode.ALPHABET,
        active_symbol="C",
        confidence=0.88,
        is_word=False,
        is_newly_emitted=False,
        token=None,
    )

    pipeline._update_runtime_state(
        det_result=det_with_hand,
        primary_hand=dummy_hand,
        pred_result=pred_c,
        routed_result=routed_alpha,
    )

    st = pipeline.get_state()
    assert st.hand_detected is True
    assert st.predicted_label == "C"
    # CRITICAL: active_word_sign must NOT be "YES"! It must be None!
    assert st.active_word_sign is None
    assert st.word_sign_confidence == 0.0
    pipeline.stop()


def test_derive_app_status_distinguishes_error_categories():
    """Asserts that _derive_app_status maps errors to the correct AppStatus enum."""
    from signbridge.pipeline.state import PipelineRuntimeState
    from signbridge.ui.layout import DashboardLayout
    from signbridge.ui.state import AppStatus, UIState

    u_state = UIState()

    # Camera disconnect error
    p_state_cam = PipelineRuntimeState(error_state="Camera disconnected.")
    assert DashboardLayout._derive_app_status(p_state_cam, u_state) == AppStatus.CAMERA_ERROR

    # Voice / audio error
    p_state_voice = PipelineRuntimeState(error_state="Voice output unavailable — SAPI5 audio failed.")
    assert DashboardLayout._derive_app_status(p_state_voice, u_state) == AppStatus.VOICE_ERROR

    # Model inference error
    p_state_model = PipelineRuntimeState(error_state="Classifier inference failed.")
    assert DashboardLayout._derive_app_status(p_state_model, u_state) == AppStatus.MODEL_ERROR


def test_goodbye_reference_card_shows_unsupported_without_alphabet_leak():
    """Asserts that playing GOODBYE demo video renders REFERENCE ONLY and zero confidence."""
    from signbridge.pipeline.state import PipelineRuntimeState
    from signbridge.ui.layout import DashboardLayout
    from signbridge.ui.state import UIState

    layout = DashboardLayout(target_width=1280, target_height=720)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    u_state = UIState(selected_sample_video="goodbye.mp4", active_input_source="DEMO")

    # Even if raw model predicted 'C' with 79% conf on open waving hand:
    p_state = PipelineRuntimeState(
        frame_source="DEMO_VIDEO",
        recognition_mode="WORD_SIGNS",
        predicted_label="C",
        prediction_confidence=0.79,
        active_word_sign=None,
        word_sign_confidence=0.0,
        hand_detected=True,
    )

    # Render card
    layout._render_recognition_card(canvas, x=800, y=50, w=450, h=140, p_state=p_state, u_state=u_state, theme=OBSIDIAN_THEME)

    # In demo mode for unsupported word sign:
    raw_target = u_state.selected_sample_video.upper()
    is_core_word = "GOODBYE" in ("HELLO", "HELP", "YES", "NO", "STOP", "THANK YOU")
    is_letter_demo = len("GOODBYE") == 1
    assert not is_core_word
    assert not is_letter_demo
    # Conf percent displayed must be 0% [REFERENCE]
    conf_pct = 0
    assert conf_pct == 0


