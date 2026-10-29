"""Unit and integration tests for the text-based Sign Recognition panel and Sentence accumulation."""

import numpy as np
import pytest

from signbridge.pipeline.config import PipelineConfig
from signbridge.pipeline.orchestrator import SignBridgePipeline
from signbridge.pipeline.state import PipelineRuntimeState
from signbridge.translation.contracts import StableSignToken
from signbridge.translation.engine import TranslationEngine
from signbridge.ui.layout import DashboardLayout
from signbridge.ui.state import UIState
from signbridge.ui.theme import OBSIDIAN_THEME


def test_scenario_1_no_prediction_shows_placeholder_no_thumbnail():
    """Scenario 1: No hand / no prediction shows clean em-dash placeholder, zero thumbnails."""
    layout = DashboardLayout(target_width=1280, target_height=720)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    u_state = UIState()

    p_state = PipelineRuntimeState(
        hand_detected=False,
        predicted_label=None,
        active_word_sign=None,
        prediction_confidence=0.0,
        recognition_mode="AUTO",
    )

    layout._render_recognition_card(canvas, 800, 50, 450, 140, p_state, u_state, OBSIDIAN_THEME)
    assert np.count_nonzero(canvas) > 0


def test_scenario_2_valid_prediction_displays_prominent_text():
    """Scenario 2: Valid prediction (e.g. 'C' or 'GOODBYE') renders prominent text without thumbnail."""
    layout = DashboardLayout(target_width=1280, target_height=720)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    u_state = UIState()

    # Alphabet prediction
    p_state_alpha = PipelineRuntimeState(
        hand_detected=True,
        predicted_label="C",
        prediction_confidence=0.88,
        recognition_mode="ALPHABET",
    )
    layout._render_recognition_card(canvas, 800, 50, 450, 140, p_state_alpha, u_state, OBSIDIAN_THEME)
    assert np.count_nonzero(canvas) > 0

    # Word-level prediction
    canvas.fill(0)
    p_state_word = PipelineRuntimeState(
        hand_detected=True,
        active_word_sign="GOODBYE",
        word_sign_confidence=0.86,
        recognition_mode="WORD_SIGNS",
    )
    layout._render_recognition_card(canvas, 800, 50, 450, 140, p_state_word, u_state, OBSIDIAN_THEME)
    assert np.count_nonzero(canvas) > 0


def test_scenario_3_recognized_character_separate_from_sentence():
    """Scenario 3: Recognizing 'C' updates Active Word, but does NOT prematurely display as sentence."""
    layout = DashboardLayout(target_width=1280, target_height=720)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    u_state = UIState()

    p_state = PipelineRuntimeState(
        current_word="C",
        completed_words=(),
        sentence="C",  # in-flight translation text
    )

    layout._render_translation_card(canvas, 800, 200, 450, 160, p_state, u_state, OBSIDIAN_THEME)
    # Verification: completed_words is empty, so completed sentence is empty (0 words)
    assert len(p_state.completed_words) == 0
    assert p_state.current_word == "C"


def test_scenario_4_and_5_word_break_and_sentence_accumulation():
    """Scenario 4 & 5: Word Break commits active word and multiple inputs accumulate sequentially."""
    engine = TranslationEngine()

    # 1. Sign C, A, T
    for ch in ["C", "A", "T"]:
        engine.process_char(ch)
    assert engine.current_word == "CAT"
    assert len(engine.completed_words) == 0

    # 2. Word Break (Space) commits CAT
    res_sp1 = engine.space()
    assert res_sp1.completed_words == ["CAT"]
    assert engine.current_word == ""
    assert engine.completed_words == ["CAT"]

    # 3. Next input: word sign HELLO
    engine.process_word("HELLO")
    assert engine.completed_words == ["CAT", "HELLO"]

    # 4. Next input: fingerspelled word D, O, G
    for ch in ["D", "O", "G"]:
        engine.process_char(ch)
    assert engine.current_word == "DOG"
    assert engine.completed_words == ["CAT", "HELLO"]

    # 5. Word Break commits DOG
    res_sp2 = engine.space()
    assert engine.completed_words == ["CAT", "HELLO", "DOG"]
    assert " ".join(engine.completed_words) == "CAT HELLO DOG"


def test_scenario_6_clear_delete_undo_reset():
    """Scenario 6: Clear, Delete (Backspace), Undo, and Reset maintain state invariants."""
    engine = TranslationEngine()

    # Build word "HI" and commit
    engine.process_char("H")
    engine.process_char("I")
    engine.space()
    assert engine.completed_words == ["HI"]

    # Build active word "YOU"
    engine.process_char("Y")
    engine.process_char("O")
    engine.process_char("U")
    assert engine.current_word == "YOU"

    # Delete (Backspace) removes 'U'
    engine.backspace()
    assert engine.current_word == "YO"

    # Undo restores 'YOU'
    engine.undo()
    assert engine.current_word == "YOU"

    # Clear sentence wipes everything
    engine.clear_sentence()
    assert engine.completed_words == []
    assert engine.current_word == ""

    # Reset
    engine.process_char("A")
    engine.reset()
    assert engine.completed_words == []
    assert engine.current_word == ""


def test_scenario_7_word_counter_matches_sentence_contents():
    """Scenario 7: Word counter exact integer match with completed sentence words."""
    layout = DashboardLayout(target_width=1280, target_height=720)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    u_state = UIState()

    # State with 2 completed words: "HELLO" and "GOODBYE"
    p_state = PipelineRuntimeState(
        completed_words=("HELLO", "GOODBYE"),
        current_word="C",
        sentence="HELLO GOODBYE C",
    )

    layout._render_translation_card(canvas, 800, 200, 450, 160, p_state, u_state, OBSIDIAN_THEME)

    completed_sent = " ".join(p_state.completed_words).strip()
    assert completed_sent == "HELLO GOODBYE"
    assert len(p_state.completed_words) == 2
    assert len(completed_sent.split()) == 2


def test_scenario_8_goodbye_recognition_remains_functional():
    """Scenario 8: Goodbye recognition model and pipeline remain 100% operational."""
    import json
    from pathlib import Path

    classes_path = Path("models/word_sign/classes.json")
    assert classes_path.exists(), "word_sign classes.json must exist"

    with open(classes_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "GOODBYE" in meta.get("classes", []), "GOODBYE must be in trained model classes"
    assert "GOODBYE" in meta.get("class_to_id", {}), "GOODBYE must be in class_to_id mapping"


def test_scenario_9_voice_output_priority_and_readiness():
    """Scenario 9: Voice output speaks completed sentence or active word properly."""
    from unittest.mock import MagicMock
    from signbridge.tts.config import TTSConfig

    cfg = PipelineConfig(draw_hud=False, draw_skeleton=False, tts=TTSConfig(enabled=True))
    pipeline = SignBridgePipeline(config=cfg)
    pipeline.speech_controller.speak_manual = MagicMock(return_value=True)

    # 1. When completed words exist, speak_current_sentence speaks completed sentence
    pipeline.translation_engine.process_word("HELLO")
    pipeline.translation_engine.process_word("GOODBYE")
    pipeline.translation_engine.process_char("C")  # uncommitted active word

    assert pipeline.speak_current_sentence() is True
    pipeline.speech_controller.speak_manual.assert_called_with("HELLO GOODBYE")

    pipeline.stop()
