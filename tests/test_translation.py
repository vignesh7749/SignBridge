"""Unit and property tests for Stage 8: Translation Engine.

Tests:
- StableSignToken contracts and factory methods
- Word and Sentence immutability, serialization, and confidence propagation
- Token ingestion and letter buffering
- Word boundaries and space idempotency (no duplicate spaces)
- Backspace on active word, word re-opening, and empty buffer safety
- Clear word and clear sentence
- State machine transitions (EMPTY, BUILDING_WORD, WORD_COMPLETED, BUILDING_SENTENCE)
- Multi-step undo and undo stack bounds
- Full state reset
- Memory bounding under heavy token streams
- Compatibility with Stage 7 StableSign
- Invariants:
  - Sentence text never contains multiple consecutive spaces
  - Reset always produces EMPTY state
  - Deterministic replay produces identical sentences
"""

import math
import pytest

from signbridge.temporal.contracts import StableSign
from signbridge.translation import (
    Sentence,
    StableSignToken,
    TranslationCommand,
    TranslationConfig,
    TranslationEngine,
    TranslationEventType,
    TranslationResult,
    TranslationState,
    Word,
    format_sentence_text,
    normalize_whitespace,
)


# ==============================================================================
# Contract & Validation Tests
# ==============================================================================

def test_stable_sign_token_contract() -> None:
    """Validates StableSignToken attributes, factory methods, and immutability."""
    t = StableSignToken(label="A", class_id=0, confidence=0.95, timestamp=1.5, duration=0.25)
    assert t.label == "A"
    assert t.class_id == 0
    assert t.confidence == 0.95
    assert t.timestamp == 1.5
    assert t.duration == 0.25

    # Factory from char
    t_char = StableSignToken.from_char("b", timestamp=2.0, confidence=0.88, class_id=1)
    assert t_char.label == "B"
    assert t_char.class_id == 1
    assert t_char.confidence == 0.88

    # Serialization
    d = t.to_dict()
    assert d["label"] == "A"
    assert d["confidence"] == 0.95


def test_stable_sign_token_validation() -> None:
    """Ensures token rejects invalid timestamps, non-finite confidences, and empty labels."""
    with pytest.raises(ValueError, match="Timestamp must be non-negative"):
        StableSignToken(label="A", class_id=0, confidence=0.9, timestamp=-1.0)

    with pytest.raises(ValueError, match="Confidence must be finite"):
        StableSignToken(label="A", class_id=0, confidence=float("nan"), timestamp=1.0)

    with pytest.raises(ValueError, match="Confidence must be in"):
        StableSignToken(label="A", class_id=0, confidence=1.5, timestamp=1.0)

    with pytest.raises(ValueError, match="label cannot be empty"):
        StableSignToken(label="", class_id=0, confidence=0.9, timestamp=1.0)

    with pytest.raises(ValueError, match="label cannot be empty"):
        StableSignToken.from_char("   ")


def test_word_and_sentence_immutability_and_confidence() -> None:
    """Validates Word and Sentence creation, confidence propagation, and immutability."""
    t1 = StableSignToken.from_char("H", timestamp=1.0, confidence=0.90)
    t2 = StableSignToken.from_char("I", timestamp=1.2, confidence=0.80)

    word = Word.from_tokens([t1, t2])
    assert word.text == "HI"
    assert word.confidence == pytest.approx(0.85, rel=1e-3)
    assert word.start_timestamp == 1.0
    assert word.end_timestamp == 1.2
    assert len(word.tokens) == 2

    # Cannot create empty word
    with pytest.raises(ValueError, match="Cannot construct a Word from an empty sequence"):
        Word.from_tokens([])

    # Sentence
    sentence = Sentence.from_words([word])
    assert sentence.text == "HI"
    assert len(sentence.words) == 1
    assert sentence.start_timestamp == 1.0
    assert sentence.end_timestamp == 1.2

    empty_sentence = Sentence.from_words([])
    assert empty_sentence.text == ""
    assert len(empty_sentence.words) == 0


def test_stage7_stablesign_conversion() -> None:
    """Verifies that Stage 7 StableSign converts seamlessly into StableSignToken."""
    stage7_sign = StableSign(
        class_id=0,
        label="A",
        confidence=0.932,
        start_timestamp=10.0,
        end_timestamp=10.27,
        duration=0.27,
        frame_count=8,
        top_margin=0.85,
    )
    token = StableSignToken.from_stable_sign(stage7_sign)
    assert token.label == "A"
    assert token.class_id == 0
    assert token.confidence == 0.932
    assert token.timestamp == 10.27
    assert token.duration == 0.27


# ==============================================================================
# Translation Engine Functional Tests
# ==============================================================================

def test_word_building_and_order() -> None:
    """Verifies that tokens accumulate into words in strict sequential order."""
    engine = TranslationEngine()
    assert engine.current_state == TranslationState.EMPTY

    r1 = engine.process_char("C")
    assert r1.state == TranslationState.BUILDING_WORD
    assert r1.current_word == "C"

    r2 = engine.process_char("A")
    assert r2.current_word == "CA"

    r3 = engine.process_char("T")
    assert r3.current_word == "CAT"
    assert engine.sentence_text == "CAT"
    assert engine.tokens_count == 3


def test_word_boundary_and_spaces() -> None:
    """Verifies space event commits active word and prevents duplicate spaces."""
    engine = TranslationEngine()
    for c in "DOG":
        engine.process_char(c)

    res_sp = engine.space()
    assert res_sp.state == TranslationState.WORD_COMPLETED
    assert res_sp.current_word == ""
    assert engine.completed_words == ["DOG"]
    assert engine.sentence_text == "DOG"

    # Consecutive spaces should be safely ignored (idempotent)
    res_sp2 = engine.space()
    assert res_sp2.event_type == TranslationEventType.IGNORED
    assert engine.completed_words == ["DOG"]
    assert engine.sentence_text == "DOG"

    # Second word
    for c in "RUNS":
        engine.process_char(c)
    assert engine.current_state == TranslationState.BUILDING_SENTENCE
    assert engine.sentence_text == "DOG RUNS"

    engine.space()
    assert engine.completed_words == ["DOG", "RUNS"]
    assert engine.sentence_text == "DOG RUNS"


def test_backspace_on_current_word() -> None:
    """Backspace deletes the last letter in active word."""
    engine = TranslationEngine()
    for c in "HELP":
        engine.process_char(c)
    assert engine.current_word == "HELP"

    res = engine.backspace()
    assert res.current_word == "HEL"
    assert res.event_type == TranslationEventType.BACKSPACE

    engine.backspace()
    assert engine.current_word == "HE"


def test_backspace_reopens_completed_word() -> None:
    """Backspace on empty current_word re-opens the preceding completed word for editing."""
    engine = TranslationEngine()
    for c in "HI":
        engine.process_char(c)
    engine.space()
    assert engine.completed_words == ["HI"]
    assert engine.current_word == ""

    # Backspace re-opens "HI"
    res = engine.backspace()
    assert engine.completed_words == []
    assert engine.current_word == "HI"
    assert res.state == TranslationState.BUILDING_WORD

    # Subsequent backspace deletes 'I'
    engine.backspace()
    assert engine.current_word == "H"


def test_empty_backspace_safety() -> None:
    """Backspace on a completely empty engine is safe and does not raise an exception."""
    engine = TranslationEngine()
    assert engine.current_state == TranslationState.EMPTY
    res = engine.backspace()
    assert res.state == TranslationState.EMPTY
    assert res.current_word == ""
    assert res.event_type == TranslationEventType.IGNORED


def test_clear_word() -> None:
    """Clear word clears the active word without disturbing completed words."""
    engine = TranslationEngine()
    for c in "YES":
        engine.process_char(c)
    engine.space()

    for c in "TYPO":
        engine.process_char(c)
    assert engine.current_word == "TYPO"
    assert engine.completed_words == ["YES"]

    res = engine.clear_word()
    assert res.current_word == ""
    assert engine.completed_words == ["YES"]
    assert engine.sentence_text == "YES"
    assert res.state == TranslationState.WORD_COMPLETED


def test_clear_sentence() -> None:
    """Clear sentence wipes both completed words and active word, setting state to EMPTY."""
    engine = TranslationEngine()
    for c in "HELLO":
        engine.process_char(c)
    engine.space()
    for c in "WORLD":
        engine.process_char(c)

    res = engine.clear_sentence()
    assert res.state == TranslationState.EMPTY
    assert engine.sentence_text == ""
    assert engine.completed_words == []
    assert engine.current_word == ""


def test_reset() -> None:
    """Reset clears words, active word, token history, and undo stack."""
    engine = TranslationEngine()
    for c in "TEST":
        engine.process_char(c)
    engine.space()

    res = engine.reset()
    assert res.state == TranslationState.EMPTY
    assert engine.sentence_text == ""
    assert len(engine.token_history) == 0
    assert engine.tokens_count == 0


def test_multi_step_undo() -> None:
    """Undo accurately unwinds actions in reverse chronological order."""
    engine = TranslationEngine()
    engine.process_char("A")
    engine.process_char("B")
    engine.process_char("C")
    assert engine.current_word == "ABC"

    # Undo 1 -> "AB"
    u1 = engine.undo()
    assert u1.current_word == "AB"

    # Undo 2 -> "A"
    u2 = engine.undo()
    assert u2.current_word == "A"

    # Undo 3 -> ""
    u3 = engine.undo()
    assert u3.current_word == ""
    assert u3.state == TranslationState.EMPTY

    # Undo when stack is empty -> IGNORED
    u4 = engine.undo()
    assert u4.event_type == TranslationEventType.IGNORED


def test_undo_across_word_boundaries() -> None:
    """Undo restores words across space boundaries."""
    engine = TranslationEngine()
    for c in "HI":
        engine.process_char(c)
    engine.space()
    assert engine.completed_words == ["HI"]
    assert engine.current_word == ""

    # Undo space -> re-opens "HI" in current_word
    res = engine.undo()
    assert engine.completed_words == []
    assert engine.current_word == "HI"
    assert res.state == TranslationState.BUILDING_WORD


def test_stage7_stablesign_ingestion() -> None:
    """Verifies that Stage 7 StableSign objects can be passed directly to process_token()."""
    engine = TranslationEngine()
    sign_a = StableSign(0, "A", 0.95, 1.0, 1.25, 0.25, 8)
    sign_b = StableSign(1, "B", 0.92, 1.5, 1.75, 0.25, 8)

    res1 = engine.process_token(sign_a)
    assert res1.current_word == "A"

    res2 = engine.process_token(sign_b)
    assert res2.current_word == "AB"


def test_special_command_token_strings() -> None:
    """Special tokens like 'SPACE', 'BACKSPACE', 'CLEAR' trigger appropriate control actions."""
    engine = TranslationEngine()
    engine.process_char("O")
    engine.process_char("K")

    # Ingest "SPACE" token
    engine.process_token(StableSignToken.from_char("SPACE"))
    assert engine.completed_words == ["OK"]
    assert engine.current_word == ""

    # Ingest "DEL" token (backspace)
    engine.process_token(StableSignToken.from_char("DEL"))
    assert engine.completed_words == []
    assert engine.current_word == "OK"


def test_bounded_memory() -> None:
    """Verifies that token history and undo stack respect max configured bounds."""
    config = TranslationConfig(max_token_history=10, max_undo_depth=5)
    engine = TranslationEngine(config)

    for i in range(50):
        engine.process_char(chr(ord("A") + (i % 26)))

    assert len(engine.token_history) <= 10
    # Undo stack should not exceed 5
    assert len(engine._undo_stack) <= 5


def test_auto_word_boundary_pause() -> None:
    """When auto_word_boundary is True, pause exceeding threshold auto-commits word."""
    config = TranslationConfig(auto_word_boundary=True, word_boundary_pause_seconds=1.0)
    engine = TranslationEngine(config)

    engine.process_token(StableSignToken("H", 0, 0.9, timestamp=1.0))
    engine.process_token(StableSignToken("I", 1, 0.9, timestamp=1.2))
    assert engine.current_word == "HI"

    # Token arrives after 2.0s pause (> 1.0s threshold)
    engine.process_token(StableSignToken("A", 2, 0.9, timestamp=3.5))
    assert engine.completed_words == ["HI"]
    assert engine.current_word == "A"
    assert engine.sentence_text == "HI A"


# ==============================================================================
# Invariant & Property Tests
# ==============================================================================

def test_invariant_no_accidental_double_spaces() -> None:
    """Asserts that sentence text never contains consecutive double spaces under any operation."""
    engine = TranslationEngine()
    for _ in range(5):
        engine.space()

    for word in ["ALPHA", "BETA", "GAMMA"]:
        for c in word:
            engine.process_char(c)
        engine.space()
        engine.space()

    text = engine.sentence_text
    assert "  " not in text
    assert text == "ALPHA BETA GAMMA"


def test_invariant_reset_always_empty() -> None:
    """RESET must always transition the engine to EMPTY state with blank sentence."""
    engine = TranslationEngine()
    for c in "COMPLEX SENTENCE":
        if c == " ":
            engine.space()
        else:
            engine.process_char(c)

    engine.reset()
    assert engine.current_state == TranslationState.EMPTY
    assert engine.current_word == ""
    assert engine.sentence_text == ""
    assert engine.completed_words == []
    assert len(engine.token_history) == 0


def test_invariant_deterministic_replay() -> None:
    """Identical input sequence processed on separate engines produces identical sentences."""
    sequence = ["H", "E", "L", "L", "O", "SPACE", "W", "O", "R", "L", "D", "SPACE", "A", "BACKSPACE"]

    def run_engine() -> str:
        e = TranslationEngine()
        for item in sequence:
            if item == "SPACE":
                e.space()
            elif item == "BACKSPACE":
                e.backspace()
            else:
                e.process_char(item)
        return e.sentence_text

    out1 = run_engine()
    out2 = run_engine()
    assert out1 == out2 == "HELLO WORLD"
