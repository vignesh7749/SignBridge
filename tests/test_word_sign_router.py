"""Tests for RecognitionRouter arbitrating between static alphabet and word-sign modes."""

from unittest.mock import MagicMock
import pytest

from signbridge.vision.detector import HandDetectionResult
from signbridge.word_sign.router import (
    RecognitionMode,
    RecognitionRouter,
    RoutedRecognitionResult,
)
from signbridge.word_sign.temporal import WordSignPrediction, WordSignTemporalSmoother


def test_recognition_mode_from_string():
    """Parses operational modes flexibly with case-insensitivity."""
    assert RecognitionMode.from_string("alphabet") == RecognitionMode.ALPHABET
    assert RecognitionMode.from_string("WORD_SIGNS") == RecognitionMode.WORD_SIGNS
    assert RecognitionMode.from_string("word signs") == RecognitionMode.WORD_SIGNS
    assert RecognitionMode.from_string("auto") == RecognitionMode.AUTO
    assert RecognitionMode.from_string("UNKNOWN") == RecognitionMode.AUTO


def test_router_alphabet_mode_exclusivity():
    """In ALPHABET mode, router forwards only alphabet tokens without running word smoother."""
    mock_smoother = MagicMock(spec=WordSignTemporalSmoother)
    router = RecognitionRouter(word_smoother=mock_smoother, initial_mode=RecognitionMode.ALPHABET)

    res = router.route_frame(
        detection=None,
        alphabet_token_candidate="A",
        alphabet_confidence=0.95,
        alphabet_is_emitted=True,
    )

    assert res.mode == RecognitionMode.ALPHABET
    assert res.token == "A"
    assert not res.is_word
    assert res.is_newly_emitted
    mock_smoother.process_frame.assert_not_called()


def test_router_word_signs_mode_exclusivity():
    """In WORD_SIGNS mode, router routes word sign and ignores alphabet fingerspelling."""
    mock_smoother = MagicMock(spec=WordSignTemporalSmoother)
    mock_smoother.process_frame.return_value = WordSignPrediction(
        word="HELLO",
        confidence=0.88,
        is_stable=True,
        is_newly_emitted=True,
        timestamp=1.0,
        top_k=[("HELLO", 0.88)],
    )

    router = RecognitionRouter(word_smoother=mock_smoother, initial_mode=RecognitionMode.WORD_SIGNS)

    res = router.route_frame(
        detection=None,
        alphabet_token_candidate="H",
        alphabet_confidence=0.99,
        alphabet_is_emitted=True,
        current_time=1.0,
    )

    assert res.mode == RecognitionMode.WORD_SIGNS
    assert res.token == "HELLO"
    assert res.is_word
    assert res.is_newly_emitted


def test_router_auto_mode_arbitration():
    """In AUTO mode, prefers high-confidence word sign, falling back to alphabet fingerspelling."""
    mock_smoother = MagicMock(spec=WordSignTemporalSmoother)
    router = RecognitionRouter(word_smoother=mock_smoother, initial_mode=RecognitionMode.AUTO)

    # 1. High confidence word sign active -> Favors word sign
    mock_smoother.process_frame.return_value = WordSignPrediction(
        word="STOP",
        confidence=0.85,
        is_stable=True,
        is_newly_emitted=True,
        timestamp=2.0,
        top_k=[("STOP", 0.85)],
    )
    res_word = router.route_frame(
        detection=None,
        alphabet_token_candidate="S",
        alphabet_confidence=0.90,
        alphabet_is_emitted=True,
        current_time=2.0,
    )
    assert res_word.token == "STOP"
    assert res_word.is_word

    # 2. No active word sign -> Falls back to alphabet fingerspelling
    mock_smoother.process_frame.return_value = WordSignPrediction(
        word="",
        confidence=0.10,
        is_stable=False,
        is_newly_emitted=False,
        timestamp=3.0,
        top_k=[],
    )
    res_alpha = router.route_frame(
        detection=None,
        alphabet_token_candidate="B",
        alphabet_confidence=0.94,
        alphabet_is_emitted=True,
        current_time=3.0,
    )
    assert res_alpha.token == "B"
    assert not res_alpha.is_word
    assert res_alpha.is_newly_emitted
