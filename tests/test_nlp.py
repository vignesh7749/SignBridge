"""Unit tests for NLP Modules 10, 11, 12 (Mapping, Word Builder, Sentence Builder)."""

from signbridge.nlp.mapping import LinguisticToken, SignTokenMapper, TokenType
from signbridge.nlp.sentence_builder import SentenceBuilder
from signbridge.nlp.word_builder import WordBuilder


def test_sign_token_mapping() -> None:
    token_a = SignTokenMapper.map_sign("a")
    assert token_a.token_type == TokenType.CHARACTER
    assert token_a.value == "A"

    token_space = SignTokenMapper.map_sign("space")
    assert token_space.token_type == TokenType.SPACE

    token_del = SignTokenMapper.map_sign("del")
    assert token_del.token_type == TokenType.BACKSPACE


def test_word_builder_dwell_time() -> None:
    builder = WordBuilder(dwell_seconds=0.4)
    char_h = LinguisticToken(TokenType.CHARACTER, "H")

    # Frame 1 at t=0.0 -> seen, not committed
    evt1 = builder.process_token(char_h, now=0.0)
    assert evt1.current_word == ""
    assert evt1.character_appended is None

    # Frame 2 at t=0.2 (under threshold) -> not committed
    evt2 = builder.process_token(char_h, now=0.2)
    assert evt2.current_word == ""

    # Frame 3 at t=0.45 (exceeds threshold) -> committed
    evt3 = builder.process_token(char_h, now=0.45)
    assert evt3.current_word == "H"
    assert evt3.character_appended == "H"

    # Frame 4 at t=0.6 (same held token) -> should NOT commit again (debounced)
    evt4 = builder.process_token(char_h, now=0.6)
    assert evt4.current_word == "H"
    assert evt4.character_appended is None


def test_sentence_builder() -> None:
    builder = SentenceBuilder()
    builder.add_word("hello")
    builder.add_word("world")
    assert builder.current_text == "Hello world"

    state = builder.add_punctuation("?")
    assert state.completed_sentence == "Hello world?"
    assert builder.current_text == ""
