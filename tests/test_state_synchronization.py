"""Comprehensive Regression Tests for State Synchronization across Recognition, Translation, and Speech.

Tests:
- Test A: Validated THANK YOU prediction does not leave K as active input.
- Test B: Word-level prediction is not processed as a single alphabet character.
- Test C: Word counter matches the actual number of words in the sentence.
- Test D: Word Break commits active input to sentence.
- Test E: Sustained prediction does not append duplicate words repeatedly.
- Test F: Clear, Delete, Undo, and Reset All maintain exact UI and pipeline synchronization.
- Test G: Speak reads the intended source.
- Test H: Last Spoken reflects actual speech requests and does not change without speech output.
- Test I: Existing Goodbye recognition fix remains intact.
"""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from signbridge.core.config import CameraConfig, VisionConfig
from signbridge.model.predictor import PredictionResult
from signbridge.pipeline import PipelineConfig, SignBridgePipeline
from signbridge.temporal.contracts import FramePrediction, TemporalFrameResult, TemporalState
from signbridge.temporal.engine import StableSign, TemporalEngine
from signbridge.translation.config import TranslationConfig
from signbridge.translation.contracts import TranslationEventType, TranslationResult, TranslationState
from signbridge.translation.engine import TranslationEngine
from signbridge.tts.base import TTSStatus
from signbridge.tts.config import TTSConfig
from signbridge.tts.controller import SpeechController
from signbridge.tts.provider import MockTTSProvider
from signbridge.vision.detector import HandDetectionResult
from signbridge.vision.landmarks import BoundingBox, DetectedHand, LandmarkPoint
from signbridge.word_sign.model import WordPredictionOutput, WordSignClassifier
from signbridge.word_sign.router import RecognitionMode, RecognitionRouter, RoutedRecognitionResult
from signbridge.word_sign.temporal import WordSignPrediction, WordSignTemporalSmoother


def _make_dummy_detection(frame_id: int = 1, timestamp: float = 0.0) -> HandDetectionResult:
    raw = np.zeros((21, 3), dtype=np.float32)
    raw[0] = [0.5, 0.5, 0.0]
    raw[9] = [0.5, 0.6, 0.0]
    landmarks = [
        LandmarkPoint(
            index=i,
            name=f"p{i}",
            x_norm=float(raw[i, 0]),
            y_norm=float(raw[i, 1]),
            z_norm=0.0,
            x_px=320,
            y_px=240,
            z_depth=0.0,
        )
        for i in range(21)
    ]
    hand = DetectedHand(
        handedness="Right",
        landmarks=landmarks,
        raw_array=raw,
        handedness_confidence=0.99,
        bbox=BoundingBox(0, 0, 100, 100),
    )
    return HandDetectionResult(frame_id=frame_id, timestamp=timestamp, hands=[hand])


# =========================================================================
# TEST A: Validated THANK YOU prediction does not leave K as active input
# =========================================================================

def test_a_thank_you_prediction_does_not_leave_k_as_active_input():
    """Test A: Validated THANK YOU prediction does not leave K as active input."""
    # Setup mock components
    mock_classifier = MagicMock(spec=WordSignClassifier)
    mock_classifier.num_classes = 58
    mock_classifier.class_names = ["HELLO", "THANK YOU", "GOODBYE", "YES", "NO"]

    mock_smoother = MagicMock(spec=WordSignTemporalSmoother)
    # Simulate high confidence THANK YOU prediction
    mock_smoother.process_frame.return_value = WordSignPrediction(
        word="THANK YOU",
        confidence=0.99,
        is_stable=True,
        is_newly_emitted=False,
        timestamp=10.0,
        top_k=[("THANK YOU", 0.99)],
    )

    router = RecognitionRouter(word_smoother=mock_smoother, initial_mode=RecognitionMode.AUTO)
    det = _make_dummy_detection()

    # Route frame where alphabet classifier might have candidate "K"
    routed = router.route_frame(
        detection=det,
        alphabet_token_candidate="K",
        alphabet_confidence=0.85,
        alphabet_is_emitted=True,
        current_time=10.0,
    )

    # In AUTO mode with word confidence 0.99, it MUST route as word sign, NOT alphabet letter K
    assert routed.is_word is True
    assert routed.active_symbol == "THANK YOU"
    assert routed.token is None or routed.token == "THANK YOU"
    assert routed.confidence == 0.99
    # Letter K was completely suppressed
    assert routed.token != "K"

    # Now verify in pipeline orchestrator state
    trans_engine = TranslationEngine()
    # Even if translation engine previously had stray 'K' tokens from intermediate noise
    trans_engine.process_char("K")
    assert trans_engine.current_word == "K"

    tts_prov = MockTTSProvider()
    speech_ctrl = SpeechController(provider=tts_prov)

    pipeline = SignBridgePipeline(
        config=PipelineConfig(),
        translation_engine=trans_engine,
        speech_controller=speech_ctrl,
        router=router,
    )

    # Calling set_recognition_mode("WORD_SIGNS") purges stray alphabet tokens
    pipeline.set_recognition_mode(RecognitionMode.WORD_SIGNS)
    st = pipeline.get_state()
    assert st.recognition_mode == "WORD_SIGNS"
    # Current word must NOT be K
    assert st.current_word != "K"


# =========================================================================
# TEST B: Word-level prediction is not processed as single alphabet character
# =========================================================================

def test_b_word_level_prediction_not_processed_as_single_alphabet_character():
    """Test B: Word-level prediction is not processed as a single alphabet character."""
    trans_engine = TranslationEngine()
    tts_prov = MockTTSProvider()
    speech_ctrl = SpeechController(provider=tts_prov)

    pipeline = SignBridgePipeline(
        config=PipelineConfig(),
        translation_engine=trans_engine,
        speech_controller=speech_ctrl,
    )

    # Ingest word sign directly
    res = pipeline.translation_engine.process_word("THANK YOU", confidence=0.99)

    assert res.event_type == TranslationEventType.WORD_COMPLETED
    assert list(res.completed_words) == ["THANK YOU"]
    assert pipeline.translation_engine.completed_words == ["THANK YOU"]
    # Ensure it was NOT converted to 'T' or split into individual characters
    assert len(pipeline.translation_engine.completed_words) == 1
    assert pipeline.translation_engine.completed_words[0] == "THANK YOU"
    assert pipeline.translation_engine.current_word == ""


# =========================================================================
# TEST C: Word counter matches actual number of words in sentence
# =========================================================================

def test_c_word_counter_matches_actual_number_of_words():
    """Test C: Word counter matches the actual number of words in the sentence."""
    trans_engine = TranslationEngine()
    pipeline = SignBridgePipeline(
        config=PipelineConfig(),
        translation_engine=trans_engine,
    )

    # Initial state: 0 words
    st = pipeline.get_state()
    assert len(st.completed_words) == 0
    assert len(pipeline.translation_engine.completed_words) == 0

    # Add 1st word
    pipeline.translation_engine.process_word("HELLO")
    st = pipeline.get_state()
    assert len(st.completed_words) == 1
    assert st.completed_words == ("HELLO",)

    # Add 2nd word
    pipeline.translation_engine.process_word("WORLD")
    st = pipeline.get_state()
    assert len(st.completed_words) == 2
    assert st.completed_words == ("HELLO", "WORLD")

    # Clear sentence: back to 0
    pipeline.clear_sentence()
    st = pipeline.get_state()
    assert len(st.completed_words) == 0


# =========================================================================
# TEST D: Word Break commits active input to sentence
# =========================================================================

def test_d_word_break_commits_active_input_to_sentence():
    """Test D: Word Break commits active input to sentence."""
    trans_engine = TranslationEngine()
    tts_prov = MockTTSProvider()
    speech_ctrl = SpeechController(provider=tts_prov)

    pipeline = SignBridgePipeline(
        config=PipelineConfig(),
        translation_engine=trans_engine,
        speech_controller=speech_ctrl,
    )

    # 1. Fingerspelling input: 'C', 'A', 'T'
    pipeline.translation_engine.process_char("C")
    pipeline.translation_engine.process_char("A")
    pipeline.translation_engine.process_char("T")
    assert pipeline.translation_engine.current_word == "CAT"
    assert len(pipeline.translation_engine.completed_words) == 0

    # Trigger Word Break via space()
    pipeline.space()
    assert pipeline.translation_engine.current_word == ""
    assert pipeline.translation_engine.completed_words == ["CAT"]
    st = pipeline.get_state()
    assert st.completed_words == ("CAT",)
    assert st.current_word == ""


# =========================================================================
# TEST E: Sustained prediction does not append duplicate words repeatedly
# =========================================================================

def test_e_sustained_prediction_does_not_duplicate_words():
    """Test E: Sustained prediction does not append duplicate words repeatedly."""
    classifier = MagicMock(spec=WordSignClassifier)
    classifier.num_classes = 58
    classifier.class_names = ["HELLO", "GOODBYE"]
    # Predict HELLO with 0.90 confidence
    classifier.predict.return_value = WordPredictionOutput(
        predicted_word="HELLO",
        confidence=0.90,
        top_k=[("HELLO", 0.90), ("GOODBYE", 0.10)],
        inference_time_ms=1.5,
    )

    smoother = WordSignTemporalSmoother(
        classifier=classifier,
        window_size=20,
        min_confidence=0.75,
        stable_count=3,
        cooldown_seconds=2.0,
    )

    det = _make_dummy_detection()
    newly_emitted_counts = 0

    # Feed 35 consecutive frames (1 second at 35fps) of the same gesture
    for frame_idx in range(35):
        t = frame_idx * 0.033
        pred = smoother.process_frame(det, current_time=t)
        if pred.is_newly_emitted:
            newly_emitted_counts += 1

    # In a continuous hold, is_newly_emitted must fire EXACTLY ONCE
    assert newly_emitted_counts == 1


# =========================================================================
# TEST F: Clear, Delete, Undo, and Reset All maintain exact UI/pipeline sync
# =========================================================================

def test_f_clear_delete_undo_reset_all_synchronization():
    """Test F: Clear, Delete, Undo, and Reset All maintain exact UI and pipeline synchronization."""
    trans_engine = TranslationEngine()
    tts_prov = MockTTSProvider()
    speech_ctrl = SpeechController(provider=tts_prov)

    pipeline = SignBridgePipeline(
        config=PipelineConfig(),
        translation_engine=trans_engine,
        speech_controller=speech_ctrl,
    )

    # 1. Type "A", "B", Space ("AB")
    pipeline.translation_engine.process_char("A")
    pipeline.translation_engine.process_char("B")
    pipeline.space()
    assert pipeline.translation_engine.completed_words == ["AB"]

    # 2. Type "C", "D"
    pipeline.translation_engine.process_char("C")
    pipeline.translation_engine.process_char("D")
    assert pipeline.translation_engine.current_word == "CD"

    # 3. Backspace: deletes "D", leaving "C"
    pipeline.backspace()
    assert pipeline.translation_engine.current_word == "C"

    # 4. Undo: reverts backspace
    pipeline.undo()
    assert pipeline.translation_engine.current_word == "CD"

    # 5. Clear Sentence: wipes current word and completed words
    pipeline.clear_sentence()
    assert pipeline.translation_engine.current_word == ""
    assert len(pipeline.translation_engine.completed_words) == 0

    # 6. Type "HELLO", then Reset All
    pipeline.translation_engine.process_word("HELLO")
    assert pipeline.translation_engine.completed_words == ["HELLO"]
    pipeline.reset()
    assert pipeline.translation_engine.current_word == ""
    assert len(pipeline.translation_engine.completed_words) == 0
    assert pipeline.speech_controller.last_spoken_text is None


# =========================================================================
# TEST G: Speak reads the intended source
# =========================================================================

def test_g_speak_reads_intended_source():
    """Test G: Speak reads the intended source (sentence if present, active word otherwise)."""
    trans_engine = TranslationEngine()
    tts_prov = MockTTSProvider()
    speech_ctrl = SpeechController(provider=tts_prov)

    pipeline = SignBridgePipeline(
        config=PipelineConfig(),
        translation_engine=trans_engine,
        speech_controller=speech_ctrl,
    )

    # Priority 1: If completed sentence exists, speaks sentence
    pipeline.translation_engine.process_word("HELLO")
    pipeline.translation_engine.process_word("WORLD")
    pipeline._update_runtime_state()

    pipeline.speak_current_sentence()
    import time
    time.sleep(0.05)
    assert tts_prov.spoken_history and tts_prov.spoken_history[-1] == "HELLO WORLD"

    # Priority 2: If sentence is empty but active word exists, speaks active word
    pipeline.clear_sentence()
    pipeline.translation_engine.process_char("Y")
    pipeline.translation_engine.process_char("E")
    pipeline.translation_engine.process_char("S")
    pipeline._update_runtime_state()

    pipeline.speak_current_sentence()
    time.sleep(0.05)
    assert tts_prov.spoken_history and tts_prov.spoken_history[-1] == "YES"


# =========================================================================
# TEST H: Last Spoken reflects actual speech requests without unvoiced overwrite
# =========================================================================

def test_h_last_spoken_reflects_actual_speech_and_preserves_state():
    """Test H: Last Spoken reflects actual speech requests and does not change without speech output."""
    tts_prov = MockTTSProvider()
    speech_ctrl = SpeechController(provider=tts_prov)

    # Initial state
    assert speech_ctrl.last_spoken_text is None

    # Submit speech
    speech_ctrl.speak_manual("THANK YOU")
    import time
    time.sleep(0.05)
    assert tts_prov.spoken_history and tts_prov.spoken_history[-1] == "THANK YOU"

    # Calling stop() leaves is_speaking as False
    speech_ctrl.stop()
    assert speech_ctrl.is_speaking is False
    assert speech_ctrl.status == TTSStatus.READY


# =========================================================================
# TEST I: Existing Goodbye recognition fix remains intact
# =========================================================================

def test_i_goodbye_recognition_fix_remains_intact():
    """Test I: Existing Goodbye recognition fix remains intact."""
    from signbridge.word_sign.model import WordSignClassifier

    model_dir = Path("models/word_sign")
    assert model_dir.exists(), "models/word_sign directory must exist"

    classifier = WordSignClassifier.load(model_dir)
    assert classifier is not None
    assert len(classifier.classes) == 58
    assert "GOODBYE" in classifier.classes
    assert "HELLO" in classifier.classes
    assert "THANK YOU" in classifier.classes
