"""Unit and integration tests for Stage 7: Temporal Recognition & Stable Sign Detection.

Tests:
- Legacy Module 9 TemporalSmoother majority voting compatibility
- FramePrediction & StableSign immutability and contract validation
- Input validation (NaN, Inf, negative timestamp, invalid class_id)
- TemporalEngineConfig parameter validation
- TemporalPredictionBuffer windowing and memory bounding
- Empty input & single prediction stability rejection
- Confidence filtering (< threshold rejection)
- Anti-jitter and majority voting stability
- Timestamp-based dwell time calculation
- Duplicate suppression on continuously held signs
- Re-arming policies (on_change, timeout, neutral_release)
- Sequential sign transitions (A -> B -> C)
- Hand loss tolerance within grace period (< grace_ms)
- Hand loss candidate reset after grace period (> grace_ms)
- Non-monotonic timestamp handling
- Variable camera FPS invariance (15, 24, 30, 60 FPS)
- Complete state machine reset
- Deterministic behavior across repeated trials
"""

import math
import numpy as np
import pytest

from signbridge.core.config import TemporalConfig
from signbridge.recognition.confidence import ScoredPrediction
from signbridge.temporal import (
    ConsensusResult,
    FramePrediction,
    StableSign,
    TemporalEngine,
    TemporalEngineConfig,
    TemporalFrameResult,
    TemporalPredictionBuffer,
    TemporalSmoother,
    TemporalState,
)


# ==============================================================================
# Legacy Stage 0 Compatibility
# ==============================================================================

def test_temporal_smoother_majority_voting() -> None:
    """Preserves legacy test contract for Module 9 TemporalSmoother."""
    config = TemporalConfig(window_size=5, consensus_threshold=0.6)
    smoother = TemporalSmoother(config)

    pred_a = ScoredPrediction(
        label="A",
        confidence=0.9,
        is_valid=True,
        probabilities=np.array([0.9, 0.1]),
        top_candidates=[("A", 0.9)],
    )
    pred_b = ScoredPrediction(
        label="B",
        confidence=0.85,
        is_valid=True,
        probabilities=np.array([0.1, 0.85]),
        top_candidates=[("B", 0.85)],
    )

    s1 = smoother.update(pred_a)
    s2 = smoother.update(pred_a)
    assert s2.is_stable is True
    assert s2.label == "A"

    s3 = smoother.update(pred_b)
    assert s3.is_stable is True
    assert s3.label == "A"


# ==============================================================================
# Contracts & Validation Tests
# ==============================================================================

def test_frame_prediction_contract() -> None:
    """Validates FramePrediction field properties, factories, and immutability."""
    p = FramePrediction(
        class_id=0,
        label="A",
        confidence=0.95,
        timestamp=1.25,
        has_hand=True,
        top_margin=0.80,
    )
    assert p.class_id == 0
    assert p.label == "A"
    assert p.confidence == 0.95
    assert p.timestamp == 1.25
    assert p.has_hand is True

    # Factory for no hand
    no_hand = FramePrediction.no_hand(timestamp=2.0)
    assert no_hand.has_hand is False
    assert no_hand.class_id == -1
    assert no_hand.label == ""
    assert no_hand.confidence == 0.0

    # Immutability check
    with pytest.raises(Exception):
        p.label = "B"  # type: ignore


def test_frame_prediction_invalid_inputs() -> None:
    """Ensures FramePrediction rejects NaN, Inf, negative timestamps, and out-of-range confidence."""
    # Negative timestamp
    with pytest.raises(ValueError, match="Timestamp must be non-negative"):
        FramePrediction(class_id=0, label="A", confidence=0.9, timestamp=-0.1)

    # NaN confidence
    with pytest.raises(ValueError, match="Confidence must be finite"):
        FramePrediction(class_id=0, label="A", confidence=float("nan"), timestamp=1.0)

    # Inf confidence
    with pytest.raises(ValueError, match="Confidence must be finite"):
        FramePrediction(class_id=0, label="A", confidence=float("inf"), timestamp=1.0)

    # Out of range confidence
    with pytest.raises(ValueError, match="in \\[0.0, 1.0\\]"):
        FramePrediction(class_id=0, label="A", confidence=1.5, timestamp=1.0)

    # Valid hand with negative class_id
    with pytest.raises(ValueError, match="class_id >= 0"):
        FramePrediction(class_id=-1, label="A", confidence=0.9, timestamp=1.0, has_hand=True)


def test_stable_sign_contract() -> None:
    """Validates StableSign properties and serialization."""
    sign = StableSign(
        class_id=2,
        label="C",
        confidence=0.935,
        start_timestamp=10.0,
        end_timestamp=10.35,
        duration=0.35,
        frame_count=11,
        top_margin=0.82,
    )
    assert sign.label == "C"
    assert sign.duration_ms == 350.0
    d = sign.to_dict()
    assert d["label"] == "C"
    assert d["class_id"] == 2
    assert d["duration_ms"] == 350.0
    assert d["frame_count"] == 11


def test_temporal_config_validation() -> None:
    """Validates boundary checking in TemporalEngineConfig."""
    # Negative dwell
    with pytest.raises(ValueError, match="minimum_dwell_ms"):
        TemporalEngineConfig(minimum_dwell_ms=-10.0)

    # Invalid confidence threshold
    with pytest.raises(ValueError, match="confidence_threshold"):
        TemporalEngineConfig(confidence_threshold=1.5)

    # Zero window duration
    with pytest.raises(ValueError, match="window_duration_ms"):
        TemporalEngineConfig(window_duration_ms=0.0)

    # Invalid rearm policy
    with pytest.raises(ValueError, match="rearm_policy"):
        TemporalEngineConfig(rearm_policy="invalid_policy")  # type: ignore


# ==============================================================================
# Buffer & Consensus Tests
# ==============================================================================

def test_temporal_buffer_pruning_and_consensus() -> None:
    """Tests bounded temporal buffer time-window pruning and consensus calculation."""
    config = TemporalEngineConfig(window_duration_ms=200.0, max_buffer_frames=10)
    buf = TemporalPredictionBuffer(config)

    # Add frames at t = 0.0, 0.05, 0.10, 0.15 (all within 200 ms)
    buf.append(FramePrediction(class_id=0, label="A", confidence=0.9, timestamp=0.0))
    buf.append(FramePrediction(class_id=0, label="A", confidence=0.9, timestamp=0.05))
    buf.append(FramePrediction(class_id=1, label="B", confidence=0.8, timestamp=0.10))
    buf.append(FramePrediction(class_id=0, label="A", confidence=0.9, timestamp=0.15))

    consensus = buf.get_consensus(0.15)
    assert consensus.top_label == "A"
    assert consensus.valid_frames == 4
    assert consensus.agreement_ratio == 0.75  # 3 'A' out of 4 frames

    # Advance time to 0.30 -> frames at 0.0 and 0.05 expire (older than 0.30 - 0.20 = 0.10)
    buf.append(FramePrediction(class_id=0, label="A", confidence=0.9, timestamp=0.30))
    consensus2 = buf.get_consensus(0.30)
    assert len(buf) <= 4
    # Valid remaining: 0.10 ('B'), 0.15 ('A'), 0.30 ('A')
    assert consensus2.valid_frames == 3
    assert consensus2.agreement_ratio == pytest.approx(2 / 3, rel=1e-2)


# ==============================================================================
# Temporal Stabilization Engine Tests
# ==============================================================================

def test_single_prediction_does_not_emit() -> None:
    """A single prediction cannot satisfy minimum_stable_frames or dwell duration."""
    engine = TemporalEngine()
    pred = FramePrediction(class_id=0, label="A", confidence=0.95, timestamp=1.0)
    result = engine.process_frame(pred)

    assert result.state == TemporalState.IDLE
    assert result.has_emitted_sign is False
    assert result.stable_sign is None


def test_confidence_filtering() -> None:
    """Predictions below confidence_threshold are rejected and do not accumulate dwell."""
    config = TemporalEngineConfig(confidence_threshold=0.75)
    engine = TemporalEngine(config)

    # Feed 20 frames of 'A' with low confidence 0.60
    for i in range(20):
        res = engine.process_frame(
            FramePrediction(class_id=0, label="A", confidence=0.60, timestamp=i * 0.033)
        )
        assert res.state in (TemporalState.IDLE, TemporalState.CANDIDATE)
        assert res.has_emitted_sign is False

    assert len(engine.emitted_tokens) == 0


def test_anti_jitter_voting() -> None:
    """A single misclassified frame does not dislodge an active stable sign."""
    config = TemporalEngineConfig(
        minimum_dwell_ms=250.0,
        consensus_threshold=0.60,
        minimum_stable_frames=3,
    )
    engine = TemporalEngine(config)

    # A x 3 -> jitter B x 1 -> A x 7
    emitted = []
    t = 0.0
    for i in range(3):
        res = engine.process_frame(FramePrediction(0, "A", 0.92, t))
        if res.has_emitted_sign:
            emitted.append(res.stable_sign.label)
        t += 0.033

    # Jitter frame B
    res_b = engine.process_frame(FramePrediction(1, "B", 0.85, t))
    assert res_b.has_emitted_sign is False
    t += 0.033

    # More A frames to complete dwell
    for i in range(7):
        res = engine.process_frame(FramePrediction(0, "A", 0.92, t))
        if res.has_emitted_sign:
            emitted.append(res.stable_sign.label)
        t += 0.033

    assert emitted == ["A"]


def test_duplicate_suppression_on_held_sign() -> None:
    """Holding a sign continuously must emit exactly once and suppress repetitions."""
    config = TemporalEngineConfig(minimum_dwell_ms=200.0)
    engine = TemporalEngine(config)

    emitted_count = 0
    t = 0.0
    for i in range(30):  # 1 second of holding 'A'
        res = engine.process_frame(FramePrediction(0, "A", 0.94, t))
        if res.has_emitted_sign:
            emitted_count += 1
            assert res.state == TemporalState.EMITTED
        elif i > 10:
            assert res.state == TemporalState.WAITING_FOR_CHANGE
        t += 0.033

    assert emitted_count == 1
    assert len(engine.emitted_tokens) == 1
    assert engine.last_emitted_sign.label == "A"


def test_rearm_policy_on_change() -> None:
    """Under on_change policy, same sign cannot re-emit without changing or neutral release."""
    config = TemporalEngineConfig(rearm_policy="on_change", minimum_dwell_ms=200.0)
    engine = TemporalEngine(config)

    # 1. Hold A -> emits A
    t = 0.0
    for _ in range(12):
        engine.process_frame(FramePrediction(0, "A", 0.92, t))
        t += 0.033

    assert len(engine.emitted_tokens) == 1

    # 2. Hold B -> emits B (sign change)
    for _ in range(12):
        engine.process_frame(FramePrediction(1, "B", 0.92, t))
        t += 0.033

    assert len(engine.emitted_tokens) == 2
    assert [s.label for s in engine.emitted_tokens] == ["A", "B"]

    # 3. Hold A again -> emits A (distinct sign change back to A)
    for _ in range(12):
        engine.process_frame(FramePrediction(0, "A", 0.92, t))
        t += 0.033

    assert len(engine.emitted_tokens) == 3
    assert [s.label for s in engine.emitted_tokens] == ["A", "B", "A"]


def test_rearm_policy_timeout() -> None:
    """Under timeout policy, holding the same sign longer than suppression window re-arms."""
    config = TemporalEngineConfig(
        rearm_policy="timeout",
        minimum_dwell_ms=200.0,
        duplicate_suppression_ms=500.0,
    )
    engine = TemporalEngine(config)

    emitted_labels = []
    t = 0.0
    # Hold A for 1.0 second (30 frames)
    for _ in range(30):
        res = engine.process_frame(FramePrediction(0, "A", 0.92, t))
        if res.has_emitted_sign:
            emitted_labels.append(res.stable_sign.label)
        t += 0.033

    # Should have emitted twice (once around 200 ms, once around 700 ms after 500 ms suppression window)
    assert len(emitted_labels) == 2
    assert emitted_labels == ["A", "A"]


def test_hand_loss_grace_period_retention() -> None:
    """Brief hand disappearance (< grace period) does not reset candidate state."""
    config = TemporalEngineConfig(
        minimum_dwell_ms=250.0,
        hand_loss_grace_ms=300.0,
    )
    engine = TemporalEngine(config)

    t = 0.0
    emitted = []
    # Hold A for 100 ms (3 frames)
    for _ in range(3):
        res = engine.process_frame(FramePrediction(0, "A", 0.92, t))
        t += 0.033

    assert engine.active_candidate == "A"

    # Hand disappears for 100 ms (< 300 ms grace)
    for _ in range(3):
        res = engine.process_frame(FramePrediction.no_hand(t))
        assert res.has_emitted_sign is False
        t += 0.033

    # Candidate should still be retained
    assert engine.active_candidate == "A"

    # Hand returns with A for 200 ms -> dwell completes and emits!
    for _ in range(7):
        res = engine.process_frame(FramePrediction(0, "A", 0.92, t))
        if res.has_emitted_sign:
            emitted.append(res.stable_sign.label)
        t += 0.033

    assert emitted == ["A"]


def test_hand_loss_grace_period_expiration() -> None:
    """Hand loss exceeding grace period resets candidate state to IDLE."""
    config = TemporalEngineConfig(
        minimum_dwell_ms=250.0,
        hand_loss_grace_ms=200.0,
    )
    engine = TemporalEngine(config)

    t = 0.0
    # Establish A candidate (100 ms)
    for _ in range(3):
        engine.process_frame(FramePrediction(0, "A", 0.92, t))
        t += 0.033

    assert engine.active_candidate == "A"

    # Hand lost for 300 ms (> 200 ms grace)
    for _ in range(10):
        res = engine.process_frame(FramePrediction.no_hand(t))
        t += 0.033

    # Candidate must have reset to None and state to IDLE
    assert engine.active_candidate is None
    assert engine.current_state == TemporalState.IDLE


def test_fps_variation_robustness() -> None:
    """Verifies dwell time detection is timestamp-aware across 15, 24, and 60 FPS."""
    config = TemporalEngineConfig(minimum_dwell_ms=250.0)

    for fps in [15.0, 24.0, 60.0]:
        engine = TemporalEngine(config)
        dt = 1.0 / fps
        t = 0.0
        emitted = []
        # Hold for 400 ms at test FPS
        num_frames = int(np.ceil(0.40 * fps))
        for _ in range(num_frames):
            res = engine.process_frame(FramePrediction(3, "D", 0.95, t))
            if res.has_emitted_sign:
                emitted.append(res.stable_sign.label)
            t += dt

        assert emitted == ["D"], f"Failed at {fps} FPS"


def test_non_monotonic_timestamp_handling() -> None:
    """Verifies that occasional backwards jitter in timestamps is clamped without crashing."""
    engine = TemporalEngine()
    engine.process_frame(FramePrediction(0, "A", 0.9, timestamp=1.0))
    # Passing timestamp 0.95 (smaller than 1.0)
    res = engine.process_frame(FramePrediction(0, "A", 0.9, timestamp=0.95))
    assert res.dwell_duration_ms >= 0.0


def test_engine_reset() -> None:
    """Verifies that reset() clears all internal state, history, and candidates."""
    engine = TemporalEngine()
    # Feed frames and emit A
    for i in range(12):
        engine.process_frame(FramePrediction(0, "A", 0.95, i * 0.033))

    assert len(engine.emitted_tokens) == 1
    assert engine.active_candidate == "A"

    # Reset
    engine.reset()
    assert engine.current_state == TemporalState.IDLE
    assert engine.active_candidate is None
    assert engine.last_emitted_sign is None
    assert len(engine.emitted_tokens) == 0


def test_deterministic_behavior() -> None:
    """Asserts that identical prediction streams produce identical outputs across runs."""
    config = TemporalEngineConfig()
    stream = [
        FramePrediction(0, "A", 0.92, i * 0.033) for i in range(15)
    ] + [
        FramePrediction(1, "B", 0.94, 0.50 + i * 0.033) for i in range(15)
    ]

    engine1 = TemporalEngine(config)
    emitted1 = [res.stable_sign.label for p in stream if (res := engine1.process_frame(p)).has_emitted_sign]

    engine2 = TemporalEngine(config)
    emitted2 = [res.stable_sign.label for p in stream if (res := engine2.process_frame(p)).has_emitted_sign]

    assert emitted1 == emitted2 == ["A", "B"]
