"""Temporal Recognition & Anti-Jitter Subsystem.

Provides frame-level prediction smoothing, rolling-window consensus,
dwell-time detection, duplicate suppression, and stateful sign token stabilization.
"""

from signbridge.temporal.buffer import ConsensusResult, TemporalPredictionBuffer
from signbridge.temporal.config import TemporalEngineConfig
from signbridge.temporal.contracts import (
    FramePrediction,
    StableSign,
    TemporalFrameResult,
    TemporalState,
)
from signbridge.temporal.engine import TemporalEngine
from signbridge.temporal.smoother import StabilizedSign, TemporalSmoother

__all__ = [
    # Legacy Stage 0 compatibility
    "StabilizedSign",
    "TemporalSmoother",
    # Stage 7 Production Temporal Subsystem
    "ConsensusResult",
    "FramePrediction",
    "StableSign",
    "TemporalEngine",
    "TemporalEngineConfig",
    "TemporalFrameResult",
    "TemporalPredictionBuffer",
    "TemporalState",
]
