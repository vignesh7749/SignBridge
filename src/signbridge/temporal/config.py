"""Centralized Configuration for Stage 7 Temporal Recognition.

Provides type-safe, validated configuration for confidence filtering,
temporal buffering, consensus voting, dwell-time detection, duplicate suppression,
and hand-loss grace periods.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TemporalEngineConfig:
    """Configuration parameters for the Stage 7 Temporal Engine.

    Attributes:
        confidence_threshold: Minimum classifier posterior confidence required to accept a prediction.
            Initial value grounded in Stage 6 measured distribution (correct mean 87%, errors 52%).
        window_duration_ms: Time window in milliseconds over which rolling predictions are evaluated.
        max_buffer_frames: Maximum number of prediction frames kept in the rolling buffer.
        consensus_threshold: Minimum agreement ratio in the rolling window [0.0, 1.0] for stability.
        minimum_stable_frames: Minimum frame count required before candidate can reach STABLE state.
        minimum_dwell_ms: Minimum duration in milliseconds a candidate must remain stable before emission.
        hand_loss_grace_ms: Maximum duration in milliseconds to tolerate temporary hand tracking loss.
        duplicate_suppression_ms: Time window in milliseconds to suppress identical sign emissions (if timeout enabled).
        rearm_policy: Strategy to re-arm emission: "on_change", "timeout", or "neutral_release".
        neutral_dwell_ms: Duration in milliseconds of neutral/no-hand needed to rearm under "neutral_release".
    """

    confidence_threshold: float = 0.70
    window_duration_ms: float = 400.0
    max_buffer_frames: int = 60
    consensus_threshold: float = 0.60
    minimum_stable_frames: int = 3
    minimum_dwell_ms: float = 250.0
    hand_loss_grace_ms: float = 250.0
    duplicate_suppression_ms: float = 1200.0
    rearm_policy: Literal["on_change", "timeout", "neutral_release"] = "on_change"
    neutral_dwell_ms: float = 150.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], got {self.confidence_threshold}"
            )
        if self.window_duration_ms <= 0.0:
            raise ValueError(f"window_duration_ms must be positive, got {self.window_duration_ms}")
        if self.max_buffer_frames <= 0:
            raise ValueError(f"max_buffer_frames must be positive, got {self.max_buffer_frames}")
        if not (0.0 < self.consensus_threshold <= 1.0):
            raise ValueError(
                f"consensus_threshold must be in (0.0, 1.0], got {self.consensus_threshold}"
            )
        if self.minimum_stable_frames < 1:
            raise ValueError(
                f"minimum_stable_frames must be >= 1, got {self.minimum_stable_frames}"
            )
        if self.minimum_dwell_ms < 0.0:
            raise ValueError(f"minimum_dwell_ms must be non-negative, got {self.minimum_dwell_ms}")
        if self.hand_loss_grace_ms < 0.0:
            raise ValueError(
                f"hand_loss_grace_ms must be non-negative, got {self.hand_loss_grace_ms}"
            )
        if self.duplicate_suppression_ms < 0.0:
            raise ValueError(
                f"duplicate_suppression_ms must be non-negative, got {self.duplicate_suppression_ms}"
            )
        if self.rearm_policy not in ("on_change", "timeout", "neutral_release"):
            raise ValueError(
                f"Invalid rearm_policy '{self.rearm_policy}'. Must be 'on_change', 'timeout', or 'neutral_release'."
            )
        if self.neutral_dwell_ms < 0.0:
            raise ValueError(f"neutral_dwell_ms must be non-negative, got {self.neutral_dwell_ms}")
