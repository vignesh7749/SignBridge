"""Stage 7 Temporal Recognition & Stabilization Engine.

Orchestrates frame-level prediction ingestion, confidence filtering, rolling-window
consensus, dwell-time accumulation, duplicate suppression, and hand-loss grace handling
via a deterministic finite state machine.
"""

from collections import deque
import time
from typing import Optional, Sequence
import numpy as np

from signbridge.core.logger import get_logger
from signbridge.temporal.buffer import TemporalPredictionBuffer
from signbridge.temporal.config import TemporalEngineConfig
from signbridge.temporal.contracts import (
    FramePrediction,
    StableSign,
    TemporalFrameResult,
    TemporalState,
)

logger = get_logger("temporal.engine")


class TemporalEngine:
    """Temporal stabilization and discrete sign recognition engine."""

    def __init__(self, config: Optional[TemporalEngineConfig] = None) -> None:
        self.config = config or TemporalEngineConfig()
        self._buffer = TemporalPredictionBuffer(self.config)

        # FSM and candidate tracking
        self._state: TemporalState = TemporalState.IDLE
        self._active_candidate: Optional[str] = None
        self._active_candidate_id: Optional[int] = None
        self._candidate_first_seen: float = 0.0
        self._candidate_frame_count: int = 0
        self._candidate_confidences: list[float] = []
        self._candidate_margins: list[float] = []

        # Duplicate suppression and re-arming state
        self._last_emitted_label: Optional[str] = None
        self._last_emitted_time: float = 0.0
        self._is_rearmed: bool = True

        # Hand loss and neutral tracking
        self._last_hand_seen_time: Optional[float] = None
        self._hand_lost_since: Optional[float] = None
        self._neutral_since: Optional[float] = None

        # Monotonic time tracking
        self._last_processed_timestamp: float = -1.0

        # Emitted history for inspection/debugging
        self._emitted_history: deque[StableSign] = deque(maxlen=100)

    @property
    def current_state(self) -> TemporalState:
        """Current lifecycle state of the temporal engine."""
        return self._state

    @property
    def active_candidate(self) -> Optional[str]:
        """Label of the candidate currently accumulating stability, or None."""
        return self._active_candidate

    @property
    def last_emitted_sign(self) -> Optional[StableSign]:
        """Most recently emitted StableSign token, or None."""
        return self._emitted_history[-1] if self._emitted_history else None

    @property
    def emitted_tokens(self) -> Sequence[StableSign]:
        """Read-only sequence of recently emitted stable sign tokens."""
        return list(self._emitted_history)

    def process_frame(self, prediction: FramePrediction) -> TemporalFrameResult:
        """Processes a single frame prediction through the temporal stabilization pipeline.

        Args:
            prediction: FramePrediction containing label, confidence, timestamp, and hand state.

        Returns:
            TemporalFrameResult containing state, active candidate, dwell time, and any emitted StableSign.

        Raises:
            ValueError: If prediction contains non-monotonic negative timestamps or non-finite values.
        """
        start_eval_time = time.perf_counter()

        # Handle non-monotonic timestamps gracefully by clamping to last seen
        current_ts = prediction.timestamp
        if self._last_processed_timestamp >= 0.0 and current_ts < self._last_processed_timestamp:
            current_ts = self._last_processed_timestamp
        self._last_processed_timestamp = current_ts

        # 1. Update hand presence telemetry & grace period tracking
        if prediction.has_hand:
            self._last_hand_seen_time = current_ts
            self._hand_lost_since = None
        else:
            if self._hand_lost_since is None:
                self._hand_lost_since = current_ts

        # 2. Check neutral / no-hand dwell for re-arming
        is_neutral = (
            not prediction.has_hand
            or not prediction.label
            or prediction.confidence < self.config.confidence_threshold
        )
        if is_neutral:
            if self._neutral_since is None:
                self._neutral_since = current_ts
            neutral_elapsed_ms = (current_ts - self._neutral_since) * 1000.0
            if neutral_elapsed_ms >= self.config.neutral_dwell_ms:
                self._is_rearmed = True
        else:
            self._neutral_since = None

        # 3. Hand loss expiration check
        if self._hand_lost_since is not None:
            hand_lost_duration_ms = (current_ts - self._hand_lost_since) * 1000.0
            if hand_lost_duration_ms > self.config.hand_loss_grace_ms:
                # Grace period expired: clear history, drop active candidate, and reset to IDLE
                self._buffer.clear()
                self._reset_candidate()
                self._state = TemporalState.IDLE
                self._is_rearmed = True

        # 4. Ingest prediction into bounded rolling buffer
        self._buffer.append(prediction)

        # 5. Compute rolling consensus
        consensus = self._buffer.get_consensus(current_ts)

        # 6. Evaluate candidate stability against thresholds
        is_stable_consensus = (
            consensus.top_label is not None
            and consensus.agreement_ratio >= self.config.consensus_threshold
            and consensus.valid_frames >= self.config.minimum_stable_frames
        )

        emitted_token: Optional[StableSign] = None
        current_dwell_ms = 0.0
        stability_ratio = consensus.agreement_ratio if consensus.top_label else 0.0

        can_emit_now = (
            prediction.has_hand
            and consensus.top_label is not None
            and prediction.label == consensus.top_label
            and prediction.confidence >= self.config.confidence_threshold
        )

        if is_stable_consensus:
            winning_label = consensus.top_label
            winning_class_id = consensus.top_class_id
            assert winning_label is not None and winning_class_id is not None

            # Check if this matches active candidate
            if winning_label == self._active_candidate:
                # Accumulate dwell duration
                current_dwell_ms = (current_ts - self._candidate_first_seen) * 1000.0
                self._candidate_frame_count += 1
                self._candidate_confidences.append(consensus.mean_confidence)
                self._candidate_margins.append(consensus.mean_top_margin)

                # Check duplicate suppression conditions
                is_duplicate = (winning_label == self._last_emitted_label) and not self._is_rearmed
                if is_duplicate and self.config.rearm_policy == "timeout":
                    suppression_elapsed_ms = (current_ts - self._last_emitted_time) * 1000.0
                    if suppression_elapsed_ms >= self.config.duplicate_suppression_ms:
                        self._is_rearmed = True
                        is_duplicate = False

                if is_duplicate:
                    self._state = TemporalState.WAITING_FOR_CHANGE
                else:
                    if current_dwell_ms >= self.config.minimum_dwell_ms and can_emit_now:
                        self._state = TemporalState.STABLE
                        mean_conf = float(np.mean(self._candidate_confidences))
                        mean_margin = (
                            float(np.mean(self._candidate_margins))
                            if self._candidate_margins
                            else 0.0
                        )

                        emitted_token = StableSign(
                            class_id=winning_class_id,
                            label=winning_label,
                            confidence=mean_conf,
                            start_timestamp=self._candidate_first_seen,
                            end_timestamp=current_ts,
                            duration=current_dwell_ms / 1000.0,
                            frame_count=self._candidate_frame_count,
                            top_margin=mean_margin,
                        )

                        self._emitted_history.append(emitted_token)
                        self._last_emitted_label = winning_label
                        self._last_emitted_time = current_ts
                        self._is_rearmed = False
                        self._state = TemporalState.EMITTED
                        logger.debug("Emitted StableSign token: %s (dwell: %.1f ms)", winning_label, current_dwell_ms)
                    else:
                        self._state = TemporalState.CANDIDATE
            else:
                # Sign candidate changed
                if not prediction.has_hand or not prediction.label or prediction.confidence < self.config.confidence_threshold:
                    # Do not initiate a new candidate on an invalid or absent hand frame
                    pass
                else:
                    if winning_label != self._last_emitted_label:
                        self._is_rearmed = True

                    self._active_candidate = winning_label
                    self._active_candidate_id = winning_class_id
                    self._candidate_first_seen = consensus.earliest_timestamp
                    self._candidate_frame_count = consensus.valid_frames
                    self._candidate_confidences = [consensus.mean_confidence]
                    self._candidate_margins = [consensus.mean_top_margin]
                    current_dwell_ms = (current_ts - self._candidate_first_seen) * 1000.0

                    is_duplicate = (winning_label == self._last_emitted_label) and not self._is_rearmed
                    if is_duplicate:
                        self._state = TemporalState.WAITING_FOR_CHANGE
                    else:
                        if current_dwell_ms >= self.config.minimum_dwell_ms and can_emit_now:
                            self._state = TemporalState.STABLE
                            mean_conf = consensus.mean_confidence
                            mean_margin = consensus.mean_top_margin

                            emitted_token = StableSign(
                                class_id=winning_class_id,
                                label=winning_label,
                                confidence=mean_conf,
                                start_timestamp=self._candidate_first_seen,
                                end_timestamp=current_ts,
                                duration=current_dwell_ms / 1000.0,
                                frame_count=self._candidate_frame_count,
                                top_margin=mean_margin,
                            )

                            self._emitted_history.append(emitted_token)
                            self._last_emitted_label = winning_label
                            self._last_emitted_time = current_ts
                            self._is_rearmed = False
                            self._state = TemporalState.EMITTED
                            logger.debug("Emitted StableSign token: %s (dwell: %.1f ms)", winning_label, current_dwell_ms)
                        else:
                            self._state = TemporalState.CANDIDATE
        else:
            # Consensus not satisfied
            if (
                self._active_candidate is not None
                and self._hand_lost_since is not None
                and (current_ts - self._hand_lost_since) * 1000.0 <= self.config.hand_loss_grace_ms
            ):
                current_dwell_ms = (current_ts - self._candidate_first_seen) * 1000.0
                # Preserve candidate state during brief hand loss within grace period
            else:
                if self._state not in (TemporalState.WAITING_FOR_CHANGE, TemporalState.IDLE):
                    self._reset_candidate()
                    self._state = TemporalState.IDLE
                elif self._state == TemporalState.WAITING_FOR_CHANGE:
                    self._reset_candidate()

        # If token was emitted on this frame, subsequent query will report WAITING_FOR_CHANGE
        result_state = self._state

        # Calculate processing latency
        elapsed_ms = (time.perf_counter() - start_eval_time) * 1000.0

        return TemporalFrameResult(
            state=result_state,
            current_candidate=self._active_candidate,
            candidate_confidence=consensus.mean_confidence if self._active_candidate else 0.0,
            dwell_duration_ms=round(current_dwell_ms, 2),
            stability_ratio=round(stability_ratio, 4),
            is_hand_present=prediction.has_hand,
            stable_sign=emitted_token,
            processing_latency_ms=round(elapsed_ms, 4),
            frame_prediction=prediction,
        )

    def _reset_candidate(self) -> None:
        """Resets active candidate tracking counters without clearing emitted history."""
        self._active_candidate = None
        self._active_candidate_id = None
        self._candidate_first_seen = 0.0
        self._candidate_frame_count = 0
        self._candidate_confidences.clear()
        self._candidate_margins.clear()

    def reset(self) -> None:
        """Completely resets all internal engine states, buffers, and suppression tracking."""
        self._buffer.clear()
        self._reset_candidate()
        self._state = TemporalState.IDLE
        self._last_emitted_label = None
        self._last_emitted_time = 0.0
        self._is_rearmed = True
        self._last_hand_seen_time = None
        self._hand_lost_since = None
        self._neutral_since = None
        self._last_processed_timestamp = -1.0
        self._emitted_history.clear()
        logger.debug("TemporalEngine completely reset.")
