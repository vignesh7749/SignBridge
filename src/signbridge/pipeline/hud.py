"""Real-Time Developer HUD and Adaptive Skeletal Overlay for SignBridge."""

import time
from typing import Optional, Sequence
import cv2
import numpy as np

from signbridge.pipeline.state import PipelineRuntimeState
from signbridge.temporal.contracts import TemporalState
from signbridge.vision.landmarks import HAND_CONNECTIONS, BoundingBox, DetectedHand


class PipelineHUD:
    """Renders developer HUD, adaptive hand skeletal feedback, and translation banners."""

    def __init__(self) -> None:
        self._last_emitted_token: Optional[str] = None
        self._token_banner_until: float = 0.0
        self._last_completed_word: Optional[str] = None
        self._word_banner_until: float = 0.0
        self._prev_completed_count: int = 0

    def update_events(self, state: PipelineRuntimeState) -> None:
        """Updates internal notification timers based on new runtime state."""
        now = time.time()
        # Token emission notification
        if state.stable_token and state.stable_token != self._last_emitted_token:
            self._last_emitted_token = state.stable_token
            self._token_banner_until = now + 1.2

        # Word completed notification
        current_completed_count = len(state.completed_words)
        if current_completed_count > self._prev_completed_count and state.completed_words:
            self._last_completed_word = state.completed_words[-1]
            self._word_banner_until = now + 1.8
        self._prev_completed_count = current_completed_count

    def reset(self) -> None:
        """Clears transient notification timers and event banners."""
        self._last_emitted_token = None
        self._token_banner_until = 0.0
        self._last_completed_word = None
        self._word_banner_until = 0.0
        self._prev_completed_count = 0

    def render(
        self,
        frame: np.ndarray,
        state: PipelineRuntimeState,
        primary_hand: Optional[DetectedHand] = None,
        draw_skeleton: bool = True,
        draw_hud: bool = True,
        draw_bbox: bool = True,
        secondary_hand: Optional[DetectedHand] = None,
        all_hands: Optional[Sequence[DetectedHand]] = None,
    ) -> np.ndarray:
        """Composites adaptive skeleton and HUD telemetry onto the video frame."""
        self.update_events(state)
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Render Hand Skeletons (Multi-hand visualization)
        if draw_skeleton:
            # Render secondary hand first so primary sits clearly on top
            if secondary_hand is not None:
                self._draw_secondary_skeleton(annotated, secondary_hand, draw_bbox=draw_bbox)
            elif all_hands and len(all_hands) > 1 and primary_hand is not None:
                for h_item in all_hands:
                    if h_item is not primary_hand:
                        self._draw_secondary_skeleton(annotated, h_item, draw_bbox=draw_bbox)
                        break

            if primary_hand is not None:
                self._draw_adaptive_skeleton(annotated, primary_hand, state, draw_bbox=draw_bbox)

        # 2. Render Developer HUD only if requested
        if draw_hud:
            self._draw_hud_panel(annotated, state, w, h)

        # 3. Render Pause Overlay
        if state.is_paused:
            self._draw_pause_overlay(annotated, w, h)

        return annotated

    def _draw_adaptive_skeleton(
        self,
        frame: np.ndarray,
        hand: DetectedHand,
        state: PipelineRuntimeState,
        draw_bbox: bool,
    ) -> None:
        """Renders 21 landmarks and connections with color adapted to recognition confidence and state."""
        # Visual State determination
        is_stable = state.temporal_state in (TemporalState.STABLE, TemporalState.EMITTED)
        is_low_conf = state.prediction_confidence < 0.60

        if is_stable:
            # Vibrant emerald highlight
            bone_color = (0, 255, 120)
            joint_color = (0, 255, 200)
            tip_color = (0, 255, 255)
            line_thickness = 3
        elif is_low_conf:
            # Amber / warning muted tone
            bone_color = (0, 160, 255)
            joint_color = (0, 200, 255)
            tip_color = (0, 120, 255)
            line_thickness = 2
        else:
            # Normal detection: Gold for Right, Cyan for Left
            is_right = (hand.handedness or "").lower() == "right"
            bone_color = (0, 220, 255) if is_right else (255, 180, 0)
            joint_color = (0, 255, 128)
            tip_color = (0, 0, 255)
            line_thickness = 2

        # Draw bone connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = (hand.landmarks[start_idx].x_px, hand.landmarks[start_idx].y_px)
            pt2 = (hand.landmarks[end_idx].x_px, hand.landmarks[end_idx].y_px)
            cv2.line(frame, pt1, pt2, bone_color, line_thickness, cv2.LINE_AA)

        # Draw joints and fingertips
        fingertip_indices = {4, 8, 12, 16, 20}
        for lm in hand.landmarks:
            pt = (lm.x_px, lm.y_px)
            if lm.index in fingertip_indices:
                cv2.circle(frame, pt, 6, tip_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 4, joint_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 5, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw Bounding Box & Label
        if draw_bbox:
            bb = hand.bbox
            cv2.rectangle(frame, (bb.x_min, bb.y_min), (bb.x_max, bb.y_max), (160, 160, 160), 1, cv2.LINE_AA)
            label = f"Primary {hand.handedness} ({hand.handedness_confidence * 100:.0f}%)"
            badge_y = max(20, bb.y_min - 8)
            cv2.putText(frame, label, (bb.x_min, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, bone_color, 1, cv2.LINE_AA)

    def _draw_secondary_skeleton(
        self,
        frame: np.ndarray,
        hand: DetectedHand,
        draw_bbox: bool,
    ) -> None:
        """Renders secondary hand skeleton with distinct warm amber styling."""
        # Distinct warm amber tone for observed secondary hand
        bone_color = (0, 165, 255)  # Amber / Orange BGR
        joint_color = (0, 205, 255)
        tip_color = (0, 245, 255)
        line_thickness = 2

        # Draw bone connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = (hand.landmarks[start_idx].x_px, hand.landmarks[start_idx].y_px)
            pt2 = (hand.landmarks[end_idx].x_px, hand.landmarks[end_idx].y_px)
            cv2.line(frame, pt1, pt2, bone_color, line_thickness, cv2.LINE_AA)

        # Draw joints and fingertips
        fingertip_indices = {4, 8, 12, 16, 20}
        for lm in hand.landmarks:
            pt = (lm.x_px, lm.y_px)
            if lm.index in fingertip_indices:
                cv2.circle(frame, pt, 5, tip_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 3, joint_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 4, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw Bounding Box & Secondary Label
        if draw_bbox:
            bb = hand.bbox
            cv2.rectangle(frame, (bb.x_min, bb.y_min), (bb.x_max, bb.y_max), (0, 165, 255), 1, cv2.LINE_AA)
            label = f"Secondary {hand.handedness} ({hand.handedness_confidence * 100:.0f}%)"
            badge_y = max(20, bb.y_min - 8)
            cv2.putText(frame, label, (bb.x_min, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bone_color, 1, cv2.LINE_AA)

    def _draw_hud_panel(self, frame: np.ndarray, state: PipelineRuntimeState, w: int, h: int) -> None:
        """Telemetry and HUD elements are rendered dynamically by the Stage 11 dashboard layout."""
        pass

    def _draw_pause_overlay(self, frame: np.ndarray, w: int, h: int) -> None:
        """Renders clear pause overlay across center of screen."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (w // 2 - 260, h // 2 - 50), (w // 2 + 260, h // 2 + 50), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (w // 2 - 260, h // 2 - 50), (w // 2 + 260, h // 2 + 50), (0, 240, 255), 2)
        cv2.putText(frame, "PAUSED", (w // 2 - 80, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 240, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'P' to resume processing", (w // 2 - 160, h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
