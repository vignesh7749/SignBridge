"""Module 3: Optical Frame Processing, Skeletal Rendering, and HUD Overlays."""

from typing import Optional, Sequence
import cv2
import numpy as np

from signbridge.core.config import VisionConfig
from signbridge.vision.detector import HandDetectionResult
from signbridge.vision.landmarks import HAND_CONNECTIONS, DetectedHand


class FrameProcessor:
    """Performs optical frame conditioning, skeletal landmark rendering, and UI HUD overlays."""

    def __init__(self, config: Optional[VisionConfig] = None) -> None:
        self.config = config or VisionConfig()

    def to_rgb(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Converts BGR image to RGB for detector inference."""
        return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    def draw_hand_skeleton(
        self,
        frame: np.ndarray,
        hand: DetectedHand,
        draw_bbox: bool = True,
        draw_keypoint_indices: bool = False,
    ) -> np.ndarray:
        """Renders 21 skeletal landmarks, bone connections, bounding box, and handedness label."""
        annotated = frame  # Modifies in-place or returns reference

        # Color scheme (BGR)
        is_right = hand.handedness.lower() == "right"
        bone_color = (0, 220, 255) if is_right else (255, 180, 0)  # Yellow/Gold for Right, Blue/Cyan for Left
        joint_color = (0, 255, 128)  # Mint green
        tip_color = (0, 0, 255)  # Red for fingertips
        bbox_color = (180, 180, 180)

        # 1. Draw Skeletal Bone Connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = (hand.landmarks[start_idx].x_px, hand.landmarks[start_idx].y_px)
            pt2 = (hand.landmarks[end_idx].x_px, hand.landmarks[end_idx].y_px)
            cv2.line(annotated, pt1, pt2, bone_color, 2, cv2.LINE_AA)

        # 2. Draw Keypoint Joints
        fingertip_indices = {4, 8, 12, 16, 20}
        for lm in hand.landmarks:
            pt = (lm.x_px, lm.y_px)
            if lm.index in fingertip_indices:
                cv2.circle(annotated, pt, 6, tip_color, -1, cv2.LINE_AA)
                cv2.circle(annotated, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(annotated, pt, 4, joint_color, -1, cv2.LINE_AA)
                cv2.circle(annotated, pt, 5, (0, 0, 0), 1, cv2.LINE_AA)

            if draw_keypoint_indices:
                cv2.putText(
                    annotated,
                    str(lm.index),
                    (lm.x_px + 6, lm.y_px - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 3. Draw Bounding Box & Hand Label
        if draw_bbox:
            bb = hand.bbox
            cv2.rectangle(annotated, (bb.x_min, bb.y_min), (bb.x_max, bb.y_max), bbox_color, 1, cv2.LINE_AA)

            # Label badge
            label = f"{hand.handedness} ({hand.handedness_confidence * 100:.0f}%)"
            badge_y = max(20, bb.y_min - 8)
            cv2.putText(
                annotated,
                label,
                (bb.x_min, badge_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                bone_color,
                2,
                cv2.LINE_AA,
            )

        return annotated

    def draw_all_hands(
        self,
        frame: np.ndarray,
        detection_result: HandDetectionResult,
        draw_bbox: bool = True,
        draw_keypoint_indices: bool = False,
    ) -> np.ndarray:
        """Renders all detected hands in the frame."""
        annotated = frame
        for hand in detection_result.hands:
            self.draw_hand_skeleton(
                annotated,
                hand,
                draw_bbox=draw_bbox,
                draw_keypoint_indices=draw_keypoint_indices,
            )
        return annotated

    def draw_stage3_hud(
        self,
        frame: np.ndarray,
        detection_result: HandDetectionResult,
        camera_fps: float,
        camera_status: str = "RUNNING",
        is_mirrored: bool = True,
    ) -> np.ndarray:
        """Renders comprehensive Heads-Up Display (HUD) for Stage 3."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw semi-transparent header bar
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, h - 35), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

        # Draw hand skeletons over frame
        self.draw_all_hands(annotated, detection_result)

        # Line 1: Title & Camera
        title_txt = f"SignBridge [Stage 3: Hand Tracking] | Camera: {camera_status}"
        cv2.putText(annotated, title_txt, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Line 2: Telemetry Metrics
        hands_count = detection_result.hand_count
        hand_info_str = "None"
        if hands_count > 0:
            hand_info_str = ", ".join(
                f"{h.handedness} ({h.score * 100:.0f}%)" for h in detection_result.hands
            )

        metrics_txt = (
            f"FPS: {camera_fps:.1f} | "
            f"Hands: {hands_count} | "
            f"Detected: [{hand_info_str}] | "
            f"Latency: {detection_result.processing_time_ms:.1f}ms | "
            f"Mirror: {'ON' if is_mirrored else 'OFF'}"
        )
        cv2.putText(annotated, metrics_txt, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2, cv2.LINE_AA)

        # Bottom bar instructions
        bottom_info = "Keys: [P] Pause | [M] Toggle Mirror | [I] Toggle Landmark Indices | [S] Snapshot | [Q/ESC] Quit"
        cv2.putText(annotated, bottom_info, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return annotated
